# TP=3 tuning sweep — DeepSeek-V4-Flash on 3× DGX Spark

Incremental, controlled measurements on the 3-node TP=3 deployment. Every row is the
same harness, checkpoint, prompt, `temperature=0`, 256 output tokens, on otherwise
identical software.

**Workload this is tuned for:** a single user with several concurrent conversations,
mostly well under the full context window. Not a many-tenant serving deployment.

---

## Baseline chain — how we got here

| Config | Transport | decode tok/s | Note |
|---|---|---:|---|
| TP=2 (2 nodes, production) | RoCE | 48.23 | the reference every candidate is measured against |
| EP=3 (3 nodes) | Socket | 19–20 | loses the B12X kernel — see EP3 doc |
| TP=3, same-port cabling | **Socket/TCP** | 24.59 | transport-bound, **not** a valid TP=3 result |
| TP=3, same-port cabling | **RoCE** | **57.73** | `NCCL_IB_SUBNET_AWARE_ROUTING=1` was the whole fix |
| TP=3, NVIDIA cross-connected | RoCE | **53.95** | re-cabled to the supported layout; ~6.5% slower |

The 24.59 figure is frequently misquoted as "TP=3 is half the speed." It is a TCP
fallback measurement. On RDMA the same build serves 53.9–57.7 tok/s.

### On the cabling

Re-cabling to NVIDIA's cross-connected reference layout cost ~6.5% (57.73 → 53.95,
reproduced twice: 5-rep and 7-rep runs both median 53.95). We kept the NVIDIA layout
anyway — it is the supported topology, it lets index-based NCCL device pairing work
without depending on subnet-aware selection, and standardising now avoids rediscovering
this on the next project. **Do not re-cable expecting a speedup.**

---

## Tuning sweep

Starting point: `MTP_NUM_TOKENS=5`, `MAX_NUM_SEQS=16`,
`GPU_MEMORY_UTILIZATION=0.85`, `MAX_MODEL_LEN=460800`.

| # | MTP | max_num_seqs | decode tok/s (median) | acceptance | KV tokens | graph capture | Notes |
|--:|---:|---:|---:|---:|---:|---|---|
| 0 | 5 | 16 | **53.95** · 53.95 | 3.21 | 3,606,027 | 11 s / 1.89 GiB | baseline; two runs agreed exactly |
| 1 | 4 | 8 | **56.63** · 55.68 | 3.02–3.13 | 3,592,058 | 9 s / 0.99 GiB | **+4.2% avg**; half the graph memory |
| 2 | 3 | 8 | *pending* | | | | testing whether draft 4 also earns its keep |

Two independent 7-rep runs per configuration — run-to-run spread is material on this
cluster (individual reps range ~41–61 tok/s), so single runs are not trustworthy.
Medians are quoted; steady-state reps cluster tighter than the full range suggests.

### Row 1 detail (MTP=4, seqs=8)

```
rep1 41.43  rep2 52.08  rep3 52.95  rep4 57.82  rep5 57.16  rep6 57.04  rep7 56.63
median 56.63   (second run: median 55.68, range 50.41-60.98)
```

Reps 4–7 sit at 56.6–57.8; the low early reps are warm-up. Correctness verified
(`391`, `finish_reason: stop`) and a coding spot check returned the idiomatic answer:

```python
def reverse_words(s):
    return ' '.join(s.split()[::-1])
```

**Why acceptance is the key metric.** Dropping the draft length 5 → 4 moved acceptance
only 3.21 → ~3.08. Draft token #5 was contributing almost nothing while costing a full
draft-model forward pass on every step. That is the waste this sweep is removing, and it
predicts draft #4 may be similarly marginal — hence row 2.

---

## Why these two knobs

### MTP acceptance is the headline finding

Live metrics at `MTP_NUM_TOKENS=5`:

```
SpecDecoding metrics: Mean acceptance length: 3.21,
  Accepted throughput: 8.03 tokens/s, Drafted throughput: 18.17 tokens/s
```

**We draft 5 tokens per step and ~3.2 are accepted.** The rejected drafts are computed
and discarded, so roughly a third of draft compute is wasted. Lowering the draft length
toward the measured acceptance should reduce waste per step.

**Speculative decoding does not change output quality.** The draft model proposes; the
full model verifies every token and substitutes its own wherever the draft is wrong. The
result is mathematically identical to decoding without MTP. `MTP_NUM_TOKENS` is purely a
speed/compute trade — relevant here because this deployment is used heavily for coding,
where correctness is non-negotiable.

### max_num_seqs was sized for a different workload

`16` assumes 16 simultaneous full-length requests. For a single user it mostly costs:

* **CUDA graph capture size is derived** as `max_num_seqs × (MTP_NUM_TOKENS + 1)`.
  At `16 × 6 = 96` that measured 11 s and 1.89 GiB. At `8 × 5 = 40` it is far smaller.
* Fewer scheduler slots means better cache locality per sequence.

Upstream's 3-Spark repo reports the opposite direction (`max_num_seqs` 6 → 16 → 32
raising *aggregate* throughput 238 → 431 → 618 tok/s). That is a multi-tenant result:
it buys aggregate throughput at high concurrency, and explicitly does **not** improve
single-stream latency. Different workload, different optimum.

---

## Memory headroom

At `GPU_MEMORY_UTILIZATION=0.85`, each 121 GiB GB10 runs with only **2–4 GB free**:

```
sparkmain avail=2GB   spark1 avail=3GB   spark-sep avail=4GB
```

Raising utilisation would add KV capacity but is genuinely risky — an over-commit on
these machines does not OOM cleanly, it wedges the node past SSH recovery and needs a
power cycle. Not attempted.

## Context length

`MAX_MODEL_LEN=460800` (~345k words) against a model maximum of 1,048,576. It was
deliberately lowered from 1M: at 1M the memory profiler reserves worst-case activation
headroom and leaves almost nothing free under load, and the freed KV pool is what pays
for the concurrency slots. 460,800 comfortably covers the stated requirement (~500k).

## Untested levers

* **Second CX-7 card.** Each Spark has four RoCE devices; only two are configured
  (the others have no IP and MTU 1500). Multi-rail could add fabric bandwidth.
* **Alternative MoE backends** (`triton`, `deep_gemm`, `machete`). Upstream reports most
  failing or rejecting the device on SM121; worth a short probe, not an afternoon.
* **Multi-QP RDMA.** The 109 Gb/s per-cable figure is a single-QP `ib_write_bw` default,
  ~55% of line rate — a harness default, not a measured ceiling.
