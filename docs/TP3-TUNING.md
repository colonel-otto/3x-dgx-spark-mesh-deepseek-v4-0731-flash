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
| 0 | 5 | 16 | 53.95 · 53.95 | 3.21 | 3,606,027 | 11 s / 1.89 GiB | starting point |
| 1 | **4** | **8** | **56.63 · 55.68** | 3.02–3.13 | 3,592,058 | 9 s / 0.99 GiB | ⭐ **best: +4.2%**, half the graph memory |
| 2 | 3 | 8 | 40.19 · 39.57 | **2.73–2.83** | 3,603,348 | 10 s / 1.86 GiB | **−29%** — well past the knee |

**The optimum is bracketed: `MTP_NUM_TOKENS=4`.** Going 5→4 gained ~4%; going 4→3 lost
29%. Four independent 7-rep runs support the MTP=3 result (two by the operator, two by a
separate measuring agent, medians 40.94/39.73 and 40.19/39.57 — agreement within 2%).

### `VLLM_USE_BREAKABLE_CUDAGRAPH=0` — tested, no measurable gain here

vLLM **auto-enables** breakable CUDA graphs for DeepSeek-V4 when the variable is absent,
and says so explicitly at startup:

```
Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out.
WARNING: VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline.
         Equivalent to -cc.mode=none.
```

**Absent is not the same as 0** for this flag. Our config never set it, so every run
above used the breakable path. MiaAI-Lab measured a **28.6%** gain from opting out on
2-Spark TP=2 (74.55 → 95.9 tok/s), so this looked like the single biggest remaining lever.

Clean A/B — only this variable changed, everything else identical:

| | breakable (auto) | **explicit 0** |
|---|---:|---:|
| decode tok/s (median, 2x7 reps) | 56.63 · 55.68 | **55.26 · 57.50** |
| graph capture | 9 s / 0.99 GiB | **5 s / 0.44 GiB** |
| acceptance | 3.02–3.13 | 3.03–3.13 |
| KV tokens | 3,592,058 | 3,591,962 |
| `cudagraph_mode` | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |

**Result: within run-to-run noise (~56 tok/s both ways).** The 28.6% uplift did not
transfer to this TP=3 deployment.

The graph capture is genuinely different — 5 s / 0.44 GiB versus 9 s / 0.99 GiB, less
than half the memory — so the flag *is* taking effect. But decode throughput did not
move, and the startup log shows why:

```
WARNING: `torch.compile` is turned on, but the model /models/dsv4-abliterated
         does not support it.
```

Opting out of breakable mode re-enables the torch.compile pipeline, but **this
checkpoint's model class does not support torch.compile**, so the pipeline has nothing
to contribute. `cudagraph_mode` stays `FULL_AND_PIECEWISE` in both cases — the graphs
were never the bottleneck here.

**Keep the flag set to 0 anyway.** It halves graph-capture memory and time at no cost,
and it removes an implicit auto-enabled behaviour from the configuration. Just do not
expect the 28.6%.

⚠️ This also means the projection `57.73 × 1.286 ≈ 74 tok/s` — which appeared to explain
the gap to the upstream 75–79 tok/s TP=3 figures — **does not hold**. The remaining gap
is something else, still unidentified.

### Acceptance explains the whole curve

| MTP | acceptance | what the marginal draft token does |
|---:|---:|---|
| 5 | 3.21 | draft #5 almost never lands — pure waste |
| **4** | **3.02–3.13** | **removing #5 costs ~0.1 acceptance and saves a forward pass** |
| 3 | 2.73–2.83 | draft #4 *was* landing — removing it costs real throughput |

Acceptance barely moved 5 → 4 (−0.1) but fell sharply 4 → 3 (−0.3). That is the
signature of a knee: the fifth draft token was speculative overhead, the fourth was
productive work. **Acceptance length is the metric to tune against**, not tok/s alone —
it explains *why* a setting is fast or slow, and it is visible live in the
`SpecDecoding metrics` log lines.

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
