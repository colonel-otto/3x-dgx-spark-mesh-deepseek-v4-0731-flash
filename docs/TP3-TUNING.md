# TP=3 tuning sweep — DeepSeek-V4-Flash on 3× DGX Spark

Incremental, controlled measurements on the 3-node TP=3 deployment. Every row is the
same harness, checkpoint, prompt, `temperature=0`, 256 output tokens, on otherwise
identical software.

> ⚠️ **Read [`BENCHMARK-METHODOLOGY.md`](BENCHMARK-METHODOLOGY.md) before quoting any
> absolute number from this page.** All figures here use one dense-prose prompt, which
> is the *worst case* for MTP speculative decoding. The same engine measures **~79
> tok/s** on a code-shaped prompt and on the upstream harness — a 1.65x swing driven
> purely by draft-acceptance rate.
>
> Because every row uses the same prompt, **the relative comparisons below are valid.**
> The absolute values describe hard-prose workloads only.

**Workload this is tuned for:** a single user with several concurrent conversations,
mostly well under the full context window. Not a many-tenant serving deployment.

---

## Baseline chain — how we got here

| Config | Transport | decode tok/s | Note |
|---|---|---:|---|
| TP=2 (2 nodes, production) | RoCE | 48.23 | the reference every candidate is measured against |
| EP=3 (3 nodes) | Socket | 19–20 | loses the B12X kernel — see EP3 doc |
| TP=3, earlier cable rotation | **Socket/TCP** | 24.59 | valid transport control; not RoCE performance |
| TP=3, earlier cable rotation | **RoCE** | **57.73** | historical best after enabling subnet-aware routing |
| TP=3, NVIDIA reference ring | RoCE | **53.95** | retained MTP=5 result on the canonical physical layout |

The 24.59 figure is a useful TP=3 TCP fallback control. It demonstrates the transport
penalty, but it does not describe TP=3 over RDMA. On RoCE, retained medians range from
53.95 to 57.73 tok/s.

### On the cabling

After re-cabling to NVIDIA's reference ring, the retained MTP=5 runs (5 and 7
repetitions) both had a 53.95 tok/s median, about 6.5% below the earlier 57.73 result.
That is an observation, not an isolated cabling effect: the cluster has wide repetition
spread and the measurements occurred at different times. Do not infer that the cable
rotation caused the entire difference.

The reference layout is retained so later work begins from NVIDIA's documented physical
topology. It does **not** remove the ring-specific NCCL settings: NVIDIA's own launcher
still exports `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_NET_PLUGIN=none`.

---

## Tuning sweep

Starting point: `MTP_NUM_TOKENS=5`, `MAX_NUM_SEQS=16`,
`GPU_MEMORY_UTILIZATION=0.85`, `MAX_MODEL_LEN=460800`.

| # | MTP | max_num_seqs | decode tok/s (median) | acceptance | KV tokens | graph capture | Notes |
|--:|---:|---:|---:|---:|---:|---|---|
| 0 | 5 | 16 | 53.95 · 53.95 | 3.21 | 3,606,027 | 11 s / 1.89 GiB | starting point |
| 1 | **4** | **8** | **56.63 · 55.68** | 3.02–3.13 | 3,592,058 | 9 s / 0.99 GiB | **best tested combined profile**, half the graph memory |
| 2 | 3 | 8 | 40.19 · 39.57 | **2.73–2.83** | 3,603,348 | 10 s / 1.86 GiB | **−29%** — well past the knee |

`MTP_NUM_TOKENS=4`, `MAX_NUM_SEQS=8` is the best tested combined profile. The sweep
isolates MTP=4 versus MTP=3 because both use eight scheduler slots, and MTP=3 is about
29% slower. It does **not** isolate MTP=4 versus MTP=5 because the MTP=5 row used 16
slots. An MTP=5/seqs=8 control is required before calling four drafted tokens optimal.
Four 7-repetition summaries support the MTP=3 result (medians 40.94, 39.73, 40.19, and
39.57 tok/s; raw files were not retained in this branch).

### `VLLM_USE_BREAKABLE_CUDAGRAPH=0` — tested, no measurable gain here

A CUDA graph records a fixed sequence of GPU kernel launches and replays it, reducing
CPU launch overhead. vLLM's ordinary piecewise path uses compile-time graph splitting to
leave unsupported operations eager. The experimental
[breakable path](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/breakable_cudagraph.py)
instead captures at runtime, ends capture around designated eager operations, and then
resumes it. Setting this flag to `0` opts out of that breakable implementation; it does
not disable every CUDA graph. The selected `cudagraph_mode` still controls full,
piecewise, or mixed capture, as described in the
[vLLM CUDA graph design](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md).

vLLM **auto-enables** breakable CUDA graphs for DeepSeek-V4 when the variable is absent,
and says so explicitly at startup:

```
Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out.
WARNING: VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline.
         Equivalent to -cc.mode=none.
```

**Absent is not the same as 0** for this flag. Our config never set it, so every run
above used the breakable path. [MiaAI-Lab's 2-Spark report](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/results/RESULTS-2026-08-14.md)
reports a **28.6%** gain from opting out on its TP=2 setup (74.55 → 95.9 tok/s), so this
looked like the single biggest remaining lever. Its result is external comparison data,
not a directly comparable baseline for this image and TP=3 configuration.

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
than half the memory — so the flag *is* taking effect. Decode throughput did not move.
The startup log also reported:

```
WARNING: `torch.compile` is turned on, but the model /models/dsv4-abliterated
         does not support it.
```

That warning is consistent with the tested model path not using `torch.compile`, but it
does not establish why the flag changed graph-capture cost without changing decode
throughput. `cudagraph_mode` remained `FULL_AND_PIECEWISE` in both cases. The causal
explanation is therefore still unknown.

**Keep the flag set to 0 anyway.** It halves graph-capture memory and time at no cost,
and it removes an implicit auto-enabled behaviour from the configuration. Just do not
expect the 28.6%.

⚠️ This also means the projection `57.73 × 1.286 ≈ 74 tok/s` — which appeared to explain
the gap to the upstream 75–79 tok/s TP=3 figures — **does not hold**. The remaining gap
is something else, still unidentified.

### Acceptance is a useful tuning signal

| MTP | acceptance | what the marginal draft token does |
|---:|---:|---|
| 5 | 3.21 | the marginal fifth draft appears to have low acceptance; seq count is confounded |
| **4** | **3.02–3.13** | **best tested profile; requires MTP=5/seqs=8 to isolate the fifth draft** |
| 3 | 2.73–2.83 | draft #4 *was* landing — removing it costs real throughput |

Acceptance barely moved between the observed MTP=5 and MTP=4 profiles (−0.1) but fell
more sharply from MTP=4 to MTP=3 (−0.3). That makes acceptance length a useful tuning
signal alongside throughput, but acceptance alone does not prove causality when another
setting changes in the same comparison.

Sample counts differ by row: the canonical MTP=5 profile has retained 5- and
7-repetition summaries; the MTP=4/breakable A/B profiles have two 7-repetition summaries
per side; and four 7-repetition summaries were recorded for MTP=3. Run-to-run spread is
material (individual repetitions span roughly 41–61 tok/s), so medians and exact sample
counts are quoted instead of relying on a single run. Only the row-1 repetitions below
remain in this branch as raw text; the other historical rows are summary evidence.

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

The MTP=4 versus MTP=3 comparison is the cleanest acceptance result in this sweep because
`MAX_NUM_SEQS=8` is held constant. The MTP=5 comparison remains a hypothesis until the
missing MTP=5/seqs=8 control is run.

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

Speculative decoding is designed to preserve the target model's distribution (and the
greedy result under matching numerical conditions) because the target model verifies
drafts. Implementation bugs, sampling details, and different numerical execution paths
can still matter, so every MTP setting in this repository requires correctness checks;
it is not assumed correct from the algorithm alone.

### max_num_seqs was sized for a different workload

`max_num_seqs` caps the number of active sequences the scheduler may process; it does not
reserve or assume 16 simultaneous full-context requests. Reducing it changed this
workload's graph-capture shape and may change scheduling/cache behavior:

* **CUDA graph capture size is derived** as `max_num_seqs × (MTP_NUM_TOKENS + 1)`.
  At `16 × 6 = 96` that measured 11 s and 1.89 GiB. At `8 × 5 = 40` it is far smaller.
* Fewer scheduler slots means better cache locality per sequence.

The [upstream 3-Spark repository](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark)
reports the opposite direction (`max_num_seqs` 6 → 16 → 32
raising *aggregate* throughput 238 → 431 → 618 tok/s). That is a multi-tenant result:
it buys aggregate throughput at high concurrency, and explicitly does **not** improve
single-stream latency. Different workload, different optimum.

---

## Memory headroom

At `GPU_MEMORY_UTILIZATION=0.85`, each 121 GiB GB10 runs with only **2–4 GB free**:

```
node1 avail=2GB   node2 avail=3GB   node3 avail=4GB
```

Raising utilisation would add KV capacity but is genuinely risky — an over-commit on
these machines does not OOM cleanly, it wedges the node past SSH recovery and needs a
power cycle. Not attempted.

## Context length

`MAX_MODEL_LEN=460800` tokens against a model maximum of 1,048,576. It was
deliberately lowered from 1M: at 1M the memory profiler reserves worst-case activation
headroom and leaves almost nothing free under load, and the freed KV pool is what pays
for the concurrency slots. This is a hard 460,800-token ceiling and therefore does not
meet a literal 500,000-token requirement; choose the limit from measured prompt needs.

## Evidence limitations and next controls

This branch preserves medians, selected repetitions, correctness summaries, and the
sanitized head-rank configuration, but not complete timestamped raw logs for every row
or worker-rank snapshots. Do not reconstruct missing evidence. The next run should use
the artifact policy in [`EXPERIMENT-LOG.md`](EXPERIMENT-LOG.md) and add, in order:

1. MTP=5 with `MAX_NUM_SEQS=8` to isolate draft length.
2. `NCCL_DEBUG=INFO` proof of `NET/IB`, selected HCAs, and GDRDMA on every rank.
3. Interleaved A/B/A runs to reduce time/order confounding.

## Untested levers

* **Second CX-7 card.** Each Spark has four RoCE devices; only two are configured
  (the others have no IP and MTU 1500). Multi-rail could add fabric bandwidth.
* **Alternative MoE backends** (`triton`, `deep_gemm`, `machete`). Upstream reports most
  failing or rejecting the device on SM121; worth a short probe, not an afternoon.
* **Multi-QP RDMA.** The 109 Gb/s per-cable figure is a single-QP `ib_write_bw` default,
  ~55% of line rate — a harness default, not a measured ceiling.
