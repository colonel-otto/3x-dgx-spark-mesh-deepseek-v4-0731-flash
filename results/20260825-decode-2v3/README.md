# Decode: 2 nodes vs 3 nodes on healthy fabric — 2026-08-25

**Status:** `CURRENT` within the provenance caveats in [`../index.yaml`](../index.yaml).

The measurement the three-node case actually rests on. Everything else — prefill, deep
concurrency, KV capacity — either ties or favours two nodes. This is where three wins.

## Result

| concurrency | TP=2 | TP=3 | 3-node |
|---:|---:|---:|---:|
| **1** | 76.2 | **89.1** | **+16.9%** |
| 4 | 192.8 | **208.8** | **+8.3%** |
| 8 | 302.7 | **322.7** | **+6.6%** |
| 16 | **481.3** | 474.8 | −1.4% |

Aggregate tok/s, median of 7 runs after 3 warm-up sweeps, `bench_tp3.py`, 18-token
code-brief prompt, 256 tokens out, temperature 0.

**Three nodes win where a single caller feels it (+17% at cc=1) and lose nothing that
matters (−1.4% at cc=16, inside run-to-run noise).** The advantage decays monotonically
with concurrency and crosses over around cc=16 — consistent with the earlier finding that
2-node wins *aggregate* throughput at the cap.

This deployment's workload is single-user interactive coding, i.e. per-stream-latency
bound. **The +17% at cc=1 is the number that matters here.**

## Why this is a fair comparison

- **Identical config on both arms** — `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=16`,
  `MTP_NUM_TOKENS=5`, `GPU_MEMORY_UTILIZATION=0.80`. Node count is the only variable.
- **Same harness, same prompt, same day**, ~20 minutes apart.
- **Both arms verified clean**: 0 RDMA completion errors, 0 preemptions, 0 prefix-cache
  hits, fabric gated at 4.6+ GB/s per pair beforehand.
- KV cache 4,457,627 (3-node) vs 1,711,307 (2-node) — a 2.6x ratio that, as established
  in issue #15, never binds.

## ⚠ The noise is a fast mode plus a stall tail — CAUSE FOUND

The harness prints `<-- wide spread, check for other traffic` at cc≥4 on both arms. **That
warning is misleading here and the medians are sound.**

> **Correction.** An earlier version of this file called the distribution *bimodal* from
> n=15. That was under-resolved. At **n=30** three gaps appear, not one:
>
> | cluster | values | n | reading |
> |---|---|---:|---|
> | fast | 85.1–91.0 | 24 (80%) | steady state |
> | slow-1 | 77.4, 79.1 | 2 | partial stall |
> | slow-2 | 68.8, 69.8, 72.1 | 3 | full ~5 s compile |
>
> It is a fast mode with a **tail of stall severities**, consistent with variable-length
> compiles rather than two discrete states. Do not quote "bimodal".

**Root cause: JIT compilation during inference.** The engine says so itself:

```
TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`   22:55:22
TileLang completes                                                        22:55:27   (5 s)
WARNING jit_monitor: CuTeDSL JIT compilation during inference: W4A16FusedMoeKernel.
  This causes a latency spike; consider extending warmup to cover this shape/config.
```

A ~5 s compile landing inside a ~3 s benchmark run produces exactly the observed dropout.

**Warming it up measurably helps** — 15 runs before vs after exercising the shapes:

| | median | slow-mode rate | worst |
|---|---:|---:|---:|
| before | 89.1 | 20% | 68.8 |
| after | **90.3** | **13%** | 77.4 |

Once a shape is compiled it stays fast: re-probing an already-compiled 4K shape gave
0.286 / 0.275 / 0.270 s, and total `JIT compilation during inference` warnings since
engine start is **1**. The residual 13% means warm-up does not yet cover every shape.

Note this makes the headline **conservative**: on the warmed median the 3-node gain at
cc=1 is **+18.5%**, not +16.9%. The table above keeps the unwarmed 89.1 because that is
what the matched TP=2 arm was measured against.

Ruled out as causes: thermal throttling (no active throttle reasons, clocks 2405–2411 MHz
at spec, 57–65 °C), prefix cache (0 hits), preemptions (0), external traffic (the LiteLLM
gateway holds an idle keepalive to `:8100` but sent nothing — completion counts matched
our sweeps exactly).

**Consequence: use the median, take ≥7 runs, and warm every shape you intend to measure.**
A mean, or a 3-run median, can land in the stall tail and manufacture a 20% difference
that is not real.

TP=2's cc=1 was unusually tight by comparison (2% spread, 7 runs within 75.8–77.4), so the
+17% is not an artifact of picking TP=3's fast mode: even TP=3's *slow* mode (69.8) against
TP=2's tight 76.2 would show 3-node losing, and the median of 7 lands in the fast mode
reliably.

## Where this leaves the three-node case

| measurement | 2-node | 3-node | winner |
|---|---:|---:|---|
| prefill 1K/8K/32K | 1913 / 2081 / 2066 | 2023 / 2070 / 2095 | parity (±2%) |
| deep concurrency TTFT | **293,987 ms** | 396,804 ms | **2-node**, 1.35x |
| decode cc=1 | 76.2 | **89.1** | **3-node**, +17% |
| decode cc=16 aggregate | **481.3** | 474.8 | 2-node, +1.4% |
| KV capacity | 1,711,307 | **4,457,627** | 3-node 2.6x — but never binds |

**Three nodes are superior for single-stream interactive work and inferior for batch
throughput.** That is a real, matched, healthy-fabric answer rather than a slogan in
either direction — and it means the deployment choice follows the workload, not the
hardware count.

## Files

| file | what |
|---|---|
| `tp3_final.json` / `.txt` | 3-node arm, 7 runs × 4 levels |
| `tp2_final.json` / `.txt` | 2-node arm, matched |
