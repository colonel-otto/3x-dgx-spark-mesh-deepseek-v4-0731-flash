# `MAX_NUM_SEQS=32` retest on healthy fabric — 2026-08-26

**The 2026-08-24 rejection is falsified. `seqs=32` does not crash, and it is worth
+46% aggregate throughput at cc=32.** Answers [#10](../../issues/10).

## Result

| concurrency | `seqs=16` | `seqs=32` | change |
|---:|---:|---:|---:|
| 1 | 89.0 | 87.6 | −1.6% |
| 4 | 205.7 | 219.4 | +6.7% |
| 8 | 341.7 | 321.3 | −6.0% |
| 16 | **502.5** | 485.0 | −3.5% |
| **32** | 468.8 | **685.9** | **+46.3%** |

Aggregate tok/s, median of 7 runs, `bench_tp3.py`, 18-token code-brief prompt, 256
tokens out, temperature 0. **Every request succeeded in both arms** — 7 runs at each of
5 levels, `ok=N/N` on all 70.

**685.9 tok/s exceeds the 618 tok/s external reference** that [#10](../../issues/10)
named as "the one number in that comparison we cannot match." It was config, not silicon.

### Read the cc=32 row carefully

At `seqs=16` the engine admits only 16 concurrent sequences, so a cc=32 offered load
queues half its requests — which is why **468.8 at cc=32 is *lower* than 502.5 at
cc=16**. Raising the cap removes that cliff. The +46% is the queueing penalty being
lifted, not a kernel getting faster.

**Below cc=32 the change is a wash** (−6.0% to +6.7%, all inside the run-to-run spread
this harness shows on this cluster). `seqs=32` buys headroom at the top, and costs
nothing meaningful below it.

## Why it was rejected before, and why that no longer applies

The 2026-08-24 run died on an `_ALLGATHER_BASE` timeout under sustained load
([`../../docs/SEQS32-AND-NCCL-FABRIC.md`](../../docs/SEQS32-AND-NCCL-FABRIC.md)). Three
conditions have changed since, and the failure mode was a **timeout** — exactly what a
starved, slow-watchdog collective produces:

| condition | at rejection | now |
|---|---:|---:|
| 3-rank collective budget | **0.49 GB/s** (degraded fabric, [#14](../../issues/14)) | **3.25+ GB/s** — 6.6x |
| `NCCL_TIMEOUT` | 600 s (default) | **3600 s** |
| `GPU_MEMORY_UTILIZATION` | 0.85 | **0.80** |

> **[#10](../../issues/10)'s risk argument is stale.** It reasons that our test is riskier
> than the reference because *"our `GPU_MEMORY_UTILIZATION` is 0.85; theirs is 0.80."*
> We now run **0.80**, so that asymmetry no longer exists.

## The startup risk did not materialise

[#10](../../issues/10)'s primary worry was CUDA-graph capture: `MAX_NUM_SEQS ×
(MTP_NUM_TOKENS + 1)` = 32 × 6 = **192**, double the 96 at `seqs=16`.

| | `seqs=16` | `seqs=32` |
|---|---:|---:|
| graph capture | 1.20 GiB / 8 s | **2.12 GiB / 25 s** |
| GPU KV cache | 4,512,769 tok | **4,431,088 tok** (−1.8%) |
| max concurrency @1M ctx | 4.30x | 4.23x |
| OOM / `IBV_WC` / `NET/Socket` / Watchdog | 0 | **0** |

Capture cost **1.77x memory for 2x the sequences** — sublinear, and it fits at 0.80
utilization with no adjustment. **KV barely moved**, so the extra concurrency headroom is
close to free. No `GPU_MEMORY_UTILIZATION` reduction was needed.

## Why this is a fair comparison

- **Same-day control.** The `seqs=16` arm was measured on today's engine ~40 minutes
  before, not quoted from the 08-25 record. The repo's older 374.2 tok/s cc=16 figure is
  pre-fabric-fix and is **not** the baseline used here.
- **Same harness**, `bench_tp3.py`, copied from the original `seqs32` run directory — the
  same instrument that produced the rejection.
- **`MAX_NUM_SEQS` is the only variable.** `MAX_MODEL_LEN=1048576`,
  `MTP_NUM_TOKENS=5`, `GPU_MEMORY_UTILIZATION=0.80`, `MAX_NUM_BATCHED_TOKENS=8192`
  unchanged and verified identical on all three ranks before restart.
- **Clean engine after load**: 0 preemptions, 0 requests stuck, and **zero**
  `ALLGATHER_BASE` / `Watchdog` / OOM / `IBV_WC` / `NET/Socket` occurrences in the full
  container log.

## Caveat on spread

This harness reports wide spreads at cc≥4 on both arms (17–70%). That is the known JIT
stall tail — TileLang/CuTeDSL compiling shapes during inference — documented in
[`../20260825-decode-2v3/`](../20260825-decode-2v3) and
[`../../docs/TTFT-AND-WARMUP.md`](../../docs/TTFT-AND-WARMUP.md). Medians of 7 are
used for exactly this reason. **The +46.3% at cc=32 is far outside that noise; the
sub-10% differences at cc=4/8/16 are not, and should be read as parity.**

## Evidence

| file | contents |
|---|---|
| `seqs16-baseline.json` / `.log` | control arm, 35 runs |
| `seqs32-test.json` / `.log` | test arm, 35 runs |

**Related:** [#10](../../issues/10) ·
[`../../docs/SEQS32-AND-NCCL-FABRIC.md`](../../docs/SEQS32-AND-NCCL-FABRIC.md) ·
[`../../docs/DEGRADED-DATA-CATALOGUE.md`](../../docs/DEGRADED-DATA-CATALOGUE.md)
