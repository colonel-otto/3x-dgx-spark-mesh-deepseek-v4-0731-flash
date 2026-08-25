# TTFT, JIT spikes, and whether you need a keep-warm ping

Measured 2026-08-25 on the live TP=3 cluster (1M / seqs 16 / MTP=5 / 0.80).

## The short answer

**You do not need a periodic keep-alive ping.** Idleness costs almost nothing. What costs
seconds is the **first time the engine sees a new prompt shape**, because a kernel is
JIT-compiled *during* that request. The cost is **per-shape, not per-idle-period**, and it
is cached for the engine's lifetime.

So: **warm up shapes at startup; do not ping to stay warm.**

## Evidence

### Idle costs ~20 ms, not seconds

Short prompt after ~10 minutes idle:

| request | latency |
|---|---:|
| 1 (after idle) | 84 ms |
| 2 | 68 ms |
| 3 | 62 ms |

A ~22 ms first-request penalty. Not a cold-start cliff.

### A new shape costs seconds — and it hits the SECOND request

2,048-token prompt, never seen before:

| request | latency |
|---|---:|
| 1 | 1.46 s |
| 2 | **7.81 s** |
| 3 | 0.32 s |
| 4 | 0.31 s |

Reproduced on a different (4,096-token) shape:

| request | latency |
|---|---:|
| 1 | 2.10 s |
| 2 | **5.10 s** |
| 3 | 0.28 s |
| 4 | 0.29 s |
| 5 | 0.28 s |

The spike lands on request **2**, not request 1 — which is why it reads as random slowness
rather than an obvious cold start.

### The engine names the cause itself

```
TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`   22:55:22
TileLang completes                                                        22:55:27   (5 s)
WARNING jit_monitor: CuTeDSL JIT compilation during inference: W4A16FusedMoeKernel.
  This causes a latency spike; consider extending warmup to cover this shape/config.
```

The 5-second compile matches the 5.10 s spike exactly. `JIT_MONITOR_MODE=warn` is what
surfaces this — leave it on.

### Once compiled, it stays fast — including across idle

Same 4K shape, re-probed immediately:

| request | latency |
|---|---:|
| 1 | 0.286 s |
| 2 | 0.275 s |
| 3 | 0.270 s |

And the same shape again after **~12 minutes idle**:

| request | latency |
|---|---:|
| 1 | 0.297 s |
| 2 | 0.275 s |
| 3 | 0.273 s |

No decay, and **no new JIT warning** — the count stayed at 1 for the engine's whole
lifetime. Compilation is cached; idleness does not evict it.

## This also explains the decode measurement noise

The decode sweeps showed a fast mode (85–91 tok/s, 80% of runs) plus a tail of slower runs
(77–79, and 69–72). A ~5 s compile landing inside a ~3 s benchmark run produces exactly
that. Exercising the shapes first moved the numbers:

| | median | slow-mode rate | worst |
|---|---:|---:|---:|
| before warm | 89.1 | 20% | 68.8 |
| after warm | **90.3** | **13%** | 77.4 |

The residual 13% means warm-up still does not cover every shape.

## Practical guidance

1. **Warm at startup, across the shapes you serve.** Sweep a few prompt lengths
   (1K/4K/16K/64K) and concurrency levels once after each restart. This is the whole fix.
2. **Do not add a keep-alive ping for TTFT.** It buys ~20 ms and adds a permanent load
   source that will contaminate benchmarks — during this investigation a gateway keepalive
   on `:8100` was an early false suspect for exactly that reason.
3. **Before benchmarking, warm the exact shape you will measure**, then discard any sweep
   whose log contains `JIT compilation during inference`.
4. **A user-facing spike is possible on genuinely novel shapes.** If p99 TTFT matters,
   pre-compile the shapes you expect rather than relying on organic traffic.

## Open

- **13% of runs still stall after warming.** Warm-up does not cover every shape/config
  combination; which ones remain uncovered is not known.
- The `cutedsl_warmup.py` log line reads *"Skipping CuTeDSL warmup because no compile
  units were requested"* — there may be a supported way to request them at startup.
  Untested.
