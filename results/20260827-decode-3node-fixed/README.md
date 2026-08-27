# Corrected 3-node decode depth curve (256-token window)

**Status:** `CURRENT` · **Nodes:** 3 (TP=3) · **Model:** `deepseek-v4-flash-0731`
**Harness:** `scripts/decode_depth_sweep.py` @ `c3e8e0d` (window pinned + asserted)
**Output tokens:** 256, asserted on every rep · **Cache hits:** 0 / 43 reps
**Run:** 2026-08-27 02:07-03:10 UTC, single live engine (PID 556480, up since 2026-08-26 16:26 UTC)
**Supersedes:** [`../20260826-decode-depth-2v3/`](../20260826-decode-depth-2v3) tp3 arm
**Issues:** [#26](../../issues/26) (window defect), [#24](../../issues/24) (131K needs more samples)

## Why this run exists

The tp3 arm of `20260826-decode-depth-2v3` requested `max_tokens=256` but its prompt
asked for one sentence, so every rep returned 25-26 tokens. Nothing asserted the
length. [`../20260826-harness-window-calibration/`](../20260826-harness-window-calibration)
measured the distortion at 8K: median went 72.6 -> 55.3 tok/s, a ~31% overstatement.

This run re-measures the **full depth curve** with the fixed harness against the
same live 3-node cluster. Nothing about the engine was changed, restarted, or
reconfigured — measurement only.

## Result: old vs new, per depth

| depth | prompt tok | old median | new median | delta | old spread | new spread | n (old/new) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,038 | 76.35 | **54.72** | **-28.3%** | 21.6% | 22.6% | 7 / 7 |
| 8,192 | 8,085 | 72.64 | **56.02** | **-22.9%** | 40.0% | 14.6% | 7 / 7 |
| 32,768 | 32,272 | 70.21 | **54.28** | **-22.7%** | 39.2% | 25.0% | 7 / 7 |
| 131,072 | 129,009 | 72.64 | **53.06** | **-27.0%** | 35.1% | 73.5% | 7 / **15** |
| 262,144 | 257,996 | 84.37 | **50.66** | **-40.0%** | 33.9% | **4318%** | 7 / 7 |

Deltas are new-vs-old median. Spread is `(max-min)/min`.

## Sorted per-rep values

Spreads are what caught the original defect, so every rep is published.

```
depth 2,048    (n=7)   48.78  49.63  50.54 [54.72] 55.52  56.60  59.81
depth 8,192    (n=7)   50.58  51.86  55.34 [56.02] 56.97  57.16  57.96
depth 32,768   (n=7)   49.21  53.03  54.12 [54.28] 54.75  57.46  61.53
depth 131,072  (n=15)  32.54  35.37  38.44  49.64  50.98  51.47  52.84
                      [53.06] 54.01  54.51  54.52  54.62  55.33  55.41  56.45
depth 262,144  (n=7)    1.17   3.28  49.78 [50.66] 51.15  51.28  51.69
```

Old arm, for comparison (all on the defective 25-26 token window):

```
depth 2,048    (n=7)   74.10  75.93  76.29 [76.35] 76.48  90.22  90.60
depth 8,192    (n=7)   56.56  62.65  68.82 [72.64] 85.54  85.59  85.61
depth 32,768   (n=7)   55.26  60.68  70.20 [70.21] 70.40  70.60  82.78
depth 131,072  (n=7)   53.79  64.70  72.28 [72.64] 73.99  75.95  79.27
depth 262,144  (n=7)   69.50  71.16  83.75 [84.37] 85.06  89.92  98.07
```

## What the corrected curve says

**The curve is flat, not rising.** Corrected medians sit in a 50.7-56.0 band across
a 128x range of context depth. The old data showed decode *improving* at 262K
(84.4 tok/s, the highest point on the curve) — that was an artifact. With a real
256-token window, 262K is the *lowest* median, not the highest.

**The overstatement is not uniform.** It ranges from 22.7% at 32K to 40.0% at 262K.
The deeper the context, the more the short window flattered the result. Any
old-data conclusion that leaned on the *shape* of the depth curve — not just its
magnitude — is affected, because the bias itself varies with depth.

## ANOMALY: intermittent decode collapse at 262K

This is the significant finding and it is **not** a harness fault.

At 262,144 the reps are bimodal. Five reps sit at 49.8-51.7 tok/s, in line with
every other depth. But **both warmups and the first two reps collapsed** to
1.2-3.3 tok/s — a ~40x slowdown, i.e. a ~200 second decode window for 256 tokens:

```
warmup 1:  3.2   <-- collapsed
warmup 2:  1.3   <-- collapsed  (SLOWER than warmup 1, so not JIT warm-up)
rep 1:     1.2   <-- collapsed
rep 2:     3.3   <-- collapsed
rep 3:    49.8   normal
rep 4:    50.7   normal
rep 5:    51.7   normal
rep 6:    51.3   normal
rep 7:    51.1   normal
```

Points that constrain the cause:

- **Not JIT.** Warmup 2 was slower than warmup 1. A compile stall gets *better*
  with repetition, not worse. The recovery also came four requests in, not on the
  second.
- **Not prefill.** TTFT is 165-181s on every rep, collapsed and normal alike, and
  matches the old arm's 178-205s. Prefill is unaffected; the collapse is purely
  in the decode phase.
- **Not the prefix cache.** `cached_tokens` is 0 on all 43 reps.
- **Not a wedged engine.** `/v1/models` returned 200 throughout and every shallower
  depth ran normally before and after.
- **Invisible to the old harness.** A 26-token window at ~50 tok/s finishes in
  ~0.5s. Whatever regime produces the stall appears to need a longer generation to
  manifest, so the old run could not have seen it at any depth.

The median (50.66) is robust to these outliers, which is why the harness reports
medians — but the spread of 4318% is the honest headline, and a user issuing a
single long-context request has a real chance of hitting the slow mode.

**This is not characterized.** Five clean reps and four collapsed ones is not
enough to establish frequency, trigger, or whether it also occurs at 131K (where
three reps did land low at 32.5/35.4/38.4 — a milder version of the same shape).
It needs a dedicated run before anyone relies on 262K numbers.

## 131K with 15 reps (issue #24)

Issue #24 called for more samples at 131,072 because the 2-node and 3-node spreads
overlapped there. With 15 reps the distribution is clearly not symmetric: twelve
reps cluster tightly at 49.6-56.5, and three sit far below at 32.5, 35.4, 38.4.
The median (53.06) sits inside the tight cluster; the mean would not. The 73.5%
spread is driven entirely by those three low reps and is the same qualitative
shape as the 262K collapse, one order of magnitude milder.

## Verification

All checks passed on all 43 reps:

| check | result |
|---|---|
| `completion_tokens == 256` | 43 / 43 (harness raises otherwise; it never did) |
| `cached_tokens == 0` | 43 / 43 |
| window assertion triggered | never |
| depth landed on nominal | 2,038 / 8,085 / 32,272 / 129,009 / 257,996 |
| engine restarted or reconfigured | no — same PID 556480 throughout |

## Files

| file | contents |
|---|---|
| `tp3-fixed-2048.jsonl` | 7 reps @ 2,048 |
| `tp3-fixed-8192.jsonl` | 7 reps @ 8,192 |
| `tp3-fixed-32768.jsonl` | 7 reps @ 32,768 |
| `tp3-fixed-131072.jsonl` | 15 reps @ 131,072 (issue #24) |
| `tp3-fixed-262144.jsonl` | 7 reps @ 262,144 |
| `tp3-fixed-*.log` | harness stdout per depth, including warmups |
| `summary.json` | sorted per-rep values, medians, spreads, invariant checks |
| `engine-config.txt` | full argv read from `/proc/556480/cmdline` on the live process |

## Live engine config

Read from the running process, not from a config file:

```
--tensor-parallel-size 3      --pipeline-parallel-size 1
--kv-cache-dtype nvfp4_ds_mla --block-size 256
--max-model-len 1048576       --max-num-seqs 32
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.80
--moe-backend flashinfer_b12x --nnodes 3 --node-rank 0
--speculative-config {"method":"dspark","num_speculative_tokens":5,...}
--enable-prefix-caching --async-scheduling --enable-chunked-prefill
```

**KV pool size is NOT captured for this boot.** The systemd unit for the current
engine does not journal, and the `/tmp` logs on sparkmain are from earlier boots.
The last recorded value on identical hardware and flags (2026-08-24 boot) was
`GPU KV cache size: 5,382,503 tokens`, `Maximum concurrency for 1,048,576 tokens
per request: 5.13x` — carried in `engine-config.txt` as a labelled reference, not
as a measurement of this run.

## Not done here

- **2-node arm.** Re-measuring TP=2 requires a cluster restart, which was out of
  scope. Until that exists, **no 2v3 comparison can be made on corrected data** —
  the old 2-node numbers carry the same window defect and cannot be compared
  against the corrected 3-node numbers above.
- **Root-causing the 262K collapse.** See the anomaly section.
