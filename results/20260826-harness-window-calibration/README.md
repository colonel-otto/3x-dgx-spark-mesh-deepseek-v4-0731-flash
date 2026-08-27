# Harness calibration: the 25-token decode window overstated decode by ~31%

**Status:** `CURRENT` · **Nodes:** 3 (TP=3) · **Depth:** 8,192 · **Reps:** 7
**Harness:** `scripts/decode_depth_sweep.py` @ `c3e8e0d` (fixed) vs `aefa594` (old)
**Output tokens:** 256 (asserted) vs 25-26 (unasserted, old)
**Issue:** [#26](../../issues/26)

## Why this run exists

`decode_depth_sweep.py` requested `max_tokens=256`, but its prompt ended
*"In one sentence, state what this describes."* The model obeyed. **All 70 reps**
of [`../20260826-decode-depth-2v3/`](../20260826-decode-depth-2v3) returned 25 or
26 tokens. Nothing asserted the length, so it went unnoticed.

This run measures how much that distorted the numbers. Same cluster, same depth,
same 3-node config, ~40 minutes apart. **Only the harness changed.**

## Result

| | old (25-tok window) | new (256-tok window) |
|---|---:|---:|
| decode window | ~0.40 s | **4.19-5.78 s** |
| **median decode** | **72.6 tok/s** | **55.3 tok/s** |
| spread (max/min) | 1.51x | 1.38x |

```
old reps:  56.6  62.6  68.8 [72.6] 85.5  85.6  85.6
new reps:  44.3  49.3  51.1 [55.3] 57.9  59.7  61.1
```

**The short window overstated decode by ~31%.** This is a systematic bias, not
noise. At MTP=5 a 25-token window covers ~5 speculative cycles, weighted toward
the opening cycles where draft acceptance is at its best; a 256-token window
includes the realistic steady-state mix.

The old distribution's top three reps — **85.5, 85.6, 85.6** — are the tell. A
window that short is quantized by draft-acceptance granularity rather than
measuring a continuous rate.

## What this invalidates

**Every decode tok/s figure published in this repository**, including the
headline `+33.6% at 131K`. Direction may survive (both arms carried the same
defect) but no magnitude does. Tracked in [#26](../../issues/26); the README
carries a provisional-data banner as of `fd9f38a`.

## Fix

`scripts/decode_depth_sweep.py` @ `c3e8e0d`:
- prompt no longer requests a short answer
- `min_tokens == max_tokens` + `ignore_eos` pin the window
- **`completion_tokens != max_tokens` now raises** — verified against the live
  engine (64 requested, 64 returned) before this run

## Evidence

| file | contents |
|---|---|
| `new-256tok-8k-tp3.jsonl` | 7 reps, fixed harness, `completion_tokens: 256` on all 7 |
| `../20260826-decode-depth-2v3/tp3-depth.jsonl` | the old arm, filter `"target_depth": 8192` |

## Caveat

One depth, one arm, one day. It establishes that the defect **is** material and
roughly how large; it does not calibrate a correction factor to apply to old
numbers. Re-measure rather than adjust on paper.
