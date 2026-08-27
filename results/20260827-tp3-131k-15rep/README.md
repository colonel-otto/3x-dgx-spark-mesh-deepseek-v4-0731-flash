# TP=3 15-rep 131K depth evaluation - 2026-08-27

**Status:** `CURRENT` • **Nodes:** 3 • **TP:** 3 • **Fabric gate:** `PRESENT-PASS` (24 pass, 0 fail, 1 expected skip)

15-repetition single-stream decode and TTFT evaluation at 131K on the winning TP=3 (Profile B) configuration for issue #24.

## Results (131K Depth, 15 Repetitions)

- Requested completion tokens: 256
- Actual completion tokens: 256 on all 15 reps (asserted)
- Cached tokens: 0 on all 15 reps (asserted)
- Median decode throughput: **51.04 tok/s** (min: 41.83, max: 57.92, spread: 31.6%)
- Median TTFT: **79.00 s** (min: 76.84, max: 85.86, spread: 11.4%)

## Sorted Distributions (n=15)

- Decode (tok/s): `[41.83, 47.93, 48.16, 48.74, 49.36, 50.31, 51.01, 51.04, 52.79, 53.07, 53.71, 53.75, 53.86, 54.80, 57.92]`
- TTFT (s): `[76.84, 76.88, 76.92, 77.49, 77.83, 78.58, 78.96, 79.00, 79.11, 80.31, 81.63, 82.44, 83.05, 83.10, 85.86]`

## Comparison to TP=2 Arm (7 reps @ 131K)

- TP=2 decode: **44.40 tok/s** (TP=3 leads by +15.0%)
- TP=2 TTFT: **70.43 s** (TP=2 leads by ~8.6 s; distributions disjoint)
- Conclusion: The third node provides a 15% decode throughput advantage; TTFT at 131K reflects the chunked-prefill all-reduce communication cost across 3 ranks.
