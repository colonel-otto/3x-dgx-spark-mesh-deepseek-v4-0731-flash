# Corrected 256-token concurrency: two nodes versus three — 2026-08-27

**Status:** `CURRENT` · **Depth:** 8,192 · **Output:** 256 tokens asserted per request ·
**Trials:** 3 per concurrency and arm

This is the fixed-window follow-up to the historical concurrency result. It confirms the
crossover direction while showing that the old approximately 480 tok/s magnitudes came
from a short-window harness and were about an order of magnitude too high.

## Result

| concurrency | TP=2 aggregate | TP=3 aggregate | winner |
|---:|---:|---:|---|
| 4 | 39.32 tok/s | **46.57 tok/s** | TP=3 |
| 8 | **53.36 tok/s** | 46.30 tok/s | TP=2 |
| 16 | **56.20 tok/s** | 52.77 tok/s | TP=2 |

Values are medians of three trial-level aggregate rates. The harness verifies every
request returned exactly 256 tokens before writing a trial record.

## Caveats

- TP=2 used `MAX_NUM_SEQS=16`; TP=3 used the production value 32. At cc=16 the TP=2 arm
  visibly queued while TP=3 could admit all requests, yet TP=2 still won aggregate rate.
- TP=2 has an engine-stopped passing pairwise gate at 9.47 GB/s. The restored TP=3 check
  was taken with its engine live, so NCCL bandwidth was intentionally skipped.
- Three trials establish the qualitative crossover but are not a high-precision estimate.

`harness/bench_miaai_cc.py` is the exact concurrency harness. Raw arms and orchestration
logs are under `tp2/` and `tp3/`.
