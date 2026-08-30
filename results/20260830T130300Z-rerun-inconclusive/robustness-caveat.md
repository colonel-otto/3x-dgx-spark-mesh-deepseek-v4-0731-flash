# Robustness of the 2026-08-30 llama-benchy result under the variance correction

Written after the n=30 re-run measured TP=3's true spread at two cells and found
n=10 had understated it in BOTH — std 2.4x low at 8K, 1.8x low at cc=1, with the
means stable (2.5% and 3.1%). That is a systematic undersampling of the upper
tail, not a fluke of one cell, so it is fair to ask what it does to the cells the
original run DID resolve.

## The test

Recompute each resolved decode cell's Welch t with the standard deviation
inflated 2.0x on **both** arms — roughly the correction the re-run measured.

| cell | 3v2 | t as run | t at 2x std | survives? |
|---|---|---|---|---|
| depth, depth=0 | +20.8% | 4.28 | 2.14 | **yes** |
| depth, depth=32K | +11.9% | 2.63 | 1.31 | **no** |
| depth, depth=131K | +13.5% | 2.36 | 1.18 | **no** |
| concurrency, cc=4 | +20.1% | 11.21 | 5.60 | **yes** |
| concurrency, cc=8 | +18.7% | 26.68 | 13.34 | **yes** |
| concurrency, cc=16 | +15.4% | 18.49 | 9.24 | **yes** |

## What this changes

**Two of six resolved decode cells — 32K and 131K — would not resolve** under
the variance the re-run actually measured. Their t values (2.63, 2.36) sat only
just above the |t|>=2 threshold to begin with, so a 2x std increase drops them
below it.

**The concurrency cells are unaffected in practice.** At t = 5.60 to 13.34 they
would survive a 5x variance inflation. Their standard deviations were also the
smallest in the run (0.71 to 1.61), measured across 10 runs of 4-16 concurrent
clients — far more underlying samples per cell than the single-stream depth
cells, which is the likely reason they estimated variance better.

**Prefill is untouched by this analysis** — it was not re-measured, but its
standard deviations were proportionally tiny (e.g. 1578.21 +- 3.34 at 8K, CV
0.2%), so it has orders of magnitude of headroom.

## What this does NOT change

- **No cell flips direction.** Under any inflation tested, zero cells favour two
  nodes. The L1 direction claim is unaffected.
- **The concurrency result stands**, and it is the stronger half of the evidence.
- **The matched run (n=30, our own harness) is unaffected** — it is a separate
  measurement at 3x the n, with its own reported significance.

## Honest restatement

The original result's "14 of 16 cells resolved, all 14 favour three nodes" should
be read as: **the direction is robust everywhere; the magnitude is well
established at concurrency and at depth 0, and provisional at 32K and 131K**
pending a higher-n depth sweep.

This does not retract the result. It narrows which parts of it carry weight, and
it is exactly the sort of thing an n=10 confirmatory arm should be expected to
surface once someone checks.

## Why this was findable at all

Only because two cells were re-measured at n=30 and the variance was compared
against the original. A confirmatory run that had simply agreed at n=10 would
have carried the same understated spreads and nobody would have looked. The
inconclusive cells were more informative than the resolved ones.
