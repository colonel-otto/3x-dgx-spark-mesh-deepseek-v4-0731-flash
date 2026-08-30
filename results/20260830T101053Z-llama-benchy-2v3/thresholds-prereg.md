# Pre-registered decision thresholds

Computed from the **TP=3 arm only**, while the TP=2 cluster was still cold-starting
and no TP=2 number existed. Recorded here so the pass/fail line cannot be moved
after seeing the answer.

For each cell: the TP=2 decode mean that would make the cell resolve as a 3-node
win under Welch |t| >= 2, assuming TP=2's CV is similar to TP=3's.

| cell | TP=3 mean | TP=3 std | TP=2 must be below | advantage needed |
|---|---|---|---|---|
| depth 0 | 47.13 | 4.80 | 43.02 | +9.5% |
| depth 8K | 42.37 | 2.43 | 40.25 | +5.3% |
| depth 32K | 43.86 | 2.93 | 41.32 | +6.2% |
| depth 131K | 44.52 | 5.28 | 40.03 | +11.2% |
| cc=1 | 46.28 | 2.77 | 43.86 | +5.5% |
| cc=4 | 50.96 | 1.61 | 49.54 | +2.9% |
| cc=8 | 52.73 | 0.71 | 52.11 | +1.2% |
| cc=16 | 53.56 | 0.97 | 52.70 | +1.6% |

## What to expect

Our matched harness puts the 2v3 decode advantage at +17-20% for 8K-131K. Every
threshold above is well under that, so if our harness is right, **every cell
should resolve** -- including 131K, whose +11.2% bar is the highest here.

The concurrency cells have the tightest bars (+1.2% to +2.9%) because their
standard deviations are small (0.71-1.61). They are the most likely to resolve,
and also the most sensitive to any systematic difference between the arms --
which is exactly why the engine assertion and the env rewrite matter.

## The trap this guards against

131K needs +11.2% here, but our own matched run measured only **+7.3%** at 131K.
If that cell returns INCONCLUSIVE it is n=10 meeting a genuinely small effect --
NOT evidence against L1, and NOT a licence to re-run until it agrees. The plan's
remedy is to raise n on that specific cell and re-measure it.
