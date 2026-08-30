# Post-hoc power check on the TP=3 arm (n=10)

Computed from the TP=3 depth sweep's own measured variance, BEFORE the TP=2 arm
ran, so this is a check on whether n=10 can resolve the effect -- not a
justification written after seeing the answer.

Two-sample, alpha=0.05, power=0.80: n = 2*(1.96+0.84)^2 * CV^2 / delta^2,
solved for the minimum detectable delta at n=10.

| depth | mean tok/s | std | CV | min detectable effect | vs our +17-20% |
|---|---|---|---|---|---|
| 0 | 47.13 | 4.80 | 10.2% | 12.8% | resolves |
| 8192 | 42.37 | 2.43 | 5.7% | 7.2% | resolves |
| 32768 | 43.86 | 2.93 | 6.7% | 8.4% | resolves |
| 131072 | 44.52 | 5.28 | 11.9% | 14.9% | resolves, but tightest |

**131K is the cell at risk.** It needs a >=14.9% effect to resolve at n=10, and
our harness puts the 131K advantage at the LOW end of the band (+7.3% in the
matched run's own table). If that cell returns INCONCLUSIVE, that is the
expected consequence of n=10 against a small effect -- not evidence against L1.
The plan's remedy applies: raise n on that cell and re-run it, rather than
reporting a marginal difference either way.

Depth 0 and 131K both show CV ~10-12% against ~6% for the middle depths. Decode
variance is not uniform across depth on this cluster.
