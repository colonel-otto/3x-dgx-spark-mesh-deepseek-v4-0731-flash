# n=10 undersampled the variance, and that weakens my earlier power claim

Written mid-run, after the TP=3 8K cell finished at n=30 but before the TP=2 arm
ran, so this is not written to fit a conclusion.

## What happened

| | n=10 (original run) | n=30 (this re-run) |
|---|---|---|
| mean | 42.37 tok/s | 43.44 tok/s |
| std | 2.43 | **5.87** |
| CV | 5.7% | **13.5%** |
| observed range | 38.5 – 46.5 | 38.9 – **61.4** |

The **mean was stable** — the two estimates agree within 2.5%, so the central
value was never in doubt. But the **standard deviation grew 2.4x**, because
n=10 never sampled the upper tail. The n=30 sample reaches 61.4 tok/s; nothing
in the n=10 sample exceeded 46.5.

## Why this matters, and what it retracts

`rerun_inconclusive_cells.sh`'s header argues that the original n=10 was **not**
underpowered for the effect under test, on the grounds that n=8 would resolve a
+15.4% effect at the measured CV of 10.8% (pooled). **That argument used a CV
that was itself an underestimate.** With the honest TP=3 CV of 13.5%, the
required n for a +15.4% effect is larger, and n=10 sits much closer to the edge
than that header implies.

The narrower claim still holds: the 95% CIs from the original run genuinely did
exclude +15.4% and include zero, and CIs are computed from the observed spread,
so they were honest about their own uncertainty at the time. What is now
doubtful is the stronger framing that the cells "failed to resolve because the
effect is genuinely smaller" **rather than** because n=10 was too small. With
the true variance this high, undersampling is back on the table as a
contributing cause. The two explanations are not exclusive and this run cannot
fully separate them.

## Consequence for this re-run

At CV 13.5%, n=30 resolves a **~9.8%** effect, not the ~8-9% projected when the
script was written. If the TP=2 arm shows a similar variance increase, the
resolvable effect will be larger still. n=30 remains the pre-committed stopping
point — this note does not license extending it — but the reporting must use the
**measured** CV from this run, not the optimistic one from n=10.

## The general lesson

A CV estimated from n=10 is not a reliable input to a power calculation for the
same cell, because the quantity it estimates is exactly what small samples get
wrong. Report power analyses with the sample size the CV came from, and treat a
CV from n<20 as a lower bound on the true spread rather than a point estimate.

Related: the original run's `power-check.md` and `thresholds-prereg.md` both
built on n=10 CVs and inherit this caveat.
