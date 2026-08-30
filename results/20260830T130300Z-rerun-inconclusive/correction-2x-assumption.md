# Correction: the 2x-both-arms inflation was wrong

## What I assumed

After the TP=3 arm of the n=30 re-run showed std 2.4x higher at 8K than n=10 had
reported, I inflated **both** arms' standard deviations by 2.0x and recomputed
the parent run's significance. Two cells (32K, 131K decode) dropped below
|t|>=2, and I published that caveat to the README and the result doc.

That inflation was an **assumption applied to both arms, not a measurement of
both arms.** Only TP=3 had been re-measured at the time.

## What the data showed

The TP=2 arm completed and moved the **opposite way**:

| 8K decode | std at n=10 | std at n=30 | ratio |
|---|---|---|---|
| TP=3 | 2.43 | 5.87 | **2.4x** |
| TP=2 | 6.45 | 4.02 | **0.62x** |

The arms **converged** (CV 13.5% and 10.4%) rather than both becoming noisier.
At n=10, TP=2 was the noisier arm and TP=3 the quieter one; at n=30 that
reversed. Ten reps mis-estimated both, in opposite directions.

## Effect on the published caveat

| cell | t as run | t (2x both, published) | t (measured) | verdict |
|---|---|---|---|---|
| depth 0 | 4.28 | 2.14 | 2.19 | survives, either way |
| depth 32K | 2.63 | 1.31 | **1.94** | still fails, but **borderline** |
| depth 131K | 2.36 | 1.18 | **1.29** | still fails |
| cc=4 | 11.21 | 5.60 | 6.72 | survives |
| cc=8 | 26.68 | 13.34 | 15.02 | survives |
| cc=16 | 18.49 | 9.24 | 9.51 | survives |

**The conclusion stands: 32K and 131K remain provisional.** But the severity was
overstated. 32K sits at t=1.94 — just under the line — not the 1.31 I published,
which read as a clear failure.

Both documents now carry the measured figures, with the superseded 2x column
shown alongside so the change is visible rather than quietly swapped.

## The lesson

Applying one arm's measured behaviour to the other arm is a guess wearing the
clothes of a measurement. I flagged the assumption when publishing, which was
right, but the honest move would have been to wait ~10 minutes for the TP=2 arm
rather than publish a two-sided correction from one-sided data.

The direction of the caveat was right. Its magnitude was not, and a reader who
saw only the first version would have thought 32K was in worse shape than it is.
