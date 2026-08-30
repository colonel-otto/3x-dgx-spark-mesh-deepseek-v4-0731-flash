# RESULT — the two inconclusive cells resolve at n=30, and refute my reading of them

**Run:** `results/20260830T130300Z-rerun-inconclusive/` (2026-08-30, 09:03–09:45 EDT)
**Parent:** [`RESULT-LLAMA-BENCHY-2V3-2026-08-30.md`](RESULT-LLAMA-BENCHY-2V3-2026-08-30.md)
**n:** 30 per arm per cell, **pre-committed before any of these numbers existed**

## Headline

Both cells resolve. **Three nodes faster in both.**

| cell | 2-node | 3-node | 3v2 | Welch t | 95% CI |
|---|---|---|---|---|---|
| 8K decode | 38.55 ± 4.02 | 43.44 ± 5.87 | **+12.7%** | 3.76 | [+6.1%, +19.3%] |
| cc=1 decode | 41.57 ± 6.18 | 47.73 ± 5.01 | **+14.8%** | 4.24 | [+8.0%, +21.7%] |

The parent run is now **16 of 16 decode/prefill cells resolved, all 16 favouring
three nodes, zero favouring two.**

## What this refutes — my own claim, not the result

When these cells came back inconclusive at n=10, I argued they were **not**
underpowered: their 95% CIs excluded the +15.4% depth-sweep effect while
including zero, so I concluded "the effect at these two shapes is genuinely
SMALLER" and called them the shapes where three nodes help least.

**That was wrong.** At n=30 both CIs *include* +15.4%:

| cell | n=10 CI | n=30 CI | includes +15.4%? |
|---|---|---|---|
| 8K decode | [-6.5%, +14.5%] | [+6.1%, +19.3%] | n=10 **no** → n=30 **yes** |
| cc=1 decode | [-2.6%, +15.2%] | [+8.0%, +21.7%] | n=10 **no** → n=30 **yes** |

Nothing is special about these two shapes. At +12.7% and +14.8% they sit
squarely inside the range of every other decode cell (+11.9% to +20.8%).

The n=10 CIs excluded +15.4% only because they were computed from TP=3 standard
deviations that were **2.4x and 1.8x too small**. A CI is honest about its own
uncertainty *given its inputs*; it cannot warn you that its inputs are wrong.
The reasoning was internally valid and the conclusion was still false.

## The variance story is more interesting than the result

The two arms did **not** move together between n=10 and n=30:

| cell / arm | std at n=10 | std at n=30 | ratio |
|---|---|---|---|
| 8K, TP=3 | 2.43 | 5.87 | **2.4x** |
| 8K, TP=2 | 6.45 | 4.02 | **0.62x** |
| cc=1, TP=3 | 2.77 | 5.01 | **1.8x** |
| cc=1, TP=2 | 5.61 | 6.18 | 1.1x |

At 8K they **swapped**: TP=2 looked like the noisy arm at n=10 and the quiet one
at n=30. Ten reps mis-estimated both arms, in *opposite* directions. Means were
stable throughout (2.5%–5.4% shifts) — it was always the spread, never the
centre.

This is why the correct statement is "n=10 **mis-estimates** variance", not
"n=10 **understates** variance". I published the latter first, from the TP=3 arm
alone, and the TP=2 arm falsified it within the hour. See
[`correction-2x-assumption.md`](../results/20260830T130300Z-rerun-inconclusive/correction-2x-assumption.md).

## Effect on the parent run's provisional cells

The published caveat marks 32K and 131K decode provisional, on the grounds that
their t values (2.63, 2.36) sit close to the threshold and the variance estimates
behind them are unreliable. **That caveat stands** — this run did not re-measure
32K or 131K, so nothing here rehabilitates them.

But its framing needs the same correction as above: the risk is that n=10
mis-estimated those cells' variance in *either* direction, not that it
necessarily understated it. Settling them requires measuring them, which this
run did not do.

## Updated depth-sweep mean

Adding the now-resolved 8K cell to the depth-sweep decode figures:

| depth | 3v2 |
|---|---|
| 8K | +12.7% (this run, n=30) |
| 32K | +11.9% (n=10, provisional) |
| 131K | +13.5% (n=10, provisional) |
| 0 | +20.8% (n=10) |
| **mean** | **+14.7%** |

Against our matched harness's +17–20% band, ~2.3 pp below the floor — still
inside the ±5 pp pre-registered tolerance, still at the low end. The direction
and rough magnitude corroborate; the exact figure does not match, which L2
predicted and which the differing depth semantics explain.

## Method notes

- **n=30 was pre-committed** in the script header before these numbers existed,
  precisely so a resolving result could not be mistaken for a stopping rule
  chosen after the fact. Both cells resolved; had they not, the tightened CI
  would have been the reportable outcome.
- **Do not re-run these cells again.** They are settled at n=30.
- Same fairness controls as the parent: node count the only variable, nine
  settings asserted against the live engine per arm, real-tokenizer proof,
  fabric gate (24/24 then 14/14), correctness 391 both arms, cooldown to ≤70 °C,
  telemetry, automatic production restore. spark2 idle at 0% / 51 °C throughout
  the TP=2 arm.
- The two cells need **different commands** — 8K decode is the depth-sweep shape
  (`--pp 2048 --depth 8192`), cc=1 is the concurrency-sweep shape
  (`--pp 8192 --depth 0 --no-cache`). Both run at concurrency 1 and are not the
  same measurement.
