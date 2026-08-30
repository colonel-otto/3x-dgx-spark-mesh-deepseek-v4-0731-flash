# Where the pre-registered thresholds were wrong

`thresholds-prereg.md` predicted **131K** would be the cell most likely to return
INCONCLUSIVE, because it had the highest bar (+11.2%) and our matched harness
puts the 131K advantage at only +7.3%.

**That prediction was wrong.** 131K resolved cleanly at +13.5%. The cell that
went inconclusive was **8K**, whose bar was the *lowest* of the four (+5.3%).

## Why

The threshold model assumed TP=2's coefficient of variation would resemble
TP=3's. At 8K it did not, by a wide margin:

| depth | TP=3 CV | TP=2 CV | ratio |
|---|---|---|---|
| 0 | 10.2% | 9.2% | 0.9x |
| 8192 | **5.7%** | **15.8%** | **2.8x** |
| 32768 | 6.7% | 12.3% | 1.8x |
| 131072 | 11.9% | 12.1% | 1.0x |

The TP=2 8K cell ranged 33.24-52.73 tok/s across 10 runs. TP=3 at the same depth
ranged 38.52-46.48. The inconclusive verdict is driven by spread in ONE arm, not
by the arms being close: the means are 40.74 vs 42.37, but TP=2's std of 6.45
swamps the 1.63 gap.

## What this does and does not mean

- It does **not** mean 8K is a depth where three nodes stop helping. Both
  neighbouring depths resolve at +11.9% and +13.5%, and prefill at 8K resolves
  at +12.5%. A real effect that vanished only at 8K and returned either side of
  it would be a strange physical claim.
- It does mean **n=10 was not enough for that cell given TP=2's actual
  variance**. The plan's remedy applies as written: raise n on that specific
  cell and re-measure, rather than reporting the +4.0% either way.
- The lesson for the threshold method: predicting the marginal cell from ONE
  arm's variance is unreliable, because the other arm's variance is unknown
  until it is measured. The thresholds were still worth pre-registering -- they
  fixed the decision rule before the data existed -- but their ranking of which
  cell was at risk carried no weight.

Recording this because the pre-registration is only worth something if its
misses are reported as plainly as its hits.
