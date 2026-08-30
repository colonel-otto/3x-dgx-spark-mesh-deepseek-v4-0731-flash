# RESULT — llama-benchy 2v3: an independent harness agrees on direction, and mostly on magnitude

**Run:** `results/20260830T101053Z-llama-benchy-2v3/` (2026-08-30, 06:10–08:40 EDT)
**Plan:** [`PLAN-LLAMA-BENCHY-2V3.md`](PLAN-LLAMA-BENCHY-2V3.md)
**Tool:** [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) `0.4.1.dev1+ge9be34457`
**Scope:** 2 nodes vs 3 nodes. **1 node was not tested and cannot be** — the
checkpoint exceeds a single GB10's 128 GB.

## Headline

A third-party harness we did not write reproduces the direction of
[`RESULT-2V3-MATCHED-2026-08-30.md`](RESULT-2V3-MATCHED-2026-08-30.md) on every
resolved cell: **three nodes are faster, on decode and on prefill, at every
depth and every concurrency that resolved.** 14 of 16 cells resolved; the other
two were underpowered at n=10, not contrary.

The claim is no longer self-certified.

## Both pre-registered expectations held

| # | Expectation | Outcome |
|---|---|---|
| L1 | Three nodes faster in every cell | **HELD** — 14/14 resolved cells; zero cells favour two nodes |
| L2 | Absolute t/s will not match ours | **As expected**, and never compared (see caveat) |
| L3 | 2v3 ratio within ~5 pp of our +17–20% | **HELD** — decode pool mean +16.7% |

## Decode — depth sweep (n=10)

| depth | 2-node | 3-node | 3v2 | verdict |
|---|---|---|---|---|
| 0 | 39.02 ± 3.58 | 47.13 ± 4.80 | **+20.8%** | 3 nodes faster |
| 8K | 40.74 ± 6.45 | 42.37 ± 2.43 | +4.0% | INCONCLUSIVE |
| 32K | 39.19 ± 4.80 | 43.86 ± 2.93 | **+11.9%** | 3 nodes faster |
| 131K | 39.23 ± 4.75 | 44.52 ± 5.28 | **+13.5%** | 3 nodes faster |

## Prefill — depth sweep (n=10)

Resolves on **all four** cells, and the advantage *grows* with depth:

| depth | 2-node | 3-node | 3v2 |
|---|---|---|---|
| 0 | 1641.3 ± 18 | 1851.9 ± 41 | **+12.8%** |
| 8K | 1578.2 ± 3 | 1776.0 ± 13 | **+12.5%** |
| 32K | 1520.0 ± 11 | 1728.2 ± 7 | **+13.7%** |
| 131K | 1357.9 ± 13 | 1571.8 ± 15 | **+15.8%** |

## Decode — concurrency sweep (pp=8192, `--no-cache`, n=10)

| cc | aggregate 2-node | aggregate 3-node | 3v2 | per-request 3v2 |
|---|---|---|---|---|
| 1 | 43.54 ± 6 | 46.28 ± 3 | +6.3% (inconclusive) | +6.3% (inconclusive) |
| 4 | 42.42 ± 2 | 50.96 ± 2 | **+20.1%** | **+22.4%** |
| 8 | 44.44 ± 1 | 52.73 ± 1 | **+18.7%** | **+21.9%** |
| 16 | 46.43 ± 1 | 53.56 ± 1 | **+15.4%** | **+18.1%** |

Reported both ways deliberately. Aggregate throughput *rises* with concurrency
(43→46 on two nodes, 46→54 on three) while per-request throughput *collapses*
(43→6 and 46→7). Quoting only the aggregate is the trap this repository already
documents; the 2v3 ratio holds either way, which is the point.

## The two inconclusive cells

Neither favours two nodes. Both are **one arm being noisy at n=10**, not the
arms being equal.

**8K decode.** TP=2 measured CV 15.8% there (range 33.24–52.73 tok/s) against
TP=3's 5.7% — a 2.8x variance asymmetry. The means differ by 1.63 tok/s; a std
of 6.45 swamps it. Both neighbouring depths resolve (+11.9%, +13.5%) and prefill
at 8K resolves at +12.5%, so an effect that vanished only at 8K and returned
either side of it is not a credible physical claim.

**cc=1 decode.** TP=2 std 6.45 against TP=3's 2.77, same pattern. Note cc=1 here
is a *different measurement* from the depth sweep's cc=1 (pp=8192 vs pp=2048),
which is why they disagree (+6.3% vs +20.8%).

Per the plan: raise n on these two cells and re-measure. Do **not** report
+4.0% or +6.3% as findings in either direction.

## Where our pre-registration was wrong

[`thresholds-prereg.md`](../results/20260830T101053Z-llama-benchy-2v3/thresholds-prereg.md)
predicted **131K** would be the marginal cell — highest bar (+11.2%), smallest
expected effect (+7.3% in our matched run). **That was wrong.** 131K resolved
cleanly at +13.5%; the cell that failed was 8K, which had the *lowest* bar (+5.3%).

The threshold model assumed TP=2's CV would resemble TP=3's. It did not, and
could not have been known before measuring the second arm. The thresholds were
still worth fixing in advance — they pinned the decision rule before the data
existed — but their *ranking* of which cell was at risk carried no weight.
Recorded in full at
[`prereg-outcome.md`](../results/20260830T101053Z-llama-benchy-2v3/prereg-outcome.md).

## The caveat that governs every number here

`llama-benchy --depth N` prefills N tokens of **cached context** and measures on
top of it. Our `decode_depth_sweep.py` sends an N-token prompt with **no**
caching. These are different measurements.

**Cross-harness absolute t/s are not comparable and are never compared here.**
Only the 2v3 ratio computed *within* each harness may be read against the other.
The analyzer enforces this structurally — it has no code path that divides one
harness's absolute by another's.

### Read this before quoting +16.7%

L3's pool mean of **+16.7%** covers six resolved decode cells — three from the
depth sweep, three from concurrency. Those two sweeps do not agree:

- depth-sweep decode: **+15.4%** (n=3 cells)
- concurrency decode: **+18.1%** (n=3 cells)

Our harness's +17–20% band was measured on **decode at depth**, so the
like-for-like comparison is **+15.4%**, which sits ~1.6 pp below our band's floor
— inside the ±5 pp tolerance, but at the low end. The single pooled figure
flatters the agreement slightly. Quote +15.4% against our depth result, or state
the pool explicitly.

## Fairness controls, all asserted rather than assumed

- **Node count was the only variable.** Both arms asserted against the *live*
  engine before measuring: TP, `max-num-seqs 32`, `gpu-memory-utilization 0.835`,
  `max-num-batched-tokens 8192`, `max-model-len 1048576`,
  `kv-cache-dtype nvfp4_ds_mla`, `long-prefill-token-threshold 1024`,
  `num_speculative_tokens 2`, `moe-backend flashinfer_b12x`, same container image
  (`dsv4-3spark:0.1.1`) on every rank.
- **This nearly failed.** `head.env`/`worker.env` were found at seqs=16, MTP=5,
  gpumem=0.80, *missing three settings entirely*, and pointing at a different
  image — seven variables adrift. See the commit for
  `scripts/match_env_for_benchy.sh`.
- **Real tokenizer proven per run**, vocab 129,280. llama-benchy falls back to
  gpt2 silently; that fallback carries an 11.6% token-count error.
- **`--exact-tg`** — every cell in all four result files records
  `response_size: 256`, and the live TP=3 counters during the run showed
  benchmark requests finishing on `length` rather than `stop`. The 25-token
  collapse cannot recur.
- **`--no-cache`** on the concurrency sweep, so prefix caching cannot serve the
  prefill (`prefix_caching_enabled: false` in both result files).
- **Fabric gate passed** both arms (24/24 on 3-node, 14/14 on 2-node — fewer
  directed pairs, correctly).
- **Coherence test passed** on every sweep; correctness gate returned 391 on both
  arms. Guards the "serves fluent nonsense" trap.
- **Cooled to ≤70 °C before each arm**; clocks sampled every 5 s.
  spark2 held 50 °C flat across all 910 samples of the TP=2 arm — independent
  confirmation the third node was genuinely idle.
- **open-webui stopped** for the duration.
- **Software stack:** driver 580.173.02, kernel 6.17.0-1029-nvidia, all three
  nodes identical. Driver is a first-class variable on GB10.

## What this does not establish

- **Nothing about 1 node.** Out of scope and impossible for this checkpoint.
- **No comparability with published Spark numbers** at different quantization,
  context depth, or driver version.
- **Not a replacement for the matched run.** Confirmatory, at n=10 vs n=30.
- **262K untested here** — excluded by the plan at ~173 s/rep; our own harness
  covers it at n=12, Cliff's δ = 1.000.
