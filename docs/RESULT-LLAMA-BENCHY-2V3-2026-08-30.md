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
depth and every concurrency.** 14 of 16 cells resolved at n=10; the remaining two
were re-measured at n=30 and **both resolved, three nodes faster** (+12.7% and
+14.8%), making it **16 of 16**.

The claim is no longer self-certified.

## Both pre-registered expectations held

| # | Expectation | Outcome |
|---|---|---|
| L1 | Three nodes faster in every cell | **HELD** — 16/16 cells once the two inconclusive ones were re-measured at n=30; zero cells favour two nodes |
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

## ⚠️ Variance correction — read before quoting the depth-sweep magnitudes

A higher-n re-run of the two inconclusive cells
([`RESULT-RERUN-INCONCLUSIVE-2026-08-30.md`](RESULT-RERUN-INCONCLUSIVE-2026-08-30.md),
`n=30`) found that **`n=10` mis-estimates the spread — in either direction**:

| cell / arm | std at n=10 | std at n=30 | ratio |
|---|---|---|---|
| 8K, TP=3 | 2.43 | 5.87 | **2.4x** |
| 8K, TP=2 | 6.45 | 4.02 | **0.62x** |
| cc=1, TP=3 | 2.77 | 5.01 | **1.8x** |
| cc=1, TP=2 | 5.61 | 6.18 | 1.1x |

At 8K the two arms **swapped**: TP=2 was the noisier arm at n=10 and the quieter
one at n=30. Means were stable throughout (2.5%–5.4%) — it was always the
spread, never the centre. Since n=10 can err either way, it bears on the cells
this run *did* resolve.

**Both re-measured cells resolved at n=30, three nodes faster** — 8K decode
**+12.7%** (t=3.76), cc=1 decode **+14.8%** (t=4.24). With those two settled,
this run's tally is **16 of 16 cells resolved, all favouring three nodes.**

**The two arms did not move together.** An earlier version of this section
assumed both arms' spreads had been understated and inflated both by 2.0x. The
completed TP=2 arm falsified that: TP=2's std at 8K went *down* at n=30
(6.45 → 4.02, **0.62x**) while TP=3's went *up* (2.43 → 5.87, **2.4x**). The
arms converged toward similar CVs (10.4% and 13.5%) rather than both worsening.

Recomputing with the **measured** per-arm ratios (TP=2 x0.62, TP=3 x2.4), and
showing the superseded 2x-both figure for comparison:

| cell | 3v2 | t as run | t (2x both, superseded) | t (measured) | survives |
|---|---|---|---|---|---|
| depth 0 | +20.8% | 4.28 | 2.14 | 2.19 | yes |
| depth 32K | +11.9% | 2.63 | 1.31 | **1.94** | **no — but borderline** |
| depth 131K | +13.5% | 2.36 | 1.18 | **1.29** | **no** |
| cc=4 | +20.1% | 11.21 | 5.60 | 6.72 | yes |
| cc=8 | +18.7% | 26.68 | 13.34 | 15.02 | yes |
| cc=16 | +15.4% | 18.49 | 9.24 | 9.51 | yes |

The conclusion is unchanged — 32K and 131K remain provisional — but the severity
was **overstated** in the 2x-both version. 32K lands at t=1.94, just under the
threshold rather than well below it.

**Honest restatement: the direction is robust everywhere — no cell flips under
any inflation tested — but the magnitude is firmly established only at
concurrency and depth 0, and is provisional at 32K and 131K.** The concurrency
cells would survive a 5x inflation; prefill's CVs were ~0.2-1% and have orders
of magnitude of headroom. The matched run (our own harness, n=30) is a separate
measurement and is unaffected.

This does not retract the result — it narrows which parts carry weight. Full
working: `results/20260830T130300Z-rerun-inconclusive/robustness-caveat.md` and
`variance-caveat.md`.

## The two inconclusive cells — now RESOLVED at n=30

Both were re-measured at n=30 and **both resolve, three nodes faster**:

| cell | n=10 (this run) | n=30 (re-run) | verdict |
|---|---|---|---|
| 8K decode | +4.0%, inconclusive | **+12.7%**, t=3.76 | 3 nodes faster |
| cc=1 decode | +6.3%, inconclusive | **+14.8%**, t=4.24 | 3 nodes faster |

Full result: [`RESULT-RERUN-INCONCLUSIVE-2026-08-30.md`](RESULT-RERUN-INCONCLUSIVE-2026-08-30.md).
**Do not re-run these cells again** — they are settled at n=30.

### A claim in an earlier version of this document was wrong

That version argued these cells were **not** underpowered, on the grounds that
their n=10 CIs excluded the +15.4% depth-sweep effect while including zero —
concluding that "the effect at these two shapes is genuinely smaller" and that
they were where three nodes help least.

**The n=30 CIs include +15.4%:**

| cell | n=10 CI | n=30 CI |
|---|---|---|
| 8K decode | [-6.5%, +14.5%] | **[+6.1%, +19.3%]** |
| cc=1 decode | [-2.6%, +15.2%] | **[+8.0%, +21.7%]** |

Nothing is special about these two shapes; at +12.7% and +14.8% they sit inside
the range of every other decode cell (+11.9% to +20.8%). The n=10 CIs excluded
+15.4% only because they were built on TP=3 standard deviations 2.4x and 1.8x
too small. **A confidence interval is honest about its uncertainty given its
inputs; it cannot tell you its inputs are wrong.** The reasoning was internally
valid and the conclusion was still false.

Note also that cc=1 here is a *different measurement* from the depth sweep's
cc=1 (pp=8192 vs pp=2048), which is why they differed (+6.3% vs +20.8%).

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
