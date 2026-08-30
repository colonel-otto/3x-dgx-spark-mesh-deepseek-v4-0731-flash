# n=30 re-run of the two unresolved cells — 2026-08-30

**Status:** `CURRENT` · **Date:** 2026-08-30 · **Nodes/TP:** 2 and 3 (`TP=2`, `TP=3`) ·
**Live config source:** `tp2/engine-config.txt`, `tp3/engine-config.txt` (live argv) ·
**Harness:** `eugr/llama-benchy` 0.4.1.dev1+ge9be34457 (commit e9be344), third-party ·
**Output tokens:** 256 asserted per request via `--exact-tg` (`response_size: 256` in
every cell) · **Reps:** 30 per cell per arm, 3 warm-ups per shape discarded ·
**Statistic:** mean ± std, Welch t and 95% CI in `analysis.json` ·
**Fabric gate:** present/pass both arms (TP=3 24/24, TP=2 14/14, NCCL bandwidth skipped —
engine live)

Re-measures the only two cells that did not resolve in
[`20260830T101053Z-llama-benchy-2v3`](../20260830T101053Z-llama-benchy-2v3/).

## Result

| cell | 2-node | 3-node | 3v2 | Welch t | 95% CI |
|---|---|---|---|---|---|
| 8K decode | 38.55 ± 4.02 | 43.44 ± 5.87 | **+12.7%** | 3.76 | [+6.1%, +19.3%] |
| cc=1 decode | 41.57 ± 6.18 | 47.73 ± 5.01 | **+14.8%** | 4.24 | [+8.0%, +21.7%] |

**Both resolved, three nodes faster.** The parent run is now 16 of 16 cells resolved,
all favouring three nodes, zero favouring two.

The two cells need **different commands** and are not the same measurement: 8K decode is
the depth-sweep shape (`--pp 2048 --depth 8192`), cc=1 is the concurrency-sweep shape
(`--pp 8192 --depth 0 --no-cache`). Both run at concurrency 1, which is why they gave
different answers at n=10 (+4.0% vs +6.3%).

## The finding that matters more than the result

**`n=10` mis-estimates variance in *either* direction.** At 8K the two arms swapped:

| cell / arm | std at n=10 | std at n=30 | ratio |
|---|---|---|---|
| 8K, TP=3 | 2.43 | 5.87 | **2.4x** |
| 8K, TP=2 | 6.45 | 4.02 | **0.62x** |
| cc=1, TP=3 | 2.77 | 5.01 | **1.8x** |
| cc=1, TP=2 | 5.61 | 6.18 | 1.1x |

Means were stable throughout (2.5%–5.4% shifts) — it was always the spread, never the
centre. Applying the measured per-arm ratios to the parent run's cells still at n=10, its
**32K (t 2.63 → 1.94) and 131K (t 2.36 → 1.29) decode magnitudes would not resolve** and
are marked provisional wherever they appear. Concurrency (t 6.7–15.0) and prefill are
unaffected. No cell flips direction under any inflation tested.

## What this retracts

An earlier version of the parent result doc argued these cells were not underpowered — that
their n=10 CIs excluded +15.4% while including zero, so "the effect at these two shapes is
genuinely smaller". **The n=30 CIs include +15.4%.** Nothing is special about these shapes;
both sit inside the range of every other decode cell. The n=10 CIs excluded +15.4% only
because they rested on standard deviations 2.4x and 1.8x too small.

A second, separate correction is recorded in
[`correction-2x-assumption.md`](correction-2x-assumption.md): the first published robustness
analysis inflated **both** arms 2x from the TP=3 arm alone, before TP=2 had been measured.

## Method notes

- **n=30 was pre-committed** in `scripts/rerun_inconclusive_cells.sh` before any of these
  numbers existed, so a resolving result cannot be mistaken for a stopping rule chosen
  after the fact. **Do not re-run these cells** — they are settled.
- Node count was the only variable: nine engine settings plus MoE backend and container
  image asserted against the **live** engine per arm. spark2 held 0% utilisation / 51 °C
  throughout the TP=2 arm.
- Correctness gate (17×23=391) passed on both arms; coherence test passed on every sweep.

## Files

- `tp2/`, `tp3/` — per-cell llama-benchy JSON, live engine argv, fabric-gate artifacts.
- `analysis.json` — Welch t, CIs, variance comparison against the parent run.
- `variance-caveat.md`, `robustness-caveat.md`, `correction-2x-assumption.md` — the
  variance finding, its effect on the parent run, and the correction to my own first
  analysis of it.
