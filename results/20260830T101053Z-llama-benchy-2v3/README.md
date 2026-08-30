# Independent llama-benchy 2v3 — 2026-08-30

**Status:** `CURRENT` · **Date:** 2026-08-30 · **Nodes/TP:** 2 and 3 (`TP=2`, `TP=3`) ·
**Live config source:** `tp2/engine-config.txt`, `tp3/engine-config.txt` (live argv) ·
**Harness:** `eugr/llama-benchy` 0.4.1.dev1+ge9be34457 (commit e9be344), **third-party —
not written by this project** · **Output tokens:** 256 asserted per request via
`--exact-tg` (`response_size: 256` in every cell) · **Reps:** 10 per cell per arm, 3
warm-ups per shape discarded · **Statistic:** mean ± std per cell ·
**Fabric gate:** present/pass both arms (TP=3 24/24, TP=2 14/14 — fewer directed pairs on
two nodes; NCCL bandwidth skipped, engine live)

Independent corroboration of [`20260830-matched-2v3-powered`](../20260830-matched-2v3-powered/)
on a harness this project did not write, so the node-count claim stops being
self-certified. Full write-up: [`docs/RESULT-LLAMA-BENCHY-2V3-2026-08-30.md`](../../docs/RESULT-LLAMA-BENCHY-2V3-2026-08-30.md).

## Result

**16 of 16 cells resolved; all favour three nodes; zero favour two.** 14 resolved here at
n=10; the remaining two resolved at n=30 in
[`20260830T130300Z-rerun-inconclusive`](../20260830T130300Z-rerun-inconclusive/).

### Decode — depth sweep

| depth | 2-node | 3-node | 3v2 |
|---|---|---|---|
| 0 | 39.02 ± 3.58 | 47.13 ± 4.80 | **+20.8%** |
| 8K | 40.74 ± 6.45 | 42.37 ± 2.43 | +4.0% → **+12.7%** at n=30 |
| 32K | 39.19 ± 4.80 | 43.86 ± 2.93 | **+11.9%** ⚠️ provisional |
| 131K | 39.23 ± 4.75 | 44.52 ± 5.28 | **+13.5%** ⚠️ provisional |

### Prefill — depth sweep (resolves on all four, advantage grows with depth)

| depth | 2-node | 3-node | 3v2 |
|---|---|---|---|
| 0 | 1641.3 ± 18 | 1851.9 ± 41 | **+12.8%** |
| 8K | 1578.2 ± 3 | 1776.0 ± 13 | **+12.5%** |
| 32K | 1520.0 ± 11 | 1728.2 ± 7 | **+13.7%** |
| 131K | 1357.9 ± 13 | 1571.8 ± 15 | **+15.8%** |

### Decode — concurrency sweep (`pp=8192`, `--no-cache`)

| cc | aggregate 3v2 | per-request 3v2 |
|---|---|---|
| 1 | +6.3% → **+14.8%** at n=30 | same cell |
| 4 | **+20.1%** | **+22.4%** |
| 8 | **+18.7%** | **+21.9%** |
| 16 | **+15.4%** | **+18.1%** |

Reported both ways deliberately: aggregate throughput *rises* with concurrency while
per-request throughput *collapses*. Quoting only the aggregate is the trap this repository
documents; the 2v3 ratio holds either way.

## Caveats that govern every number here

- ⚠️ **Cross-harness absolute t/s are NOT comparable.** `llama-benchy --depth N` prefills
  *cached* context; our `decode_depth_sweep.py` does not. Only the 2v3 ratio computed
  **within** each harness may be read against the other.
- **Quote +14.7%, not +16.7%**, against our +17–20% band. The like-for-like figure is the
  decode-at-depth mean across all four depths (+20.8 / +12.7 / +11.9 / +13.5). The pooled
  +16.7% mixes depth-sweep with concurrency decode and flatters the agreement.
- ⚠️ **32K and 131K decode magnitudes are PROVISIONAL.** The n=30 re-run showed n=10
  mis-estimates variance in either direction; under the measured per-arm ratios those two
  cells would not resolve (t 2.63 → 1.94 and 2.36 → 1.29). Direction is unaffected.
- **Confirmatory, not a replacement** for the matched run: n=10 here against n=30 there.
- **1 node is out of scope and impossible** for this checkpoint — it exceeds a single
  GB10's 128 GB.

## Method notes

Node count was the only variable: nine engine settings plus MoE backend and container image
asserted against the **live** engine per arm before measuring. The real tokenizer was proved
loaded per run (vocab 129,280) — llama-benchy silently falls back to `gpt2`, an 11.6%
token-count error. `--no-cache` on the concurrency sweep, confirmed by
`prefix_caching_enabled: false` in all four result files. Cooled to ≤70 °C before each arm;
clocks sampled every 5 s. open-webui stopped throughout. Driver 580.173.02, kernel
6.17.0-1029-nvidia, identical on all three nodes.

## Files

- `tp2/`, `tp3/` — depth and concurrency JSON, live engine argv, fabric-gate artifacts.
- `analysis.json` — per-cell 3v2, Welch t, verdicts.
- `thresholds-prereg.md` — decision thresholds fixed from the TP=3 arm **before** TP=2 ran.
- `prereg-outcome.md` — where that pre-registration was wrong, recorded rather than dropped.
- `power-check.md` — post-hoc power from the TP=3 arm's own variance.
- `software-stack.txt` — driver, kernel, image, harness commit.
