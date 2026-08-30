# Matched, powered 2-node vs 3-node comparison — 2026-08-30

**Status:** `CURRENT`. **This is the settled node-count comparison.** It supersedes the
decode and concurrency rows of `20260827-decode-2v3-fixed` and
`20260827-decode-concurrency-2v3-fixed`.

Analysis: [`docs/RESULT-2V3-MATCHED-2026-08-30.md`](../../docs/RESULT-2V3-MATCHED-2026-08-30.md)
Pre-registration (written before measuring): [`docs/PREREGISTRATION-2V3-MATCHED.md`](../../docs/PREREGISTRATION-2V3-MATCHED.md)

## What makes this one different

Every prior 2v3 comparison in this repository differed in **six** engine settings and ran
at n=7. This one is configuration-identical and powered:

| | prior arms | this run |
|---|---|---|
| `MAX_NUM_SEQS` | 16 vs 32 | **32 both** |
| `MTP_NUM_TOKENS` | 5 vs 2 | **2 both** |
| `GPU_MEMORY_UTILIZATION` | 0.80 vs 0.835 | **0.835 both** |
| `LONG_PREFILL_TOKEN_THRESHOLD` | unset vs 1024 | **1024 both** |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | unset vs 2 | **2 both** |
| `KV_CACHE_DTYPE` | unset vs nvfp4_ds_mla | **nvfp4_ds_mla both** |
| reps | 7 | **30** (12 at 262K, 15 per cc cell) |

Both arms verified against the **live engine** before measuring (`MATCH CONFIRMED` in
`orchestration*.log`). Nodes cooled to ≤70 °C before each arm; clocks sampled every 5 s
throughout (`clocks-*.csv`), because GB10 clocks cannot be locked.

## Results

Decode, cc=1, median tok/s:

| Depth | TP=3 | TP=2 | Delta | p | Cliff's δ |
|---:|---:|---:|---:|---:|---:|
| 2K | 46.59 | 43.68 | +6.7 % | 6.0e-05 | 0.604 |
| 8K | 51.07 | 43.62 | +17.1 % | 3.5e-10 | 0.944 |
| 32K | 50.83 | 42.29 | +20.2 % | 3.0e-11 | **1.000** |
| 131K | 47.38 | 39.92 | +18.7 % | 3.3e-11 | 0.998 |
| 262K | 45.04 | 39.79 | +13.2 % | 4.0e-05 | **1.000** |

Aggregate throughput @8K:

| cc | TP=3 | TP=2 | Delta | Cliff's δ |
|---:|---:|---:|---:|---:|
| 4 | 41.85 | 35.28 | +18.6 % | +1.000 |
| 8 | 49.83 | 40.76 | +22.3 % | +1.000 |
| 16 | 54.20 | 44.45 | +21.9 % | +1.000 |

KV pool, each arm's own init log: TP=3 **4,688,072** vs TP=2 **2,217,166** tokens
(**2.11×**, not the 2.6× previously published).

Warm TTFT favours TP=3 from 32K up (−11.9 % / −14.4 % / −26.6 % at 32K/131K/262K,
δ = −1.000). At 2K and 8K the arms overlap — reported as ties.

**No measured workload in this bundle favours two nodes.**

## Hygiene

- `EXCLUSIVITY_PASS delta=654 expected=654` on **both** arms (`tp*/exclusivity.json`).
- Every rep returned exactly 256 completion tokens; `cached_tokens=0` throughout.
- Correctness `17 × 23 = 391` verified per arm before measuring.
- Fabric gate artifacts committed per arm (`tp*/fabric-gate.json`).

## Layout

- `tp2/`, `tp3/` — per-arm depth JSONL + summaries, concurrency JSON, engine config, KV
  pool init log, exclusivity record, fabric gate, clock telemetry.
- `tp3-n7/` — the earlier n=7 TP=3 arm, retained rather than deleted.
- `orchestration.log`, `orchestration-tp2.log` — full run logs.
- `patch-hashes-*.txt` — TP=3 padding patch sha256, verified identical across all 3 nodes.

## Caveats

- TTFT figures are **warm** (3 warm-ups per shape, both arms), not cold-start.
- Prefill throughput and the APC warm path were **not** re-tested here.
- Raw per-cell spread stays wide (GB10 clocks float); n=30 is what makes this resolvable.
