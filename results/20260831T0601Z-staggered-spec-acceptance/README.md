# Staggered ragged-context load — MTP acceptance and KV stability (2026-08-31)

**Status:** `CURRENT` · **Engine:** `eugr/spark-vllm-b12x:latest`, `eugr.service` on port 8100,
nst=5 / mnbt=8192, persistent kernel caches · **Nodes:** 3 (TP=3), `--kv-cache-dtype fp8`,
`--max-num-seqs 16`, `--gpu-memory-utilization 0.82`, prefix caching on · **Config id:**
`eugr-tp3-seqs16-dspark5-cached` (the serving config, same as the K-sweep winner) ·
**Fabric gate:** inherited from `20260830T2245Z-eugr-ksweep`'s pre-boot gate (30/30 incl. NCCL
bandwidth) on the same service boot; no NCCL warnings in the service log.

This is a **correctness and stability gate, not a throughput result.** See the warning below
before quoting any tok/s number from this bundle.

## What it settles

`scripts/benchmark_staggered_spec_acceptance.py` sends Poisson-arrival requests whose prompt
lengths are drawn log-uniformly from a tiered mixture (50% 1–8K, 35% 8–32K, 15% 32–131K), so
requests of wildly different depth arrive and retire while the batch is in motion. 150 measured
requests (30 per tier) plus 2 warmups, `min_tokens=max_tokens=256`, `ignore_eos=True`.

| cc | requests | errors | window fails | preemptions | acceptance | TTFT median |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 0 | 0 | 0 | 41.3% | 3.53 s |
| 4 | 30 | 0 | 0 | 0 | 40.8% | 7.79 s |
| 8 | 30 | 0 | 0 | 0 | 41.6% | 12.49 s |
| 16 | 30 | 0 | 0 | 0 | 41.3% | 61.88 s |
| 32 | 30 | 0 | 0 | 0 | 40.0% | 61.04 s |

**VERDICT: PASS** — 0 errors, 0 window failures, 0 preemptions across all 150 requests;
minimum acceptance 40.0% against a 30% floor.

1. **MTP draft acceptance survives asynchronous continuous batching.** Acceptance is flat
   (40.0–41.6%) from c=1 to c=32 — twice `--max-num-seqs 16`. There is no collapse at depth,
   so the nst=5 draft depth is not degraded by ragged arrival.
2. **Ragged-context churn is safe.** Prompts from 1,154 to 128,275 tokens interleaving in the
   same batch produced no HTTP errors and no short completions. The KV allocator handled the
   churn with **zero preemptions**, so no request was ever evicted and recomputed.
3. **The virtual-TP zero-fill attention sink is stable under dynamic KV compaction** — the
   condition that would corrupt it (slot reuse across disparate context depths) is exactly what
   this workload generates, and correctness held.

## Do not read decode throughput off this run

The per-request `Decode_med` printed in `run.log` falls 51.8 → 1.5 tok/s across the tiers. That
is **not** engine decode speed: it is per-request wall-clock while sharing a batch, and it is
dominated by deep-context prefill blocking the batch. This workload is prefill-bound by
construction — each tier generates a fixed 7,680 output tokens against 442K–1.02M prompt tokens,
a 58–132x prefill:decode ratio:

| cc | tier wall | output tok | aggregate tok/s | prompt tok | prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 334.4 s | 7,680 | 23.0 | 488,440 | 1,460.7 |
| 4 | 310.1 s | 7,680 | 24.8 | 628,353 | 2,026.4 |
| 8 | 304.9 s | 7,680 | 25.2 | 673,653 | 2,209.1 |
| 16 | 441.3 s | 7,680 | 17.4 | 1,017,018 | 2,304.6 |
| 32 | 219.1 s | 7,680 | 35.1 | 442,236 | 2,018.9 |

The ~1,460–2,300 tok/s prefill rate is the meaningful throughput figure here and is consistent
with the rates in `docs/PREFILL-MEASURED.md`. **Decode throughput for this config remains the
K-sweep's numbers** (84.3 at c=1 rising to 252.9 at c=8) — those were measured on a
decode-shaped workload. Quoting 17–35 tok/s as "eugr throughput" would be the aggregate-metric
trap recorded in `docs/BENCHMARK-POLICY.md`.

Total wall time 26.8 min.

## Files
`run.log` (harness stdout with the per-tier verdict lines), `staggered_spec_acceptance.json`
(per-request records: prompt/completion tokens, TTFT, decode rate, errors, plus per-tier
spec-decode counter deltas), `engine-cmdline.txt` (the serving process argv as measured),
`eugr-unit-env.txt` (pinned `EUGR_NST` / `EUGR_MNBT`), `metrics-after.txt` (full Prometheus
scrape at run end).

The `--master-addr` value in `engine-cmdline.txt` is redacted to the documented placeholder
range (`192.168.10.1`); every flag that bears on the measurement is verbatim as run.
