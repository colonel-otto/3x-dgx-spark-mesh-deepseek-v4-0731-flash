# Issue #29 — the warm path: multi-turn prefix caching, first measurement

**Status:** `CURRENT` · **Nodes:** 3 · **Date:** 2026-08-28 (UTC)
**Harness:** `scripts/multiturn_apc.py` (new, this bundle).
**Config:** live 3-node TP=3 Profile B, baked image `dsv4-3spark:0.1.1`
(`d72817dc7657`), `MAX_NUM_BATCHED_TOKENS=8192`, `GPU_MEMORY_UTILIZATION=0.835`,
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`. Engine untouched — live serving run.
**Cache regime:** declared and asserted per record (`cache_regime: cold|warm`),
zero regime violations across 63 turns.

## Headline

| depth | cold TTFT (median, n=3) | warm TTFT (median, n=15) | speedup | warm hit ratio |
|---:|---:|---:|---:|---:|
| 8K | 4.14 s | **0.417 s** | **9.9x** | 94.7% |
| 32K | 16.97 s | **0.455 s** | **37.3x** | 99.0% |
| 131K | 78.09 s | **0.731 s** | **106.8x** | 99.8% |

The cold 131K reference (78.09 s) independently matches the published
`20260827-tp3-131k-15rep` value (79.0 s), which cross-validates the harness.

## Retention boundary (issue #29 success criterion 2)

Think-time gaps of **30 s and 120 s** between turns at 32K: no degradation at all
(warm TTFT 0.42–0.45 s, hit 98.9–99.1% — identical to gap 0). The prefix survives at
least 2 minutes of idle under Profile B. The boundary, if one exists, is beyond 120 s;
untested further because 6-turn sessions at larger gaps cost wall-clock linearly.

## Findings beyond the numbers

1. **Per-request `cached_tokens` reporting is broken in this build.** Usage
   `prompt_tokens_details` is null / `cached_tokens=0` even on a turn with a
   sub-second TTFT over 131K tokens. The engine-level Prometheus counters
   (`vllm:prefix_cache_{queries,hits}_total`) do track correctly, and the harness
   measures hits from per-turn counter deltas (valid on this single-user cluster —
   requests are strictly sequential). Any future warm-regime assertion must use the
   metrics endpoint, not the OpenAI usage block.
2. **Occasional warm-turn stall, ~5 s:** 2 of 30 gap-0 warm turns (one at 8K, one at
   32K) had TTFT ≈ 5.1 s with a NORMAL hit ratio — the cache hit, but first token
   took ~5 s. Magnitude matches the known JIT-compile-inside-a-request stall
   (`JIT_MONITOR_MODE=warn` exists for exactly this). Not investigated further here.
3. **One unexplained cold-ish warm turn in the smoke run** (`smoke.jsonl`, retained on
   sparkmain, not in this bundle): the very first multi-turn session after harness
   deployment re-prefilled turn 2 in full, then hit from turn 3 on. Never reproduced
   in 63 subsequent turns. Recorded as an anomaly, cause unknown.

## Provenance

- Raw: `apc-gap0.jsonl` (54 turns), `apc-gap30.jsonl`, `apc-gap120.jsonl` (6 each),
  with `-summary.json` and full stdout logs.
- No fabric gate: gate requires stopping the engine; this was a live-service run.
  TTFT/cache measurements are not fabric-sensitive the way collective-bound
  throughput arms are, but the bundle should be indexed `gate absent` like other
  live runs.
- Decode windows pinned and asserted at 256 tokens (issue #26 guard) on every turn;
  decode tok/s in the raw records is consistent with the published 51–56 band.
- Sessions salted uniquely; sessions cannot warm each other.

## What the user actually feels

Turn 1 of a 131K-context coding session costs ~78 s. Every turn after it costs
**under a second** until the conversation grows past the KV pool. This is the
largest single perceived-latency effect measured in this repo — two orders of
magnitude — and it was previously invisible because every bundle asserted the
cold path.
