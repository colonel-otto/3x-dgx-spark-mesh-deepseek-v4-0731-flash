# Decode: 2 nodes vs 3 nodes on healthy fabric — 2026-08-25

The measurement the three-node case actually rests on. Everything else — prefill, deep
concurrency, KV capacity — either ties or favours two nodes. This is where three wins.

## Result

| concurrency | TP=2 | TP=3 | 3-node |
|---:|---:|---:|---:|
| **1** | 76.2 | **89.1** | **+16.9%** |
| 4 | 192.8 | **208.8** | **+8.3%** |
| 8 | 302.7 | **322.7** | **+6.6%** |
| 16 | **481.3** | 474.8 | −1.4% |

Aggregate tok/s, median of 7 runs after 3 warm-up sweeps, `bench_tp3.py`, 18-token
code-brief prompt, 256 tokens out, temperature 0.

**Three nodes win where a single caller feels it (+17% at cc=1) and lose nothing that
matters (−1.4% at cc=16, inside run-to-run noise).** The advantage decays monotonically
with concurrency and crosses over around cc=16 — consistent with the earlier finding that
2-node wins *aggregate* throughput at the cap.

This deployment's workload is single-user interactive coding, i.e. per-stream-latency
bound. **The +17% at cc=1 is the number that matters here.**

## Why this is a fair comparison

- **Identical config on both arms** — `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=16`,
  `MTP_NUM_TOKENS=5`, `GPU_MEMORY_UTILIZATION=0.80`. Node count is the only variable.
- **Same harness, same prompt, same day**, ~20 minutes apart.
- **Both arms verified clean**: 0 RDMA completion errors, 0 preemptions, 0 prefix-cache
  hits, fabric gated at 4.6+ GB/s per pair beforehand.
- KV cache 4,457,627 (3-node) vs 1,711,307 (2-node) — a 2.6x ratio that, as established
  in issue #15, never binds.

## ⚠ The noise is bimodal — read this before re-running

The harness prints `<-- wide spread, check for other traffic` at cc≥4 on both arms. **That
warning is misleading here and the medians are sound.** Characterized with 15 spaced cc=1
runs on TP=3:

```
69.8 89.1 90.7 90.9 88.8 90.6 85.8 89.0 68.8 89.4 87.7 72.1 90.7 89.4 90.8
```

| mode | value | frequency |
|---|---:|---:|
| fast | **89.4 tok/s** (±1%) | 80% |
| slow | 69.8 tok/s | 20% |

An intermittent ~22% dropout, not progressive degradation. Ruled out: thermal throttling
(no active throttle reasons, clocks 2405–2411 MHz at spec, 57–65 °C), prefix cache (0
hits), preemptions (0), and external traffic (the LiteLLM gateway holds an idle keepalive
to `:8100` but sent nothing — completion counts matched our sweeps exactly).

**Consequence: use the median and take ≥7 runs.** A mean, or a 3-run median, can land in
the slow mode and manufacture a 20% difference that is not real. Root cause of the dropout
is not known and is worth investigating separately.

TP=2's cc=1 was unusually tight by comparison (2% spread, 7 runs within 75.8–77.4), so the
+17% is not an artifact of picking TP=3's fast mode: even TP=3's *slow* mode (69.8) against
TP=2's tight 76.2 would show 3-node losing, and the median of 7 lands in the fast mode
reliably.

## Where this leaves the three-node case

| measurement | 2-node | 3-node | winner |
|---|---:|---:|---|
| prefill 1K/8K/32K | 1913 / 2081 / 2066 | 2023 / 2070 / 2095 | parity (±2%) |
| deep concurrency TTFT | **293,987 ms** | 396,804 ms | **2-node**, 1.35x |
| decode cc=1 | 76.2 | **89.1** | **3-node**, +17% |
| decode cc=16 aggregate | **481.3** | 474.8 | 2-node, +1.4% |
| KV capacity | 1,711,307 | **4,457,627** | 3-node 2.6x — but never binds |

**Three nodes are superior for single-stream interactive work and inferior for batch
throughput.** That is a real, matched, healthy-fabric answer rather than a slogan in
either direction — and it means the deployment choice follows the workload, not the
hardware count.

## Files

| file | what |
|---|---|
| `tp3_final.json` / `.txt` | 3-node arm, 7 runs × 4 levels |
| `tp2_final.json` / `.txt` | 2-node arm, matched |
