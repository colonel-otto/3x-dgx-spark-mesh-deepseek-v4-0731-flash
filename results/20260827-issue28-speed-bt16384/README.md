# Speed profile sweep (MAX_NUM_BATCHED_TOKENS=16384) — 2026-08-27

**Status:** `CURRENT` · **Target:** Speed optimization by scaling prefill batch tokens · **Output:** 256 tokens asserted on all reps · **Cache:** zero cached tokens

This bundle records the Issue #28 experiment trading unneeded KV cache headroom to investigate whether doubling `MAX_NUM_BATCHED_TOKENS` from 8,192 to 16,384 improves TTFT and decode throughput.

## Configuration
- `MAX_NUM_BATCHED_TOKENS=16384`
- `MAX_MODEL_LEN=460800`
- `NCCL_BUFFSIZE=16777216`
- `GPU_MEMORY_UTILIZATION=0.835`
- `LONG_PREFILL_TOKEN_THRESHOLD=1024`
- `DSPARK_MAX_INFLIGHT_PREFILLS=2`
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`
- **Resulting KV Pool**: **1,954,299 tokens** (33.44 GiB available per GPU).

## Single-Stream Depth Sweep (7 reps per depth)

| Target Depth | Actual Prompt | Median Decode (tok/s) | Min (tok/s) | Max (tok/s) | Spread | Median TTFT (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,037 | 52.90 | 46.77 | 62.09 | 29.0% | 1.09 |
| 8,192 | 8,083 | 55.98 | 49.42 | 58.74 | 16.6% | 4.17 |
| 32,768 | 32,271 | 54.91 | 51.08 | 57.98 | 12.6% | 18.46 |
| 131,072 | 129,007 | 51.18 | 43.21 | 56.85 | 26.6% | 92.46 |
| 262,144 | 257,995 | **51.39** | 37.18 | 53.72 | 32.2% | 228.64 |

## Starvation Probe (5 trials under concurrent prefill)
- **Median Decoder TTFT**: 149.32 s
- **Median Decoder Decode**: 46.28 tok/s
- **Median Max Event Gap**: 0.107 s
- **Max Event Gap Across All Trials**: 0.341 s

## Findings
1. **Decode at Extreme Depth**: 262K decode throughput improved from 46.08 tok/s to **51.39 tok/s (+11.5%)**.
2. **TTFT Penalty**: Sizing chunk batches to 16,384 increased activation tensor sizes per layer to ~235 MB. On the GB10 unified memory bus (273 GB/s), transferring and reducing 235 MB tensors created memory bus saturation and RoCE serialization, increasing 131K TTFT from 75.7s to 92.5s (+22.1%) and 262K TTFT from 177.3s to 228.6s (+29.0%).
3. **Conclusion**: `MAX_NUM_BATCHED_TOKENS=8192` remains the optimal chunk size on this hardware topology.
