# 2-Node TP=2 Qwen 3.8-27B Dense Baseline Run (2026-09-01)

**Status: CURRENT.**

This bundle records the baseline benchmark sweep on `RadixArk/Qwen3.8-27B-NVFP4` without speculative decoding across **2 nodes** (`sparkmain` + `spark1`, TP=2) on 2x DGX Spark (GB10).

## Method & Configuration

| Parameter | Value |
|---|---|
| Date | 2026-09-01 |
| Nodes / TP | `sparkmain` + `spark1`, TP=2, PP=1 |
| Target Model | `RadixArk/Qwen3.8-27B-NVFP4` |
| Speculative Drafter | None (Dense Baseline) |
| Image | `eugr/spark-vllm-b12x:latest` |
| Max Num Batched Tokens | 8192 |
| Trials | 5 repetitions per concurrency level ($c \in \{1, 4, 8, 16\}$) |

## Benchmark Results

| Concurrency ($c$) | Median Decode (tok/s) | Aggregate Throughput (tok/s) | Median TTFT (ms) |
|---|---|---|---|
| 1 | 20.3 | 19.9 | 146 |
| 4 | 18.6 | 69.5 | 492 |
| 8 | 16.7 | 120.5 | 854 |
| 16 | 14.9 | 192.6 | 2062 |
