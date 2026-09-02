# 3-Node TP=3 Qwen 3.8-27B Dense Baseline Run (2026-09-01)

**Status: CURRENT.**

This bundle records the baseline benchmark sweep on `RadixArk/Qwen3.8-27B-NVFP4` without speculative decoding across **3 nodes** (`sparkmain`, `spark1`, `spark2`, TP=3) on 3x DGX Spark (GB10).

## Method & Configuration

| Parameter | Value |
|---|---|
| Date | 2026-09-01 |
| Nodes / TP | `sparkmain`, `spark1`, `spark2`, TP=3, PP=1 |
| Target Model | `RadixArk/Qwen3.8-27B-NVFP4` |
| Speculative Drafter | None (Dense Baseline) |
| Image | `eugr/spark-vllm-b12x:latest` |
| Virtual TP Sharding | Padded GQA heads (24 -> 36), KV heads (4 -> 6), GDN key heads (16 -> 18), GDN value heads (48 -> 54), intermediate size (17408 -> 17424) |
| Max Num Batched Tokens | 8192 |
| Trials | 5 repetitions per concurrency level ($c \in \{1, 4, 8, 16\}$) |

## Benchmark Results

| Concurrency ($c$) | Median Decode (tok/s) | Aggregate Throughput (tok/s) | Median TTFT (ms) |
|---|---|---|---|
| 1 | 25.2 | 24.5 | 157 |
| 4 | 23.2 | 85.6 | 471 |
| 8 | 21.4 | 150.8 | 805 |
| 16 | 17.5 | 225.4 | 1765 |
