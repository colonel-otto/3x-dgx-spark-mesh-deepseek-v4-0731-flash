# 2-Node TP=2 Qwen 3.8-27B + DFlash 2 Speculative Decoding Run (2026-09-02)

**Status: CURRENT.**

This bundle records the full speculative benchmark sweep on `RadixArk/Qwen3.8-27B-NVFP4` with `z-lab/Qwen3.8-27B-DFlash2` block diffusion drafting ($k=7$) across **2 nodes** (`sparkmain` + `spark1`, TP=2) on 2x DGX Spark (GB10).

## Method & Configuration

| Parameter | Value |
|---|---|
| Date | 2026-09-02 |
| Nodes / TP | `sparkmain` + `spark1`, TP=2, PP=1 |
| Target Model | `RadixArk/Qwen3.8-27B-NVFP4` |
| Speculative Drafter | `z-lab/Qwen3.8-27B-DFlash2` (num_speculative_tokens=7) |
| Image | `eugr/spark-vllm-b12x:latest` |
| Max Num Batched Tokens | 8192 |
| Acceptance Battery | 6/6 Passed (100%) on `qwen-quick-validate.sh` |
| Trials | 5 repetitions per concurrency level ($c \in \{1, 4, 8, 16\}$) |

## Benchmark Results

| Concurrency ($c$) | Median Decode (tok/s) | Aggregate Throughput (tok/s) | Median TTFT (ms) |
|---|---|---|---|
| 1 | 28.4 | 26.9 | 246 |
| 4 | 22.8 | 71.9 | 689 |
| 8 | 16.8 | 107.2 | 1114 |
| 16 | 12.6 | 149.3 | 2415 |
