# 3-Node TP=3 Qwen 3.8-27B + DFlash 2 Speculative Decoding Run (2026-09-02)

**Status: CURRENT.**

This bundle records the full speculative benchmark sweep on `RadixArk/Qwen3.8-27B-NVFP4` with `z-lab/Qwen3.8-27B-DFlash2` block diffusion drafting ($k=7$) across **3 nodes** (`sparkmain`, `spark1`, `spark2`, TP=3) on 3x DGX Spark (GB10).

## Method & Configuration

| Parameter | Value |
|---|---|
| Date | 2026-09-02 |
| Nodes / TP | `sparkmain`, `spark1`, `spark2`, TP=3, PP=1 |
| Target Model | `RadixArk/Qwen3.8-27B-NVFP4` |
| Speculative Drafter | `z-lab/Qwen3.8-27B-DFlash2` (num_speculative_tokens=7) |
| Image | `eugr/spark-vllm-b12x:latest` |
| Target Virtual TP | Padded GQA heads (24 -> 36), KV heads (4 -> 6), GDN key heads (16 -> 18), GDN value heads (48 -> 54), intermediate size (17408 -> 17424) |
| Draft Virtual TP | Padded attention heads (32 -> 36), KV heads (8 -> 9), intermediate size (17408 -> 17409), vocab storage (248320 -> 248448) |
| Max Num Batched Tokens | 8192 |
| Acceptance Battery | 6/6 Passed (100%) on `qwen-quick-validate.sh` |
| Trials | 5 repetitions per concurrency level ($c \in \{1, 4, 8, 16\}$) |

## Benchmark Results

| Concurrency ($c$) | Median Decode (tok/s) | Aggregate Throughput (tok/s) | Median TTFT (ms) |
|---|---|---|---|
| 1 | 32.0 | 30.2 | 239 |
| 4 | 26.3 | 89.7 | 647 |
| 8 | 16.9 | 108.1 | 1048 |
| 16 | 12.1 | 145.1 | 2125 |
