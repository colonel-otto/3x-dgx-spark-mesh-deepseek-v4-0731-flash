# 2-Node TP=2 Eugr Benchmark Run — eugr-spark-vllm-b12x (2026-09-01)

**Status: CURRENT.**

This bundle records the full benchmark battery on the updated `eugr/spark-vllm-b12x:latest` engine (vLLM main dev g b5f995e73) across **2 nodes** (`sparkmain` + `spark1`, TP=2) on 2x DGX Spark (GB10).

## Method & Configuration

| Parameter | Value |
|---|---|
| Date | 2026-09-01 |
| Nodes / TP | `sparkmain` + `spark1`, TP=2, PP=1 |
| Image | `eugr/spark-vllm-b12x:latest` (digest `7dc02f16`) |
| Speculator | DSpark `nst=5` (`dspark_block_size=5`), `mnbt=8192` |
| KV Cache Dtype | `fp8` (`fp8_ds_mla`), utilization 0.82 |
| KV Cache Pool | 5.02 GiB / 13,107 tokens (`max_model_len` auto-fitted to 11,948) |
| Persistent Caches | Warm mounts `/opt/eugrcache-*` (vllm, flashinfer, triton, tilelang) |
| Harness | `scripts/eugr-ab/bench-miaai.py` (concurrency) and `scripts/eugr-ab/eugr-remaining-cells-v2.py` (prompt-effect & context) |
| Trials | 5 repetitions per cell, warm-up discarded, verified cold nonces |

## Benchmark Results

### 1. Concurrency Sweep ($c \in \{1, 4, 8, 16\}$)

| c | Median Decode (tok/s) | Median Aggregate (tok/s) | Median TTFT (ms) |
|---|---|---|---|
| 1 | 71.9 | 60.0 | 302 |
| 4 | 47.2 | 144.3 | 790 |
| 8 | 37.3 | 218.1 | 1171 |
| 16 | 15.7 | 199.9 | 1877 |

### 2. Prompt Effect & Context Depth

| Test | Prompt Tokens | Output Tokens | Median Decode (tok/s) | Median TTFT (ms) |
|---|---|---|---|---|
| `code-brief` | 18 | 256 | 77.3 | 282.7 |
| `dense-prose` | 23 | 256 | 44.3 | 215.2 |
| 8K Context | 8200 | 256 | 42.0 | 3364.9 |

- **Prompt Ratio (`code-brief` / `dense-prose`):** **1.74x** on 2-node Eugr.
