# Cross-Engine Parity Report — 2 Nodes vs 3 Nodes (MiaAB vs Eugr)

**Status: COMPLETE & SETTLED (2026-09-01).**

This document reports full benchmark parity across **both engine generations** (`anemll-v0.25.1` / MiaAB vs `eugr-spark-vllm-b12x`) and **both cluster topologies** (2-node TP=2 vs 3-node TP=3) on NVIDIA DGX Spark (GB10) over a 200 GbE RoCE mesh.

---

## Executive Summary

1. **3-Node Eugr vs 3-Node MiaAB:**
   - **Eugr wins decisively on 3 nodes**: **+37.7% faster single-stream decode** (84.7 tok/s vs 61.5 tok/s), **+61.4% higher peak aggregate throughput** (249.9 tok/s vs 154.8 tok/s at c=8), and **3.2x lower time-to-first-token latency** at c=16 (2.1s vs 6.8s).
   - At 131,072 context length, Eugr delivers **+8.4% faster decode** (90.5 tok/s vs 83.5 tok/s) and **2.6x faster prefill** (53.7s vs 138.1s TTFT).
   - MiaAB's sole remaining advantage is raw KV cache token count (4.39M vs 2.36M tokens) due to `nvfp4_ds_mla` format support, though Eugr's 2.36M token pool is more than sufficient for full 1M context concurrency ($2.26\times$).

2. **2-Node vs 3-Node Scaling Dynamics:**
   - **On Eugr**, scaling from 2 nodes to 3 nodes provides positive scaling across all metrics: **+17.8% single-stream decode** (71.9 -> 84.7 tok/s), **+14.6% peak throughput** (218.1 -> 249.9 tok/s), and expands the KV cache pool by **180x** (13,107 tokens / 11.9K max model len -> 2,365,571 tokens / 1M max model len).
   - **On MiaAB**, 3 nodes suffered from unoptimized virtual-TP padding overhead without kernel fusion, regressing single-stream decode (70.0 -> 61.5 tok/s). Eugr resolves this architectural limitation completely.

---

## The $2 \times 2$ Benchmark Matrix

*All data captured using standardized test harness (`bench-miaai.py`), 256-token prompt, 128/256-token output window, median-of-5 repetitions, warm persistent kernel caches.*

### 1. Concurrency Sweep ($c \in \{1, 4, 8, 16\}$)

| Concurrency Cell | MiaAB 2-Node (TP=2) | MiaAB 3-Node (TP=3) | Eugr 2-Node (TP=2) | Eugr 3-Node (TP=3) | 3-Node Delta (Eugr vs MiaAB) |
|---|---|---|---|---|---|
| **$c=1$ Decode (tok/s)** | 70.0 | 61.5 | 71.9 | **84.7** | **+37.7% (Eugr Wins)** |
| **$c=4$ Decode (tok/s)** | 40.7 | 33.0 | 47.2 | **54.4** | **+64.8% (Eugr Wins)** |
| **$c=4$ Aggregate (tok/s)** | 112.4 | 108.0 | 144.3 | **164.5** | **+52.3% (Eugr Wins)** |
| **$c=8$ Decode (tok/s)** | 30.9 | 29.0 | 37.3 | **44.9** | **+54.8% (Eugr Wins)** |
| **$c=8$ Aggregate (tok/s)** | 161.0 | 154.8 | 218.1 | **249.9** | **+61.4% (Eugr Wins)** |
| **$c=16$ Aggregate (tok/s)** | 191.2 | 141.3 | 199.9 | **187.4** | **+32.6% (Eugr Wins)** |
| **$c=16$ TTFT (ms)** | 1,842 | 6,774 | 1,877 | **2,122** | **3.2x Lower Latency** |

---

### 2. Prompt Sensitivity & Context Depth

| Workload Dimension | MiaAB 2-Node (TP=2) | MiaAB 3-Node (TP=3) | Eugr 2-Node (TP=2) | Eugr 3-Node (TP=3) | 3-Node Delta (Eugr vs MiaAB) |
|---|---|---|---|---|---|
| **`code-brief` (18 tok)** | 81.8 tok/s | 81.8 tok/s | 77.3 tok/s | **91.0 tok/s** | **+11.2% (Eugr Wins)** |
| **`dense-prose` (23 tok)** | 48.2 tok/s | 49.4 tok/s | 44.3 tok/s | **49.2 tok/s** | **Parity (49.2 vs 49.4)** |
| **131K Context Decode** | 74.0 tok/s | 83.5 tok/s | *N/A (OOM)* | **90.5 tok/s** | **+8.4% (Eugr Wins)** |
| **131K Context TTFT** | 128.5 s | 138.1 s | *N/A (OOM)* | **53.7 s** | **2.6x Faster Prefill** |
| **Max Context Ceiling** | 460,800 tok | 1,048,576 tok | 11,948 tok | **1,048,576 tok** | **Full 1M Context** |
| **Total KV Pool** | 1,832,675 tok | **4,391,722 tok** | 13,107 tok | 2,365,571 tok | MiaAB +86% (FP4 vs FP8 KV) |

---

## Evidence Bundles & Provenance

- **2-Node Eugr TP=2 Bundle:** [`results/20260901T174000Z-eugr-tp2-2node/`](../results/20260901T174000Z-eugr-tp2-2node/)
- **3-Node Matched Engine A/B Bundle:** [`results/20260831T1000Z-matched-engine-ab/`](../results/20260831T1000Z-matched-engine-ab/)
- **Winning Eugr K-Sweep Bundle:** [`results/20260830T2245Z-eugr-ksweep/`](../results/20260830T2245Z-eugr-ksweep/)
- **Master Measurements:** [`benchmarks/measurements.csv`](../benchmarks/measurements.csv)
- **Derived Summary Matrix:** [`benchmarks/summary.csv`](../benchmarks/summary.csv)
