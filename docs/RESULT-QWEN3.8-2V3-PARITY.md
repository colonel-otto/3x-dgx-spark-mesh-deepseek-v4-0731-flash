# Qwen 3.8-27B NVFP4 Multi-Node DGX Spark Benchmark & 2-Node vs 3-Node Parity Report

**Date:** September 1, 2026  
**System:** 3x NVIDIA DGX Spark (Grace Blackwell GB10, 128 GB Unified Memory per node, 4x 100 Gbps CX-7 RoCE Mesh Interconnect)  
**Model:** `RadixArk/Qwen3.8-27B-NVFP4` (Hybrid GQA + Gated DeltaNet architecture, NVFP4 Cutlass Linear Kernels, FP8 KV-Cache)  
**Engine:** `eugr/spark-vllm-b12x:latest` (vLLM 0.11.0 with Blackwell SM120/SM121 FlashInfer NVFP4 & Triton GDN kernels)  
**Harness:** `bench-miaai.py` (256 prompt tokens, 128 output tokens, 5 repeats per concurrency level)

---

## 1. Executive Summary

This report establishes the multi-node distributed serving recipes and benchmark parity for **Qwen 3.8-27B NVFP4** across dual-node ($TP=2$) and triple-node ($TP=3$) DGX Spark mesh configurations.

### Key Highlights
1. **Decode Throughput Scaling:**
   - **2-Node Baseline ($TP=2$):** **20.3 tok/s** at $c=1$, scaling to **192.6 tok/s** aggregate at $c=16$.
   - **3-Node Mesh ($TP=3$):** **25.2 tok/s** at $c=1$ (**+24.1% speedup**), scaling to **225.4 tok/s** aggregate at $c=16$ (**+17.0% higher throughput**).
2. **Time to First Token (TTFT):**
   - High-concurrency TTFT improves significantly on 3 nodes: **1,765 ms** vs **2,062 ms** at $c=16$ (**-297 ms / 14.4% latency reduction**).
3. **Cluster Virtual TP Patches:**
   - Successfully parallelized the hybrid GQA + Gated DeltaNet language backbone across $TP=3$ using virtual-TP padding while keeping the vision encoder in replicated data mode (`--mm-encoder-tp-mode data`).
4. **100% Acceptance & Correctness Battery Pass:**
   - Evaluated models on 3-node cluster: Capital lookup, exact arithmetic ($17 \times 23 = 391$), color reasoning ($7$), 1.5K-token needle retrieval (`FALCON42`), and text quality/degeneration tests all achieved **100% PASS (6/6)**.

---

## 2. 2-Node vs 3-Node Benchmark Parity Comparison Matrix

All tests conducted with standard MiaAI-Lab benchmark harness (`bench-miaai.py`):
- **Input Prompt:** 256 tokens
- **Output Target:** 128 tokens
- **Trials:** 5 repeats per concurrency level ($c \in \{1, 4, 8, 16\}$)
- **Batching:** `max_num_batched_tokens = 8192`

### Measured Results Table

| Metric | Concurrency ($c$) | 2-Node Mesh ($TP=2$) | 3-Node Mesh ($TP=3$) | Scaling Factor / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Median Single-Stream Decode** | **$c=1$** | 20.3 tok/s | **25.2 tok/s** | **+24.1% faster** (1.24x) |
| | **$c=4$** | 18.6 tok/s | **23.2 tok/s** | **+24.7% faster** (1.25x) |
| | **$c=8$** | 16.7 tok/s | **21.4 tok/s** | **+28.1% faster** (1.28x) |
| | **$c=16$** | 14.9 tok/s | **17.5 tok/s** | **+17.4% faster** (1.17x) |
| **Aggregate Throughput** | **$c=1$** | 19.9 tok/s | **24.5 tok/s** | **+23.1%** |
| | **$c=4$** | 69.5 tok/s | **85.6 tok/s** | **+23.2%** |
| | **$c=8$** | 120.5 tok/s | **150.8 tok/s** | **+25.1%** |
| | **$c=16$** | 192.6 tok/s | **225.4 tok/s** | **+17.0%** |
| **Time to First Token (TTFT)** | **$c=1$** | **146 ms** | 157 ms | +11 ms (interconnect hop) |
| | **$c=4$** | 492 ms | **471 ms** | **-21 ms faster** |
| | **$c=8$** | 854 ms | **805 ms** | **-49 ms faster** |
| | **$c=16$** | 2062 ms | **1765 ms** | **-297 ms faster (-14.4%)** |

---

## 3. Architecture & Virtual-TP Sharding Plan

Qwen 3.8-27B combines standard Transformer attention layers with Gated DeltaNet (GDN) linear attention layers. To shard across 3 nodes ($TP=3$), virtual-TP padding was implemented for dimensions not evenly divisible by 3:

| Architectural Dimension | Original Size | Padded Size ($TP=3$) | Local Size per Rank | Divisibility Note |
| :--- | :--- | :--- | :--- | :--- |
| **Attention Heads** | 24 | **36** | 12 | Padded to match $q\_heads\_per\_kv = 6$ |
| **KV Heads** | 4 | **6** | 2 | $6 \div 3 = 2$ |
| **GDN Key Heads** | 16 | **18** | 6 | $18 \div 3 = 6$, $value\_heads\_per\_key = 3$ |
| **GDN Value Heads** | 48 | **54** | 18 | $54 \div 3 = 18$ |
| **GDN Conv1d Channel Dim** | 10,240 | **11,520** | 3,840 | $18 \times 128 \times 2 + 54 \times 128 = 11520$ |
| **Dense Intermediate Size** | 17,408 | **17,424** | 5,808 | 16-aligned for Cutlass NVFP4 linear kernels ($5808 \div 16 = 363$) |
| **Vocab Size** | 248,320 | **248,448** | 82,816 | Padded with $\text{LCM}(64, 3) = 192$ |
| **Vision Attention Heads** | 16 | **16** | 16 | Replicated per node via `--mm-encoder-tp-mode data` ($TP=1$) |

---

## 4. Acceptance & Quality Verification

Full test battery executed via `qwen-quick-validate.sh` against the 3-node cluster:

```text
=== Qwen Validation Battery ===
== 1. Models Endpoint ==
PASS  models endpoint serves qwen3.8-27b-nvfp4,qwen3.8,qwen-3.8-flash
== 2. Core Reasoning & Correctness ==
PASS  capital lookup (Paris)
PASS  17 x 23 (391)
PASS  red/blue (7)
== 3. Needle In Haystack Retrieval ==
PASS  needle ~1.5k tok (FALCON42)
== 4. Text Quality & Degeneration Check ==
PASS  no degeneration (unique-word ratio)

=== Result: 6 PASS / 0 FAIL ===
```

---

## 5. Deployment Artifacts & Recipes

### 1. 2-Node Recipe (`configs/qwen3.8-27b-nvfp4-tp2.yaml`)
```yaml
model: RadixArk/Qwen3.8-27B-NVFP4
served_model_name:
  - qwen3.8-27b-nvfp4
  - qwen3.8
  - qwen-3.8-flash
tensor_parallel_size: 2
mm_encoder_tp_mode: data
kv_cache_dtype: fp8
block_size: 256
max_model_len: 131072
max_num_seqs: 16
max_num_batched_tokens: 8192
gpu_memory_utilization: 0.82
enable_prefix_caching: true
load_format: instanttensor
reasoning_parser: qwen3
tool_call_parser: qwen3_xml
enable_auto_tool_choice: true
```

### 2. 3-Node Recipe (`configs/qwen3.8-27b-nvfp4-tp3.yaml`)
```yaml
model: RadixArk/Qwen3.8-27B-NVFP4
served_model_name:
  - qwen3.8-27b-nvfp4
  - qwen3.8
  - qwen-3.8-flash
tensor_parallel_size: 3
mm_encoder_tp_mode: data
kv_cache_dtype: fp8
block_size: 256
max_model_len: 131072
max_num_seqs: 16
max_num_batched_tokens: 8192
gpu_memory_utilization: 0.82
enable_prefix_caching: true
load_format: instanttensor
reasoning_parser: qwen3
tool_call_parser: qwen3_xml
enable_auto_tool_choice: true
```

### 3. Cluster Launcher Scripts
- **2-Node Boot:** `~/qwen-boot-tp2.sh [mnbt]`
- **3-Node Boot:** `~/qwen-boot-tp3.sh [mnbt]`
- **Cluster Stop:** `~/qwen-stop.sh`
- **Validation:** `~/qwen-quick-validate.sh`
- **Benchmark Sweep:** `~/qwen-sweep.sh <nodes> <mnbt>`
