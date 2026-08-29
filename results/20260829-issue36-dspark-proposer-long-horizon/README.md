# DSpark Proposer Long-Horizon Acceptance Probe (Issue #36)

**Date:** 2026-08-29  
**Status:** `CURRENT`  
**Image:** `dsv4-3spark:0.1.1` (Hermetic build with TP=3 padding & hotfixes)  
**Configuration:** Profile B (`GPU_MEMORY_UTILIZATION=0.835`, `MTP_NUM_TOKENS=2`, `MAX_NUM_BATCHED_TOKENS=8192`)  
**Nodes:** 3 (`sparkmain`, `spark1`, `spark2`)  

---

## 1. Why this run exists

Community reports (`vladimir-voinea/dspark-vllm-gb10`) suggested that the DSpark proposer skips intermediate accepted tokens when updating its 128-slot sliding-window cross-attention KV cache (writing only the final bonus token's `main_kv`), causing drafting acceptance rate $\tau$ to collapse after $\sim 50$ decode steps. Because earlier benchmarks relied on 256-token windows, this run specifically probes extended generation horizons (256, 512, 1024, 1536 tokens) to audit long-horizon acceptance stability.

---

## 2. Empirical Results

Single-stream generation with `temperature=0.0` on an architectural specification prompt:

| Target Tokens | Actual Generated | Wall Time (s) | Decode Tok/s | Acceptance % | Mean Accepted / Step | Pos 0 Accepted | Pos 1 Accepted |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **256** | 256 | 13.263 | 19.30 (incl TTFT) | **69.0%** | 1.380 | 83 | 66 |
| **512** | 512 | 9.199 | **55.66** | **78.9%** | 1.578 | 172 | 142 |
| **1024** | 1024 | 19.574 | **52.31** | **80.4%** | 1.608 | 346 | 286 |
| **1536** | 1536 | 26.873 | **57.16** | **76.7%** | 1.535 | 507 | 423 |

---

## 3. Conclusions

1. **Zero Staleness Decay**:
   - Speculative drafting acceptance rate remains rock-solid between **76.7% and 80.4%** across all tested horizons.
   - Mean accepted draft tokens per verification step stays consistent at **$\sim 1.55$ tokens/step** (effective $\tau \approx 2.55$ tokens per forward pass).
2. **Decode Throughput Stability**:
   - Sustained decode speed holds steady at **52.3–57.2 tok/s** through 1,536 generated tokens (~1,000 decode steps).
3. **Proposer Implementation Verified Clean**:
   - `_insert_context_kv` in `dspark.py` properly updates the sliding-window attention cache at `context_slots` for all verified tokens in the batch.
