# Issue #35 — GuideLLM Standard Concurrency & Latency Sweep

**Status:** `CURRENT` · **Nodes:** 3 (TP=3) · **Date:** 2026-08-28 (UTC)
**Harness:** `guidellm 0.7.3` (`scripts/run_guidellm_suite.py`)
**Config:** `dsv4-3spark:0.1.1` on Profile B (`MTP_NUM_TOKENS=2`, `MAX_NUM_BATCHED_TOKENS=8192`, `GPU_MEMORY_UTILIZATION=0.835`)

---

## 1. Concurrency Sweep Results ($cc \\in \\{1, 4, 8, 16, 32\\}$)

| Concurrency ($cc$) | Median TTFT (ms) | Median ITL (ms) | Output Tok/Iter | Output Tok / Sec | Total Tok / Sec |
|---:|---:|---:|---:|---:|---:|
| **1** | 1,314.2 | 21.3 | 2.0 | 38.0 | 343.0 |
| **4** | 1,610.5 | 50.0 | 2.2 | 69.8 | 727.5 |
| **8** | 3,520.3 | 68.8 | 2.2 | 91.7 | 1,089.3 |
| **16** | 7,774.8 | 107.7 | 2.3 | 101.8 | 1,562.2 |
| **32** | 1,582.0 | 179.8 | 2.7 | **109.8** | **1,541.4** |

---

## 2. Key Findings

1. **Streaming Inter-Token Latency (ITL)**:
   - At $cc=1$, median ITL is **21.3 ms**, delivering immediate, fluid streaming to interactive clients.
   - ITL scales smoothly as batch size expands, reaching **107.7 ms** at $cc=16$.
2. **Peak Token Generation Ceiling**:
   - Output token throughput scales from 38.0 tok/s at $cc=1$ to **109.8 tok/s at full saturation ($cc=32$)**.
   - Input + output total processed throughput peaks at **1,562 tokens/second** at $cc=16$.
3. **Speculative Drafting Efficiency**:
   - Under $MTP_NUM_TOKENS=2$, the model consistently emits **2.0 to 2.7 output tokens per stream iteration**, verifying high acceptance across all concurrency levels.

---

## 3. Artifacts & Reproducibility

- `report.json`: Complete request-level timing records and percentiles.
- `report.html`: Interactive GuideLLM visual performance dashboard.
