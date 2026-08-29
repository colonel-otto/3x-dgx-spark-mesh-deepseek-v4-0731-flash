# Issue #36: Locked-Clocks Master Benchmark Suite (2026-08-28)

**Status:** `CURRENT` · **Nodes:** 3 · **TP:** 3 · **Fabric gate:** `ABSENT`

Evidence bundle evaluating serving determinism, MTP $K=2$ concurrency, prefill depth, and Automatic Prefix Caching (APC) under verified PCIe Gen5 x4 ConnectX-7 interconnects.

> ⚠️ **The "locked 3003 MHz" claim in this bundle is WITHDRAWN (2026-08-29).**
> GB10 does not honour `nvidia-smi -lgc`: it reports `Supported Clocks: N/A` and no
> settable power limit, so the command returns success and pins nothing. **This bundle's
> own `gpu_clocks.csv` shows it** — a single row reading **2522 MHz**, not 3003, sampled
> at 50 °C / 15.57 W (idle, before the run) despite being described below as telemetry
> "during execution". The measurements here still stand as measurements; what is
> withdrawn is the claim that they were taken under controlled clock conditions. Full
> analysis: [`../../docs/GPU-CLOCKS-NOT-LOCKABLE.md`](../../docs/GPU-CLOCKS-NOT-LOCKABLE.md).

---

## 1. Hardware Configuration
- **Cluster**: 3x NVIDIA DGX Spark (Grace-Blackwell GB10 SoC).
- **GPU Clocks**: `sudo nvidia-smi -lgc 3003,3003` was issued with persistence mode
  enabled and **reported success, but did not take effect** — see the withdrawal notice
  above. Actual clock during this suite is unmeasured; the one committed sample reads
  2522 MHz at idle. Sustained clock on GB10 under MoE decode settles around 2420–2520 MHz,
  governed by SW power capping (cumulative ~3 h) rather than thermal slowdown (~1.2 s).
- **Interconnect**: Switchless 3-node ring over dual 200 GbE ConnectX-7 links. All 12 controllers verified at PCIe Gen5 x4 (`Speed 32GT/s, Width x4`).
- **Engine**: `dsv4-3spark:0.1.1` hermetic image, `TP=3`, `PP=1`, `MAX_MODEL_LEN=1048576`, `MAX_NUM_BATCHED_TOKENS=8192`, `MTP_NUM_TOKENS=2`, `GPU_MEMORY_UTILIZATION=0.835`.

---

## 2. Summary of Results

### A. Serving Noise Floor & Parity (`logprob_parity.json`)
- **Status**: **PASS** (all 5 passages within empirical noise tolerance floor <= 12.0%).
- Prose: **5.31%** spread ($NLL=0.8628 \pm 0.0208$)
- Code: **7.46%** spread ($NLL=0.3288 \pm 0.0091$)
- Math: **3.54%** spread ($NLL=0.6782 \pm 0.0100$)
- Structured: **4.18%** spread ($NLL=0.6455 \pm 0.0110$)
- Multilingual: **5.39%** spread ($NLL=0.5677 \pm 0.0128$)

### B. MTP $K=2$ Concurrency Matrix (`mtp_k2_concurrency.json`)
*Context Depth: 8192, Output Window: 256 tokens asserted.*

| Concurrency ($cc$) | Median Aggregate (tok/s) | Median Stream (tok/s) | Median TTFT (s) | Draft Acceptance Ratio |
|---|---:|---:|---:|---:|
| **$cc=1$** | 28.51 | **56.05** | 4.28 | **70.0%** |
| **$cc=4$** | **41.65** | 20.27 | 11.05 | **67.2%** |
| **$cc=8$** | **50.12** | 12.27 | 18.46 | **67.0%** |
| **$cc=16$** | **54.31** | 6.66 | 34.22 | **65.8%** |

### C. Prefill TTFT Depth & Decode Sweep (`prefill_depth.json`)
*Output Window: 256 tokens asserted.*

| Nominal Depth | Actual Prompt Tokens | Median TTFT (s) | TTFT Spread (s) | Median Decode (tok/s) |
|---|---:|---:|---:|---:|
| **2048** | 2,040 | **1.11** | 0.03 | **50.25** |
| **8192** | 8,086 | **4.23** | 0.02 | **54.95** |
| **32768** | 32,272 | **17.23** | 0.17 | **50.91** |
| **65536** | 64,517 | **35.32** | 0.20 | **50.54** |
| **131072** | 129,010 | **77.23** | 2.96 | **51.03** |

### D. Multi-Turn Automatic Prefix Caching (APC) (`apc_warm_path.json`)
*Think-time pause: 5.0s between turns.*

| Context Depth | Cold Turn 1 TTFT (s) | Warm Turn 2+ TTFT (s) | Speedup Factor | Median Cache Hit Ratio |
|---|---:|---:|---:|---:|
| **8K** | 4.22s | **0.462s** | **9.1x** | **94.8%** |
| **32K** | 17.06s | **0.479s** | **35.6x** | **99.0%** |
| **131K** | 74.04s | **0.760s** | **97.4x** | **99.8%** |

---

## 3. Raw Evidence Files
- `gpu_clocks.csv`: Live hardware clock telemetry during execution.
- `logprob_parity.json`: 5-passage 5-rep teacher-forced logprob parity metrics.
- `mtp_k2_concurrency.json`: Full per-rep concurrency observations and Prometheus scraper metrics.
- `prefill_depth.json`: Multi-depth TTFT and forced 256-token decode samples.
- `apc_warm_path.json`: Multi-session multi-turn prefix cache retention telemetry.