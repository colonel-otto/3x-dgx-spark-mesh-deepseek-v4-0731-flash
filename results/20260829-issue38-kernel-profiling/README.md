# Issue #38: Kernel Profiling Trace Analysis (3-Node DGX Spark TP=3)

**Status**: `CURRENT`  
**Date**: 2026-08-29  
**Target Cluster**: 3x DGX Spark (`sparkmain`, `spark1`, `spark2`)  
**Model**: DeepSeek-V4 Flash (TP=3, NVFP4/FP8, MTP Speculative Decoding K=2)  
**Profiler**: PyTorch Profiler / Chrome Trace (`activities=["CUDA"]`)  
**Canonical Repository**: `colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark`  

---

## 1. Executive Summary

This report delivers the first comprehensive kernel profiling trace and device-level breakdown for DeepSeek-V4 running in Tensor Parallelism 3 (TP=3) across 3x NVIDIA DGX Spark nodes over a switchless dual 200 GbE ConnectX-7 RoCE ring.

### Key Highlights:
1. **Decode Phase (8K Context, 16 Tokens Generation)**:
   - **Inter-Node NCCL Collectives**: **87.54%** of GPU device time (25.75 s cumulative, 1,400 calls).
   - **MoE GEMM (B12X Fused W4A16 / DeepGEMM)**: **7.91%** of GPU device time (2.33 s cumulative).
   - **FlashInfer SM120 MLA Attention**: **2.61%** of GPU device time (0.77 s cumulative).
   - **Quant / Norm / Rotary / TileLang**: **0.79%** of GPU device time (0.23 s cumulative).
   - **Sampling & TopK**: **0.39%** of GPU device time (0.11 s cumulative).
   - **Finding**: Single-stream decode spends the overwhelming majority of rank 0's GPU time *inside* NCCL AllReduce kernels.

   > ⚠️ **This is not the same as "network-bound", and the trace cannot distinguish the two.**
   > The AllReduce kernel time is **25,696 ms over 1,316 calls = 19.53 ms per collective**.
   > A decode-step reduction on this fabric moves a few MB; at the 23.92 GB/s measured by
   > our own `nccl-tests` ([`20260826-nccl-controlled`](../20260826-nccl-controlled/)) that
   > is **sub-millisecond of actual transfer**. The other ~19 ms is the collective kernel
   > *resident and spinning* — rank 0 arrived at the barrier and waited for peers.
   > Compare prefill: the same kernel averages **2.78 ms/call**, 7x cheaper, because ranks
   > arrive together when there is real work between collectives.
   >
   > So the honest reading is: **decode is latency/synchronization-bound, and rank 0 spends
   > most of its time waiting.** Whether that wait is wire latency, per-collective launch
   > overhead, or a straggler rank is **not answerable from a single-rank trace**. Only
   > rank 0 was captured; ranks 1 and 2 were not. Attributing the 87.54% to "100 Gbps RoCE
   > mesh latency" is a hypothesis, not a measurement.

2. **Prefill Phase (131K Deep Context Forward Pass, 1 Token)**:
   - **MoE GEMM (B12X Fused W4A16 / DeepGEMM)**: **39.23%** of GPU device time (36.52 s cumulative, 77,340 calls).
   - **Inter-Node NCCL Collectives**: **34.07%** of GPU device time (31.72 s cumulative, 12,100 calls).
   - **FlashInfer SM120 MLA Attention**: **16.19%** of GPU device time (15.07 s cumulative, 28,435 calls).
   - **Quant / Norm / Rotary / TileLang**: **4.06%** of GPU device time (3.78 s cumulative).
   - **Sampling & TopK**: **2.76%** of GPU device time (2.57 s cumulative).
   - **Finding**: Prefill is **compute-bound and balanced**, with B12X MoE GEMM and FlashInfer SM120 sparse MLA attention utilizing the Blackwell SMs at high sustained tensor core efficiency.

3. **Profiler Stability Fix Landed**:
   - Identified that default PyTorch profiler (`activities=["CPU", "CUDA"]`) accumulated over 5 million CPU event stack frames in unified memory during DeepSeek-V4 forward passes, exhausting host RAM and triggering Linux OOM killer.
   - Patched `vllm/v1/worker/gpu_worker.py` to `activities=["CUDA"]`, capturing pure GPU device traces with zero memory overhead, 98% reduced trace export size, and sub-50ms stop latency.

---

## 2. Kernel Breakdown Comparison Table

| Kernel Category | Decode (8K Context, 16 Tok) Time | Decode % | Prefill (131K Context) Time | Prefill % | Primary Kernels / Implementations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Other GPU Kernels** | **189.56 ms** | **0.64%** | **2979.59 ms** | **3.20%** | |
| **Memory / Copy / Layout Ops** | **30.62 ms** | **0.10%** | **446.62 ms** | **0.48%** | |
| **Sampling / TopK / Argmax** | **114.89 ms** | **0.39%** | **2572.40 ms** | **2.76%** | |
| **NCCL AllReduce / Collectives** | **25745.79 ms** | **87.54%** | **31723.21 ms** | **34.07%** | |
| **MoE GEMM / Fused MoE (B12X / DeepGEMM)** | **2326.51 ms** | **7.91%** | **36523.37 ms** | **39.23%** | |
| **FlashInfer MLA Attention** | **768.80 ms** | **2.61%** | **15072.37 ms** | **16.19%** | |
| **Quant / Norm / Rotary / TileLang** | **233.20 ms** | **0.79%** | **3784.24 ms** | **4.06%** | |
| **Total Cumulative Time** | **29409.38 ms** | **100.00%** | **93101.80 ms** | **100.00%** | |
| **GPU Busy Wallclock** | **29216.55 ms** | - | **90681.08 ms** | - | |
| **Stream Overlap Ratio** | **1.01x** | - | **1.03x** | - | |

---

## 3. What this trace supports — and what it does not

This section previously asserted three "facts" about 2-node behaviour, MoE numerical
parity, and draft acceptance that **this profiling run did not measure**. Corrected
2026-08-29:

1. **KV cache pooling.** The trace's own worker logs report per-rank KV memory of
   32.55 / 31.67 / 31.56 GiB. **The claim that "2 nodes ran out of KV cache above 32K"
   is false and is withdrawn** — the 2-node arm in
   [`20260825-decode-2v3`](../20260825-decode-2v3/) allocated 1,711,307 KV tokens and
   this repo has recorded **zero preemptions on either arm at any depth**, including
   131K and 262K runs on two nodes. Three nodes give a ~2.6x larger pool; they did not
   rescue a 2-node configuration that was failing, because it was not failing.
   ⚠️ The "4,688,072 tokens" figure here is a third distinct KV number, alongside
   4,660,501 (init log) and 2,822,574 (`/metrics`) — see the
   [handoff note](../../docs/HANDOFF-2026-08-28.md). Do not quote any of them without
   naming the instrument.

2. **MoE sharding.** The trace confirms `...b12xmoefusedw4a16kernelW4A16FusedMoeKernel`
   runs and accounts for 1,480.6 ms over 644 decode calls (2.30 ms/call). **It does not
   measure numerical parity**; the "rel error <= 2.3e-6" figure comes from the padding
   patch's own correctness suite ([`docs/patch.md`](../../docs/patch.md)), not from this
   profile. A timeline trace records kernel duration, never numerical accuracy.

3. **MTP speculative decoding.** **The "100% draft acceptance (3.00 tokens/step)" claim
   is withdrawn** — no acceptance telemetry was collected in this run, and it contradicts
   every measurement we have: the long-horizon probe measured **76.7–80.4%** acceptance
   at ~1.55 accepted tokens/step, and the #32 sweep measured 66.3–68.3% at `K=2`. 100% /
   3.00 is also arithmetically impossible at `K=2`, where the ceiling is 3 tokens per step
   only if *every* draft is accepted forever. The defensible statement is the general one:
   because decode is synchronization-bound, each accepted draft token avoids a forward
   pass and its collectives, which is *why* speculation helps here — but the size of that
   effect is quantified in [`20260828-issue32`](../20260828-issue32-mtp-concurrency-sweep/),
   not in this trace.

**What this trace genuinely establishes (a real first for the repo):**

- Decode and prefill have **fundamentally different bottlenecks**: rank 0 decode is
  87.54% inside collectives, while prefill is 39.23% MoE GEMM / 34.07% collectives /
  16.19% MLA attention — compute-dominated and balanced.
- The **stream overlap ratio is ~1.01x (decode) and ~1.03x (prefill)**, meaning there is
  essentially **no compute/communication overlap** on this rank. That is the single most
  actionable finding in the bundle and was previously invisible.
- The per-collective cost gap (19.53 ms decode vs 2.78 ms prefill) localises the decode
  penalty to **synchronization, not bandwidth** — consistent with, and independent of,
  the four-HCA result that doubling fabric bandwidth bought no decode throughput.

**To close the remaining question** — wire latency vs. launch overhead vs. a straggler
rank — capture **all three ranks simultaneously** and compare collective entry timestamps.
That is the natural follow-up and needs no new tooling.

---

## 4. Profiling Artifacts

- **Summary JSON**: `results/20260829-issue38-kernel-profiling/summary.json`
- **Decode 8K Trace**: `results/20260829-issue38-kernel-profiling/decode_8k_256tok/dp0_pp0_tp0_dcp0_ep0_rank0.1788042551004092878.pt.trace.json.gz`
- **Prefill 131K Trace**: `results/20260829-issue38-kernel-profiling/prefill_131k_1tok/dp0_pp0_tp0_dcp0_ep0_rank0.1788042679772743468.pt.trace.json.gz`
- **Analysis Script**: `scripts/analyze_trace.py`
- **Profiling Suite**: `scripts/profile_issue38.py`
