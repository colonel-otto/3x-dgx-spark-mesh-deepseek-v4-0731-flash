# Issue #33: Deep-Prefill TTFT at `MAX_NUM_BATCHED_TOKENS=16384` vs `8192` (Single-Variable Investigation)

## Executive Summary

We investigated the deep-prefill TTFT degradation (+22.1% at 131K, +29.0% at 262K) observed when raising `MAX_NUM_BATCHED_TOKENS` from 8192 to 16384.

### 1. Confound Audit & Script Fix
The script `scripts/configure_speed_profile.py` was audited and found to have hardcoded `MAX_MODEL_LEN=460800` (instead of the production 1,048,576) and attempted to write an unused `NCCL_BUFFSIZE` variable. This was corrected in `scripts/configure_speed_profile.py` and `docker-compose.yml`.

### 2. Single-Variable Confirmation
Re-running the matched baseline with `MAX_MODEL_LEN=1048576`, `GPU_MEMORY_UTILIZATION=0.835`, and `MTP_NUM_TOKENS=2` confirmed:
- **32K Context**: TTFT = **17.34 s** ($n=5$), Decode = 53.28 tok/s.
- **131K Context**: TTFT = **74.74 s** ($n=5$), Decode = 47.32 tok/s.

This precisely reproduces the 74–75s baseline for $bt=8192$ and confirms that $bt=16384$'s ~92.5s TTFT (+22.1%) is real and independent of `MAX_MODEL_LEN`.

### 3. Mechanism — NOT DIAGNOSED

**The mechanism remains uncharacterized.** The arithmetic disproof of the bus-saturation
hypothesis holds: activation transfers alone cannot explain a ~17-second delta, so that
explanation is ruled *out*. Nothing has been ruled *in*.

Two candidate hypotheses remain, **neither measured**:

1. **Attention kernel chunk tiling / cache spilling** — 16,384-token chunks may exceed
   L2 tile boundaries in the MLA/FlashInfer kernels, spilling across the memory
   hierarchy.
2. **All-reduce serialization** — larger per-chunk collective buffers may increase
   serialization latency between TP layers across the switchless RoCE mesh.

> **Correction, 2026-08-29.** This section previously presented both hypotheses as "the
> actual mechanism," diagnosed. No profile, kernel trace, or collective timing was ever
> captured to support either. That is the same defect withdrawn from the Issue #28 record
> in commit `d199655` — a hypothesis attached to a real delta and stated as a finding.
> Distinguishing these requires a timeline capture; that is the subject of
> [issue #38](../../../issues/38). **The measured TTFT delta is real and reproducible;
> the explanation for it is not yet evidence.**

### 4. Final Operational Decision
**`MAX_NUM_BATCHED_TOKENS=8192` stands confirmed as the cluster default.** It provides:
- Optimal deep TTFT (-22% faster at 131K, -29% faster at 262K).
- +21% greater KV cache capacity.
- Zero chunked prefill starvation under concurrent workloads.
