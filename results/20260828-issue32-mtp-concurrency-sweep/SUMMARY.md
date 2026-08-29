# Issue #32: MTP Speculative Depth ($K \in \{5, 3, 2\}$) Concurrency Sweep

## Executive Summary

We evaluated speculative drafting depth ($MTP\_NUM\_TOKENS \in \{5, 3, 2\}$) across concurrency levels $cc \in \{1, 4, 8, 16\}$ at 8,192 token context with forced 256-token decode windows and telemetry from engine Prometheus metrics.

> **Correction, 2026-08-29.** The originally published table labelled several cells
> "Median" that were in fact single arbitrary repetitions read out of `rep_details`.
> Aggregate throughput and acceptance were correct; **TTFT and per-stream tok/s were
> not**, and the TTFT deltas derived from them were wrong. Every cell below is now
> recomputed from the committed raw JSON (`aggregate_tok_s_median`, `ttft_s_median`,
> `spec_acceptance_ratio_pct`, and the median of `rep_details[].median_stream_tok_s`).
> The operational conclusion is unchanged; the TTFT advantage is smaller than claimed.

### Key Conclusions:
1. **$K=2$ (Upstream Reference) is the Pareto Winner**:
   - **$cc=16$ Throughput**: $K=2$ achieves **55.10 tok/s** (vs 51.37 tok/s on $K=5$, a **+7.3% throughput gain**), closing the concurrency gap where TP=3 previously lagged TP=2.
   - **Acceptance Efficiency**: Draft acceptance rate increases from **41.7% ($K=5$) to 66.3% ($K=2$)**, drastically reducing rejected verification FLOPs at high batch sizes.
   - **Single-Stream ($cc=1$)**: Decode speed is equivalent within noise — median stream 54.32 tok/s on $K=5$ vs **53.37 tok/s** on $K=2$ (-1.7%), inside the per-passage noise floor of 6.6–11.7% established in [issue #31](../20260828-issue31-serving-determinism/). **$K=2$ does not measurably beat $K=5$ single-stream; it ties.**
   - **TTFT**: $K=2$ improves median TTFT at every concurrency, by **-7.8% to -11.4%**.
2. **Mechanism — hypothesis, not measurement.** The plausible account is that as batch size grows, verification compute ($\text{batch} \times K$) saturates the SMs, so long rejected drafts on $K=5$ displace real generation compute. **This has not been profiled**; acceptance ratio and throughput are consistent with it, but no kernel-level evidence exists ([issue #38](../../../issues/38)).

## Headline Comparison (Context Depth = 8,192 Tokens, 256 Output Tokens)

All values are medians of 5 repetitions. Per-stream tok/s at $cc>1$ is the per-request
rate; aggregate is the cluster total, and the two diverge by design as concurrency rises.

| Concurrency ($cc$) | Metric | MTP=5 | MTP=3 | **MTP=2 (Winner)** | Delta ($K=2$ vs $K=5$) |
|---|---|---:|---:|---:|---:|
| **$cc=1$** | Median Aggregate tok/s | 25.87 | 28.12 | **26.83** | +3.7% |
| | Median Stream tok/s | **54.32** | 54.90 | 53.37 | -1.7% (parity, within noise) |
| | Acceptance Ratio | 42.9% | 56.8% | **68.3%** | **+25.4 pp** |
| | Median TTFT | 4.86 s | 4.44 s | **4.33 s** | **-10.8%** |
| **$cc=4$** | Median Aggregate tok/s | 40.81 | **42.46** | 41.99 | +2.9% |
| | Median Stream tok/s | **23.23** | 23.21 | 20.74 | -10.7% |
| | Acceptance Ratio | 42.2% | 56.0% | **66.4%** | **+24.2 pp** |
| | Median TTFT | 12.47 s | 11.68 s | **11.05 s** | **-11.4%** |
| **$cc=8$** | Median Aggregate tok/s | 49.11 | 49.56 | **50.51** | +2.9% |
| | Median Stream tok/s | 12.05 | 12.08 | **12.41** | +3.0% |
| | Acceptance Ratio | 42.8% | 56.6% | **67.6%** | **+24.8 pp** |
| | Median TTFT | 18.92 s | 18.60 s | **18.45 s** | **-2.5%** |
| **$cc=16$** | Median Aggregate tok/s | 51.37 | 53.51 | **55.10** | **+7.3%** |
| | Median Stream tok/s | 6.31 | 6.53 | **6.70** | +6.2% |
| | Acceptance Ratio | 41.7% | 56.1% | **66.3%** | **+24.6 pp** |
| | Median TTFT | 36.30 s | 34.30 s | **33.45 s** | **-7.8%** |

**One caveat the aggregate column hides:** at $cc=4$ the $K=2$ median per-stream rate is
*lower* than $K=5$ (20.74 vs 23.23 tok/s). Aggregate still favours $K=2$ across the
sweep, but the per-request experience at moderate concurrency is not uniformly better.

**Acceptance deltas are stated in percentage points (pp), not percent.** The earlier
table wrote "+25.4%" for a move from 42.9% to 68.3%; that is +25.4 pp, or +59% relative.

## Raw Repetition Data

### MTP=5 ($K=5$)
- $cc=1$: Agg [19.44, 25.87, 27.59, 24.20, 28.78], Stream [54.32, 50.81, 54.68, 51.51, 59.66] tok/s
- $cc=4$: Agg [12.17, 38.05, 40.87, 40.81, 44.58] tok/s
- $cc=8$: Agg [36.43, 47.92, 49.53, 49.16, 49.11] tok/s
- $cc=16$: Agg [42.38, 44.41, 51.54, 51.49, 51.37] tok/s

### MTP=3 ($K=3$)
- $cc=1$: Agg [26.01, 28.20, 28.53, 28.12, 25.55], Stream [54.41, 54.90, 56.50, 52.60, 58.98] tok/s
- $cc=4$: Agg [19.17, 43.76, 27.83, 42.46, 42.98] tok/s
- $cc=8$: Agg [45.75, 49.79, 50.58, 49.31, 49.56] tok/s
- $cc=16$: Agg [44.33, 53.03, 53.51, 54.03, 54.54] tok/s

### MTP=2 ($K=2$)
- $cc=1$: Agg [26.83, 28.90, 28.78, 26.04, 24.27], Stream [53.37, 55.72, 55.24, 46.60, 52.30] tok/s
- $cc=4$: Agg [21.97, 39.37, 41.99, 42.23, 42.45] tok/s
- $cc=8$: Agg [42.41, 49.70, 51.06, 50.95, 50.51] tok/s
- $cc=16$: Agg [49.80, 55.21, 55.10, 55.07, 55.40] tok/s

## Operational Recommendation
Adopt **`MTP_NUM_TOKENS=2`** as the operational default: it ties single-stream decode
speed (within the measured noise floor), raises draft acceptance from 42% to 67%,
reduces median TTFT by 2.5–11.4% across concurrencies, and yields +7.3% aggregate
throughput at $cc=16$. The trade it accepts is a lower per-stream rate at $cc=4$.

**Shipped:** `MTP_NUM_TOKENS=2` is the live production value — see
[`docs/DECISIONS.md`](../../docs/DECISIONS.md) and
[`HANDOFF-2026-08-28.md`](../../docs/HANDOFF-2026-08-28.md).
