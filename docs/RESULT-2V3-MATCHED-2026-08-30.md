# RESULT — matched 2-node vs 3-node comparison, 2026-08-30

**This is the answer.** First configuration-identical, same-session,
thermally-equalised, adequately-powered comparison in this repository's history.

Pre-registered in [`PREREGISTRATION-2V3-MATCHED.md`](PREREGISTRATION-2V3-MATCHED.md) —
hypotheses, tie band, outlier rule, and adjudication test all fixed **before** measurement.

---

## 1. Headline: three nodes decode faster. Confirmed.

Node count was the **only** variable. Both arms ran `MAX_NUM_SEQS=32`,
`MTP_NUM_TOKENS=2`, `GPU_MEMORY_UTILIZATION=0.835`, `MAX_NUM_BATCHED_TOKENS=8192`,
`MAX_MODEL_LEN=1048576`, `LONG_PREFILL_TOKEN_THRESHOLD=1024`,
`DSPARK_MAX_INFLIGHT_PREFILLS=2`, `KV_CACHE_DTYPE=nvfp4_ds_mla`,
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` on the same baked image, verified against each
live engine before measuring (`MATCH CONFIRMED` in the run log).

**n=30 per cell per arm** (n=12 at 262K), 256-token asserted windows, 3 warm-ups per shape,
nodes cooled to ≤70 °C before each arm, clocks sampled every 5 s throughout.

| Depth | TP=3 median | TP=2 median | Delta | Mann-Whitney p | Permutation p | Cliff's δ | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2,048 | **46.59** | 43.68 | **+6.7 %** | 6.0×10⁻⁵ | 1×10⁻⁵ | 0.604 | **SIGNIFICANT** |
| 8,192 | **51.07** | 43.62 | **+17.1 %** | 3.5×10⁻¹⁰ | 1×10⁻⁵ | 0.944 | **SIGNIFICANT** |
| 32,768 | **50.83** | 42.29 | **+20.2 %** | 3.0×10⁻¹¹ | 1×10⁻⁵ | **1.000** | **SIGNIFICANT** |
| 131,072 | **47.38** | 39.92 | **+18.7 %** | 3.3×10⁻¹¹ | 1×10⁻⁵ | 0.998 | **SIGNIFICANT** |

**All four cells are significant by two independent tests.** The permutation p-values sit
at the 100,000-resample floor.

**Cliff's δ = 1.000 at 32K means every one of the 30 TP=3 reps beat every one of the 30
TP=2 reps** — complete separation, no overlap. At 131K δ=0.998 and at 8K δ=0.944 are near-
total separation.

262K and the concurrency arm were still running when this was written; they are appended
below once complete.

## 2. KV cache: 2.11×, not 2.6×

Read from each arm's own init log, same instrument, same MTP depth:

| | TP=2 | TP=3 | Ratio |
|---|---:|---:|---:|
| GPU KV cache size | 2,217,166 tokens | 4,688,072 tokens | **2.11×** |
| Max concurrency @ 1M tokens/request | 2.11× | 4.47× | 2.12× |

The advantage is real, but the published **2.6×** is inflated. That figure came from arms
differing in `GPU_MEMORY_UTILIZATION` (0.80 vs 0.835) and `MTP_NUM_TOKENS` (5 vs 2), both
of which move the pool. Matching them costs ~19 % of the claimed ratio.

## 3. Why this run succeeded where the published comparison failed

The published comparison reported **+7.3 % to +16.7 %** — directionally right, and this run
confirms three nodes wins. But it could not *support* that claim, for two reasons now fixed:

**Six confounds, of which only two were disclosed.** `LONG_PREFILL_TOKEN_THRESHOLD`,
`DSPARK_MAX_INFLIGHT_PREFILLS`, and `KV_CACHE_DTYPE` — all **Profile B** settings — were
set on the 3-node arm and unset on the 2-node one, alongside the known `MAX_NUM_SEQS`,
`MTP_NUM_TOKENS`, and the undisclosed `GPU_MEMORY_UTILIZATION`. The old comparison was
"3 nodes tuned vs 2 nodes untuned".

**n=7 could not resolve the effect.** At the observed CVs, four of five published rows
failed a significance test (131K: p=0.535). n=30 fixes this: CVs here are 2.6–8.7 %
against effects of 6.7–20.2 %, so every cell separates decisively.

> **The published numbers were not wrong in direction — they were unsupported in
> evidence.** This run supplies the evidence, and the true effect at 8K–131K
> (**+17 % to +20 %**) is *larger* than the published +14.2 % / +7.3 %, because the old
> 2-node arm was running an untuned profile that flattered it at some depths and the noise
> swamped the rest.

## 4. Spreads — wide, but n=30 absorbs them

| Depth | TP=3 spread / CV | TP=2 spread / CV |
|---:|---:|---:|
| 2K | 37.1 % / 8.7 % | 11.0 % / 3.1 % |
| 8K | 23.9 % / 6.1 % | 11.0 % / 2.6 % |
| 32K | 15.4 % / 3.2 % | 16.2 % / 4.2 % |
| 131K | 15.8 % / 3.9 % | 24.1 % / 5.9 % |

Raw spread stays wide — GB10 clocks float with a package power budget and cannot be locked
([`GPU-CLOCKS-NOT-LOCKABLE.md`](GPU-CLOCKS-NOT-LOCKABLE.md)) — but **CV is what determines
resolvability**, and at n=30 every cell clears its effect size comfortably. This is the
power analysis working as designed: n was chosen per cell from CV and effect size, not by
convention.

## 5. Hygiene

- **`EXCLUSIVITY_PASS delta=654 expected=654`** on the TP=2 arm — exact to the request,
  no foreign traffic.
- All reps returned exactly 256 completion tokens; `cached_tokens = 0` throughout.
- Correctness `17 × 23 = 391` verified on each arm before measuring.
- Clock envelope, TP=2 arm (2,260 samples/node): sparkmain 2442 MHz / 77 °C, spark1
  2478 MHz / 80 °C, spark2 2515 MHz / 50 °C (idle — not in the TP=2 ring, as expected).

## 6. What to tell users

> **Run three nodes if you have them.** On matched configuration, three-node `TP=3` decodes
> **+17 % to +20 % faster than two nodes at 8K–131K context**, and **+6.7 % at 2K**, with
> complete or near-complete separation across 30 reps per cell. It also pools **2.11×** the
> KV cache. Both results are measured with node count as the only variable.

Remaining caveats to publish alongside:
- 262K and high-concurrency aggregate are appended when those cells complete; the older
  claim that **two nodes win aggregate throughput at cc≥8** was measured under the six
  confounds and has not yet been re-tested here.
- Cold deep-prefill TTFT past ~100K previously favoured two nodes; also not yet re-tested
  matched.
