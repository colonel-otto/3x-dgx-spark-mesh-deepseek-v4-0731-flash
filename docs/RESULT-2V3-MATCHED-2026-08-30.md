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
| 262,144 *(n=12)* | **45.04** | 39.79 | **+13.2 %** | 4×10⁻⁵ | 2.3×10⁻⁴ | **1.000** | **SIGNIFICANT** |

**All five cells are significant by two independent tests.** The permutation p-values at
2K–131K sit at the 100,000-resample floor.

**Cliff's δ = 1.000 at 32K and 262K means every TP=3 rep beat every TP=2 rep** — complete
separation, no overlap. At 131K δ=0.998 and 8K δ=0.944 are near-total separation.

### 262K reverses a published claim

The published table said **two nodes win cold deep context**. Matched, they do not — three
nodes win 262K by **+13.2 %** with zero overlap:

```
TP=3: 42.4 43.5 44.0 44.0 44.5 44.7 45.4 45.5 45.9 46.1 47.8 64.3
TP=2: 31.5 33.8 35.3 38.3 38.9 39.5 40.1 40.3 40.5 40.6 40.8 41.7
      TP=3 min 42.4  >  TP=2 max 41.7
```

That cell's raw spread reads 48.6 %, driven entirely by the single 64.3 rep against a
42–48 band — the outlier pattern §5e of the pre-registration anticipated and fixed a rule
for in advance. The median is unaffected and the separation is total.

*(Note: cold **TTFT** at depth is a different measurement from decode rate and is not
settled by this table — see §6.)*

The concurrency arm was still running when this was written; it is appended once complete.

## 1b. TTFT: the "two nodes win deep prefill" claim is REVERSED

The published table states **two nodes reach first token 22.3 s sooner at 131K** and
14.1 s sooner at 262K, and recommends two nodes for cold deep-context ingestion. **Matched,
the opposite is true at every depth**, and three nodes' advantage *grows* with context:

| Depth | TP=3 TTFT | TP=2 TTFT | Delta | p | Cliff's δ |
|---:|---:|---:|---:|---:|---:|
| 2,048 | **1.13 s** | 1.23 s | **−7.9 %** | — | — |
| 8,192 | **4.22 s** | 4.79 s | **−11.8 %** | — | — |
| 32,768 | **17.26 s** | 19.59 s | **−11.9 %** | — | — |
| 131,072 | **74.90 s** | 87.50 s | **−14.4 %** | 3.0×10⁻¹¹ | **−1.000** |
| 262,144 | **166.68 s** | 227.24 s | **−26.6 %** | 3.7×10⁻⁵ | **−1.000** |

At 262K the separation is total — **TP=3's worst TTFT (176.8 s) beats TP=2's best
(204.2 s) by 27 seconds**:

```
TP=3 TTFT range: 165.8 – 176.8 s
TP=2 TTFT range: 204.2 – 258.7 s
```

The published finding was an artefact of the six confounds — principally that the 2-node
arm never ran Profile B, whose `LONG_PREFILL_TOKEN_THRESHOLD=1024` and
`DSPARK_MAX_INFLIGHT_PREFILLS=2` exist precisely to tune long-prefill behaviour, and whose
`GPU_MEMORY_UTILIZATION=0.835` gives prefill more room.

> **Caveat on wording:** these are **warm** TTFTs (3 warm-ups per shape), so they measure
> steady-state time-to-first-token, not first-request-after-restart cold start. Both arms
> were warmed identically, so the comparison is valid; the absolute numbers are not
> "cold-start" figures.

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
> faster than two nodes at **every tested depth from 2K to 262K** — **+6.7 %** at 2K and
> **+13 % to +20 %** from 8K through 262K — with complete or near-complete separation
> across 30 reps per cell (12 at 262K). It also pools **2.11×** the KV cache. Every result
> is measured with node count as the only variable.

Remaining caveats to publish alongside:
- **High-concurrency aggregate is not yet settled.** The older claim that **two nodes win
  at cc≥8** was measured under all six confounds; the matched arm is appended when it
  completes.
- **The deep-prefill TTFT claim is reversed, not merely unsettled** (§1b). Three nodes
  reach first token sooner at every depth, by 26.6 % at 262K with zero overlap. The
  published recommendation to prefer two nodes for deep ingestion should be withdrawn.
