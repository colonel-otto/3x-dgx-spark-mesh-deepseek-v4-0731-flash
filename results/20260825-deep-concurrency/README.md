# Deep-concurrency re-run on healthy fabric — 2026-08-25

Re-runs the 2026-08-21 "4 × 200,000-token" test that [issue #15](../../issues/15)
flagged as suspect, because the original was taken while spark1's RDMA fabric ran at
~15% of its sibling's collective bandwidth.

## Result

Matched depth, byte-identical prompts on both arms, same afternoon, same harness.

| | 2-node TP=2 | 3-node TP=3 |
|---|---:|---:|
| TTFT then (degraded) | 539,666 ms | 553,113 ms |
| TTFT now (healthy) | **293,987 ms** | **396,804 ms** |
| improvement | **1.84x** | **1.39x** |
| wall | 7.8 min | 10.2 min |
| decode tok/s | 1.0 | 0.9 |
| preemptions | **0** | **0** |
| prefix-cache hits | 0 / 800,185 | 0 / 800,185 |
| prompt tokens (engine-verified) | 200,045–200,048 | 200,045–200,048 |

**What survives:** "KV capacity is not the binding constraint." Preemptions are 0 on
both arms, the pool never fills, and that was true then and is true now.

**What does not:** "Equally unusable on both." They are no longer equal — 2-node reaches
first token **1.35x sooner**. The old parity (1.025x apart) was two configurations being
throttled to a similar floor by the same degraded link, and spark1 sat in the 3-node arm,
so the handicap fell disproportionately there. Removing it separated them, in 2-node's
favour. Both are still unusable at this depth: ~5 and ~6.6 minutes to first token.

**Not re-tested:** whether the "serialized prefill" *mechanism* is unchanged. Prefill got
materially faster, so the magnitude attributed to serialization was inflated by the
fabric — but the mechanism itself was never measured directly and is not restated here as
settled.

## Depth is steeply non-linear — read this before comparing anything

On the 3-node arm, TTFT went **275 s at 185.5K → 397 s at 200K: +44% for +8% depth.**

The first two runs here missed the target depth (185.5K, then 172.2K) because the prompt
builder used an *estimated* tokens-per-word ratio, and the first "correction" moved the
constant the wrong way — it divides the word target, so raising it shortens the prompt.
The ratio is now measured against the engine's `/tokenize` (1.2056, flat from 150K–240K),
and every run records `prompt_tokens_actual` from the engine.

Those two off-depth runs are kept (`tp3_run1.json`, `warm_tp3.json`) because they are
what establishes the non-linearity. They are **not** comparable to the 2026-08-21 rows.

## Files

| file | what |
|---|---|
| `deepconc.py` | the harness — streams, defeats the prefix cache, reads preemptions from the engine |
| `tp2_200k.json` / `tp3_200k.json` | **the headline runs**, matched depth |
| `warm_tp2.json` / `warm_tp3.json` | warm-up sweeps; TP=2's agrees with its measured run within 0.9% |
| `tp3_run1.json` | off-depth (172.2K) supporting run |
| `fabric-gate-full.json` | `scripts/fabric_gate.sh --nccl=full` taken between the two arms: 13/13 pass, pairs at 4.59–4.65 GB/s |

## Method

The fabric was gated before the runs and again between the arms, with the engine stopped
so NCCL bandwidth could actually be measured — see `scripts/fabric_gate.sh`. Both arms ran
the 2026-08-21 settings (`MAX_MODEL_LEN=460800`, `MAX_NUM_SEQS=16`, `MTP_NUM_TOKENS=4`,
`GPU_MEMORY_UTILIZATION=0.85`) so that **the fabric is the only variable that changed**
versus the suspect rows. That is deliberately *not* the current production profile
(1M / MTP=5 / 0.80), which was restored afterwards and verified from the engine.

KV cache read back per arm: 3,588,422 tokens (3-node) vs 1,815,356 (2-node) — a 1.98x
ratio, matching the 1.95x the original comparison reported.
