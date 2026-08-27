# Long-context decode: 2 nodes vs 3 nodes on healthy fabric — 2026-08-26

**Status:** `VOID-25-token-window`; retained only as a diagnostic baseline. See
[`../index.yaml`](../index.yaml).

The measurement the advertising claim actually needs, and the one that did not exist
until today.

## Why this run exists

The repo's headline number for the third node — **"+8–17% per-stream from 2K context
upward"** — came from 2026-08-21, on the degraded fabric ([#14](../../issues/14)). The
healthy-fabric re-run on 2026-08-25 confirmed the *direction* (+16.9% at cc=1) but used
an **18-token prompt**. So the two arms of the claim were split across two runs:

| run | depth coverage | fabric | problem |
|---|---|---|---|
| 2026-08-21 | ✅ 2K–131K | ❌ degraded | right shape, unusable numbers |
| 2026-08-25 | ❌ 18 tokens | ✅ healthy | right numbers, wrong shape |

**Nobody buys a 1M-context cluster to send 18 tokens.** This run closes that gap: matched
arms, healthy fabric, five depths from 2K to 262K.

## Result

**The third node is worth nothing below 32K and a great deal above it.**

| context | TP=2 | TP=3 | 3-node gain |
|---:|---:|---:|---:|
| 2,036 | 75.8 | 76.3 | +0.8% |
| 8,081 | 72.4 | 72.6 | +0.3% |
| 32,268 | 70.8 | 70.2 | −0.9% |
| **129,006** | **54.4** | **72.6** | **+33.6%** |
| **257,993** | **71.5** | **84.4** | **+17.9%** |

Per-stream decode tok/s, median of 7, measured from first content token to last (prefill
and queueing excluded). Zero cache hits, 0 preemptions, 70 measured reps total.

### This supersedes "+8–17% from 2K upward"

The old claim was **degraded-fabric data** ([#14](../../issues/14)) and it is wrong in
both directions:

| | old claim (2026-08-21, degraded) | measured healthy (this run) |
|---|---|---|
| 2K–32K | +14%, +8%, +17% | **parity** (+0.8%, +0.3%, −0.9%) |
| 131K | +13% | **+33.6%** |
| 262K | not measured on 2 nodes | **+17.9%** |

Below 32K the advantage **does not exist** — the three cells are inside run-to-run noise
and one is negative. Above 100K it is **more than double** what was claimed. The
degradation had compressed a strongly depth-dependent effect into a flat ~13% band.

### The crossover is between 32K and 131K

That boundary is the useful fact for a purchasing decision, and it is mechanistically
sensible: the 2-node KV pool is **1,844,001 tokens** against the 3-node
**~4.5M**. Below 32K neither is under pressure and per-stream decode is bound by
per-token compute, which the third node does not improve (consistent with prefill being
at parity). Past ~100K the smaller pool costs real work per decode step.

**Confidence at 131K, the load-bearing cell.** The distributions barely overlap:

```
TP=2 @131K:  37.0  47.1  54.2 [54.4] 54.7  64.0  64.3
TP=3 @131K:  53.8  64.7  72.3 [72.6] 74.0  76.0  79.3
```

**Six of seven TP=3 reps beat the TP=2 median.** Both medians sit mid-distribution, not
in a stall cluster — the wide spreads are the JIT tail (trap 2 below), not bimodality.

### A second finding: decode gets *faster* past 131K

On three nodes the curve is a **U**, not a decay: 76.3 → 72.6 → 70.2 → 72.6 → **84.4**.
At 262K this cluster decodes **faster than at 8K**.

This is not a stall artifact — every one of the seven reps at 262K (69.5–98.1) beat the
*median* at 32K. The likely mechanism is MTP speculative decoding: a longer context gives
the draft model more signal, so acceptance rises and offsets the growing attention cost.
It is consistent with the established finding that decode rate is content-dependent
(a 1.65x swing from prompt shape alone —
[`../../docs/BENCHMARK-METHODOLOGY.md`](../../docs/BENCHMARK-METHODOLOGY.md)).

Two nodes show the same upturn (54.4 → 71.5), so the effect is not node-count-specific.

### TTFT: the third node costs prefill latency at depth

| context | TP=2 TTFT | TP=3 TTFT | |
|---:|---:|---:|---|
| 129,006 | **72.4 s** | 77.1 s | 2-node 6% sooner |
| 257,993 | **158.4 s** | 181.6 s | 2-node **13% sooner** |

Consistent with prefill parity plus a third node's added collective cost. **If your
workload is one-shot long prompts with short answers, two nodes reach first token sooner.
If it generates substantial output at depth, three nodes finish first.**

## Why this is a fair comparison

- **Node count is the only variable.** Both arms: `MAX_MODEL_LEN=1048576`,
  `MAX_NUM_SEQS=16`, `MTP_NUM_TOKENS=5`, `GPU_MEMORY_UTILIZATION=0.80`,
  `--kv-cache-dtype nvfp4_ds_mla`, `--pipeline-parallel-size 1`. Verified by reading
  `ps -eo args` on the live engine in **both** arms, not from config files.
- **Same harness, same prompts, same day**, ~30 minutes apart.
- **Fabric gated between the arms with the engine stopped** — the only moment NCCL
  bandwidth is measurable. 33 passed / 0 failed; pairs at 7.78 / 9.19 / 9.33 GB/s, all
  `via NET/IB/*`, zero `NET/Socket`. See [`fabric-gate-pre-tp2.txt`](fabric-gate-pre-tp2.txt).
- **Zero prefix-cache hits** in either arm: every prompt carries a unique session header,
  and `cached_tokens` is asserted 0 on every one of the 35 measured reps per arm. The
  engine-side `prefix_cache_hits_total` counter was unchanged across the TP=3 sweep.
- **0 preemptions** in both arms.

## Three traps this harness is built to avoid

`scripts/decode_depth_sweep.py` exists because the obvious way to run this measures the
wrong thing three different ways.

**1. The prefix cache.** Ascending depths where each prompt is a prefix of the next turn
every run after the first into a cache hit. Every prompt here gets a unique
`[session <label>-<rep>-<depth>]` header, and the result is checked rather than assumed.

**2. The JIT stall tail.** TileLang/CuTeDSL compile kernels *during inference* the first
time they see a shape, costing ~5 s. Landing one inside a ~3 s run manufactures a 20%
difference that is not real. Observed live at 2K on an idle engine with zero competing
requests: three reps at ~91 tok/s, then 75.4 and 57.3 — a **38% spread from JIT alone**.
Mitigation: warm every shape before measuring it, take 7 reps, report the **median**.
This is why the per-depth spreads below are wide and why the medians are still sound.

**3. Token targeting.** The naive 4-chars-per-token rule undershot by **38%** — a "2048"
prompt tokenized to 1,276 — because repeated English prose tokenizes far denser.
Recalibrated to 6.42 chars/token, landing within 1.4% of nominal. Depths that do not land
on their nominal values are not comparing the same shape between arms.

## Evidence

| file | contents |
|---|---|
| `tp3-depth.jsonl` / `-summary.json` / `.log` | 3-node arm, 35 measured reps + 10 warmups |
| `tp2-depth.jsonl` / `-summary.json` / `.log` | 2-node arm, matched |
| `fabric-gate-pre-tp2.txt` | full gate, engine stopped, between the arms |

Harness: [`../../scripts/decode_depth_sweep.py`](../../scripts/decode_depth_sweep.py).

**Related:** [#14](../../issues/14) · [`../20260825-decode-2v3/`](../20260825-decode-2v3)
(the 18-token healthy run) · [`../../docs/WHY-THREE-NODES.md`](../../docs/WHY-THREE-NODES.md)
