# Pre-registration — configuration-identical 2-node vs 3-node comparison

**Written before any measurement.** Per [`BENCHMARK-POLICY.md`](BENCHMARK-POLICY.md) and
the instruction in [`HANDOFF-2026-08-27.md`](HANDOFF-2026-08-27.md): *"Pre-register the
tables before running. It is the cheapest thing that converts 'we expect three nodes to
win' from a bias into a stated hypothesis."*

**Status:** tables empty, hypotheses stated, fairness rules fixed. Nothing measured yet.

---

## 1. The question

Should a user with three DGX Sparks run DeepSeek-V4 Flash on all three (`TP=3`) or on two
(`TP=2`)?

Every published 2v3 row in this repository is currently **confounded**. The existing TP=2
arm ([`20260827-decode-2v3-fixed`](../results/20260827-decode-2v3-fixed/)) ran a different
engine configuration from the production TP=3 arm. This run removes that confound.

## 2. The confound being removed

Read from the live 2-node env files and the running 3-node engine on 2026-08-29:

| Knob | TP=2 arm (as run) | TP=3 arm (as run) | Confounded |
|---|---|---|---|
| `MAX_NUM_SEQS` | 16 | **32** | yes — documented |
| `MTP_NUM_TOKENS` | 5 | **2** | yes — documented |
| `GPU_MEMORY_UTILIZATION` | 0.80 | **0.835** | yes — **not previously disclosed** |
| `MAX_MODEL_LEN` | 1048576 | 1048576 | no |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 8192 | no |

> **The third row is a new finding.** `DECISIONS.md` and `BENCHMARK-2V3-NODES.md` disclose
> only the first two. `GPU_MEMORY_UTILIZATION` differs as well, and per Issue #25 that
> single knob is worth ~35% of the KV pool and -10.7% starvation TTFT. The published 2v3
> table is therefore confounded by **more than it admits**. This document supersedes the
> two-confound description in both files.

`head.env` sets `GPU_MEMORY_UTILIZATION=0.80` explicitly, which **overrides** the
`docker-compose.yml` default of `0.80`→ and must be edited to `0.835`, not merely left
unset, to match.

## 3. Fairness rules — fixed before running

1. **Node count is the only independent variable.** Both arms run
   `MAX_NUM_SEQS=32`, `MTP_NUM_TOKENS=2`, `GPU_MEMORY_UTILIZATION=0.835`,
   `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=1048576`,
   `KV_CACHE_DTYPE=nvfp4_ds_mla`, `BLOCK_SIZE=256`.
2. **Same image, verified.** `dsv4-3spark:0.1.1`. Patch files verified byte-identical
   across all three nodes by sha256 before the run (image *IDs* differ — local builds —
   but `/opt/dsv4-tp3/*.py` hashes match).
3. **Fabric gate before each arm**, engine stopped, artifact committed. A failed gate
   aborts; it is not retried into a pass.
4. **256-token asserted window**, `min_tokens` + `ignore_eos`, hard failure when
   `completion_tokens != max_tokens`.
5. **Exclusivity enforced** (Requirement 5). `open-webui` runs on sparkmain against this
   engine; it must be stopped for the duration or the run is void. Request-count deltas
   recorded per arm.
6. **Warm, not cold.** ≥2 warmups per prompt shape per arm. Cold TTFT reported separately
   and labelled.
7. **7 reps per cell**, per-rep values published, spread published, median never alone.
8. **Config read from the live process** (`ps -eo args`), not from the env file.
9. **Both arms measured in the same session**, same day, same fabric state.

## 4. Hypotheses — stated before measurement

| # | Prediction | Confidence | What would falsify it |
|---|---|---|---|
| H1 | TP=3 retains a single-stream decode advantage at cc=1 across 2K–262K, but **smaller** than the currently published +7.3%–+16.7% | medium | TP=3 advantage unchanged or larger; or TP=2 wins |
| H2 | The `cc=16` aggregate row **flips or closes** to within the noise floor once TP=2 also runs `MTP=2` | medium | TP=2 retains a >6.5% aggregate win at `cc=16` |
| H3 | TP=2 retains the cold deep-prefill TTFT advantage past ~100K | high | TP=3 reaches first token sooner at 131K |
| H4 | Prefill throughput at 32K stays at parity (±2%) | high | Either arm wins by >2% |

**H1 and H2 are the ones that matter.** They are the two rows a user's decision turns on,
and they are the two most likely to move.

## 5. Pre-registered result tables — TO BE FILLED

Noise floor from Issue #31 is **6.6%–11.7%** per-passage. Any delta inside that band is
declared a **tie**, decided before seeing data.

> **Observed in-run, logged before any delta was computed (2026-08-29).** The TP=3 arm's
> 8K cell measured a **12.2%** per-rep spread (47.7–53.9 tok/s, n=7), marginally above the
> Issue #31 ceiling. TTFT was flat across all seven reps (4.1–4.7 s), so this is B12X MoE
> non-determinism, not JIT contamination or a stall. **The tie band is NOT being widened
> to accommodate it** — that would be exactly the post-hoc adjustment this document exists
> to prevent. It is recorded here so that if a final delta lands between 11.7% and 12.2%,
> the reader knows the per-rep spread was already that wide before the comparison was made,
> and can discount the result accordingly.

### 5a. Single-stream decode, cc=1 (tok/s, median of 7, spread published)

| Depth | TP=3 (32/MTP=2/0.835) | TP=2 (32/MTP=2/0.835) | Delta | Verdict |
|---|---|---|---|---|
| 2,048 | | | | |
| 8,192 | | | | |
| 32,768 | | | | |
| 131,072 | | | | |
| 262,144 | | | | |

### 5b. Aggregate throughput at concurrency, 8K prompt (tok/s)

| Concurrency | TP=3 | TP=2 | Delta | Verdict |
|---|---|---|---|---|
| cc=4 | | | | |
| cc=8 | | | | |
| cc=16 | | | | |

### 5c. Cold TTFT (s, labelled cold, reported separately)

| Depth | TP=3 | TP=2 | Delta | Verdict |
|---|---|---|---|---|
| 131,072 | | | | |
| 262,144 | | | | |

### 5d. Capacity (from each boot's own init log, with MTP depth stated)

| Metric | TP=3 | TP=2 |
|---|---|---|
| KV pool (tokens, init log) | | |
| KV pool (`/metrics`, if it disagrees) | | |
| Max concurrency | | |

## 6. Stopping rule

The run completes all cells in 5a–5d, or it reports which cells are missing and why. A
partial result is published as partial. **No cell is dropped because it came out the wrong
way.**

If the matched arm shows two nodes winning a workload, **that is the finding**, and it
goes in the README scoreboard with the same prominence as a three-node win.

## 7. What this run does NOT settle

- The **mechanism** of any delta. Issue #38 profiling is rank-0 and 3-node only; no
  matched all-rank trace exists. Every "why" in the results doc stays marked as a
  hypothesis until that trace is captured.
- The `bt=16384` deep-prefill mechanism (#33), still uncharacterized.
- The KV instrument disagreement (init log vs `/metrics`), tracked separately.
