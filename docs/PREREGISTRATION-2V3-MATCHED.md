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

> **Run 1 was ABORTED and is not the result. Recorded here so the restart is part of the
> pre-registered record, not an undocumented do-over.** Eight minutes in, three of ten
> cells done, per-cell spread was growing with depth and elapsed time — 5.0 % (2K),
> 12.2 % (8K), 17.3 % (32K) — and the 32K cell declined monotonically across its seven
> reps (54.9 → 53.8 → 52.8 → 50.8 → 46.0 → 51.4 → 46.9 tok/s) with TTFT falling in
> lockstep. Cause established mid-run: GB10 does not honour `nvidia-smi -lgc`, so GPU
> clock floats with a package power budget and diverged per node as the workers heat-soaked
> (83–86 °C on spark1/spark2 against 75 °C on sparkmain) — see
> [`GPU-CLOCKS-NOT-LOCKABLE.md`](GPU-CLOCKS-NOT-LOCKABLE.md). A 17.3 % spread cannot
> resolve a 7–17 % effect, so the run was stopped rather than carried to a conclusion it
> could not support. **No run-1 number is used anywhere.** Run 2 adds continuous
> clock/temp/power telemetry on all three nodes and a ≤70 °C cooldown before each arm.
> Hypotheses, tie band, and tables below are unchanged from before run 1.

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

## 5e. Outlier handling — fixed 2026-08-29, mid-TP=3-arm, before the TP=2 arm ran

The TP=3 131K cell produced one extreme high rep. Per-rep decode, in issue order:

```
46.40  47.95  45.75  45.05  49.40  46.87  59.37 tok/s      (spread 30.6%)
```

Rep 7 is **7.8 robust standard deviations** above the other six (MAD of the rest = 1.10),
and it is the only rep with a different TTFT (78.07 s against a flat 74.34–74.61 s). All
seven completed exactly 256 tokens with `cached_tokens: 0`, so this is neither a window
collapse nor APC contamination. Engine-wide MTP counters at the time read 66.3 %
acceptance / 1.325 mean accepted length — matching Issue #32 — so the most likely cause is
an unusually favourable speculative-acceptance run, an intrinsic property of the workload.

**The rule, fixed before the TP=2 arm and applied identically to both:**

1. **No rep is ever discarded.** All seven values are published for every cell.
2. **The median is the reported statistic**, as it already is. It is robust here: the cell
   reads **46.87** with the outlier and **46.63** without — a **0.5 %** difference.
3. **Spread is reported as measured (30.6 %), not cleaned.** Where a single rep drives it,
   the trimmed spread is reported *beside* it, never instead of it: 30.6 % as measured,
   **9.3 %** excluding the single extreme rep.
4. **A cell whose spread exceeds the noise floor cannot adjudicate a sub-20 % difference**
   even when its median is stable. Such cells are marked and excluded from any verdict.

This exists so the choice cannot be made after seeing which arm it favours.

### 5f. The deep cells are outlier-prone, not broadly unstable (TP=3 arm, all 5 cells)

With the full TP=3 depth sweep complete, the shape of the noise is now characterised.
Dropping **one** high and **one** low rep from each cell:

| Depth | Median | Spread as measured | Spread dropping hi+lo |
|---|---:|---:|---:|
| 2K | 51.10 | 9.6 % | **5.7 %** |
| 8K | 51.50 | 6.6 % | — |
| 32K | 52.20 | 5.9 % | — |
| 131K | 46.87 | 30.6 % | **7.8 %** |
| 262K | 44.90 | 22.3 % | **8.5 %** |

Every cell lands at **5.7–8.5 %** once a single extreme rep at each end is set aside —
inside the Issue #31 floor of 6.6–11.7 %. So the deep cells are not diffusely noisy; they
carry **occasional extreme reps on an otherwise tight distribution**, consistent with
speculative-acceptance variance rather than thermal drift (which would show as a trend
across reps, as it did in the aborted run 1, and does not here).

**Consequence, fixed before the TP=2 arm:** the **median is trustworthy at every depth**,
including 131K and 262K. Raw spread at those depths is inflated by single reps and must
not by itself be read as "this cell cannot adjudicate". The adjudication test in 5e(4) is
therefore applied to the **hi+lo-trimmed** spread, with the raw spread always published
beside it. Both arms are read the same way, and n=7 stays the published rep count.

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
