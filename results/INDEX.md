# Provenance index

**Every published number should be traceable to how it was measured.** This page is
generated from [`index.yaml`](index.yaml), which is the machine-readable source of truth.
If the two disagree, the YAML wins.

Provenance here has been forgotten once already. The point of this index is that a reader
can go from any number in the repo to the committed file it came from — or discover that
no such file exists.

---

## Read this first: two rules that govern the whole page

### 1. A result without a fabric gate is not trustworthy

This repo lost multiple benchmark rounds to a fault that **no indicator caught**. One node
ran at roughly **15% of its collective bandwidth** with every counter green, no errors, no
NCCL warnings, and a healthy-looking engine ([#14](../../issues/14)). Forty-five files were
spent investigating an engine that was fine
([`20260824-prefill`](20260824-prefill)), and a good configuration was rejected because the
link, not the config, was failing ([`20260824-seqs32-nccl`](20260824-seqs32-nccl)).

[`scripts/fabric_gate.sh`](../scripts/fabric_gate.sh) exists to catch that, and
[`scripts/run_experiment.sh`](../scripts/run_experiment.sh) calls it and **refuses to
benchmark on failure**. Every row below therefore carries a `fabric_gate` column:

| Value | Meaning |
|---|---|
| ✅ `PRESENT-PASS` | A gate artifact is committed **in that directory** and it passed. The fabric state is measured. |
| ❌ `PRESENT-FAIL` | A gate artifact is committed and it failed. *(No row currently holds this value.)* |
| ⚠️ `ABSENT` | No gate artifact committed. **The fabric state of this run is an assumption, not a measurement.** Not automatically void — but not verified either. |
| ⏳ `PRE-DATES-GATE` | The run predates the gate script. No artifact could exist. All seven of these are also `VOID-degraded-fabric`. |

**Two things that look like gates and are not.** A *setup* script that configures the mesh
(`mesh-master.sh`) does not verify that the mesh performs. A *correctness* gate proves the
config serves right answers, not that the link runs at speed — a degraded link serves
correct answers slowly. And having bandwidth data is not the same as having a gate: the
`agbench` sweeps in `20260824-seqs32-nccl` *measured* 0.49 GB/s and that number was used to
condemn the config instead of the fabric. **A gate is a measurement with a threshold
attached, taken before the run, that stops the run.**

Bandwidth is only measurable with the **engine stopped** — it holds the NICs otherwise.
A gate taken against a live engine comes back `nccl_mode: "skip"`, which is correct
behaviour, not a pass.

### 2. VOID rows are kept on purpose, as diagnostic baselines

> *"We can keep bad results but it needs to be specified that it's only kept as a baseline,
> so others who fail and get comparable results will know."*

A voided number is still a **fingerprint of a specific failure**, and matching that
fingerprint is often the fastest way to identify the failure. So nothing here is deleted or
quietly corrected. Every `VOID` and `SUPERSEDED` row carries a **`baseline_value`** field in
[`index.yaml`](index.yaml) answering one question:

> **If someone reproducing this gets numbers like these, what does it tell them?**

Those answers are reproduced in full under each status section below. Read them as
troubleshooting entries, not as tombstones. The short version:

| If you measure… | It probably means… | Baseline row |
|---|---|---|
| ~1,034 tok/s prefill at 32K on 3 nodes, engine tests all clean | One degraded fabric link. Same shape hit 2,095 after a reboot, no config change. | [`20260824-prefill`](20260824-prefill) |
| TP=3 cc=1 around 78–80 tok/s where healthy measures ~88–90 | Degraded fabric, ~12% shortfall, zero indicators. | [`…tp3-upstream-harness`](20260821T142000Z-3spark-tp3-upstream-harness) |
| Decode 30–40% high, clustered on a few repeated values | Short completion window. Check **per-request** `completion_tokens`. | [`20260826-decode-depth-2v3`](20260826-decode-depth-2v3) |
| `_ALLGATHER_BASE` timeout when you raise concurrency | Fabric symptom presenting as a config failure. Gate before you roll back. | [`20260824-seqs32-nccl`](20260824-seqs32-nccl) |
| EP=3 per-stream collapsing to ~1/3 of your TP baseline | The B12X MoE kernel does not take the EP path. Stop tuning. | [`…3spark-ep3`](20260821T031300Z-3spark-ep3) |
| PP=3 raising `SupportsPP` or a `strides[0] … divisible by 16` error | A hard block, not a fabric problem. No PP tok/s exists. | [`…3spark-pp3`](20260821T133000Z-3spark-pp3) |
| TP=3 serving fluent but *wrong* output | The `o_groups` padding patch. Stock TP=3 silently loses 2 of 8 groups. | [`…3spark-tp3`](20260821T133000Z-3spark-tp3) |

---

## What the statuses mean

| Status | Meaning |
|---|---|
| 🔴 `VOID-degraded-fabric` | Measured **before 2026-08-25**. One node ran at ~15% of its collective bandwidth with every indicator green until a 2026-08-25 reboot ([#14](../../issues/14)). Conclusions may survive when arms were matched; **magnitudes do not**. |
| 🟠 `VOID-25-token-window` | Decode measured over a **~25-token completion window** instead of the requested 256 ([#26](../../issues/26)). Too short to average MTP speculative-decode acceptance. Verified per directory by grepping `completion_tokens` in the raw data — never assumed. |
| 🟡 `SUPERSEDED-BY-<dir>` | A later run replaced it. The directory stays frozen as the decision trail. |
| 🟢 `CURRENT` | Healthy fabric **and** sound methodology. |

A directory can qualify for VOID on several grounds. All reasons are listed; the **most
severe** becomes the status. Severity order: `VOID-degraded-fabric` → `VOID-25-token-window`
→ `SUPERSEDED-BY` → `CURRENT`.

> **The critical column is "output tokens (actual)".** It is read out of the raw data, not
> from the README. A harness that *requests* 256 tokens and *receives* 25 produces a decode
> rate that is roughly 31% too high — measured, not estimated
> ([`20260826-harness-window-calibration/`](20260826-harness-window-calibration)).

> **⚠️ Aggregate vs per-request — the trap in that column.** The `bench_tp3.py` family
> writes `output_tokens` as an **aggregate** (`max_tokens × concurrency`), so
> `output_tokens: 4096` at cc=16 means 16 requests × 256, *not* one 4096-token request. The
> jsonl harnesses (`decode_depth_sweep.py`, `deepconc.py`, `soak.py`, `kvab.py`) write a
> genuine **per-request** count. **Only the per-request field can detect the #26 defect.**
> Reading an aggregate as per-request will hide it. Rows are labelled with which they carry.

## Tally

| Status | Count |
|---|---:|
| 🟢 `CURRENT` | 11 |
| 🔴 `VOID-degraded-fabric` | 7 |
| 🟡 `SUPERSEDED-BY-…` | 2 |
| 🟠 `VOID-25-token-window` | 1 |
| ⚪ `IN-PROGRESS` | 1 |
| **Total** | **22** |

`20260824-kv-quality` and `20260824-seqs32-nccl` are the two `SUPERSEDED` rows;
`20260824-prefill` is counted under `VOID-degraded-fabric` (its `superseded_by` field
points at `20260825-fabric-fix` as a pointer, not as its status).

### Fabric-gate tally

| `fabric_gate` | Count | Directories |
|---|---:|---|
| ✅ `PRESENT-PASS` | **5** | `20260825-deep-concurrency` (13/13, `nccl=full`) · `20260825-prefill-2v3` (12/12, `nccl=pairs`) · `20260825-upper-mesh` (**4 files**: 26/26 ×2 + 21/0 + 24/0) · `20260826-decode-depth-2v3` (33/0) · `20260826-nccl-controlled` (**is** the fabric measurement) |
| ❌ `PRESENT-FAIL` | **0** | — |
| ⚠️ `ABSENT` | **12** | `20260824-kv-quality` · `20260824-mtp5-1m` · `20260824-prefill` · `20260824-seqs32-nccl` · `20260825-decode-2v3` · `20260825-fabric-fix` · `20260826-four-hca-throughput` · `20260826-harness-window-calibration` · `20260826-kv-dtype-ab` · `20260826-seqs32-retest` · `20260827-decode-3node-fixed` |
| ⏳ `PRE-DATES-GATE` | **5** | all five `20260821*` directories — every one of which is also `VOID-degraded-fabric` |
| **Total** | **22** | |

*(`20260826-nccl-controlled` counts as `PRESENT-PASS`: it commits official `nccl-tests
all_gather_perf` output with the engine stopped, which exceeds the gate's own standard.
`PRE-DATES-GATE` is exactly the five 2026-08-21 directories; the four 2026-08-24 rows fall
under `ABSENT` because they predate the *discipline* without predating the *tool*.)*

**The uncomfortable finding: only 5 of 22 directories carry their own passing gate**, and
one of them is the directory that is void for an unrelated reason. Two `ABSENT` rows are
worth singling out:

- **[`20260825-decode-2v3`](20260825-decode-2v3)** — the repo's **headline concurrency
  result** — claims in its README that the fabric was "gated at 4.6+ GB/s per pair
  beforehand" but **commits no gate file**. The claim is plausible: two gates from the same
  day, in neighbouring directories, do read 4.59–4.67 GB/s. But a same-day gate elsewhere in
  the tree is corroboration, not provenance. Marked `ABSENT` deliberately.
- **[`20260826-four-hca-throughput`](20260826-four-hca-throughput)** — a finding *entirely
  about fabric bandwidth* — cross-references a gate in another directory rather than
  committing its own. The cited gate is real and passing (verified, 33/0); the provenance is
  still not local.

---

## 🟢 CURRENT

Healthy fabric, sound methodology. Numbers from these directories are quotable — subject
to the per-row warnings.

| Directory | Date | Description | Nodes / TP | KV pool | Harness | Output tokens (actual) | Reps / stat | Fabric gate | Raw data |
|---|---|---|---|---|---|---|---|---|---|
| [`20260825-decode-2v3`](20260825-decode-2v3) | 08-25 | Decode 2v3, cc=1/4/8/16, 18-token prompt — headline on the concurrency axis | 2 & 3 / TP=2,3 | 1,711,307 / 4,457,627 (README only) | `bench_tp3.py` (not local) | **256 ✓** *(aggregate)* on all 28 records/arm | 7/level, median | ⚠️ `ABSENT` — claimed in README, no file | ✅ `tp{2,3}_final.json/.txt` |
| [`20260825-deep-concurrency`](20260825-deep-concurrency) | 08-25 | 4×200K deep-concurrency re-run for [#15](../../issues/15) | 2 & 3 / TP=2,3 | 1,815,356 / 3,588,422 | `deepconc.py` ✅ | **128 ✓** *(per-request)* on all streams | 1 run/arm + warmups | ✅ `PRESENT-PASS` — `fabric-gate-full.json` 13/13, `nccl=full` | ✅ `tp{2,3}_200k.json` |
| [`20260825-fabric-fix`](20260825-fabric-fix) | 08-25 | The fix itself — matched prefill/decode either side of the reboot | 2 & 3 / TP=2,3 | — | `benchmark_prefill.py`, `bench_tp3.py` ✅ | 1 *(per-request, by design)* / 256 *(aggregate)* | 3/shape, median | ⚠️ `ABSENT` — but `pairwise-agbench/` is the characterisation | ✅ many |
| [`20260825-prefill-2v3`](20260825-prefill-2v3) | 08-25 | Prefill 2v3 — parity ±2% at 1K/8K/32K | 2 & 3 / TP=2,3 | — | `benchmark_prefill.py` (not local) | **1 ✓** *(per-request)* by design | 3/depth, median | ✅ `PRESENT-PASS` — `fabric-gate.json` 12/12, `nccl=pairs` | ⚠️ TP=2 arm only |
| [`20260825-upper-mesh`](20260825-upper-mesh) | 08-25 | Four-HCA mesh at 2.0x, 408-request soak, 0 RDMA deltas — [#17](../../issues/17) | 3 / TP=3 | 4,502,448 | `fabric_gate.sh` + `soak.py` ✅ | 22–87 *(per-request)*, asked 1000 — **no tok/s published** | 408 req / 20.5 min | ✅ `PRESENT-PASS` — **4 gate files**, 26/26 ×2 + 21/24 skip-mode | ✅ `soak-results.jsonl` |
| [`20260826-decode-depth-2v3`](20260826-decode-depth-2v3) → see 🟠 | 08-26 | *(listed under VOID-25-token-window)* | | | | | | ✅ `PRESENT-PASS` 33/0 | |
| [`20260826-four-hca-throughput`](20260826-four-hca-throughput) | 08-26 | 2x bandwidth buys **no** decode throughput — a published null result | 3 / TP=3 | — | `bench_tp3.py` (not local) | **256 ✓** *(aggregate)* on all 28 records | 7/level, median | ⚠️ `ABSENT` — cross-ref'd to another dir's gate | ✅ `4hca-warmed.json/.log` |
| [`20260826-harness-window-calibration`](20260826-harness-window-calibration) | 08-26 | The 25-token window overstated decode by ~31% — the control for [#26](../../issues/26) | 3 / TP=3 | — | `decode_depth_sweep.py` @ `c3e8e0d` | **256 ✓ on all 7** *(per-request; harness now raises otherwise)* | 7, median | ⚠️ `ABSENT` — same-engine A/B, largely cancels | ✅ `new-256tok-8k-tp3.jsonl` |
| [`20260826-kv-dtype-ab`](20260826-kv-dtype-ab) | 08-26 | `nvfp4_ds_mla` vs `fp8_ds_mla` — 23/24 cells byte-identical | 3 / TP=3 | 4,451,877 / 4,504,137 / 4,483,281 | `kvab.py` ✅ | **512 ✓ on all 100** *(per-request, field `ctok`)* — **corrected, see below** | quality n=3 & n=12; speed median of 5 | ⚠️ `ABSENT` — correctness gate ≠ fabric gate | ✅ quality `.jsonl`, speed `.json` |
| [`20260826-nccl-controlled`](20260826-nccl-controlled) | 08-26 | Official `all_gather_perf` — **no bandwidth gap ever existed**, 23.92 GB/s — [#18](../../issues/18) | 2 & 3 ranks | — | `nccl-tests` 2.30.7 ✅ | n/a — engine stopped | 12 runs, 5 variants | ✅ *is* the fabric measurement, and exceeds the gate | ✅ `raw/*.log` |
| [`20260826-near-ceiling-prefill`](20260826-near-ceiling-prefill) | 08-26 | 967,286 prompt tokens (92% of ceiling) served without incident | 3 / TP=3 | — | ⚠️ ad-hoc, **not committed** | 1 (prefill), no token field in raw | 1/depth, single | ⚠️ `ABSENT` — self-flagged | ✅ `raw.jsonl` |
| [`20260826-seqs32-retest`](20260826-seqs32-retest) | 08-26 | `MAX_NUM_SEQS=32` retest — +46.3% at cc=32, [#10](../../issues/10) rejection falsified | 3 / TP=3 | 4,512,769 / 4,431,088 | `bench_tp3.py` (not local) | **256 ✓** *(aggregate)* on all 35 records/arm | 7/level/arm, median | ⚠️ `ABSENT` — same-day matched control instead | ✅ both arms `.json/.log` |

### Warnings on CURRENT rows

- **`20260826-kv-dtype-ab` — the earlier "unverifiable window" caveat is WITHDRAWN.** The
  first audit pass reported no `completion_tokens` in any committed file and flagged the
  speed table as unverifiable. That was a **grep artifact, not a data gap**: `kvab.py` names
  its field **`ctok`**, not `completion_tokens`. Re-checked directly — `nvfp4-speed.json`
  and `fp8-speed.json` hold 100 per-request values between them and **every one is exactly
  512**. The window was requested and honoured on both arms. *Lesson for future audits:
  grep the harness's own field name, not just the two canonical ones.*
- **`20260826-decode-depth-2v3`'s gate file contradicts itself — trust the body, not the
  trailer.** `fabric-gate-pre-tp2.txt` ends `"GATE PASSED (with skips). Bandwidth unverified
  -- see above."` **That trailer is wrong here.** It is a stock line emitted whenever *any*
  check skips; the three skips were engine-RDMA checks correctly skipped because the engine
  was stopped. Bandwidth *was* measured, in the same file: 9.09 / 9.01 / 9.52 GB/s per pair
  and 5.04 GB/s 3-rank, all `via NET/IB/*`, zero `NET/Socket`. This is the strongest gate in
  the index — which is exactly why that directory is void for its **token window** and not
  for its fabric.
- **`20260825-decode-2v3` — the published cc=1 median is not in the file.** The README and
  six other places quote **89.1 tok/s** for TP=3 at cc=1. `tp3_final.txt` says
  **`median=81.7 tok/s spread=28%`**, from reps `88.6 90.2 89.3 81.7 67.5 67.9 73.0`. The
  other three TP=3 levels and all four TP=2 levels match exactly. See the untraceable list
  below.
- **`20260825-prefill-2v3` — half the comparison lives elsewhere, and its gate is
  `pairs`-mode.** Only the TP=2 arm is here; the TP=3 arm is
  `../20260825-fabric-fix/anemll_fresh.json` (verified present: 2022.6 / 2069.5 / 2094.9).
  Its gate is `nccl_mode: "pairs"` — the 3-rank `gate-all` collective was *not* run.
  Pairwise health does not by itself prove 3-rank collective health.
- **`20260825-fabric-fix` is `ABSENT` on a technicality.** It commits no
  `fabric_gate.sh` artifact, but `pairwise-agbench/` (nine all-gather captures) is the
  strongest fabric documentation in the repo — it is where the fault was characterised. It
  deliberately includes *failing* pre-reboot states, because documenting the failure is the
  directory's purpose.
- **`20260826-near-ceiling-prefill` — self-flagged preliminary, and correctly so.** n=1, no
  fabric gate, uncommitted harness, hex content instead of the token-ID pool (which is why
  every depth overshot 10–20×: 8,192 → 81,124 and 100,000 → 967,286). Kept CURRENT for the
  capability datapoint, not the tok/s.
- **`20260826-four-hca-throughput` is single-arm.** Its 2-HCA comparison column is quoted
  from `20260825-decode-2v3`, including the untraceable 89.1. The "flat" conclusion survives
  regardless — it rests on the 2-HCA median sitting inside the 4-HCA spread, and 81.7 does
  too.
- **`20260825-upper-mesh` completion tokens look like the #26 defect and are not.** Requests
  asked for 1000 tokens and returned 22–87. Two reasons it stands: the soak publishes **no
  tok/s** (its metrics are request success, latency percentiles and RDMA counter deltas,
  none of which depend on completion length), and the distribution is **broad and smooth**
  — modal value 28 with 44 hits, tapering to single hits above 60 — not the tight
  two-value cluster that marks the real defect. Its 5.80 GB/s figure is superseded as a
  *capability* claim by `20260826-nccl-controlled`; the 2.0× *ratio* is a same-harness
  comparison and stands.
- **`20260826-seqs32-retest` mitigates its missing gate by design.** Both arms ran the same
  day, ~40 min apart, with only `MAX_NUM_SEQS` varying, so a common fabric fault would
  largely cancel out of the +46.3% ratio. That is a legitimate substitute for gating a
  **comparison** — not for gating an **absolute**.

---

## 🟠 VOID-25-token-window — *kept as the canonical short-window baseline*

> **This is retained deliberately.** It is the reference fingerprint for a decode
> measurement that is silently 30–40% too high. If your numbers match this shape, this row
> tells you what happened.

| Directory | Date | Description | Nodes / TP | KV pool | Harness | Output tokens (actual) | Reps / stat | Fabric gate | Raw data |
|---|---|---|---|---|---|---|---|---|---|
| [`20260826-decode-depth-2v3`](20260826-decode-depth-2v3) | 08-26 | Long-context decode 2v3, 2K–262K at cc=1 — headline on the depth axis | 2 & 3 / TP=2,3 | 1,844,001 / 4,512,769 | `decode_depth_sweep.py` @ `aefa594` | 🔴 **25 (×43) and 26 (×27)** *(per-request)* against a requested 256 — all 70 reps | 7/depth/arm, median | ✅ `PRESENT-PASS` 33/0, pairs at 9.0–9.5 GB/s | ✅ `tp{2,3}-depth.jsonl` |

**Verified by grep, both arms, in both audit passes:** `tp2-depth.jsonl` → 25×22, 26×13.
`tp3-depth.jsonl` → 25×21, 26×14. 35 lines per arm. Not one rep in either arm reached 256.

### 📌 Baseline value — what these numbers tell you if you reproduce them

**If your decode numbers look implausibly high — roughly 30–40% above what a matched run
gives — and they cluster on a small set of suspiciously repeated values, check the
per-request `completion_tokens` before you believe them.**

The fingerprint is exact and easy to match: 70 of 70 reps returned either 25 or 26 tokens
against a requested 256, i.e. a **~0.4 s measurement window**. The calibration run
([`20260826-harness-window-calibration`](20260826-harness-window-calibration)) puts the
resulting overstatement at **~31%** at 8K on TP=3.

**Root cause worth recognising:** the prompt ended *"In one sentence, state what this
describes"*. The model emitted one sentence and stopped. **`max_tokens` is a ceiling, not a
contract** — and nothing in the old harness asserted the floor. The fix, in
`decode_depth_sweep.py` @ `c3e8e0d`: set `min_tokens == max_tokens`, set `ignore_eos`, and
**raise** if `completion_tokens != max_tokens`.

**Two cautions.**
1. **Do not apply a 31% correction on paper.** That calibration is one depth, one arm, one
   day. Re-measure.
2. **Do not discard the TTFT column.** TTFT is measured to the *first* token and is entirely
   independent of how many follow. The TTFT finding (2 nodes 6–15% sooner past ~100K)
   stands, as do the fabric gate, the KV pool readback and the zero-cache-hit assertions.

**What this voids:** every decode **magnitude** here, including the published **+33.6% at
131K** and **+17.9% at 262K** — the repo's headline case for the third node. Directions may
survive because both arms carried the identical defect, but no size does.

**Not yet superseded.** A matched re-run of all five depths on both arms does not exist.
[`20260827-decode-3node-fixed`](20260827-decode-3node-fixed) is aimed at this, but runs
`MAX_NUM_SEQS=32` against this run's `16` — see the caveat there.

---

## ⚪ IN-PROGRESS

| Directory | Date | Description | Nodes / TP | KV pool | Harness | Output tokens (actual) | Reps / stat | Fabric gate | Raw data |
|---|---|---|---|---|---|---|---|---|---|
| [`20260827-decode-3node-fixed`](20260827-decode-3node-fixed) | 08-27 | 3-node decode re-run with the fixed 256-token harness — **config capture only so far** | 3 / TP=3 | ⚠️ **not captured for this boot** | expected `decode_depth_sweep.py` @ `c3e8e0d`, not yet local | — no results file exists | not yet run | ⚠️ `ABSENT` — take one **before** the run | `engine-config.txt` only |

Appeared during this audit and is untracked in git. Three things to flag before it is used:

- **Its `engine-config.txt` cannot report this boot's KV pool.** The file says so honestly —
  *"not present in any reachable log for the current engine boot"* — and then quotes a
  **2026-08-24** value (5,382,503) from a different boot as the last known figure. That
  fallback must not be recorded as this run's pool.
- **It runs `MAX_NUM_SEQS=32`**, the post-retest production profile.
  `20260826-decode-depth-2v3` — the run it would replace — used **16**. Unless that is
  reconciled, or a matched TP=2 arm is taken at the same setting, this will not be a clean
  replacement for the voided depth sweep; it will be a third configuration.
- **Gate it before benchmarking, with the engine stopped.** `scripts/run_experiment.sh` does
  this automatically and refuses to proceed on failure — this directory is the natural place
  to prove the workflow. Note the engine was *already running* at capture time (PID 556480,
  started 2026-08-26 16:26), so a gate taken now would return `nccl_mode: "skip"` with
  bandwidth unmeasured.

When the run lands: assert `completion_tokens == 256` on every rep before publishing
anything from it. That assertion is what the fixed harness exists to provide.

---

## 🟡 SUPERSEDED — *kept as the decision trail*

> **Retained deliberately.** A superseded row records a conclusion that a later run
> overturned. Knowing *why* the earlier conclusion looked right is how you avoid reaching it
> again.

| Directory | Date | Description | Nodes / TP | KV pool | Harness | Output tokens (actual) | Reps / stat | Fabric gate | Raw data | Superseded by |
|---|---|---|---|---|---|---|---|---|---|---|
| [`20260824-kv-quality`](20260824-kv-quality) | 08-24 | NVFP4 KV quality to 464K — single-arm, no comparison ([#16](../../issues/16)) | 3 / TP=3 | 5,444,869 | `kvquality.py` ✅ | n/a — retrieval task, no tok/s | 1/depth, single | ⚠️ `ABSENT` | summary `.txt` only | [`20260826-kv-dtype-ab`](20260826-kv-dtype-ab) |
| [`20260824-seqs32-nccl`](20260824-seqs32-nccl) | 08-24 | `seqs=32` + NCCL sweeps; the rejection, plus the `_ALLGATHER_BASE` crash logs | 3 / TP=3 | 5,382,503 | sweep + `agbench.py` (partial) | **256 ✓** *(aggregate)* per level | 3/level, median | ⚠️ `ABSENT` — bandwidth data existed and was **misread** | ✅ `seqs32.json`, `crash/` | [`20260826-seqs32-retest`](20260826-seqs32-retest) |

Both also carry `VOID-degraded-fabric` (measured 2026-08-24). `SUPERSEDED` is shown because
a specific later run replaced their **conclusion**; the fabric reason is listed in
[`index.yaml`](index.yaml) under `void_reasons`.

### 📌 Baseline values

**[`20260824-kv-quality`](20260824-kv-quality) — long-context KV quality.**
If you are probing NVFP4 KV-cache quality with needle retrieval and you get **3/3 needles at
every depth out to ~464K actual prompt tokens**, with no CJK, no special-token leakage and
no pathological repetition, you have matched this record and **your KV dtype is not your
problem**. If needles start dropping at depth, or garble flags fire, suspect the KV dtype or
a context-length misconfiguration rather than the fabric — *a degraded link makes retrieval
slow, not wrong*, which is why this row's quality PASS survives its own VOID tag. The
methodological lesson is why it was superseded: it was **single-arm**, so it could show
nvfp4 passes but not that nvfp4 is *no worse than* fp8. A quality claim needs both arms.

**[`20260824-seqs32-nccl`](20260824-seqs32-nccl) — the repo's worked example of a FALSE
NEGATIVE: a good config rejected because the fabric was bad.**
If you raise `MAX_NUM_SEQS` to 32 and the engine dies with an **`_ALLGATHER_BASE` watchdog
timeout** (full crash logs from all three nodes are committed under `crash/`), **do not
conclude that seqs=32 is unstable on this hardware.** That is the conclusion recorded here
and it is **falsified**: the same change on healthy fabric is stable and gives +46.3%
aggregate at cc=32. The real signal sits in the bandwidth numbers alongside the crash — a
3-rank collective budget of **0.49 GB/s** against **3.25+ GB/s** healthy, a 6.6× shortfall.
Raising concurrency raises collective traffic; a degraded link that merely *slows* a light
workload will *time out* a heavy one. **An allgather timeout that appears only when you
increase concurrency is a fabric symptom presenting as a config failure.** Gate the fabric
before you roll back the config — and note that *having* bandwidth data is not the same as
having a threshold that stops the run.

---

## 🔴 VOID-degraded-fabric — *kept as diagnostic baselines*

> **These are not failures being hidden; they are the failure catalogue.** Every row here
> was measured before the 2026-08-25 reboot, when one node ran at ~15% of its collective
> bandwidth with every indicator green. Knowing what bad data looked like is how you spot
> the next batch. See [`../docs/DEGRADED-DATA-CATALOGUE.md`](../docs/DEGRADED-DATA-CATALOGUE.md)
> and [#14](../../issues/14).

All seven are `PRE-DATES-GATE` or `ABSENT` on the fabric column — which is the point: **the
gate exists because of these rows.**

| Directory | Date | Description | Nodes / TP | KV pool | Harness | Output tokens (actual) | Reps / stat | Fabric gate | Raw data |
|---|---|---|---|---|---|---|---|---|---|
| [`20260821T001024Z-2spark-baseline`](20260821T001024Z-2spark-baseline) | 08-21 | The frozen 2-node TP=2 reference bundle | 2 / TP=2 *(manifest.env verified)* | 1,771,152 (from CSV, not this dir) | `ours-bench.py` — **not committed** | 🔴 **10 on all 90 reps** *(per-request)* | 90 requests; published figure is single-observation | ⏳ `PRE-DATES-GATE` | ✅ `benchmark.jsonl` |
| [`20260821T031300Z-3spark-ep3`](20260821T031300Z-3spark-ep3) | 08-21 | EP=3 (TP=1/DP=3): per-rank configs, routes, sweep | 3 / EP=3 | 1,632,510 / 1,771,152 / 4,065,871 | unnamed — **not committed** | ⚠️ **none recorded** | unrecorded | ⏳ `PRE-DATES-GATE` | summary `results.json` only |
| [`20260821T133000Z-3spark-pp3`](20260821T133000Z-3spark-pp3) | 08-21 | PP=3 across several shapes — a hard block, not a perf number | 1/2/3 / TP=1,2 | — | n/a | n/a — no tok/s exists | n/a | ⏳ `PRE-DATES-GATE` *(moot — all shapes failed at config/warmup)* | summary `results.json` only |
| [`20260821T133000Z-3spark-tp3`](20260821T133000Z-3spark-tp3) | 08-21 | TP=3 rank config + mesh setup | 3 / TP=3 *(`TP_SIZE=3 NNODES=3`)* | — | n/a (setup script) | n/a | n/a | ⏳ `PRE-DATES-GATE` — `mesh-master.sh` is **setup, not verification** | ❌ none |
| [`20260821T142000Z-3spark-tp3-upstream-harness`](20260821T142000Z-3spark-tp3-upstream-harness) | 08-21 | Upstream's harness, unmodified; note `warmup-discarded.json` | 3 / TP=3 | 3,591,962 (from CSV) | `benchmark_tp3.py` upstream — not committed | **256 ✓** *(aggregate)* per level | 3/level × 2 sweeps, median | ⏳ `PRE-DATES-GATE` | ✅ `sweep{1,2}.json` |
| [`20260824-mtp5-1m`](20260824-mtp5-1m) | 08-24 | MTP=5 vs 4 at 1M context + acceptance reps — matched arms | 3 / TP=3 | 5,412,285 / 5,433,516 | sweep + `accept.py` (partial) | **256 ✓** *(aggregate)* per level | 3/level, median | ⚠️ `ABSENT` | ✅ four arm `.json` |
| [`20260824-prefill`](20260824-prefill) | 08-24 | The 45-file prefill investigation; the ~2× "gap" was one degraded node | 2 & 3 / TP=2,3 | — | `benchmark_prefill.py` + ad-hoc `pf3–pf8.py` (partial) | 1 *(per-request, ×111, by design)* / 256 *(aggregate)* | 2–3/shape, median | ⚠️ `ABSENT` — `tcpbw.sh` ≠ a gate | ✅ many |
| *(plus the two SUPERSEDED rows above, which are also degraded-fabric)* | | | | | | | | | |

### 📌 Baseline values — the failure catalogue

**[`20260824-prefill`](20260824-prefill) — the canonical degraded-fabric baseline.**
The most useful VOID directory in the repo, because it is 45 files of someone correctly
measuring the wrong thing. **If you measure ~1,034 tok/s prefill at 32K on 3 nodes and
cannot explain why it is roughly half what the hardware should do, stop investigating the
engine** — that exact number became **2,095 tok/s** after a reboot with zero config changes,
and the before/after pair is committed in [`20260825-fabric-fix`](20260825-fabric-fix). The
pattern to recognise: a ~2× prefill deficit, no errors, no NCCL warnings, all counters
green, and every plausible engine-side explanation (GFLOPs, memory bandwidth, chunked
prefill sizing, logits width, prefix caching, TP degree) testing clean one after another.
**That pattern *is* the fingerprint — when every component measures healthy but the system
does not, the fault is in the link.** Run `scripts/fabric_gate.sh` with the engine stopped
before spending a day on `gflops.py` and `membw.py` as this directory did. Note that
`tcpbw.sh` **passing did not clear the fabric**: TCP throughput and NCCL collective
bandwidth are different measurements, and only the latter caught it.

**[`20260821T142000Z-3spark-tp3-upstream-harness`](20260821T142000Z-3spark-tp3-upstream-harness) — the cleanest degraded fingerprint.**
Useful precisely because the methodology is *sound* — upstream's own unmodified harness,
pinned 256-token window, honoured. Everything is right except the fabric. **If you run the
upstream TP=3 harness on 3 nodes and see cc=1 around 78–80 tok/s where a healthy 3-node arm
measures ~88–90**, that ~12% shortfall with no error, no NCCL warning and every counter
green is the signature of [#14](../../issues/14). Do not tune the engine. Stop it and run
`scripts/fabric_gate.sh --nccl=full`: a healthy 3-Spark ring gates at **4.6+ GB/s per pair**
on 2 HCAs (**9.0–9.5** on 4), and the degraded state showed a 3-rank collective budget of
**0.49 GB/s**. If a pair is an order of magnitude low, **reboot the node** — the fault
survives everything short of that, and no counter reports it.

**[`20260821T001024Z-2spark-baseline`](20260821T001024Z-2spark-baseline) — two failures in
one row.**
*(1) Fabric:* if your TP=2 single-stream decode sits around **44–52 tok/s** where a healthy
2-node arm measures ~76 aggregate at cc=1, suspect a degraded link.
*(2) Short window:* **all 90 requests returned exactly 10 output tokens.** That is a more
extreme instance of the #26 defect than the 25-token case — a 10-token window cannot average
MTP speculative-decode acceptance at all, so any tok/s from it is noise *before* the fabric
is considered. The prompt is a needle-retrieval shape whose `output_preview` is a single
needle code, which explains it. **If your per-request `completion_tokens` come back as a
small constant regardless of the `max_tokens` you asked for, you are measuring startup
overhead, not steady-state decode.** Assert `completion_tokens == max_tokens`, set
`min_tokens == max_tokens` and `ignore_eos`, and re-run.
This is the frozen reference the whole 2-vs-3 comparison was originally anchored to.

**[`20260824-mtp5-1m`](20260824-mtp5-1m) — why matched arms are worth the extra hour.**
Both MTP=5 and MTP=4 arms ran on the same degraded fabric, so the comparison between them is
still valid and **MTP=5 still wins** — but no absolute tok/s here is quotable. If you are
tuning MTP, the acceptance data in `acceptance-reps.txt` is the more durable artifact,
because acceptance rate is a property of the model and the prompt, not of the link: it shows
accept-len near the k ceiling for structured content (`json` ~4.93/5, `count` 5.00/5, `code`
~4.28/5) and collapsing for prose (~1.05–1.35/5). **If your own tok/s swings 1.65× between
prompts and you have not changed a flag, you are seeing MTP acceptance vary by content, not
a regression** — always state the workload alongside a tok/s number. If instead your
accept-len matches these values but your tok/s does not, the gap is transport: gate it.

**[`20260821T031300Z-3spark-ep3`](20260821T031300Z-3spark-ep3) — expert parallelism.**
If you configure EP=3 (TP=1/DP=3) on a B12X MoE backend and **your per-stream decode
collapses to roughly a third of your TP baseline** — committed here as `per_stream_tok_s`
**17.23 / 12.34 / 11.95 / 8.65** at cc=1/4/8/16 — you are not looking at a misconfiguration
you can tune out, and not at a node-count penalty. **The B12X MoE kernel does not take the
EP path.** The 256 experts *do* shard 86/85/85 and the KV pool *does* grow ~2.3×, so the
config "works" and will mislead you. Stop tuning and change parallelism strategy. The
magnitudes were measured on degraded fabric and must not be quoted; the ~2.5×-slower
*direction* is kernel behaviour and is not fabric-sensitive.

**[`20260821T133000Z-3spark-pp3`](20260821T133000Z-3spark-pp3) — a hard block that
reproduces exactly, regardless of your fabric.**
Try PP=3 on this model and image and you will hit one of two errors, both committed verbatim
in `results.json`:
- **With MTP on** — `NotImplementedError: Pipeline parallelism is not supported for this
  model`, raised because `DeepSeekMTP` (`deepseek_mtp.py:223`) does not implement
  `SupportsPP`, via `create_speculative_config → draft model verify_with_parallel_config`
  (`config/model.py:1223`).
- **With MTP off** — `ValueError: Invalid state_cache.strides[0] … expected to be divisible
  by 16` at warmup, a DSA stride constraint.

Neither is a fabric problem, neither is fixed by retrying, and **there is no PP tok/s number
anywhere in this repo because none was ever produced.** Note this *scopes* the EP finding:
B12X **does** survive pipeline parallelism, so "the kernel refuses everything but TP" is the
wrong lesson.

**[`20260821T133000Z-3spark-tp3`](20260821T133000Z-3spark-tp3) — a configuration baseline,
not a performance one.**
No benchmark output exists in this directory at all. Its value is the TP=3 rank env, which
documents the **`o_groups` padding patch** that TP=3 requires on this image. Stock vLLM
rejects 64 heads / TP=3; if you bypass only that validation then `n_local_groups = 8 // 3 ==
2`, which represents 6 global groups and **silently loses 2** — the server starts, serves,
and returns *fluent nonsense with no error*. **If you are running TP=3 and your outputs are
grammatical but wrong, check this before anything else**, and verify the patch pads the
*group count* while holding `heads_per_group` at 8 so the `o_proj` BMM `r` dim stays 4096 to
match `wo_a=[8192,4096]`. Also note the two published TP=3 figures from this era
(**53.95–57.73 tok/s**) trace to **no file** under `results/` — do not treat them as a
target to reproduce.

---

## Cross-reference: harness location

Several directories reference a harness they do not carry. The canonical copies live in
[`20260825-fabric-fix/harness/`](20260825-fabric-fix/harness) (`bench_tp3.py`,
`benchmark_prefill.py`, `bench_full.py`, `compare_prefill.py`) and in
[`../scripts/`](../scripts) (`decode_depth_sweep.py`, `fabric_gate.sh`).

Per [`README.md`](README.md)'s own rule — *"a result whose harness does not exist as a file
cannot be checked"* — the one genuine gap is
[`20260826-near-ceiling-prefill`](20260826-near-ceiling-prefill), whose ad-hoc
`bench_compare.py` was never committed anywhere, and the 2026-08-21 `ours-bench.py` behind
the 2-Spark baseline.

---

## Numbers in the top-level README that do not trace to committed raw data

Checked by grepping every figure in [`../README.md`](../README.md) against the raw files in
this tree. **These are unsupported claims in our own repo.** They are listed here rather
than quietly corrected, and no file outside this index was edited.

| Number | Where it is claimed | What the raw data says |
|---|---|---|
| **89.1 tok/s** — TP=3 decode cc=1 | `README.md` L220, plus `docs/HANDOFF.md` L69, `docs/POSTMORTEM-2026-08-25.md` L40, `docs/TTFT-AND-WARMUP.md` L93, `benchmarks/measurements.csv` L70, `20260826-four-hca-throughput/README.md` L18 | `20260825-decode-2v3/tp3_final.txt` reports **`median= 81.7 tok/s spread=28%`** at cc=1, from reps `88.6 90.2 89.3 81.7 67.5 67.9 73.0`. **89.1 appears nowhere** in either committed arm file. The other three TP=3 levels and all four TP=2 levels match exactly, so this is one cell, not a wholesale mismatch. It propagates into the derived **"+17%"** / **"+16.9%"** headline. |
| **48.23 tok/s** — the frozen 2-Spark baseline | `README.md` L87 ("2-Spark baseline (`TP=2`) 48.2"), `benchmarks/measurements.csv` | `20260821T001024Z-2spark-baseline/benchmark.jsonl` has nine cc=1 reps: `44.31 50.69 50.86 51.24 51.48 53.61 56.53 65.33 66.32` (median **51.48**). **48.23 is not among them** and is not their median or mean. It is labelled `single-observation` in the CSV, so it may be an unretained run — but no committed file contains it. |
| **1,711,307 / 4,457,627** — KV pool, 2-node vs 3-node | `README.md` L226, `docs/HANDOFF.md` L75, `docs/POSTMORTEM-2026-08-25.md` L46 | Present only in `20260825-decode-2v3/README.md` prose. **No engine readback is committed** in that directory. Contrast `20260826-decode-depth-2v3`, which commits `engine-config.txt` showing `1,844,001` and `4,512,769` — different numbers for the same two configurations, unexplained. |
| **"131K" — max verified context, 2-node column** | `README.md` L47 (Max verified context: 2 Spark = 131K) | The 2-Spark baseline bundle only ever ran `context_target` of `2048`, `8192`, `32768`. The 131K figure for **two** nodes traces to no committed 2-node file. (The 3-node `967,286` **does** trace, to `20260826-near-ceiling-prefill/raw.jsonl`.) |
| **19–20 tok/s** — EP=3 decode cc=1 | `README.md` L88 | `20260821T031300Z-3spark-ep3/results.json` records `per_stream_tok_s` of `17.23 12.34 11.95 8.65` and `aggregate_tok_s` of `17.23 49.37 95.57 138.42`. **Nothing reads 19–20.** The cc=16 aggregate of `138` in the same row does trace (138.42). |
| **53.9–57.7 tok/s** — TP=3 + padding patch, 08-21 | `README.md` L90 | Quoted as `53.95–57.73` in `docs/BENCHMARK-METHODOLOGY.md`, `docs/DEGRADED-DATA-CATALOGUE.md` and `docs/EXPERIMENT-LOG.md`, but **no file under `results/` contains either value**. The run directory for this configuration (`20260821T133000Z-3spark-tp3`) commits **no benchmark output at all** — only a rank env file and a mesh setup script. |
| **~80 tok/s** — MTP=5 + 1M, 08-21/24 row | `README.md` L92 | Approximate by construction ("~80"), and `20260824-mtp5-1m` commits four arm files, but no single committed median is identified as the source. Traceable in spirit, not to a specific cell. |

### Numbers that **do** trace, checked in the same pass

Recorded so the list above is not mistaken for a general indictment: `23.92 GB/s`
(`20260826-nccl-controlled/summary.json`), `685.9` and the full seqs32/seqs16 median rows
(`20260826-seqs32-retest/*.log`), `967,286` (`near-ceiling-prefill/raw.jsonl`), all five
depth-sweep medians and TTFTs (`20260826-decode-depth-2v3/*-summary.json`), the prefill
parity triples `1913/2081/2066` and `2023/2070/2095`
(`20260825-prefill-2v3/tp2_prefill.json` and `20260825-fabric-fix/anemll_fresh.txt`),
`293,987 / 396,804 ms` (`20260825-deep-concurrency/tp{2,3}_200k.json`), `491.0` and `374.2`
(`20260825-fabric-fix/decode_parity3.txt`, `20260824-mtp5-1m/ours_tp3_1m_mtp4_matched.json`),
`1,034.3 → 2,094.9` (`20260824-prefill/anemll_run.txt`, `20260825-fabric-fix/anemll_fresh.txt`),
`408/408` and the RDMA deltas (`20260825-upper-mesh/soak-*`), and the KV dtype pool sizes
(`20260826-kv-dtype-ab/*-engine-facts.txt`).

Added in the second pass: all four committed gate JSONs verify against their stated
pass/fail counts (12/0, 13/0, 26/0 ×2, plus two skip-mode gates at 21/0 and 24/0), the
33/0 text gate verifies including its per-pair bandwidth, and every rep count in the table
above was re-derived by counting records rather than read from a README.

---

## Result READMEs lacking a provenance header block

Reported for a follow-up to add; **no README was edited by this pass.** A provenance header
is the block `20260826-harness-window-calibration/README.md` demonstrates: status, node
count, depth, reps, harness **with its commit**, output tokens, **fabric gate**, and the
issue it answers.

**Missing a README entirely (10):**

`20260821T001024Z-2spark-baseline`, `20260821T031300Z-3spark-ep3`,
`20260821T133000Z-3spark-pp3`, `20260821T133000Z-3spark-tp3`,
`20260821T142000Z-3spark-tp3-upstream-harness`, `20260824-kv-quality`, `20260824-mtp5-1m`,
`20260824-prefill`, `20260824-seqs32-nccl`, `20260825-fabric-fix`

Plus `20260827-decode-3node-fixed`, which is still in progress — it should get a README
before it gets results, not after.

`20260825-fabric-fix` is the priority of these ten: it is the only healthy-fabric directory
with no README, it is the before/after pair the whole degraded-fabric story rests on, and it
holds the canonical harness copies four other directories depend on.

**Has a README, but no structured provenance header (10):**

| Directory | What its README opens with instead |
|---|---|
| `20260825-decode-2v3` | Prose framing, then the result table. Config appears under "Why this is a fair comparison"; no harness commit, no output-token line, and its fabric-gate claim has no artifact. |
| `20260825-deep-concurrency` | Prose. Config in a "Method" section at the bottom; no harness commit. |
| `20260825-prefill-2v3` | Prose, then the result table. Harness md5 is given (good) but no commit, no reps line up front. |
| `20260825-upper-mesh` | Has a partial header (`**Date:** … **Fabric:** … **Engine:**`) — closest to compliant; missing harness, reps, output tokens. |
| `20260826-decode-depth-2v3` | Prose. Config under "Why this is a fair comparison"; **no output-token line — which is exactly how the #26 defect went unnoticed.** |
| `20260826-four-hca-throughput` | Prose, then the measurement. Harness named inline, no commit; gate cross-referenced, not local. |
| `20260826-kv-dtype-ab` | Has an issue/date/config line; no reps, no harness commit, no output tokens (though `ctok` **is** in the raw data). |
| `20260826-seqs32-retest` | Prose. Config under "Why this is a fair comparison"; no harness commit. |
| `20260826-near-ceiling-prefill` | Two stacked caveat banners; full config is committed further down, harness explicitly uncommitted. |
| `20260826-nccl-controlled` | Has a good header line (date, engine state, harness **with version and flags**) — second-closest to compliant. |

**Compliant (1):** `20260826-harness-window-calibration` — use it as the template, and add a
`fabric_gate:` line to it.
