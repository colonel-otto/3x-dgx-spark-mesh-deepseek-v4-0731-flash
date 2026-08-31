# Handoff — 2026-08-31: the matched engine A/B is done

## 1. The question, answered

**Is the eugr image stronger than our anemll image?** Yes, decisively, for
serving. Matched, one variable, both engines measured back-to-back today:

| c | metric | anemll-v0.25.1 | eugr-spark-vllm-b12x | delta |
|---:|---|---:|---:|---:|
| 1 | decode | 61.5 | **84.7** | +37.7 % |
| 4 | decode | 33.0 | **54.4** | +64.8 % |
| 8 | decode | 29.0 | **44.9** | +54.8 % |
| 16 | decode | **18.2** | 15.0 | −17.6 % |
| 1 | aggregate | 53.8 | **70.7** | +31.4 % |
| 4 | aggregate | 108.0 | **164.5** | +52.3 % |
| 8 | aggregate | 154.8 | **249.9** | +61.4 % |
| 16 | aggregate | 141.3 | **187.4** | +32.6 % |
| — | KV cache tokens | **4,391,722** | 2,357,009 | −46 % |

Bundle: `results/20260831T1000Z-matched-engine-ab/` (raw per-trial logs for all
8 cells). Rows: `2026-08-31T10:00:00Z` in `benchmarks/measurements.csv`.

**anemll keeps exactly two things**, and both are worth knowing:

1. **1.86× the KV pool** (4.39M vs 2.36M tokens; 4.19× vs ~2.25× max concurrency
   at 1M context). This is **permanent** — `nvfp4_ds_mla` is refused by a
   `VllmConfig` validator on every MLA backend in the eugr build. If a workload
   needs more than ~2.25× concurrency at 1M context, eugr cannot serve it at all.
2. **Per-stream decode at the c=16 cap** (18.2 vs 15.0). eugr trades per-stream
   latency for aggregate there and still moves +33 % more total tokens in the
   same cell. Which matters depends on whether you are optimising one user's
   latency or the box's total output.

anemll was also markedly **less stable under load**: 84 % trial spread at c=16
(6.9 → 22.2 tok/s) against eugr's 14.7 %, and TTFT swinging 1.2 s → 8.1 s at c=8.

## 2. Why the previous answer was wrong (and wrong in eugr's favour)

`docs/ENGINE-AB-3NODE.md` was marked COMPLETE, but its anemll column was quoted
from **2026-08-21** rows while every eugr row was 08-30/31. Two defects:

- **Unmatched** — 10 days, a different boot, a different fabric state. The
  standing rule against this exists because the same pattern once *reversed* a
  headline conclusion.
- **Under-powered** — the repo's own 8-rep noise study on an *unchanged* anemll
  engine recorded 66.6–88.5 tok/s at c=1 (27 % spread) against a **12 % parity
  tolerance**. The old table's single-stream cells (+5 %, +8 %, +11 %) were all
  *inside* that band. They were never resolved by the data.

Both defects pushed the same way and **understated** eugr: c=1 decode moved from
"+5 %, unresolved" to **+38 %, decisive**. It could as easily have gone the other
way — that is the point of matching.

## 3. What changed in the repo

- `results/20260831T1000Z-matched-engine-ab/` — new bundle, raw logs, gate
  recorded **ABSENT** (not inherited: an engine-swap A/B never frees the GPUs
  for the NCCL check, and a gate from another boot is not evidence for this one).
- `benchmarks/measurements.csv` — 8 matched rows, both arms, `2026-08-31T10:00Z`.
- `scripts/generate_summary.py` — **bug fixed**: its hardcoded allowlist silently
  dropped all 8 new rows, the same failure that once hid every eugr row including
  the served engine. Both arms are now emitted for all four concurrencies.
- `docs/ENGINE-AB-3NODE.md` — status rewritten to SETTLED with the matched table
  and an explicit note that the old numbers understated eugr.
- `docs/troubleshooting.md` — two entries: `MAX_NUM_SEQS` drift silently making
  an A/B two-variable; a delta inside the noise floor is not a result.

## 4. Method notes worth carrying forward

- The live anemll `tp3.env` had **drifted to `MAX_NUM_SEQS=32`** while the
  reference rows say `seqs16`. Set to 16 on **all three ranks** for the run
  (a rank mismatch hangs startup forever with no error) and restored to 32
  afterwards. Without that step the A/B would have been two-variable.
- **Correctness gate ran first on the anemll arm** (7/7, virtual-TP active)
  before any throughput cell — a throughput number from an unvalidated TP=3
  engine may be the speed of generating garbage.
- The **eugr arm was re-measured, not quoted**, and reproduced the 08-30 rows
  within ±8 % on every cell (84.7 vs 84.3, 54.4 vs 53.2, 44.9 vs 44.1, 15.0 vs
  15.5). So the swing in this comparison is the anemll reference being stale,
  not eugr drifting.
- **Permanent confound**: anemll MTP K=2 vs eugr DSpark nst=5, and the checkpoint
  *refuses* nst<5. Every cell measures engine+speculator. A pure engine A/B is
  not constructible on this checkpoint — say so on every row.

## 5. Service state at handoff — all verified

- `eugr.service` **active**, `/health` 200, correctness **7/7**.
- Gateway **4/4 hops** ("GATEWAY ROUTE VERIFIED"): both served names on :8100,
  LiteLLM :4000 lists and completes, manifest :8771 auto-discovers.
- anemll `tp3.env` **restored to `MAX_NUM_SEQS=32`** on all three ranks.
- **Zero leaked containers** — exactly one `vllm_node` per node, all eugr.
- Repo suite **25 passed / 52 subtests**.

## 6. Still open

1. **Merge the open PRs** — #44, #45, #46 (→ main) and #47 (→
   `feat/upstream-vllm-image`), all MERGEABLE/CLEAN with CI green. Merging is
   blocked for the agent by the permission classifier, so it needs a human.
2. **Enable `dsv4-fabric-reconcile.service`** on all three nodes — files and
   `/etc/dsv4-fabric-map` are installed and hand-run successfully, but
   `systemctl enable` was classifier-blocked, so the unit is still `disabled`:
   `sudo systemctl daemon-reload && sudo systemctl enable --now dsv4-fabric-reconcile.service`
3. **The NCCL bandwidth fabric gate** is still the one check never run today.
   It needs the GPUs free; the next engine-down window is the moment for it.
4. **The remaining unmatched cells** (131K decode, the prompt-effect pair,
   4×200K deep concurrency) still carry 2026-08-21/25 anemll references. The
   four concurrency cells that mattered are now matched; these three would need
   another engine swap to match.

   **4×200K is tracked as [issue #49](../../issues/49)** and is the one worth
   keeping open, because it is the cell where anemll's 1.86× KV pool *should*
   matter most and the current data cannot adjudicate it. Note what it does
   NOT say: eugr did not fail there. Both engines complete with 0 errors and
   both are unusable — eugr at 1.40 agg tok/s / 227 s TTFT against anemll's
   0.8–1.1 / 275–397 s, i.e. eugr is ~40 % faster and still unusable. It is a
   workload-shape limit (~800K tokens of prefill), not an engine defect, and
   there is nothing to tune at that prompt:decode ratio.
