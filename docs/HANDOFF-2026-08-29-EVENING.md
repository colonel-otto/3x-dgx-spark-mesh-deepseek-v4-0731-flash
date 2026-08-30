# Handoff — 2026-08-29 evening

Written for a fresh conversation with no prior context. Read §1 before touching the
cluster; it contains a state trap that will otherwise be misdiagnosed.

---

## 1. Cluster state — READ FIRST

**The cluster is UP and serving. `systemctl` will tell you it is not. Believe the engine,
not systemd.**

```
curl http://127.0.0.1:8100/health        -> 200          ← the truth
systemctl is-active dsv4.service         -> failed       ← misleading
docker ps (all 3 nodes)                  -> Up, running since 2026-08-30T01:30:22Z
17 x 23                                  -> 391          ← verified correct
```

Live shape: `TP=3`, `max-model-len 1048576`, `max-num-seqs 32`,
`gpu-memory-utilization 0.835`, `MTP=2`. This is the production profile.

**Why the mismatch:** `dsv4.service` is `Type=oneshot` with `RemainAfterExit=yes`. Its
start script gave up waiting for the health endpoint before the engine finished booting,
so systemd recorded `exit-code / failed` — but the containers it had already launched kept
starting and came up fine minutes later. The unit is orphaned from the containers it owns.

**Consequences:**
- `systemctl stop dsv4.service` may not stop these containers. To tear down cleanly, stop
  the containers directly on each node, or `systemctl start` then `stop` to re-sync.
- Do **not** "fix" this by restarting the engine unless you need to. It is serving.
- Timeouts were raised (§4) so this should not recur, but the current process predates
  that fix.

**Env files on disk are the 2-node values** (`MAX_NUM_SEQS=16`, `MTP_NUM_TOKENS=5`,
`GPU_MEMORY_UTILIZATION=0.80` in `config/head.env` / `worker.env`). That is correct and
expected — those files configure the TP=2 arm, not the running 3-node service. The live
3-node engine gets its config from `config/tp3.env` via the systemd unit.

---

## 2. The goal, and where it actually stands

**Goal:** tell users whether to run DeepSeek-V4 on 3 nodes or 2, with correct information.

**Status: NOT ANSWERED. And the previously published answer is not defensible.**

The headline finding of this session is negative and it matters:

> **Four of the five published 2v3 decode deltas do not survive a significance test.**

Exact two-sided Mann-Whitney U on the repository's own committed per-rep data
([`20260827-decode-2v3-fixed`](../results/20260827-decode-2v3-fixed/), n=7 vs n=7, α=0.05):

| Depth | Published delta | p | |
|---:|---:|---:|---|
| 2K | **+16.7 %** | 0.0973 | not significant |
| 8K | +14.2 % | **0.0006** | significant |
| 32K | +11.0 % | 0.0728 | not significant |
| 131K | **+7.3 %** | 0.5350 | not significant |
| 262K | +10.0 % | 0.2086 | not significant |

Every delta is smaller than the spread of at least one arm it was computed from; the 2K
TP=3 cell that anchors the headline spans **38.22–64.54 tok/s** across 7 reps.

**An independent bundle agrees.** `20260825-decode-2v3` (status `CURRENT`, real 256-token
window, same harness/day, ~20 min apart) gives **0 of 4 cells significant**: cc=1 shows
+7.2 % at **p=0.68**. Its TP=2 arm is also visibly bimodal (428.4/429.8 against a 279–305
cluster at cc=8; 686.9/687.8 against 462–485 at cc=16) — that arm has a contamination
signature of its own.

**The third bundle cannot help.** `20260826-decode-depth-2v3` — the 70-rep run often
remembered as "the headline result" — is `VOID-25-token-window`. It is the 25-token
defect the team already caught, which overstated decode 30–39 %.

### What IS still true

- Three nodes measured **faster in essentially every cell**. The ordering is not in
  dispute; the *statistical separation* is.
- **KV cache really is ~2.6× larger** on 3 nodes (4,457,627 vs 1,711,307 tokens). That is
  arithmetic, not a noisy benchmark. It has never been the binding constraint in any
  measured workload.
- TP=3 is correct (14/14 acceptance, quality suite passing), reproduces within **~±8 %**
  across 48 hours, and serves 2K–262K without OOM.

### The honest public line, until a powered comparison lands

> Three-node TP=3 works, is numerically correct, and is the shape we run. We are **not
> currently able to claim a decode speedup over two nodes** — our own published deltas do
> not survive a significance test at n=7, and the arms were not configuration-identical.

`README.md`, `docs/BENCHMARK-2V3-NODES.md`, and the FAQ have been corrected to say this.

---

## 3. Why the published comparison could not settle it

**SIX confounds, not one.** A full key-by-key diff of `config/tp3.env` against
`config/head.env` (run 2026-08-29 late, after a TP=2 bringup crash forced a closer look)
shows the published arms differ in **six** engine settings:

| Knob | TP=2 arm | TP=3 arm | Disclosed before this session? |
|---|---:|---:|---|
| `MAX_NUM_SEQS` | 16 | 32 | yes |
| `MTP_NUM_TOKENS` | 5 | 2 | yes |
| `GPU_MEMORY_UTILIZATION` | 0.80 | 0.835 | **no** |
| `LONG_PREFILL_TOKEN_THRESHOLD` | *unset* | 1024 | **no** |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | *unset* | 2 | **no** |
| `KV_CACHE_DTYPE` | *unset* (default) | `nvfp4_ds_mla` | **no** |

The last three are **Profile B** settings — the winning profile from Issue #25, credited
with −10.7 % starvation TTFT and a +35 % KV pool. The 2-node arm was never running
Profile B at all. **Only two of six confounds were ever disclosed.**

This makes the published comparison weaker still: it is not "3 nodes vs 2 nodes", it is
"3 nodes on the tuned production profile vs 2 nodes on an untuned one". All six are now
matched in the TP=2 env files for the next run.

The third was found this session by reading the live env files against the running engine.
Per Issue #25 that knob alone is worth ~35 % of the KV pool and −10.7 % starvation TTFT. So
even the one significant row (8K) measures *three-nodes-at-one-config vs
two-nodes-at-another*, not node count.

**No run in this repository had controlled GPU clocks.** GB10 does not honour
`nvidia-smi -lgc`: it reports `Supported Clocks: N/A` and no settable power limit, so the
command returns success and pins nothing. Measured over 690 samples/node: clock mean
**2426–2469 MHz, never reaching 3003**, governed by SW power capping (~3 h cumulative) not
thermal slowdown (~1.2 s). Full analysis in
[`GPU-CLOCKS-NOT-LOCKABLE.md`](GPU-CLOCKS-NOT-LOCKABLE.md). The
`20260828-issue36-locked-clocks-suite` bundle's "locked 3003 MHz" claim is **withdrawn** —
its own committed `gpu_clocks.csv` holds a single row reading **2522 MHz at idle**.

---

## 4. Bugs found and fixed this session

All committed. Each was silently defeating a guard the repo believes is enforced.

| # | Bug | Impact | Commit |
|---|---|---|---|
| 1 | `exclusivity.py` **overwrote instead of summed** `request_success_total` across its five finish-reason series, keeping only `repetition` | Requirement 5 read a delta of 0 and passed/failed for the wrong reason. Every asserted-window request finishes as `length` | `8852d5e` |
| 2 | `benchmark_mtp_concurrency.py` scraped `spec_decode_draft_iterations_total`, which **does not exist** in this build (it is `spec_decode_num_drafts_total`) | Mean accepted length reported as a silent **0.00** in every bundle. Now reads 1.32–1.35 | `8852d5e` |
| 3 | Same harness **never warmed the shape it measured** — one single-stream warmup, then concurrent batches | **2.5× error**: cc=4 read 15.10 tok/s cold vs 41.63 warm | `8852d5e` |
| 4 | Orchestrator's exclusivity ledger counted only the depth sweep | False `EXCLUSIVITY_FAIL` claiming 198 foreign requests | `e2b99f7` |
| 5 | `cluster_tp2.sh` + `dsv4-service-start` waited **15 min** for a cold start that takes **~30 min** | Aborted two healthy bringups; reported a working cluster as failed | `1e1f5ef` |
| 6 | **TP=2 could not start at all.** `docker-compose.yml:117` passes `VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-}"` — an **empty default**. `tp3.env` sets it to 4096; the TP=2 env files never set it, so workers received `""` and vLLM's `enable_envs_cache()` died on `int('')` → `ValueError: invalid literal for int() with base 10: ''` | Every TP=2 worker crashed ~4 min into startup, after loading 79.17 GiB of weights. The orchestrator then sat out its full bringup budget waiting on a dead engine. **This, not the timeout, is the likely cause of the earlier "TP=2 bringup failed" too.** | env fix |

On #6: the crash is **latent in the 2-node config**, unrelated to node count, memory, or
any edit made this session — TP=2 as committed could not have started. Fixed by setting
the variable in both `head.env` and `worker.env`. Note the failure mode: the engine loads
the full model first and *then* dies on env parsing, so it looks like a slow startup for
several minutes before failing.

On #5: raised to 45 min, with `dsv4.service` `TimeoutStartSec` 2400 → 3300 s so the script
fails first and emits container logs. The exited-container check is untouched, so a real
crash still fails fast. `~/bin/dsv4-service-start` on sparkmain was backed up before
editing (`.bak.<stamp>`).

---

## 5. Data collected — what exists and is trustworthy

**Bundle:** `~/bench-repo/results/20260829T234009Z-matched-2v3/` (on sparkmain; **not yet
copied into the git repo**).

### `tp3/` — complete, verified clean, n=7

| Depth | Median tok/s | Spread | Trimmed spread |
|---:|---:|---:|---:|
| 2K | 51.06 | 9.5 % | 5.7 % |
| 8K | 51.50 | 6.6 % | — |
| 32K | 52.20 | 5.9 % | — |
| 131K | 46.87 | 30.6 % | 7.8 % |
| 262K | 44.92 | 22.3 % | 8.5 % |

Concurrency @8K, `MTP=2`, n=5: cc=4 **41.07**, cc=8 **49.08**, cc=16 **54.12** tok/s;
acceptance **66.7–67.7 %**, mean accepted length **1.33–1.35**.

> **This contradicts a published claim.** `BENCHMARK-2V3-NODES.md` §D cites 76.7–80.4 %
> acceptance at ~1.55 tokens/step. That is the **single-stream** figure. At concurrency it
> is ~67 % / ~1.34, matching Issue #32's 66.3 %. Barrier-reduction arithmetic built on 1.55
> overstates the effect by ~15 %. Corrected in that doc.

**Contamination check — the arm is clean.** It logged `EXCLUSIVITY_FAIL ... 198 foreign
requests`; that was bug #4, confirmed four ways:
1. Ledger closes exactly from raw `rep_details`: 196 concurrency + 2 warmups = 198, plus
   45 from the depth sweep = **243**, the observed delta, **zero unexplained**.
2. `open-webui` (the only local client) stopped 19:40:16, restored 20:39:59 — all
   measurement inside that window.
3. 35/35 reps returned exactly 256 completion tokens; 35/35 `cached_tokens = 0`.
4. No anomalous reps (cc=16: 53.1/54.3/54.2/54.1/53.6).

**Deep cells are outlier-prone, not unstable.** Dropping one high and one low rep brings
every cell to 5.7–8.5 %, inside the Issue #31 floor (6.6–11.7 %). The 131K 30.6 % is one
rep at **7.8 robust SDs** (59.37 against a 45.05–49.40 band). Median moves only 0.5 % when
it is removed, so medians are trustworthy at every depth. This is speculative-acceptance
variance, not thermal drift — drift showed as a *monotonic trend* in an earlier aborted
run and is absent here.

### `tp2/` — NOT MEASURED

Contains only `fabric-gate.json` (passed 15/15) and `cluster-up.log`. **This is the missing
half of the comparison.**

---

## 6. Two aborted runs — what killed them

1. **Run 1** (~8 min in, deliberately stopped). Per-cell spread grew with depth and elapsed
   time — 5.0 % → 12.2 % → **17.3 %** — and the 32K cell declined monotonically across reps
   (54.9 → 46.9) as workers heat-soaked to 83–86 °C against 75 °C on the head. Under TP=3
   every rank waits at the barrier for the slowest. A 17.3 % spread cannot resolve a 7–17 %
   effect, so it was stopped rather than carried to a conclusion it could not support.
   **Fix applied:** cool all nodes to ≤70 °C before each arm, sample clocks every 5 s
   throughout. Result: 8K spread 12.2 % → 6.6 %, 32K 17.3 % → 5.9 %, medians moving <2 %.

2. **Run 2** (TP=3 arm complete, died in transition). **Operator error:** the running script
   was overwritten by `scp` while bash was still executing it. Bash reads scripts
   incrementally by byte offset, so the file shifted underneath the interpreter and it hit a
   syntax error at line 260.
   **NEVER `scp` over a script that is currently running.**

3. **Run 3** (powered attempt, never measured). Died on the 15-minute bringup timeout —
   bug #5, now fixed.

---

## 7. THE NEXT STEP

Run the powered comparison. Everything is staged and validated.

```bash
# ON sparkmain, under nohup. Do NOT edit this script while it runs.
nohup bash ~/bench-repo/scripts/run_matched_2v3_powered.sh \
  > ~/bench-repo/results/powered2v3-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
```

**Preconditions the script verifies itself** (it dies rather than proceed): patch parity
across all 3 nodes, matched config on both TP=2 ranks with `.prematched.*` backups present,
no stale vLLM containers, fabric gate, correctness `391`, and live-engine assertions that
each arm really is at `seqs=32 / gpumem=0.835 / MTP=2`.

**It measures BOTH arms** — TP=2 first (cluster is torn down anyway), then restores
production and re-measures TP=3 at the same n. The existing n=7 TP=3 arm is preserved as
`tp3-n7/`, not deleted. Comparing n=7 against n=30 would defeat the point of raising n.

**n is sized per cell by power analysis**, not convention:
`n = 2(1.96+0.84)² · CV² / δ²` at α=0.05, power=0.80, using CVs measured on the cooled arm
against the effects the published table claims.

| Cell | CV | Claimed effect | n needed | n used |
|---|---:|---:|---:|---:|
| 2K | 3.8 % | +16.7 % | ~1 | 30 |
| 8K | 2.6 % | +14.2 % | ~1 | 30 |
| 32K | 2.3 % | +11.0 % | ~1 | 30 |
| **131K** | 7.5 % | +7.3 % | **~17** | 30 |
| 262K | 6.0 % | +10.0 % | ~6 | 12 |
| cc=4/8/16 | — | — | — | 15 |

131K is the binding cell. 262K needs only ~6 but costs ~173 s/rep, so uniform n=30 would
run 5.8 h for both arms with 262K alone eating 95 min/arm. Scaled: **~1.6 h per arm**, more
power where it matters. If conditions degrade to the published arms' CVs (16–19 %), 131K
would need n≈76 — which is why the arm is cooled and clock-sampled.

**Runtime:** ~3.5 h for both arms plus ~30 min of cluster transitions. The cluster is down
for most of it and auto-restores at the end.

**Read the pre-registration first:**
[`PREREGISTRATION-2V3-MATCHED.md`](PREREGISTRATION-2V3-MATCHED.md). Hypotheses, the tie
band (6.6–11.7 % from Issue #31), the outlier rule, and the adjudication test were all
fixed **before** any measurement, specifically so they cannot be chosen after seeing which
arm they favour. **If two nodes win a workload, that is the finding** and it goes in the
README with the same prominence as a three-node win.

---

## 8. After the run

1. Fill the pre-registered tables in §5 of the pre-registration.
2. Run the same Mann-Whitney U test on both arms (method in
   [`ANALYSIS-2V3-2026-08-29.md`](ANALYSIS-2V3-2026-08-29.md) §1).
3. Copy the bundle from `~/bench-repo/results/` into the git repo and index it in
   `results/INDEX.md` + `index.yaml`.
4. Rewrite the README scoreboard from the result — including a null result.
5. Reconcile the two acceptance figures (67 % at concurrency vs 76.7–80.4 % single-stream)
   and state concurrency alongside every acceptance number.

## 9. Standing traps

- **`systemctl` lies about this cluster right now** (§1). Check `/health`.
- **Never `scp` over a running script.**
- **Cold start is ~30 min**, not 7. Ports closed and `shm_broadcast: no available shared
  memory broadcast block found in 60 seconds` are normal during boot, not a hang.
- **`clocks_throttle_reasons.active` reads `0x0`** even while the part is power-capped.
  Read the cumulative counters instead.
- **Publish spread with every median.** Four published claims died because spread was
  omitted.
- **`open-webui` runs on sparkmain against this engine.** Stop it before benchmarking; a
  foreign client once caused a 3.5× false regression.
