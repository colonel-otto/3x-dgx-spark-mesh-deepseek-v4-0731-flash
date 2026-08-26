# Documentation index

Every page is tagged with the fabric state it was measured on, because that is the single
biggest determinant of whether a number here is still true.

**2026-08-25 is the dividing line.** Before it, one node ran at ~15% of its collective
bandwidth with zero error indicators ([#14](../../issues/14)). Anything measured earlier
is marked **provisional** — not wrong, but not trustworthy without a re-run.

| Status | Meaning |
|---|---|
| ✅ **Settled** | Measured on healthy fabric, matched arms, still current |
| ⚠️ **Provisional** | Measured on degraded fabric; conclusion may not survive a re-run |
| 🧊 **Frozen** | A dated record of one experiment. Never update it — supersede it |
| ❌ **Falsified** | Kept deliberately so the dead end is not re-proposed |

### The three tiers a number can be in

Every page states which tier its figures belong to, in its top banner.

| Tier | What it is | Quote it? |
|---|---|---|
| **Our results** | Healthy fabric, RDMA confirmed `via NET/IB/*`, matched arms | Yes — this is what we advertise |
| **Diagnostic signature** | A number from a known-broken setup, kept so you can recognise the fault when reproducing | Only as a **symptom to match against** |
| **Falsified claim** | Something we asserted that later proved wrong | Never — kept so it is not re-proposed |

If your reproduction disagrees with our results, start at
[`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md) — it maps symptom → cause →
confirmation → fix.

---

## Start here

| Doc | Status | What it answers |
|---|---|---|
| [`HANDOFF.md`](HANDOFF.md) | ✅ | **What is running, how to operate it, what is open.** Read first. |
| [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md) | ✅ | Four classes of silent failure that produced plausible-but-wrong numbers, and how each is now gated. **Read before adding a benchmark.** |
| [`DECISIONS.md`](DECISIONS.md) | ✅ | Every settled config value and the measurement that settled it. |
| [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md) | ✅ | **Troubleshooting guide: your numbers don't match ours.** Symptom → cause → confirmation → fix, plus every degraded number itemized as a signature to match against. |

## The results that stand

| Doc | Status | What it answers |
|---|---|---|
| [`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3) | ✅ | **Is the third node worth it? Only past 32K.** Matched arms, healthy fabric, 2K–262K, 7 reps per depth. Parity below 32K, **+33.6% at 131K**, +17.9% at 262K — and TTFT favours *two* nodes at depth. Supersedes the "+8–17% from 2K upward" headline. |
| [`TTFT-AND-WARMUP.md`](TTFT-AND-WARMUP.md) | ✅ | Why TTFT spikes on request 2, and why you should **not** add a keep-alive ping. |
| [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) | ✅ | The 6.8x RDMA degradation, how it hid, and the parity re-run. |
| [`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh) | ✅ | **Four-HCA fabric measures 2.0x and is gate-clean.** **Adopted** — soak passed 2026-08-26: 408 requests, zero RDMA counter deltas, zero log events ([#17](../../issues/17)). Throughput benefit is separately unmeasured. |
| [`PREFILL-MEASURED.md`](PREFILL-MEASURED.md) | ✅ | Prefill has **two rates, ~30x apart**. The long one. |
| [`NCCL-TESTS-BUILD.md`](NCCL-TESTS-BUILD.md) | ✅ | How to build a **version-matched** nccl-tests. The container ships two NCCL versions; the wrong one invalidates the comparison. |
| [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md) | ✅ | **RESOLVED — there was never a gap.** The same config reads 5.80 GB/s under our harness and **23.92** under official `nccl-tests`, above the 20.84 published reference. Four hypotheses falsified on the way. |
| [`BANDWIDTH-NEXT-TEST.md`](BANDWIDTH-NEXT-TEST.md) | ⏳ | **The matched test that would settle it.** Four variables still differ; the public result started at our number and recovered via *bootstrap*. |
| [`KV-QUALITY-LONG-CONTEXT.md`](KV-QUALITY-LONG-CONTEXT.md) | ⚠️ | NVFP4 KV quality, clean to 464K — but **single-arm**, no comparison. Open as [#16](../../issues/16). |
| [`MTP5-1M-AND-UPSTREAM-COMPARISON.md`](MTP5-1M-AND-UPSTREAM-COMPARISON.md) | ✅ | Why MTP=5 and 1M context; the upstream gap is workload, not deficit. |

## Pages whose numbers are diagnostic signatures

Measured before the 2026-08-25 fabric fix. **Their figures are symptoms to match against
when a reproduction goes wrong, not results to beat.** Each page's banner states what
survives, what is void, and where the healthy number lives.

| Doc | Status | Note |
|---|---|---|
| [`WHY-THREE-NODES.md`](WHY-THREE-NODES.md) | ⚠️ | The 2-vs-3 case. **Fully superseded 2026-08-26** — the depth sweep landed and its "+8–17% from 2K upward" headline is wrong in *both* directions: parity below 32K, **+33.6% at 131K**. Only the shape of the argument survives. Quote [`../README.md`](../README.md#is-the-third-node-worth-it) instead. |
| [`TP3-TUNING.md`](TP3-TUNING.md) | ⚠️🧊 | The tuning sweep at the old `460800`/`seqs=8`/`MTP=4` profile. |
| [`BASELINE-2SPARK.md`](BASELINE-2SPARK.md) | ⚠️🧊 | The frozen 2-node reference — **and it ran entirely across the degraded link.** KV accounting and correctness survive; every tok/s is void. |
| [`results.md`](results.md) | 🧊 | Earliest TP=2/TP=3 comparison, pre-rewire cabling. Its 24.59 arm is a TCP-fallback signature. |
| [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) | ⚠️ | `seqs=32` rejected against a communication budget now known to be **6.6x too small**. Worth revisiting — [#10](../../issues/10). |

## Dead ends — kept so they are not re-proposed

| Doc | Status | Why it failed |
|---|---|---|
| [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) | ❌⚠️ | EP=3 shards correctly but the B12X kernel refuses EP — a source-code check, which **survives**. This also blocks EPLB/expert placement, whose prerequisite is EP. **Worst provenance in the repo:** ran on degraded fabric *and* over TCP fallback, so no tok/s on it is usable, and its "3-node RoCE needs a switch" claim is **retracted** — we now run 3-node RDMA at 23.92 GB/s. |
| [`PP3-PIPELINE-PARALLEL.md`](PP3-PIPELINE-PARALLEL.md) | ❌ | Blocked by MTP + a DSA stride constraint. No PP tok/s exists. |

## Method and setup

| Doc | Purpose |
|---|---|
| [`setup.md`](setup.md) · [`topology.md`](topology.md) · [`patch.md`](patch.md) | Build the cluster. **`patch.md` is not optional** — TP=3 serves nonsense without it. |
| [`BENCHMARK-METHODOLOGY.md`](BENCHMARK-METHODOLOGY.md) · [`reproduction-methodology.md`](reproduction-methodology.md) | How to measure so the result is comparable. |
| [`troubleshooting.md`](troubleshooting.md) | Symptom → cause. |
| [`EXPERIMENT-LOG.md`](EXPERIMENT-LOG.md) | The durable PR-by-PR decision trail. |
| [`ACCEPTANCE.md`](ACCEPTANCE.md) · [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Original project gates and build order. Historical. |

---

## Before you add a document

1. **Date it and state the fabric/config it was measured on** in the first three lines.
   Every provisional page above is provisional because that was missing.
2. **Do not edit a frozen result page.** Write a new dated one and mark the old superseded.
3. **Run the gate first:** `make gate-full CONFIG=configs/3spark-live.env` with the engine
   stopped. `scripts/run_experiment.sh` refuses to benchmark without it.
4. **Warm your shapes**, then discard any run whose log contains
   `JIT compilation during inference` — see [`TTFT-AND-WARMUP.md`](TTFT-AND-WARMUP.md).
