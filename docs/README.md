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

---

## Start here

| Doc | Status | What it answers |
|---|---|---|
| [`HANDOFF.md`](HANDOFF.md) | ✅ | **What is running, how to operate it, what is open.** Read first. |
| [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md) | ✅ | Four classes of silent failure that produced plausible-but-wrong numbers, and how each is now gated. **Read before adding a benchmark.** |
| [`DECISIONS.md`](DECISIONS.md) | ✅ | Every settled config value and the measurement that settled it. |
| [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md) | ✅ | **What bad data looked like.** Every wrong number, how far off it was, and how to recognise the shape next time. |

## The results that stand

| Doc | Status | What it answers |
|---|---|---|
| [`TTFT-AND-WARMUP.md`](TTFT-AND-WARMUP.md) | ✅ | Why TTFT spikes on request 2, and why you should **not** add a keep-alive ping. |
| [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) | ✅ | The 6.8x RDMA degradation, how it hid, and the parity re-run. |
| [`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh) | ⚠️ | **Four-HCA fabric measures 2.0x and is gate-clean.** Not adopted until it survives an engine run. |
| [`PREFILL-MEASURED.md`](PREFILL-MEASURED.md) | ✅ | Prefill has **two rates, ~30x apart**. The long one. |
| [`KV-QUALITY-LONG-CONTEXT.md`](KV-QUALITY-LONG-CONTEXT.md) | ⚠️ | NVFP4 KV quality, clean to 464K — but **single-arm**, no comparison. Open as [#16](../../issues/16). |
| [`MTP5-1M-AND-UPSTREAM-COMPARISON.md`](MTP5-1M-AND-UPSTREAM-COMPARISON.md) | ✅ | Why MTP=5 and 1M context; the upstream gap is workload, not deficit. |

## The results that are provisional

Measured before the 2026-08-25 fabric fix. Conclusions may hold; the numbers should not
be quoted without a re-run.

| Doc | Status | Note |
|---|---|---|
| [`WHY-THREE-NODES.md`](WHY-THREE-NODES.md) | ⚠️ | The 2-vs-3 case. **The README table supersedes its numbers** — the direction survived, the magnitudes moved and the cc=16 crossover was not visible. |
| [`TP3-TUNING.md`](TP3-TUNING.md) | ⚠️🧊 | The tuning sweep at the old `460800`/`seqs=8`/`MTP=4` profile. |
| [`BASELINE-2SPARK.md`](BASELINE-2SPARK.md) | 🧊 | The frozen 2-node reference. |
| [`results.md`](results.md) | 🧊 | Earliest TP=2/TP=3 comparison, pre-rewire cabling. |
| [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) | ⚠️ | `seqs=32` rejected against a communication budget now known to be **6.6x too small**. Worth revisiting — [#10](../../issues/10). |

## Dead ends — kept so they are not re-proposed

| Doc | Status | Why it failed |
|---|---|---|
| [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) | ❌ | EP=3 shards correctly but the B12X kernel refuses EP: **2.5x slower**. This also blocks EPLB/expert placement, whose prerequisite is EP. |
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
