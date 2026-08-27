# Experiment and pull-request log

This file is the durable decision trail. Negative results remain in the repository
because they explain why the next branch exists and prevent the same dead ends from
being repeated.

## Progression

Chronological. Negative results stay because they explain why the next row exists.

### Phase 1 — establish and shard (2026-08-20 → 08-21)

| # | PR | Experiment | Outcome | Evidence |
|---:|---:|---|---|---|
| 1 | [#1](../../pull/1) | Benchmark harness | Repeatable environment, fabric, API, correctness and throughput collection | `scripts/`, `tests/` |
| 2 | [#2](../../pull/2) | 2-Spark baseline | Historical `48.23 tok/s` reference; later voided by degraded fabric | [`BASELINE-2SPARK.md`](BASELINE-2SPARK.md) |
| 3 | [#3](../../pull/3) | 3-Spark **EP=3** | ❌ Sharding works; losing the B12X MoE path costs **2.5x** | [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) |
| 4 | [#4](../../pull/4) | 3-Spark **PP=3** | ❌ Blocked by MTP + a DSA stride constraint. No PP tok/s exists | [`PP3-PIPELINE-PARALLEL.md`](PP3-PIPELINE-PARALLEL.md) |
| 5 | [#6](../../pull/6) | 3-Spark **TP=3** | ✅ Padding patch passes correctness; historical speed comparison later voided | [`TP3-TUNING.md`](TP3-TUNING.md) |
| 6 | [#5](../../pull/5) | Reproducibility package | Publication report + artifact schema | [`reproduction-methodology.md`](reproduction-methodology.md) |
| 7 | [#7](../../pull/7) | CSV reconciliation | Two benchmark CSVs reconciled by data grain | `benchmarks/` |

### Phase 2 — tune (2026-08-22 → 08-24)

| # | PR | Experiment | Outcome | Evidence |
|---:|---:|---|---|---|
| 8 | [#8](../../pull/8) | Batched-tokens scope | `MAX_NUM_BATCHED_TOKENS=16384` is a **trap**: 43% of KV for zero gain. The later “two prefill rates” interpretation was refuted | [`PREFILL-MEASURED.md`](PREFILL-MEASURED.md) |
| 9 | [#9](../../pull/9) | MTP=5 + 1M context | ✅ 1M context is **free** (memory-bound, not comms-bound); MTP=5 beats MTP=4 | [`MTP5-1M-AND-UPSTREAM-COMPARISON.md`](MTP5-1M-AND-UPSTREAM-COMPARISON.md) |
| 10 | [#9](../../pull/9) | `seqs=32` | Initial rejection overturned; healthy-fabric retest made 32 the production value | [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) |
| 11 | [#9](../../pull/9) | NVFP4 KV quality | Clean to 464K single-arm; later matched dtype A/B found no material difference | [`KV-QUALITY-LONG-CONTEXT.md`](KV-QUALITY-LONG-CONTEXT.md) |

### Phase 3 — the fabric was lying (2026-08-25)

Everything above this line was measured with one node at ~15% of its collective bandwidth.

| # | PR | Experiment | Outcome | Evidence |
|---:|---:|---|---|---|
| 12 | [#9](../../pull/9) | **Fabric degradation found** | ☠️ spark1 at 0.69 vs 4.6 GB/s — **6.8x**, with every error counter reading 0. A reboot fixed it | [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) |
| 13 | [#9](../../pull/9) | Fabric gate | ✅ `scripts/fabric_gate.sh` — 9 checks, each verified by **injecting the real fault** | [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md) |
| 14 | [#9](../../pull/9) | `roceP2p` HCAs | ⚠️ **OVERTURNED 2026-08-26** — rejected here as "unusable, no IPv4"; the wedge was the degraded fabric, not the HCAs. All four addressed and **now in production**. Bandwidth doubled; decode throughput benefit: **none** | [`../results/20260826-four-hca-throughput/`](../results/20260826-four-hca-throughput) |
| 15 | [#9](../../pull/9) | `/30` normalization | ✅ All six fabric addresses consistent; netplan persistence gated | [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md) |
| 16 | [#9](../../pull/9) | **2v3 re-run, healthy — concurrency axis** | Supporting evidence only: short-prompt result lacks a committed gate artifact and one headline cell | [`../README.md`](../README.md#current-evidence) |
| 17 | [#9](../../pull/9) | TTFT / warm-up | ✅ JIT is **per-shape, not per-idle**. Warm at startup; do **not** add a keep-alive ping | [`TTFT-AND-WARMUP.md`](TTFT-AND-WARMUP.md) |
| 18 | — | **2v3 re-run, healthy — depth axis** | ❌ `VOID-25-token-window`: every request stopped at 25–26 tokens, so the former depth comparison is withdrawn | [`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3) |

**Net effect after later validation:** Phase 3 found the fabric fault and produced useful
diagnostic signatures, but it did not settle the 2-vs-3 performance question. A corrected
3-node depth arm exists; the matching corrected 2-node arm does not. Every invalid number
is itemized in [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md).

## Open questions

| Question | Tracked | Why it matters |
|---|---|---|
| ~~3-rank collective **5.80 GB/s** vs a published ring at **18.70 @32MB**~~ | [#18](../../issues/18) | ✅ **CLOSED 2026-08-26 — there was never a gap.** The same unchanged config reads **23.92 GB/s** under official `all_gather_perf`, *above* the 20.84 reference. The only variable that mattered was the harness. Bootstrap was worth +0.1%, NIC-merge −0.3%. See [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md) |
| ~~`seqs=32` decided against a 6.6x-too-small budget~~ | [#10](../../issues/10) | ✅ **CLOSED** — healthy-fabric retest succeeded; 32 is production |
| ~~`nvfp4_ds_mla` vs `fp8_ds_mla` quality~~ | [#16](../../issues/16) | ✅ **CLOSED within tested scope** — matched A/B found no material difference |
| Decode/JIT tail still needs clean re-measurement | [#24](../../issues/24), [#26](../../issues/26) | Warm-up does not cover every shape, and the old decode window was invalid |
| Root cause of spark1's degradation | [#14](../../issues/14) | Never determined. If it recurs, a reboot cadence is warranted |

## Result labels

Use these labels consistently in reports and filenames:

| Label | Meaning |
|---|---|
| `tp2-roce-baseline` | Two-node reference, 48.23 decode tok/s |
| `tp3-socket-control` | Three-node TP=3 with TCP fallback, 24.59 decode tok/s |
| `tp3-roce-rotated` | Historical pre-rewire ring, 57.73 decode tok/s |
| `tp3-roce-canonical` | NVIDIA physical ring layout, 53.95 decode tok/s in the retained MTP=5 run |
| `tp3-roce-mtp4-seq8` | Best tested combined profile, medians 56.63 and 55.68 tok/s |
| `tp3-1m-mtp4-seq16` | 1M context, MTP=4, seqs=16; 374.2 tok/s aggregate at cc=16 |
| `tp3-1m-mtp5-seq16` | 1M context, MTP=5, seqs=16; **current recommendation** for structured/agentic use |

The 57.73 result is a historical best, not the current canonical-layout result. The
24.59 result remains useful as a transport control; it must not be presented as RoCE
performance.

## Evidence policy

Every new performance claim should retain:

1. UTC experiment ID and source commit.
2. Sanitized per-rank configuration snapshots, image digest, driver/CUDA/NCCL versions,
   model revision, and patch revision.
3. Physical topology (`lldpcli`, interface-to-RDMA mapping, IPs, MTU, link rate) and
   `NCCL_DEBUG=INFO` lines proving `NET/IB` and the selected HCAs.
4. Raw per-repetition benchmark output, summary statistics, warm-up policy, exact prompt,
   sampling parameters, and concurrency.
5. Correctness output and failure logs, including controls that failed.

The PR #6 medians predate this complete artifact policy. Where raw output was not
retained, this repository says so rather than reconstructing it from summaries.

## Merge policy

PRs #1–#7 are merged. Retain the individual experiment commits: **do not squash a
sequence into one success-only commit.** The dead ends are the value.

For new work: branch, gate, measure, and open a PR whose title states the *finding*, not
the activity. A PR that changes a documented value must update
[`DECISIONS.md`](DECISIONS.md) in the same commit.
