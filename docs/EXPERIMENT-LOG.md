# Experiment and pull-request log

This file is the durable decision trail. Negative results remain in the repository
because they explain why the next branch exists and prevent the same dead ends from
being repeated.

## Progression

| Order | PR | Experiment | Recorded outcome | Evidence |
|---:|---:|---|---|---|
| 1 | [#1](https://github.com/colonel-otto/3spark-dsv4/pull/1) | Benchmark harness | Added repeatable environment, fabric, API, correctness, and throughput collection | scripts and tests |
| 2 | [#2](https://github.com/colonel-otto/3spark-dsv4/pull/2) | 2-Spark baseline | 48.23 tok/s reference used by the later TP=3 comparison; larger baseline sweep also retained | [`BASELINE-2SPARK.md`](BASELINE-2SPARK.md) |
| 3 | [#3](https://github.com/colonel-otto/3spark-dsv4/pull/3) | 3-Spark EP=3 | Sharding worked, but the loss of the B12X MoE path made it substantially slower | [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) |
| 4 | [#4](https://github.com/colonel-otto/3spark-dsv4/pull/4) | 3-Spark PP=3 | Correctness path explored; rejected as the performance route | PR artifacts; merge after #3 |
| 5 | [#6](https://github.com/colonel-otto/3spark-dsv4/pull/6) | 3-Spark TP=3 | Padding patch passed correctness; RoCE made TP=3 faster than the 48.23 tok/s reference and nearly doubled KV capacity | [`TP3-TUNING.md`](TP3-TUNING.md) |
| 6 | [#5](https://github.com/colonel-otto/3spark-dsv4/pull/5) | Reproducibility package | Adds a publication-oriented report and artifact schema; integrate after the experiment chain | PR artifacts |

## Result labels

Use these labels consistently in reports and filenames:

| Label | Meaning |
|---|---|
| `tp2-roce-baseline` | Two-node reference, 48.23 decode tok/s |
| `tp3-socket-control` | Three-node TP=3 with TCP fallback, 24.59 decode tok/s |
| `tp3-roce-rotated` | Historical pre-rewire ring, 57.73 decode tok/s |
| `tp3-roce-canonical` | NVIDIA physical ring layout, 53.95 decode tok/s in the retained MTP=5 run |
| `tp3-roce-mtp4-seq8` | Best tested combined profile, medians 56.63 and 55.68 tok/s |

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

Merge the stacked experiment PRs with merge commits in order: **#1, #2, #3, #4, #6**.
After each base lands, retarget the next PR to `main` and rerun CI. This retains both the
individual experiment commits and GitHub's PR linkage. Rebase PR #5 onto the resulting
`main`, reconcile overlapping documentation, then merge it last as the reproducibility
layer. Do not squash the sequence into one success-only commit.
