# Documentation index

The current performance comparison is open: the corrected three-node decode arm exists,
but the matching corrected two-node arm does not. Start with the dated handoff and check
the provenance index before quoting any benchmark.

## Read first

| Document | Purpose |
|---|---|
| [Current handoff](HANDOFF-2026-08-27.md) | Current cluster, corrected results, benchmark defect, and next work |
| [Repository and data map](REPOSITORY-MAP.md) | What belongs on GitHub, what remains local, and where each artifact goes |
| [Results index](../results/README.md) | Readable catalogue of every frozen run bundle |
| [Provenance index](../results/INDEX.md) | Which evidence is current, void, superseded, gated, or ungated |
| [Benchmark policy](BENCHMARK-POLICY.md) | Required gate, output window, spreads, and live config capture |
| [Post-mortem](POSTMORTEM-2026-08-25.md) | Silent failures that produced plausible but invalid numbers |

## Current evidence

| Evidence | Status | What it establishes |
|---|---|---|
| [Corrected 3-node decode curve](../results/20260827-decode-3node-fixed/) | Current, single arm, gate absent | 256-token medians and full spreads from 2K–262K; intermittent slow mode at depth |
| [3-node quality suite](../results/20260827-quality-suite-3node/) | Current quality evidence, gate absent | RULER-lite 12/12, tools 7/7, deep tools 8/8, garble clean through 131K |
| [Controlled NCCL run](../results/20260826-nccl-controlled/) | Current, directly measured | 23.92 GB/s at 16 GiB; no bandwidth gap versus reference |
| [Four-HCA throughput](../results/20260826-four-hca-throughput/) | Current with provenance caveat | Doubling measured fabric bandwidth produced no decode-throughput gain |
| [KV dtype A/B](../results/20260826-kv-dtype-ab/) | Current quality A/B, gate absent | No material tested difference; 23/24 matched cells byte-identical |
| [Sequence-cap retest](../results/20260826-seqs32-retest/) | Current matched A/B, gate absent | `MAX_NUM_SEQS=32` improves aggregate throughput at cc=32 |

The former long-context 2-vs-3 headline is not current. The
[2026-08-26 depth sweep](../results/20260826-decode-depth-2v3/) returned 25–26 output
tokens instead of 256 and is retained as `VOID-25-token-window`. Its `+33.6% at 131K`
claim is withdrawn until a corrected two-node arm is measured.

## Build and operate

| Document | Purpose |
|---|---|
| [Setup](setup.md) | Reproduction setup |
| [Topology](topology.md) | Three-node switchless ring |
| [TP=3 patch](patch.md) | Required attention-group padding; stock TP=3 can serve wrong output |
| [Troubleshooting](troubleshooting.md) | Operational symptom-to-cause guide |
| [Degraded-data catalogue](DEGRADED-DATA-CATALOGUE.md) | Failure fingerprints from known-bad runs |
| [Current handoff](HANDOFF-2026-08-27.md) | Dated operational state; the old handoff is now a redirect only |

## Measure and interpret

| Document | Purpose |
|---|---|
| [Benchmark policy](BENCHMARK-POLICY.md) | Publication requirements |
| [Decisions](DECISIONS.md) | Settled configuration choices and their evidence |
| [Experiment log](EXPERIMENT-LOG.md) | Durable chronological decision trail |

## Focused findings

| Topic | Document |
|---|---|
| Fabric fault and recovery | [Fabric-fix parity](FABRIC-FIX-PARITY.md) |
| Official collective comparison | [Bandwidth comparison](BANDWIDTH-COMPARISON.md) |
| Repetition-loop failure | [Repetition loop](REPETITION-LOOP.md) |

## Retired redirects and historical reports

Redirects keep links from frozen result bundles working without carrying a second copy of
obsolete conclusions. Each redirect states what survived, points to the current owner and
raw evidence, and links to the last full revision for readers who need the original report.

| Document | Classification |
|---|---|
| [Old handoff](HANDOFF.md) | Redirect to the current handoff, setup, policy, and decisions |
| [Why three nodes](WHY-THREE-NODES.md) | Redirect; former 2-vs-3 recommendation withdrawn |
| [Early controlled results](results.md) | Redirect to the results and provenance indexes |
| [Bandwidth next-test plan](BANDWIDTH-NEXT-TEST.md) | Redirect to the completed controlled result |
| [2-Spark baseline](BASELINE-2SPARK.md) | Redirect; timing values void, raw bundle preserved |
| [TP=3 tuning](TP3-TUNING.md) | Redirect; patch result survives, performance claim does not |
| [Sequence-cap and NCCL investigation](SEQS32-AND-NCCL-FABRIC.md) | Redirect; original rejection overturned by later retest |
| [Prefill investigation](PREFILL-MEASURED.md) | Redirect to the resolved fabric finding and raw evidence |
| [MTP and 1M context](MTP5-1M-AND-UPSTREAM-COMPARISON.md) | Redirect to current decisions and frozen evidence |
| [KV long-context quality](KV-QUALITY-LONG-CONTEXT.md) | Redirect from single-arm study to matched A/B evidence |
| [Prompt-sensitivity study](BENCHMARK-METHODOLOGY.md) | Redirect to benchmark policy and schema |
| [Old reproduction protocol](reproduction-methodology.md) | Redirect to current policy, setup, and artifact schema |
| [NCCL build note](NCCL-TESTS-BUILD.md) | Redirect to setup and controlled bandwidth result |
| [TTFT and warm-up note](TTFT-AND-WARMUP.md) | Redirect to current warm-up policy |

The [expert-parallel](EP3-EXPERT-PARALLEL.md) and
[pipeline-parallel](PP3-PIPELINE-PARALLEL.md) reports remain focused technical
explanations because their source-level blockers are unique.

## Before adding documentation or evidence

1. Put reusable narrative in `docs/`, frozen raw evidence in a dated `results/` bundle,
   and normalized observations in `benchmarks/`.
2. Give every result a status, node/TP shape, live config source, harness/version, actual
   output-token count, reps, statistic, gate status, and raw-file links.
3. Never rewrite a frozen run. Supersede it and preserve the failure signature.
4. Update both `results/index.yaml` and `results/README.md`.
5. Follow the [repository publication checklist](REPOSITORY-MAP.md#publication-checklist).
6. Run `py scripts/audit_docs.py --fail-links` and follow the
   [iterative review loop](REPOSITORY-MAP.md#iterative-documentation-review).
