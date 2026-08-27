# DeepSeek-V4-Flash on three DGX Sparks

Reproducible deployment notes and benchmark evidence for serving DeepSeek-V4-Flash on
three NVIDIA DGX Spark systems with tensor parallelism (`TP=3`) over a switchless 200 GbE
RoCE ring.

The three-node deployment is working and passes the available correctness and long-context
quality suites. The performance question—whether three nodes beat two—remains open because
the corrected 256-token three-node sweep exists, but the matching corrected two-node sweep
does not.

> [!WARNING]
> Do not cite the older 2-vs-3 decode claims. The 2026-08-26 sweep requested 256 output
> tokens but returned only 25–26, biasing both its magnitudes and its curve. The affected
> bundle is retained as a diagnostic baseline and marked `VOID-25-token-window` in the
> [provenance index](results/INDEX.md).

## Start here

1. [Current handoff](docs/HANDOFF-2026-08-27.md) — current cluster state, reliable
   findings, open work, and the benchmark defect.
2. [Documentation index](docs/README.md) — setup, operations, method, decisions, and
   historical investigations.
3. [Results index](results/README.md) — one readable row per frozen run bundle.
4. [Provenance index](results/INDEX.md) — status, gate coverage, raw evidence, and caveats.
5. [Repository and data map](docs/REPOSITORY-MAP.md) — what belongs on GitHub and what
   should stay local.

## Current evidence

| Question | Current answer | Evidence |
|---|---|---|
| Does TP=3 serve correct output? | Yes; the attention-group padding patch is required. | [Patch](docs/patch.md), [quality suite](results/20260827-quality-suite-3node/) |
| Does three-node quality hold at long context? | Yes in the tested suite: RULER-lite 12/12, tool battery 7/7, deep-context tools 8/8, garble sweep clean through 131K. | [Quality suite](results/20260827-quality-suite-3node/) |
| What is corrected three-node decode speed? | Median 50.7–56.0 tok/s from 2K–262K with 256 output tokens; 131K and 262K show intermittent slow modes. | [Corrected curve](results/20260827-decode-3node-fixed/) |
| Is three-node faster than two-node? | **Unknown.** A method-matched corrected two-node arm has not been run. | [Current handoff](docs/HANDOFF-2026-08-27.md#open-problem-1--the-2-node-arm-does-not-exist) |
| Is the fabric below the published reference? | No. Official `nccl-tests` measured 23.92 GB/s at 16 GiB. | [Controlled NCCL run](results/20260826-nccl-controlled/) |
| Does four-HCA addressing improve decode throughput? | No measurable benefit despite doubling fabric bandwidth. | [Four-HCA result](results/20260826-four-hca-throughput/) |
| Does KV dtype change quality? | No material difference in the tested A/B; 23/24 matched cells were byte-identical. | [KV dtype A/B](results/20260826-kv-dtype-ab/) |

The corrected decode curve is a single three-node arm, not a purchasing comparison. Its
262K median also hides a severe bimodal distribution: two measured reps fell to 1.2–3.3
tok/s while five were near 50 tok/s. Read the spread, not only the median.

## How benchmark data is organized

| Location | Purpose | Edit policy |
|---|---|---|
| [`results/YYYYMMDD-<subject>/`](results/) | Frozen raw run bundle: harness, config capture, gate, logs, and per-rep output | Never rewrite; supersede with a new dated bundle |
| [`results/index.yaml`](results/index.yaml) | Machine-readable provenance and status for every run bundle | Source of truth for result status |
| [`results/INDEX.md`](results/INDEX.md) | Human-readable provenance summary | Keep aligned with the YAML |
| [`benchmarks/measurements.csv`](benchmarks/measurements.csv) | Append-only normalized observations | Add measured points with prompt and harness attribution |
| [`benchmarks/summary.csv`](benchmarks/summary.csv) | Generated headline metrics | Regenerate; do not hand-edit |
| [`docs/`](docs/) | Method, setup, operations, decisions, and interpretation | Update living docs; freeze dated reports |

Before publishing a benchmark, follow the [benchmark policy](docs/BENCHMARK-POLICY.md):
commit a passing fabric gate, force and verify the output window, publish every rep and
its spread, and capture config from the live process.

## Deployment shape

```text
rank 0 -- 200 GbE -- rank 1
   \                  /
    +---- 200 GbE ---+
          rank 2
```

The TP=3 patch pads eight attention groups to nine for sharding, then trims the padding
after gather. Without it, stock integer division silently drops groups and can produce
fluent but wrong output. See [topology](docs/topology.md), [setup](docs/setup.md), and
[patch details](docs/patch.md).

## Reproduce or add a result

```bash
# First copy and edit the tracked example; the live file remains local.
cp configs/3spark-live.env.example configs/3spark-live.env

# Engine stopped: measure the full fabric and retain the artifact.
make gate-full CONFIG=configs/3spark-live.env

# Run the repository test suite and sensitive-data scan before publishing.
make test
make check-sensitive
```

New runs must use a dated directory with a provenance header and include raw per-rep
evidence. See [Adding a run](results/README.md#adding-a-run).

## Scope and credit

This repository contributes the controlled comparison work, TP=3 integration notes,
failure gates, and evidence trail. It builds on Anemll's DGX Spark vLLM runtime, MiaAI-Lab
benchmark and deployment work, localaiguyy's TP=3 attention-group padding, and NVIDIA's
DGX Spark playbooks and `nccl-tests`. See [CREDITS.md](CREDITS.md) for attribution.

Security and redaction rules are in [SECURITY.md](SECURITY.md). Do not commit live `.env`
files, credentials, management addresses, usernames, serials, or unredacted environment
captures.
