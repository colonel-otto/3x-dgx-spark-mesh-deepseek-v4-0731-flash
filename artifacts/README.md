# Benchmark artifact bundles

The original exploratory measurements are represented in `benchmarks/summary.csv` and
are explicitly marked `historical-summary`; incomplete raw samples must not be invented.

For every new benchmark, create one immutable directory:

```text
artifacts/YYYY-MM-DD-config-name/
  manifest.json
  rendered-compose.yaml
  environment-redacted.txt
  startup-rank0.log
  startup-rank1.log
  startup-rank2.log
  benchmark-raw.jsonl
  benchmark-summary.json
  correctness.json
  nccl-info.log
  rdma-counters-before.json
  rdma-counters-after.json
  nccl-tests.csv
  sha256sums.txt
```

The manifest should record node model, DGX OS, kernel, driver, CUDA, CX-7 firmware,
container digest, live NCCL runtime, vLLM revision, checkpoint revision, patch revision,
all serving parameters, harness revision and timestamps.

Validate `manifest.json` against [`manifest.schema.json`](manifest.schema.json). Fields
such as image digest and checkpoint revision must be real immutable identifiers, not a
mutable tag or `latest`.

Redact the bundle according to [`../SECURITY.md`](../SECURITY.md), inspect it manually,
then calculate checksums after redaction.
