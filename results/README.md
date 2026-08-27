# Results catalogue

Every subdirectory is a frozen experiment or quality bundle. Do not edit an old bundle to
make a newer claim true; create a new dated bundle and mark the old one `VOID` or
`SUPERSEDED`.

The machine-readable source of truth is [`index.yaml`](index.yaml). The concise provenance
view is [`INDEX.md`](INDEX.md).

## Status key

| Status | Meaning |
|---|---|
| `CURRENT` | Useful evidence under its stated scope and caveats |
| `VOID-25-token-window` | Decode window collapsed; keep only as a failure fingerprint |
| `VOID-degraded-fabric` | Measured before the fabric fix; magnitudes are not citable |
| `SUPERSEDED` | Replaced by a newer run; retained as decision history |

`CURRENT` does not mean “perfect.” Check the gate column and the bundle README. An absent
fabric gate weakens performance evidence but does not invalidate pass/fail quality results.

## Current and citable within stated scope

| Bundle | Nodes | Gate | What it establishes |
|---|---:|---|---|
| [20260827-tp3-131k-15rep](20260827-tp3-131k-15rep/) | 3 | Present/pass | 15-rep 131K single-stream evaluation for issue #24; 51.04 tok/s median |
| [20260827-issue25-profile-b](20260827-issue25-profile-b/) | 3 | Present/pass | Winning Profile B tuning bundle for issue #25; starvation resistance & depth sweep |
| [20260827-decode-concurrency-2v3-fixed](20260827-decode-concurrency-2v3-fixed/) | 2/3 | TP=2 present/pass; TP=3 live check | Corrected-window concurrency crossover at 8K |
| [20260827-decode-2v3-fixed](20260827-decode-2v3-fixed/) | 2/3 | TP=2 present/pass; TP=3 live check | Corrected-window cc=1 node-count sweep; 7 reps per depth and arm |
| [20260827-decode-3node-fixed](20260827-decode-3node-fixed/) | 3 | Absent | Corrected 256-token depth curve; single arm; intermittent slow mode at 131K/262K |
| [20260827-quality-suite-3node](20260827-quality-suite-3node/) | 3 | Absent | Quality suite passes through 131K; pass/fail evidence, not timing evidence |
| [20260826-nccl-controlled](20260826-nccl-controlled/) | 2/3 ranks | Direct measurement | Official `all_gather_perf`; 23.92 GB/s at 16 GiB |
| [20260826-kv-dtype-ab](20260826-kv-dtype-ab/) | 3 | Absent | NVFP4 vs FP8 KV quality and speed A/B |
| [20260826-seqs32-retest](20260826-seqs32-retest/) | 3 | Absent; matched control | Sequence-cap A/B; cc=32 aggregate improvement |
| [20260826-four-hca-throughput](20260826-four-hca-throughput/) | 3 | Absent; cross-reference | Four-HCA throughput null result |
| [20260826-harness-window-calibration](20260826-harness-window-calibration/) | 3 | Absent; same-engine A/B | Quantifies the short-window benchmark defect |
| [20260826-near-ceiling-prefill](20260826-near-ceiling-prefill/) | 3 | Absent | Preliminary capability result: 967,286 prompt tokens served |
| [20260825-upper-mesh](20260825-upper-mesh/) | 3 | Present/pass | Four-HCA mesh and 408-request soak |
| [20260825-prefill-2v3](20260825-prefill-2v3/) | 2/3 | Present/pass; TP=2 raw elsewhere | Prefill comparison through 32K |
| [20260825-fabric-fix](20260825-fabric-fix/) | 2/3 | Absent; diagnostic A/B | Before/after reboot characterization of the fabric fault |
| [20260825-deep-concurrency](20260825-deep-concurrency/) | 2/3 | Present/pass | Four concurrent 200K requests |
| [20260825-decode-2v3](20260825-decode-2v3/) | 2/3 | Absent | Short-prompt concurrency comparison; one headline cell lacks raw traceability |

## Void: short output window

| Bundle | Why retained |
|---|---|
| [20260826-decode-depth-2v3](20260826-decode-depth-2v3/) | All 70 reps returned 25–26 tokens instead of 256. Useful for recognizing 30–40% inflated, high-variance decode results; not valid 2-vs-3 evidence. |

## Void: incomplete run

| Bundle | Why retained |
|---|---|
| [20260827-decode-2node-failed](20260827-decode-2node-failed/) | The engine passed a live sanity request, then every benchmark request returned HTTP 404 or reset. No valid samples; logs preserve the operational failure. |

## Superseded

| Bundle | Replaced by |
|---|---|
| [20260827-issue25-profile-a](20260827-issue25-profile-a/) | [20260827-issue25-profile-b](20260827-issue25-profile-b/) |
| [20260824-kv-quality](20260824-kv-quality/) | [20260826-kv-dtype-ab](20260826-kv-dtype-ab/) |
| [20260824-seqs32-nccl](20260824-seqs32-nccl/) | [20260826-seqs32-retest](20260826-seqs32-retest/) |

## Void: degraded fabric

| Bundle | Diagnostic value |
|---|---|
| [20260824-prefill](20260824-prefill/) | ~1,034 tok/s prefill at 32K versus ~2,095 after reboot fingerprints a degraded link |
| [20260824-mtp5-1m](20260824-mtp5-1m/) | Older MTP/1M experiment; matched conclusions may survive, magnitudes do not |
| [20260821T142000Z-3spark-tp3-upstream-harness](20260821T142000Z-3spark-tp3-upstream-harness/) | Older TP=3 upstream-harness signature |
| [20260821T133000Z-3spark-tp3](20260821T133000Z-3spark-tp3/) | TP=3 setup record; patch requirement survives |
| [20260821T133000Z-3spark-pp3](20260821T133000Z-3spark-pp3/) | PP hard-block evidence; no performance number exists |
| [20260821T031300Z-3spark-ep3](20260821T031300Z-3spark-ep3/) | EP/B12X incompatibility signature |
| [20260821T001024Z-2spark-baseline](20260821T001024Z-2spark-baseline/) | Frozen 2-node baseline with degraded fabric and a 10-token decode window |

## Adding a run

1. Name the bundle `YYYYMMDD-<subject>` or `YYYYMMDDTHHMMSSZ-<subject>`.
2. Add a README header containing status, date, nodes/TP, live config source,
   harness/version, actual output tokens, reps/statistic, and fabric-gate result.
3. Run the fabric gate before each performance arm with the engine stopped and commit its
   artifact. Pass/fail-only quality runs must still state when the gate is absent.
4. Force the generation window (`min_tokens == max_tokens`, `ignore_eos`) and assert the
   actual per-request completion count.
5. Commit raw per-rep data and publish sorted values or an equivalent spread; never retain
   only the median or only the best run.
6. Keep the harness beside the output or identify an immutable repository commit.
7. Add the bundle to `index.yaml` and this catalogue. Append normalized observations to
   `../benchmarks/measurements.csv` when applicable.
8. Run the tests and sensitive-data check before publishing.

See the full [benchmark policy](../docs/BENCHMARK-POLICY.md) and
[repository/data map](../docs/REPOSITORY-MAP.md).
