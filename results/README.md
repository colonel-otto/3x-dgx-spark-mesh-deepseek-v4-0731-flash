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
| [20260830T101053Z-llama-benchy-2v3](20260830T101053Z-llama-benchy-2v3/) | 2/3 | Present/pass both arms | **Independent third-party corroboration** on `eugr/llama-benchy` 0.4.1.dev1+ge9be34457: 14 of 16 cells resolved at n=10, all favouring three nodes, zero favouring two; the last two resolved at n=30 in the re-run below, making it 16 of 16. Confirmatory only; cross-harness absolute t/s are not comparable. 32K/131K decode magnitudes are PROVISIONAL — see the re-run's variance finding |
| [20260830T130300Z-rerun-inconclusive](20260830T130300Z-rerun-inconclusive/) | 2/3 | Present/pass both arms | **n=30 re-measurement of the two cells that did not resolve above.** Both resolved, three nodes faster: 8K decode +12.7% (t=3.76), cc=1 decode +14.8% (t=4.24). Its real finding is that **n=10 mis-estimates variance in either direction** — at 8K the arms swapped (TP=3 std x2.4, TP=2 std x0.62) — which makes the parent's 32K/131K magnitudes provisional. n=30 was pre-committed; do not re-run these cells |
| [20260830-matched-2v3-powered](20260830-matched-2v3-powered/) | 2/3 | Present/pass | **The settled node-count comparison**; node count the only variable, n=30/cell, three nodes win decode, TTFT, aggregate throughput and KV pool |
| [20260830T194550Z-engine-ab-eugr](20260830T194550Z-engine-ab-eugr/) | 3 | Absent (NCCL clean; gate the matched A/B) | **Engine A/B arm 1 — eugr/spark-vllm-b12x on the same weights.** Correctness gate PASSED on the byte-identical suite (garble ALL CLEAN, RULER-lite 16/16 incl. 262K, deep-context 8/8, tools 6/7 with a valid-JSON `forced_choice` semantics diff) — proves the image's native virtual-TP is correct at TP=3, no padding patch. Throughput vs anemll tp3-seqs16: c=1 parity (82.1 vs 80.4), **c=4 +41%, c=8 +20% aggregate**, c=16 −17% from a scheduling cliff the engine attributes to nst=5 draft slots. Config deltas recorded (dspark nst=5, kv fp8, V2 runner). Not yet a same-day matched A/B. **⚠️ Superseded same-day by [20260830T2245Z-eugr-ksweep](20260830T2245Z-eugr-ksweep/): the c=16 cliff was mostly JIT contamination, not draft slots, and every throughput number here is a cold-cache LOWER BOUND** |
| [20260830T2245Z-eugr-ksweep](20260830T2245Z-eugr-ksweep/) | 3 | **PASS 30/30** (incl. NCCL bandwidth, 9.08–9.27 GB/s/pair) | **DSpark depth sweep + persistent kernel caches — winner nst=5/mnbt=8192, now pinned in `eugr.service`.** Three findings. (1) The sweep space is **{5,7}, not {2,3,5,7}**: nst<5 is *rejected* because the checkpoint sets `dspark_block_size: 5` ("produce incorrect output"), so **K=2 parity with the anemll MTP arm is impossible** — the cross-engine delta is permanent. nst=5 won *every* cell, so "high K wins single-stream" does not transfer. (2) **Arm 1's c=16 cliff is retracted**: at identical nst=5, cache persistence alone gave c=8 +47% and c=16 +48%, TTFT 7000→1755 ms. (3) `mnbt=16384` is the anemll KV trap on fp8 KV too: +8% on two cells for **−52% KV cache** (2.30×→1.11× max concurrency @1M). Correctness 7/7, virtual-TP active |
| [20260829-issue38-kernel-profiling](20260829-issue38-kernel-profiling/) | 3 | Absent | First device-level kernel trace for TP=3; decode is 87.5% NCCL AllReduce bound |
| [20260829-issue36-dspark-proposer-long-horizon](20260829-issue36-dspark-proposer-long-horizon/) | 3 | Absent | DSpark proposer long-generation audit (256-1536 tokens); confirms 76.7%-80.4% acceptance (tau=2.55) with 0 staleness decay |
| [20260828-issue36-locked-clocks-suite](20260828-issue36-locked-clocks-suite/) | 3 | Absent | Master suite under hardware-locked 3003 MHz clocks: determinism, MTP K=2 concurrency, prefill depth, and APC warm path |
| [20260828-issue35-guidellm-concurrency](20260828-issue35-guidellm-concurrency/) | 3 | Absent | GuideLLM industry standard concurrency and latency sweep across cc=1..32 |
| [20260828-issue33-deep-prefill-bt-sweep](20260828-issue33-deep-prefill-bt-sweep/) | 3 | Absent | Single-variable deep-prefill TTFT evaluation confirming bt=8192 sweet spot at MAX_MODEL_LEN=1M |
| [20260828-issue32-mtp-concurrency-sweep](20260828-issue32-mtp-concurrency-sweep/) | 3 | Absent | MTP K in {5,3,2} concurrency sweep; K=2 delivers +7.3% at cc=16 with 66.3% draft acceptance |
| [20260828-issue31-serving-determinism](20260828-issue31-serving-determinism/) | 3 | Absent | Serving path determinism & logprob noise floor quantification; established numerical parity gate tolerance |
| [20260828-issue29-apc-warm-path](20260828-issue29-apc-warm-path/) | 3 | Absent | Multi-turn prefix caching (APC) warm-path benchmark (106.8x TTFT speedup at 131K) |
| [20260827-issue28-speed-bt16384](20260827-issue28-speed-bt16384/) | 3 | Present/pass | Speed profile tuning with MAX_NUM_BATCHED_TOKENS=16384 for issue #28 |
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
| [20260829-matched-2v3-ABORTED](20260829-matched-2v3-ABORTED/) | Matched 2v3 attempt aborted mid-transition by operator error; no complete arm. The settled result is 20260830-matched-2v3-powered |
| [20260830-llama-benchy-ABORTED](20260830-llama-benchy-ABORTED/) | First llama-benchy attempt stopped at the fabric gate before any request — the gate working as designed. Completed run: 20260830T101053Z-llama-benchy-2v3 |
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
