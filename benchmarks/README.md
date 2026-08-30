# Benchmark data — schemas and valid comparisons

> ## ⚠️ A tok/s number without its prompt is not a measurement
>
> On this deployment the **benchmark prompt alone** moves single-stream decode by
> **1.65x**. Same script, same engine, minutes apart:
>
> | prompt | decode tok/s | MTP acceptance length | draft acceptance |
> |---|---:|---:|---:|
> | `Write a Python function that merges two sorted lists. Explain briefly.` | **81.8** | 4.44–4.67 | 86–92% |
> | `Write a detailed technical explanation of how pipeline parallelism differs from tensor parallelism…` | **49.4** | 2.89–3.25 | 47–56% |
>
> MTP speculative decoding drafts predictable code well and novel dense prose
> poorly, so throughput tracks draft-acceptance length. **Never compare two
> numbers produced by different prompts.** Every row here records the harness and
> prompt that produced it, and `tests/test_benchmark_schemas.py` fails the build
> if a new measurement omits them.

> The same rule now applies to the **engine**: every row carries an `engine`
> column, and the same cell on a different engine is a different measurement.
> Everything through 2026-08-30 is `anemll-v0.25.1` (the Anemll
> dspark-vllm-gx10 image / our rebuild of it). The new-engine arm
> (`eugr-spark-vllm-b12x`, 3-node only) is defined in
> [`../docs/ENGINE-AB-3NODE.md`](../docs/ENGINE-AB-3NODE.md) — including the
> **hard gate**: TP=3 on an unpatched engine silently serves garbage, so no
> eugr 3-node number is valid before the padding patch is ported, applied, and
> the 14/14 correctness validation passes on that image.

---

## Three files, three different grains

These are **not** interchangeable and must never be concatenated. One aggregated
experiment is not equivalent to one sweep point.

| File | Grain | One row is… | Hand-edited? |
|---|---|---|---|
| [`measurements.csv`](measurements.csv) | **observation** | one measured point, at one concurrency, from one harness | yes — append only |
| [`historical-summary.csv`](historical-summary.csv) | **aggregated experiment** | a whole experiment, carrying its own min/max/repetitions | yes — legacy, frozen |
| [`summary.csv`](summary.csv) | **derived** | one headline result, with provenance and comparability | **NO — generated** |

`summary.csv` is produced by [`../scripts/generate_summary.py`](../scripts/generate_summary.py).
Edit a source file, then regenerate:

```bash
python3 scripts/generate_summary.py
python3 tests/test_benchmark_schemas.py
```

The test fails if `summary.csv` is stale or hand-edited.

---

## Why the split exists

Two benchmark records were produced independently and merged from different PRs.
They describe the same cluster but at different grains, so neither could
overwrite the other without losing information:

- The **historical** rows (48.23 TP=2, 24.59 TP=3-socket, 57.73 TP=3-RoCE) are
  aggregated experiment summaries that predate prompt attribution. Their prompt
  is **not recoverable**, so it is recorded as `unrecorded` and is **never
  inferred**.
- The **measurement** rows are individual sweep points, each tagged with its
  harness and prompt shape.

Combining all 48 rows into one table would make a 5-repetition aggregated
experiment look like one benchmark run. The merge conflict is preserved in the
git history rather than being force-pushed away.

---

## Comparability values

Stated explicitly per row in `summary.csv`, so a reader never has to infer it:

| Value | Meaning |
|---|---|
| `prompt-matched` | Same harness **and** prompt shape as the rows beside it. A direct comparison is valid. |
| `historical-only` | Prompt `unrecorded`. Use as a within-file trend only. **Do not divide against a prompt-matched value** — the prompt alone is worth ~1.65x. |
| `external` | Published by a third party on their own hardware. Never mixed silently with local rows. |
| `capacity-metric` | Not a throughput number (e.g. KV cache tokens). Read from the engine startup log, prompt-independent, so comparable across configs without prompt matching. |

### The comparison this protects against

`57.73` (historical, prompt unrecorded) and `82.1` (bench-miaai,
synthetic-numbered-words) are both real, both from this cluster. Dividing them
would manufacture a **1.42x "speedup" that was never observed** — it is mostly
the prompt. That is exactly the error this schema exists to prevent.

---

## Harnesses

| `harness` | Prompt | Sampling | Metric definition |
|---|---|---|---|
| `bench-miaai` | Unique nonce cold prefix + *"Return exactly 128 numbered lowercase English words, then stop."* | `temperature 0.6`, `top_p 0.95`, `min_tokens=max_tokens=128`, `ignore_eos`, `thinking:false` | per-stream decode **after** first token |
| `benchmark_tp3` | *"Write a Python function that merges two sorted lists. Explain briefly."* (18 tokens) | `temperature 0`, `max_tokens 256` | `total_out / wall` — **includes** prefill + TTFT |
| `ours-bench.py` | code-brief or dense-prose (see `prompt_shape`) | `temperature 0`, `max_tokens 256`, streaming | decode only, **excludes** TTFT |

`decode_tok_s` and `aggregate_tok_s` are deliberately **separate columns** and
must never be merged: aggregate is what the cluster delivers in total, decode is
what one caller experiences.

⚠️ `benchmark_tp3` includes prefill while the others do not. On an 18-token
prompt that is only ~3% of wall, so its `cc=1` values are close to comparable —
but the definitions differ and should not be divided at longer prompts.

---

## Measurement noise

Single-stream decode on this cluster has a **~33% run-to-run spread** at
concurrency 1: 8 reps on an unchanged engine gave
`88.3 / 67.5 / 77.0 / 88.5 / 66.6 / 83.7 / 88.1 / 75.1` (median 80.4, range
66.6–88.5).

**Any single-stream difference under roughly 20% is not meaningful without
repeat measurement.** Use median-of-N with N ≥ 5.

Aggregate at higher concurrency is far tighter — `c=16` read 161.0 vs 162.0
across two *different* configurations — so **aggregate is the more reliable
comparator here**. This rule already caught one false positive: see the
`MAX_NUM_BATCHED_TOKENS=16384` entry in [`CHANGELOG.md`](CHANGELOG.md), where an
apparent single-stream gain (87.3 vs 71.6) turned out to sit inside the noise
band.

---

## Column definitions

### `measurements.csv`

| Column | Meaning |
|---|---|
| `timestamp_utc` | When the measurement was taken |
| `config_id` | Stable id for the engine configuration (see `CHANGELOG.md`) |
| `engine` | Which serving image produced the row: `anemll-v0.25.1` (all rows through 2026-08-30) or `eugr-spark-vllm-b12x`. New engines must be added to `VALID_ENGINE` in the schema test deliberately, with an `ENGINE-AB-3NODE.md` entry. |
| `nodes`, `tp_size`, `pp_size` | Cluster shape |
| `max_model_len`, `max_num_seqs`, `mtp_num_tokens`, `gpu_mem_util` | Engine settings |
| `kv_cache_gib`, `kv_cache_tokens`, `max_concurrency_x` | Engine-reported capacity at startup |
| `observation_type` | `sweep-point` · `acceptance-observation` · `correctness-check` |
| `statistic` | `median` · `single-observation` · `mean` — what the value *is* |
| `source` | `local-measurement` · `external-published` |
| `reverted` | `true` if the config was rolled back. **Reverted experiments are preserved and marked, never deleted.** |
| `harness`, `prompt_shape`, `prompt_tokens` | Attribution — mandatory for any throughput value |
| `concurrency` | Simultaneous requests |
| `decode_tok_s` | Per-stream decode rate |
| `aggregate_tok_s` | Total cluster output rate |
| `ttft_ms`, `accept_rate_pct`, `accept_len` | Latency and speculative-decoding health |
| `notes` | Free text |

### `summary.csv` (generated)

`result_id`, `source_file`, `config_id`, `engine`, `metric`, `statistic`,
`value`, `prompt_shape`, `harness`, `comparability`, `evidence_status`,
`notes`.

Every summarized result carries a stable `result_id`, the `source_file` it came
from, and an explicit `statistic` and `comparability`.

---

## Structural result: what the third node buys

Prompt-independent, so directly comparable:

| | 2-Spark TP=2 | 3-Spark TP=3 | gain |
|---|---:|---:|---:|
| KV cache | 19.52 GiB | 37.72 GiB | 1.93x |
| KV cache tokens | 1,771,152 | 3,581,724 | **2.02x** |
| Max concurrency @460,800 | 3.84x | 7.77x | **2.02x** |

**Caveat:** `vllm:num_preemptions_total` stayed at **0** through every test,
including concurrency 32. The KV pool has never been the binding constraint —
the sequence cap was. The third node currently buys capacity headroom, not
proportional aggregate throughput.
