# Engine A/B — 3-node TP=3, anemll-v0.25.1 vs eugr-spark-vllm-b12x

**Status: PROTOCOL DEFINED, new-engine arm NOT YET RUN.** Image pulled to all
three nodes 2026-08-30; no eugr-engine measurement exists yet.

## Scope decision (2026-08-30)

Only the **3-node TP=3** shape gets a new-engine arm. The 2-node baseline is
NOT being re-run on the new engine at this time — the 2-node numbers remain
old-engine-only and are labeled as such by the `engine` column. The question
this A/B answers is narrow: *on our production 3-node shape, does the new
engine regress or improve anything?*

## The engine is part of a measurement's identity

Every CSV row now carries an `engine` column:

- `anemll-v0.25.1` — everything through 2026-08-30: ghcr Anemll image or our
  `Dockerfile.runtime` rebuild of it. Stock vLLM v0.25.1 + b12x + 13-file
  overlay + our patch set.
- `eugr-spark-vllm-b12x` — the community image vLLM's own recipe site
  (recipes.vllm.ai) recommends for DSV4-Flash on DGX Spark. Built from vLLM
  main + FlashInfer main by eugr/spark-vllm-docker. **`:latest` moves — record
  the image digest (`docker images --digests`) in the run's results/ bundle or
  the row is not reproducible.**

Cross-engine comparison is valid ONLY cell-by-cell with everything else
matched: same harness, same prompt_shape, same concurrency, same context, same
day, healthy fabric (NCCL sanity check first — see the silent-RDMA-degradation
history). One variable: the engine.

## ⛔ Gate before ANY eugr 3-node number is recorded

**Stock vLLM at TP=3 on this model SILENTLY serves nonsense** — o_groups=8
does not divide by 3, and without the padding patch the engine starts, serves,
and returns garbage with no error. The eugr image does not carry our patch.

Before the first measurement:

1. Run `patches/apply_tp3_patch.py --check` against the eugr image's vLLM tree
   (its vLLM is newer than v0.25.1, so expect anchor MISSes; port before
   applying).
2. Apply the patch inside the image (or a derived layer) on ALL THREE nodes.
3. Run the full 14/14 correctness validation and record the result in the
   results/ bundle. **A throughput number from an unvalidated TP=3 engine is
   not a measurement — it may be the speed of generating garbage.**

Also pin before measuring (upstream defaults moved since v0.25.1):

- `--max-num-batched-tokens 8192` (newer default is 16384 — measured here as
  −43% KV for zero gain, see the reverted rows in measurements.csv)
- Note whether async scheduling auto-enabled (newer vLLM turns it on with
  draft models); record the speculative method and depth actually in effect —
  the eugr/recipe default is `dspark nst=7 probabilistic`, ours is MTP K=2, and
  those are different measurements.
- Record `prefix_cache_retention_interval` / prefix-cache state, kv-cache-dtype
  and `gpu_memory_utilization` as launched.

## The cells

Old-engine reference values are QUERIED from benchmarks/summary.csv (they are
restated here for convenience; summary.csv wins on any discrepancy). Fill the
new-engine column by appending rows to measurements.csv with
`engine=eugr-spark-vllm-b12x` and the SAME config/harness/prompt columns —
the comparison table then falls out of the data instead of being hand-kept.

| cell | harness | prompt_shape | c | anemll-v0.25.1 | eugr-spark-vllm-b12x |
|---|---|---|---|---:|---:|
| single-stream decode (256-tok prompt) | bench-miaai | synthetic-numbered-words | 1 | 80.4 (tp3-seqs16) | — |
| peak useful aggregate (seqs cap) | bench-miaai | synthetic-numbered-words | 16 | 161.0 | — |
| decode at 131,072-token context | bench-miaai | synthetic-numbered-words | 1 | 83.5 | — |
| prompt-effect: code-brief | ours-bench.py | code-brief | 1 | 81.8 | — |
| prompt-effect: dense-prose | ours-bench.py | dense-prose | 1 | 49.4 | — |
| deep concurrency 4×~200K (usability) | deepconc.py | synthetic-numbered-words | 4 | 0.9 (unusable) | — |
| KV cache tokens (capacity, prompt-independent) | n/a | n/a | n/a | 3,588,422 | — |

Decode rates depend on the speculative path (the prompt-effect pair exists
because MTP acceptance moves decode 1.65x). If the eugr arm runs a different
speculative config than MTP K=2, the decode cells measure *engine+speculator*,
not engine — still useful, but say so in the row notes.

## Where results go

- Raw evidence: a `results/<UTC>-engine-ab-eugr/` bundle (digest, launch
  command, engine startup log, harness outputs, NCCL sanity check).
- Rows: append to `benchmarks/measurements.csv` with the new engine label;
  regenerate summary.csv (`python3 scripts/generate_summary.py`); schema test
  enforces the engine whitelist (`tests/test_benchmark_schemas.py`).
