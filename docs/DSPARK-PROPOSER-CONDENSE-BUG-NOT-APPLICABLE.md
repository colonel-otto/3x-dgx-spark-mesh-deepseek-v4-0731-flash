# MiaAI's draft-KV condense bug: why this deployment does not have it

**Status:** `CLOSED — NOT APPLICABLE` · **Investigated:** 2026-08-30 ·
**Branch:** `investigate/miaai-batching-and-backports`

## The bug (theirs)

[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
`docs/PATCHES.md` documents three correctness patches to their speculative-decoding
proposer (`recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`):

- **Patch 1 — request-stable KV slot.** Their draft keeps a persistent cross-step
  ring buffer `main_kv_cache[max_num_seqs, window, head_dim]` holding each
  sequence's sliding-window KV history, indexed by **batch row**
  (`main_kv_cache[:batch_size]`) with no request identity. Under vLLM-v1
  continuous batching, when a request finishes the running set is *condensed* —
  a later request moves into the freed row, but the ring-buffer row does not
  move with it. The request then reads another request's draft history →
  silent draft-acceptance collapse. No crash.
- **Patch 2 / 2b — ragged context path.** Their `prepare_context` reshaped flat
  hidden states into a rectangular `[batch, seq, H]`, asserting uniform
  per-request row counts. Chunked prefill mixes prefill and decode rows in one
  step, so counts differ → HTTP 500
  `got N rows for batch_size=M`.

## Why we are not exposed

We verified against the production image `dsv4-3spark:0.1.1`
(vLLM `0.25.2.dev0+g752a3a504.d20260714`, Anemll lineage), read-only via
`docker run --rm` on sparkmain, 2026-08-30.

### 1. The patched file does not exist in our image

- `vllm/v1/spec_decode/dspark_proposer.py` — **absent**. `DSparkProposer` is not
  defined anywhere in the tree.
- `main_kv_cache`, `store_main_kv`, `_view_by_request` (the crash site) —
  **zero occurrences** in the entire installed vLLM package.

MiaAI's proposer is their own hand-written overlay for vLLM 0.26/0.27, `COPY`'d
into their image by `Dockerfile.gb10-dsv4-dspark`. It never existed in the
Anemll 0.25.2 lineage we run. Their patches repair their own code.

### 2. Our `method="dspark"` routes to a structurally immune implementation

`--speculative-config '{"method":"dspark",...}'` dispatches via
`vllm/v1/worker/gpu/spec_decode/__init__.py::init_speculator` to
**`DSparkSpeculator`** (`v1/worker/gpu/spec_decode/dspark/speculator.py`,
170 lines, subclass of `DFlashSpeculator`). Its design removes both failure
modes at the architectural level:

| Failure mode (theirs) | Our implementation |
|---|---|
| Private per-request ring buffer indexed by batch row | **Draft KV lives in vLLM's paged KV cache** — `kv_cache_groups`, `block_tables`, `build_slot_mappings_by_layer` (`dflash/speculator.py:113–157`). Block tables are per-request *by construction*; batch-row condensing cannot cross-contaminate them. |
| Cross-step draft history that must survive condensing | **No cross-step draft history.** DSpark here drafts a whole block in one parallel pass over the *current* step's target hidden states, then applies a sequential Markov bias. The upstream docstring: *"DSpark does not use the same pre-allocated buffer that DeepSeek-V4's MTP uses."* |
| Rectangular per-request reshape assuming uniform lengths | No `_view_by_request` equivalent exists on this path. |

### 3. Every persistent tensor audited

All cross-step tensors in `DSparkSpeculator` + its `DFlashSpeculator` parent:

| Tensor | Lifecycle | Verdict |
|---|---|---|
| `self.hidden_states` | Fully overwritten from the current step's target forward before every read (`dflash/speculator.py:291`: `.copy_(hidden_states[:num_target_tokens])`) | scratch |
| `_step_cols`, `_anchor_idx` | Constant index maps | safe |
| `_draft_scatter_buf` | `-inf` once; per-step `index_copy_` overwrites all consumed columns | scratch |
| `draft_tokens[:num_reqs, i]` | Written then consumed within the same step | scratch |

**None carries per-request content across engine steps outside the paged KV.**

## Consequences

- MiaAI Patches 1/2/2b: **nothing to port, nothing analogous to fix.**
- Our published matched 2v3 results and the ~66 % draft-acceptance figure are
  **unthreatened** by this bug class.
- The uniform-prompt limitation of our concurrency harness
  (`scripts/benchmark_mtp_concurrency.py`) is real but was **not** hiding this
  bug; there is no bug on this path to hide.

## What remains open from the same investigation

1. Four upstream perf backports (#50298, #48957, #50312, #49486) written for
   our byte-exact vLLM build — dry-run pending.
2. Issue #22 (`nvfp4_ds_mla` 600K+ decode collapse): our "does not reproduce"
   exemption was measured at `max_model_len=460800`; we now ship 1048576.
   Re-measure or lower the ceiling.
3. MiaAI issue #141 (stochastic sparse-MLA verify stalls on the same pinned
   Anemll SM120 adapter) — relevance to TP=3 unknown.
