# Backport dry-run — MiaAI perf hotfixes vs our TP=3 image (2026-08-30)

Ephemeral-container dry-run of the four MiaAI-Lab anchor-verified perf hotfixes
against `dsv4-3spark:0.1.1`. **Nothing persistent was modified**: every apply ran
inside `docker run --rm` (image-only, no volumes) on sparkmain, and the container
was discarded. The engine was down (port 8100 refused) for the whole exercise; no
service was started, stopped, or restarted.

## Method

- Scripts read in full from the read-only checkout
  `../MiaAI-Lab-2spark/patches/` (local checkout verified **byte-identical to
  `origin/main`** for all four scripts after `git fetch` — see
  "Version check" below).
- Each script's **default action applies immediately** (no dry-run flag), but the
  apply is a whole-script staged transaction: every hunk validates against an
  in-memory view first, nothing is written unless all hunks validate, and a
  failed commit rolls back byte-exactly. Each script also has a read-only
  `--status` self-check and auto-runs it after apply.
- All four were applied for real **inside an ephemeral container**, followed by
  their `--status` self-checks, `python3 -m py_compile` of every edited file,
  and `patches/apply_tp3_patch.py --check` before and after to detect TP=3
  anchor consumption.
- Note: the `.sh` scripts fail with `set: pipefail: invalid option name` if
  copied from a Windows checkout with CRLF line endings. Strip CRLF
  (`sed -i 's/\r$//'`) before shipping them to a node.

## Image state (verified in-container)

- vLLM `0.25.2.dev0+g752a3a504.d20260714` — exactly the build all four scripts
  target.
- TP=3 padding patch is baked in: 8 `.tp3bak` backups present; 14 of 15
  `apply_tp3_patch.py --check` edits report "already patched".
- One pre-existing, unrelated wrinkle: TP=3 edit 13 (`gpu_worker.py`: profile
  CUDA only) reports `MISS` — but the image's `gpu_worker.py:1078` already
  contains `activities=["CUDA"]` (the edit's *effect*) without the marker
  comment, and there is no `gpu_worker.py.tp3bak`. The edit was evidently baked
  into the image build in a form that lacks the tag. Functionally fine;
  `--check` will always exit 1 on this image until the marker or the check is
  reconciled. **Not caused by, and not affecting, any hotfix below.**

## Active attention backend caveat (matters for #50298)

`docker-compose.yml` passes no `--attention-backend`, so
`nvidia/model.py::_select_dsv4_attn_cls` takes its default branch: on device
capability major == 12 (GB10 is SM121) it selects
**`DeepseekV4FlashInferSM120Attention`** (`nvidia/flashinfer_sparse.py`), *not*
`DeepseekV4FlashMLAAttention` (`nvidia/flashmla.py`). The SM120 class has its own
`forward_mqa` and `_forward_prefill` (built on
`compute_global_topk_indices_and_lens` + FlashInfer kernels);
`combine_topk_swa_indices` is called **only** from `nvidia/flashmla.py:326`.
Confirm which class the live boot actually instantiates before crediting any
FlashMLA-path hotfix with a speedup.

## Results per hotfix

| Hotfix | Target files (under `vllm/`) | Anchors | Apply in ephemeral | Self-check | TP=3 overlap | Verdict / next step |
|---|---|---|---|---|---|---|
| `hotfix-dsv4-flashmla-workspace-50298.sh` (1.88x kernel, workspace reuse) | `models/deepseek_v4/nvidia/flashmla.py` (5 hunks), `models/deepseek_v4/common/ops/cache_utils.py` (1 hunk) | 6/6 matched at expected counts | Committed 6 hunks / 2 files, exit 0 | `--status` 6/6 APPLIED; py_compile OK | **None** — neither file is TP=3-patched; TP=3 `--check` unchanged after apply | Applies cleanly and is safe, **but likely inert on our deployment**: the patched `_forward_prefill` belongs to the FlashMLA attention class, and our default GB10 path is the SM120 FlashInfer class (see caveat above). Apply only with an A/B that can detect "no change"; verify the live attention class first. |
| `hotfix-dsv4-skip-empty-c128-48957.sh` (~2x kernel, skip empty C128 launches) | `models/deepseek_v4/compressor.py` (6 hunks) | 6/6 matched | Committed 6 hunks / 1 file, exit 0 | `--status` 5/5 APPLIED; py_compile OK | **None** — `compressor.py` is not TP=3-patched | **Best candidate.** The compressor is shared by every attention class, so it is on our active path; at long context ~127/128 decode steps skip the C128 kernel. Pure compute saving, KV budget must be unchanged (`--after` asserts this). Apply on all 3 nodes, restart, run the 17x23 correctness check. |
| `hotfix-dsv4-mtp-buffer-50312.sh` (448 MiB GPU freed) | `models/deepseek_v4/nvidia/model.py` (2 hunks), `v1/worker/gpu/model_runner.py` (1 hunk, expect=2 sites) | 3/3 matched (runner hunk found both sites) | Committed 3 hunks / 2 files, exit 0 | `--status` 3/3 APPLIED; py_compile OK | **File overlaps, regions do not**: `nvidia/model.py` *is* TP=3-patched (expert assert + attn_sink window), but both hotfix anchors sit in untouched regions — they matched at expected counts with the TP=3 patch present, and TP=3 `--check` is unchanged after apply | **Real memory win here.** Our compose builds `--speculative-config {"method":"dspark",...}`, and `use_eagle()`/`uses_draft_model()` are both False for dspark, so the 256 MiB/rank buffer is genuinely freed and the per-step `copy_` removed. The two runner None-guards are the crash-safety part — do not apply the model.py hunks without them (the script applies all-or-nothing). Validate with `--before`/`--after`: KV blocks should INCREASE. |
| `hotfix-dsv4-skip-topk-49486.sh` (+#52492 guard; skip indexer topk on short contexts) | `models/deepseek_v4/attention.py` (3 hunks) | 3/3 matched | Committed 3 hunks / 1 file, exit 0 | `--status` 4/4 APPLIED incl. the #52492 CUDA-graph capture guard; py_compile OK | **File overlaps, regions do not**: `attention.py` carries 3 TP=3 edits (all in `DeepseekV4Attention.__init__` / `wo_a`/`wo_b` construction); the hotfix touches imports, a module-level Triton kernel, and `DeepseekV4Indexer.forward` — disjoint; anchors matched with TP=3 present; TP=3 `--check` unchanged after | Safe, but low priority for us: trigger window is prompts ≤ 2048 tokens (C4, topk 512 x ratio 4) and this cluster's raison d'être is long context. Apply opportunistically together with the others; the #52492 capture guard is already included (do not port an older revision without it). |

All four are idempotent, and applying all four back-to-back plus the issue-141
workaround produced zero anchor conflicts and a clean `py_compile` across every
edited file.

### Version check against MiaAI origin/main

`git fetch` + `git diff HEAD origin/main -- patches/` in the MiaAI checkout:
**zero diffs** on all four scripts (and on `hotfix-nvfp4-ds-mla-issue22.sh` and
`hotfix-dsv4-issue141-sparse-mla-decode-chunk.py`). Newer upstream patch activity
is unrelated: a new `hotfix-vllm-issue138-responses-history.py` and a 2-character
change to `hotfix-dsv4-issue27-partial-prefill-concurrency.py`.

### Rollout order, when the user decides to apply

1. `48957` (active path, pure compute) and `50312` (real memory) first — these
   are the two with expected measurable effect.
2. `49486` bundled in the same restart (harmless, small win on short prompts).
3. `50298` only with an explicit before/after A/B, since our default backend
   probably never executes the patched path.
4. Same restart discipline as always: apply on ALL THREE nodes, `--before`
   snapshot, restart by the user, `--after` diff, then the 17x23 correctness
   check and a short depth sweep per `docs/BENCHMARK-POLICY.md`.

---

## Issue #141 relevance — sparse-MLA verify-decode chunking

MiaAI issue #141: stochastic verify-decode stalls inside FlashInfer's SM120
sparse-MLA **paged/prefill orchestrator**. The pinned Anemll adapter sends every
verify row through one FlashInfer call; calls of ≤64 rows go to the standalone
DSv4 decode kernel, larger calls to the orchestrator implicated in the stalls.
Their `hotfix-dsv4-issue141-sparse-mla-decode-chunk.py` splits >64-row calls into
ordered ≤64-row slice views (workaround, opt-in, not a root-cause fix).

### (a) Does the patched code path exist verbatim in our image? YES

- `--status` in the ephemeral container:
  `issue141 sparse-MLA decode chunk : NOT APPLIED (compatible)` — this is a
  byte-exact source lock: the **entire** pinned
  `DeepseekV4FlashInferSM120Attention._forward_decode` method matches at
  `models/deepseek_v4/nvidia/flashinfer_sparse.py`, and all five FlashInfer
  guard fragments validate in `flashinfer/mla/_core.py` and
  `flashinfer/mla/_sparse_mla_sm120.py`.
- The 64-row boundary is present as pinned:
  - `_sparse_mla_sm120.py:74` — `_DECODE_MAX_TOKENS = 64` (dispatch predicate at
    lines 225/246, prefill routing at 629)
  - `_core.py:326` — `if num_tokens > 64: return None, None` (decode workspace
    cutoff)
- Applying it in the ephemeral container succeeded
  (`APPLIED + VERIFIED`) and the file still compiles.
- **This is our active decode path**: with no `--attention-backend` flag,
  `_select_dsv4_attn_cls` returns `DeepseekV4FlashInferSM120Attention` on
  capability-12 devices (GB10). The class the hotfix patches is the class we run.

### (b) Our verify-row exposure: EXCEEDS 64

Rows per verify call = decode tokens in the step =
`MAX_NUM_SEQS x (MTP_NUM_TOKENS + 1)` at full decode concurrency (the same
formula the compose file uses for `--max-cudagraph-capture-size`).

| Config | SEQS | MTP K | Max rows/verify | >64? |
|---|---|---|---|---|
| **Live production** (HANDOFF-2026-08-28; MTP=2 shipped via issue #32) | 32 | 2 | **96** | **yes** |
| `config/tp3.env.example` as written (stale, still says K=5) | 32 | 5 | 192 | yes |

The 64-row boundary is crossed as soon as **≥22 sequences** are in a decode step
at K=2 (22 x 3 = 66). Anything above modest concurrency routes every verify call
through the paged/prefill orchestrator implicated in the stalls.

(Aside found while checking this: `config/tp3.env.example` still carries
`MTP_NUM_TOKENS=5` and its "MTP=5 beats 4" comment; the live cluster shipped
K=2 on 2026-08-28. The example file should be updated separately.)

### (c) Conflict with TP=3 patch sites? NONE

The workaround touches `nvidia/flashinfer_sparse.py`, `flashinfer/mla/_core.py`,
and `flashinfer/mla/_sparse_mla_sm120.py` — none are TP=3-patched files.
`apply_tp3_patch.py --check` output was identical before and after applying it
in the ephemeral container, and `py_compile` passed.

### Verdict: EXPOSED (conditionally, by decode concurrency)

- The exact adapter method, dispatch predicate, and 64-row cutoff MiaAI pinned
  are byte-identical in our image, on our default attention class.
- Our production shape exceeds the 64-row boundary whenever ≥22 requests decode
  concurrently — routine at `MAX_NUM_SEQS=32`.
- The condition: #141 is a **stochastic TP=2 stall** on MiaAI's pair (one pair
  survived 3.1M tokens split-at-64 while 65 failed the first round; the second
  pair never repeated the A/B, and the mechanism is unknown). We run TP=3 and
  have not catalogued a matching stall signature; structural exposure is proven,
  live reproduction is not.
- Recommendation: keep the workaround **staged, not applied**. If a stochastic
  decode stall with the #141 signature appears at high concurrency, this is the
  first candidate. If applied, measure the chunked-call overhead at cc=32 first
  (sequential ≤64-row slices serialize what was one kernel launch).

---

## Issue #22 exemption is STALE — see `docs/ISSUE22-EXEMPTION-STALE.md`

The suspect dispatch line is confirmed present verbatim in our image at
`v1/attention/backends/mla/flashmla_sparse.py:880`:

```
use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
```

and MiaAI's `hotfix-nvfp4-ds-mla-issue22.sh --status` reports `NOT APPLIED` in
the ephemeral container. Our "does not reproduce" exemption was measured under
`max_model_len=460800` (decode flat 75–99 tok/s to 409,600 tokens); we now ship
`MAX_MODEL_LEN=1048576`, which reaches the 600K+ regime where MiaAI's ~1 tok/s
collapse bites. Re-measure with `scripts/rerun_issue22_deep_decode.sh` once the
engine is up, before deciding on the hotfix.
