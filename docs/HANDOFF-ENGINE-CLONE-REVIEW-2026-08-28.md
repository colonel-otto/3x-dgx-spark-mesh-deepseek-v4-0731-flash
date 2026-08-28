# Handoff: engine-clone review, 2026-08-28 — VERIFICATION REQUESTED

**Purpose of this document.** A prior agent session surveyed public DeepSeek-V4-Flash
engine clones for work bearing on our 3-node blockers and reported one significant
finding. This handoff exists so a **second agent can independently confirm or refute
every claim**. Treat nothing here as settled. Each claim below carries the exact command
that checks it.

**Author state:** repo `3spark-dsv4` at `acbb2fefece8433c024c97fdf9c1b0b8c3520686`.
Survey performed via `gh api` + WebFetch on 2026-08-28. No code was run on any Spark;
no configuration was changed. This is a literature review only.

---

## The headline claim to check

> **Our documented PP=3 "blocker one" — that MTP and PP are mutually exclusive by class
> hierarchy — is falsified by a public patch set. The draft model never needed
> `SupportsPP`.**

If this survives verification, `docs/PP3-PIPELINE-PARALLEL.md:132` is wrong and PP=3
becomes retestable. If it does not survive, our doc stands and no action follows.

---

## Verification anchors (pinned)

| Thing | Pinned value |
|---|---|
| Our repo HEAD | `acbb2fefece8433c024c97fdf9c1b0b8c3520686` |
| `docs/PP3-PIPELINE-PARALLEL.md` | sha256 `a040a68a8c96f1278572915696411f8dc798082508df32ba1c95602c166e9ad2`, 221 lines |
| `allover326/deepseek-v4-cmp170hx` HEAD | `3dd2d8817e7deae00d998edde0d227e7254ea71e` (pushed 2026-08-13T03:18:01Z, 96 stars) |
| Their base fork | `haosdent/vllm@dsv4-flash-a100`, commit `f8ea5bb` (README says later updated to `c3046d1`) |

Patch blob SHAs (`gh api repos/allover326/deepseek-v4-cmp170hx/contents/patches`):

| Blob SHA | Size | File |
|---|---|---|
| `05a771d74e272749909c1ce16bc7d983ac48d7a2` | 1486B | `0001-sparse_attn_indexer.patch` |
| `9e2458d32aeaad009c6dde46be4d50b0a5724a1e` | 885B | `0002-speculative.patch` |
| `cb6ec9c44ecac197e235fa62939a216ee56020c9` | 5272B | `0003-pp_utils.patch` |
| `97f15b7c944a64d2848482380d8fbcfeb9c608ac` | 3010B | `0004-model_runner.patch` |
| `531f7debc65d47c0f5dbf7cb1fde7d3e168ec688` | 5727B | `0005-dspark-utils.patch` |
| `6056e06b5ef203db24d1d86f274b29acf5277755` | 7201B | `0005a-prefill-topk-torch-fallback.patch` |
| `add877934e5e0be9773d7cfe7065d5f373e3005f` | 3991B | `0006-logits-row-chunk.patch` |

If a blob SHA differs, the upstream changed after this review — re-read before trusting.

---

## Claim-by-claim, with the command that checks it

### C1 — `0002-speculative.patch` sets the draft's PP size to 1 (CONFIRMED, verbatim)

```bash
gh api repos/allover326/deepseek-v4-cmp170hx/contents/patches/0002-speculative.patch \
  --jq '.content' | base64 -d
```

Expected, verbatim, added to `vllm/config/speculative.py` near line 1078:

```
+                if self.method == "dspark":
+                    # The DSpark draft is NOT pipelined: the model runner builds
+                    # it on the last PP rank only and it runs there whole, so it
+                    # is a pp_size=1 model regardless of the target's split.
+                    # Inheriting the target's pipeline_parallel_size would make
+                    # verify_with_parallel_config demand SupportsPP from the
+                    # draft architecture, which it neither implements nor needs.
+                    self.draft_parallel_config.pipeline_parallel_size = 1
```

**Confidence: high.** Read byte-for-byte from the blob, not summarized.

### C2 — This contradicts our PP3 doc (CONFIRMED locally)

```bash
sed -n '64,90p;130,134p' docs/PP3-PIPELINE-PARALLEL.md
```

Ours says (line 132): `MTP and PP are mutually exclusive in this runtime, by class
hierarchy.` and (line 79) tables `DeepSeekMTP` (draft) at `deepseek_mtp.py:223` as not
implementing `SupportsPP`. Both documents agree the draft lacks `SupportsPP`; they differ
on whether that is *dispositive*. Their claim is that vLLM asks the wrong question — the
draft is built on the last PP rank only and runs whole there, so `pp_size=1` is correct
for it and the interface check should never have fired.

**Confidence: high that the two documents conflict. UNVERIFIED that their reading is
correct** — nobody has run this on GB10. See "What is NOT verified".

### C3 — Three refusal sites, patched in `0002` / `0004` / `0005`

```bash
for p in 0002-speculative 0004-model_runner 0005-dspark-utils; do
  gh api repos/allover326/deepseek-v4-cmp170hx/contents/patches/$p.patch --jq '.content' | base64 -d
done
```

`0004` verbatim keeps the guard for other methods and exempts only dspark:

```
-                if self.use_pp:
+                if self.use_pp and self.speculative_config.method != "dspark":
                     raise ValueError(
                         f"{self.speculative_config.method} with pipeline parallel "
                         "is not supported."
```

`0005` drops a `NotImplementedError` in `vllm/v1/worker/gpu/spec_decode/dspark/utils.py`
and adds checkpoint-loading of the draft's token embedding, because under PP the target's
`embed_tokens` lives on rank 0 while the drafter runs on the last rank — so there is
nothing local to alias.

**Confidence: high** (read from blobs). The "three sites" framing is their
`patches/README.md` prose; the three code changes are directly visible.

### C4 — `0004` also carries two silent-corruption fixes

Verbatim comment from `0004`:

> Relay the proposed draft tokens to the non-last PP ranks so their next-step
> `combine_sampled_and_draft_tokens` reads real values instead of zero-init (otherwise
> acceptance ~= 0 and the output is garbage).

**Why this matters to us:** this is the same failure *class* as the TP=3 `o_groups` bug —
it runs, it looks healthy, and it emits nonsense. Any PP=3 attempt must be quality-gated,
not just throughput-gated. **Confidence: high** (verbatim).

### C5 — The layer-tap argument applies to our split (quote CONFIRMED; mapping is INFERENCE)

Verbatim from `0004`:

> for DeepSeek-V4 the aux-hidden-state taps (`dspark_target_layer_ids = [40,41,42]` of 43
> layers) plus `lm_head` all land on that same last rank. eagle3/dflash keep the guard --
> untested, and their aux layers are spread across ranks.

Our PP=3 split is `14,15,14` (`docs/PP3-PIPELINE-PARALLEL.md:111`), so layers 40–42 would
fall on rank 2. **The quoted comment is verified. The mapping onto our split is MY
INFERENCE from arithmetic (14+15=29, so rank 2 holds layers 29–42). A verifier should
confirm vLLM's actual layer-to-rank assignment rather than trust that arithmetic.**

### C6 — Their measured numbers (CONFIRMED from README, but NOT our hardware)

```bash
gh api repos/allover326/deepseek-v4-cmp170hx/contents/README.md --jq '.content' | base64 -d
```

- decode single stream: 50.8 → **98.1 tok/s**
- decode @ 64 concurrent: 472.0 → **712.8 tok/s**
- PP+spec worth **1.93×**; "keeps winning all the way to 64 concurrent"

**Hardware is 4× NVIDIA CMP 170HX — GA100 silicon, sm_80, VRAM-unlocked to 64 GB,
PCIe Gen2 x4, no P2P.** Not GB10, not our B12X image. **These numbers are not transferable
to us and must never enter `benchmarks/measurements.csv` except as
`source=external-published`.**

---

## Second repo: `vladimir-voinea/dspark-vllm-gb10`

Pinned: pushed `2026-06-28T19:59:44Z`, **0 stars**. GB10/DGX Spark — our hardware family.
Verified from its README:

- Single-user DSpark gains by workload: **2.20×** predictable text, **1.32×** JSON,
  **1.08×** novel coding. Their framing: DSpark's headline "60–85%" is a *concurrent-load*
  number; single-user collapses to acceptance-length gain. Independently echoes our
  `project_dsv4_tokens_depend_on_prompt` finding from different hardware.
- **Stale cross-attention window bug:** the proposer stored only the bonus's `main_kv` per
  step, skipping intermediate accepted tokens; at τ≈2.4 that is ~1.4 positions/step never
  written, so after ~50 steps the 128-slot window went >half stale and acceptance
  collapsed. Fix took τ **2.4 → 3.5** and coding from a **0.85× regression to ~1.1–1.2×**.
- int32 UE8M0 scale-unpack bug: 4 exponents packed little-endian, only 1 read; attention
  cosine **0.97 → 0.9997**.

**Actionable for us:** the staleness bug manifests as *decode degrading over long
generations*. No benchmark of ours measures that — our harness runs are short. Worth
checking whether our image carries it. **Note the 0 stars: unreviewed, unreplicated work.
Treat as a hypothesis to test, not a result to cite.**

---

## Corrections to what the prior session told the user verbally

Two errors, both caught while writing this up:

1. **"drowzeys' GLM-5.3 repo was updated today"** — the *name* was misreported. The real
   repo is `keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated`, pushed
   `2026-08-27T12:51:17Z`. The date was right; the earlier listing came from a summarizing
   WebFetch that garbled repo names. **Lesson for the verifier: the `?tab=repositories`
   WebFetch output proved unreliable — use `gh api users/<user>/repos` instead.**
2. **Freshness** — `allover326` was presented as if currently active; it was last pushed
   **2026-08-13**, two weeks stale.

---

## What is NOT verified (the honest list)

- **Nobody has run these patches on GB10.** Everything above is code reading.
- **They will not apply cleanly.** Base is `haosdent/vllm@dsv4-flash-a100`; ours is the
  Anemll `dspark-vllm-gx10` B12X image. Patches `0001`/`0005a` are sm_80 top-k fallbacks,
  **irrelevant to us**. Line offsets will not match.
- **Our blocker two is untouched.** `ValueError: Invalid state_cache.strides[0]`
  (`PP3-PIPELINE-PARALLEL.md:97,111,121`) is a DSA-compressor shape failure, independent of
  node count and layer partition. Their patches say nothing about it. Our own doc already
  names the **`--block-size` sweep** as the untested next step — that remains the gating
  item, and PP=3 will likely still fail there even if C1–C5 hold.
- **B12X interaction unknown.** Per `project_b12x_ep_incompatibility_root_cause`, B12X is
  vendored FlashInfer. Whether it tolerates PP is a separate question from whether vLLM's
  config layer permits it. (Our PP3 doc line 6 does state the MoE kernel survives PP.)
- **`0003-pp_utils.patch` was not read in full** — only its role (adds `broadcast_draft()`
  and token padding) via the patches README. Read it before porting.

---

## Recommended next step (not done, not authorized)

1. Amend `docs/PP3-PIPELINE-PARALLEL.md:132` — the class-hierarchy claim is falsified as
   *reasoning*, even though the observed error was real. Per
   `feedback_dont_declare_impossible_without_measuring`, it should not sit uncorrected.
2. Port `0002`/`0004`/`0005` onto our B12X image; skip `0001`/`0005a`; read `0003` first.
3. Retry PP=3 with MTP on, **expecting to hit `state_cache.strides[0]` next** — pair with
   the `--block-size` sweep.
4. Quality-gate any PP=3 run (see C4): acceptance rate + byte-comparison, not tok/s alone.

**No Spark was touched and no configuration changed in producing this document.**

---

## Addendum 2026-08-28: `tonyd2wild/Deepseek-V4-Flash-TP4-4x-DGX-Spark` — reviewed and DISMISSED

Audited at user request ("last time I checked tony's repo it had some invalid claims" —
confirmed). Pinned: pushed `2026-08-24T12:17:36Z`, 6 stars, **exactly two commits**
(`1433fe1` initial recipe, `2dd40b9` title-case rename). **No patches directory, no
engine code, no fork** — five docs plus a `docker run` launcher and an annotated NCCL
env file, all on the prebuilt image `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`.

**Why it needs no patches (and is therefore not a lead):** at TP=4 everything divides
evenly — 64 heads -> 16/rank, `o_groups` 8 -> 2/rank. The padding problem that makes our
TP=3 require the localaiguyy patch does not exist at degree 4.

**Invalid claims, with receipts:**

1. *"DeepSeek-V4-Flash has 128 attention heads"* (launcher header + README flag table).
   The model's `config.json` on HF says `num_attention_heads: 64`:
   `curl -s https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json | grep num_attention_heads`
2. *"TP=3 is NOT [valid] ... for 3 nodes you'd need pipeline-parallel"* — falsified by
   this repository's running TP=3 with the `o_groups` 8->9 padding patch (14/14 correct;
   see `CREDITS.md` and `patch.md`). Divisibility of the head count is not even the
   operative constraint.
3. Its own `TROUBLESHOOTING.md` §8 concedes *"'NVFP4' is a mirage — the weights are
   actually FP8."*

**Kept from it (data points, not code):**

- `VLLM_USE_B12X_MOE=1` at `--tensor-parallel-size 4` reportedly serves — extends the
  B12X finding's shape (TP scales where EP is architecturally blocked) to degree 4,
  on someone else's rig, unverified by us.
- The `drop_caches`-before-relaunch note (their §6): GB10 driver can hold ~100 GiB of
  unified memory after a crash; relaunching over the stale cache driver-OOMs at boot.
  Recorded in our `troubleshooting.md` as an unverified external report.
- Their config is untuned by our standards (MTP=2 vs our 5, `fp8` KV, `max_num_seqs=6`,
  gmu 0.78), so their ~120 tok/s count-prompt headline is not a comparison anchor.

**Verdict: reviewed-and-dismissed. Do not re-treat as a lead.**

---

## Addendum 2026-08-28: publication landscape for 3-Spark DSv4 (survey result)

Question: does HF or anywhere else publish a 3-Spark DSv4 recipe? **No.** Sweep of HF,
the official vLLM recipe, NVIDIA playbooks, and 40 GitHub repos sorted by recency:

- **Official vLLM recipe** (https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash):
  documents **2 nodes only** ("Use 2 nodes ... `--tensor-parallel-size 2`, one GB10 per
  node"). No TP=3, no 3-node anything.
- **NVIDIA `dgx-spark-playbooks`**: officially documents the 3-node **ring wiring**
  ("Connect Three DGX Spark in a Ring Topology") plus switch and NCCL playbooks — but
  **no model-serving guidance above 2 nodes**. NVIDIA blesses the wiring, not the deploy.
- **HuggingFace**: model-card discussions cover 1x and 2x Spark deploys;
  `0xSero/deepseek-v4-flash-0731-spark` is a single-device EXL3 quant. Nothing 3-node.
- **GitHub**: published node counts are 1, 2, and 4 (the TP4 repo above). The only
  public 3-Spark work is `localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark`
  (pushed 2026-08-09, 2 stars — already in `CREDITS.md`) and this repository.

**Implication:** the public record effectively claims TP=3 is impossible (the TP4 repo
says so outright); this repo holds a working counterexample. That gap is publishable.

---

## Sources

- https://github.com/allover326/deepseek-v4-cmp170hx (HEAD `3dd2d88`)
- https://github.com/vladimir-voinea/dspark-vllm-gb10
- https://github.com/haosdent/vllm/tree/dsv4-flash-a100 (their base, `f8ea5bb`)
- https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- https://github.com/drowzeys/vllm-gb10-spin-wait-fix
- https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m
- https://github.com/eugr/spark-vllm-docker
- https://github.com/yhfgyyf/vllm-deepseek-v4-sm89
- https://github.com/tonyd2wild/Deepseek-V4-Flash-TP4-4x-DGX-Spark (audited, dismissed)
- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash (official recipe, 2-node only)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json (head-count receipt)
- Already credited in `CREDITS.md`, not re-reviewed: MiaAI-Lab 2x, localaiguyy 3x,
  NVIDIA dgx-spark-playbooks, Anemll.
