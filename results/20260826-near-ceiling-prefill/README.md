# Near-ceiling prefill, single run — 2026-08-26

> ⚠ **PRELIMINARY — NEEDS MORE INVESTIGATION. Do not cite these tok/s numbers as
> settled, and do not use them to revisit the §2/§4a prefill-parity conclusions.**
> This was an incidental byproduct of an unrelated Claude-assisted debugging session
> (fixing an SSE-chunk-undercounting bug in a throwaway benchmark script), not a
> designed experiment. Kept because the depth reached (~967K tokens, essentially at
> the model's context ceiling) is untested anywhere else in this repo, and because
> one config value observed live contradicts an existing "falsified" finding (see
> "Needs investigation" below) — not because the throughput numbers themselves are
> trustworthy.

## What was measured

Three single-shot streaming chat-completion requests against the live production
endpoint (`localhost:8100`, unmodified, no other traffic), `max_tokens=1`, timing
client-side TTFT and reading the server-side `vllm:prompt_tokens_total` counter
delta for actual prompt size. TP=3, current production config (see full flags
below).

| requested depth | **actual prompt tokens** | TTFT | prefill tok/s |
|---:|---:|---:|---:|
| 8,192 | 81,124 | 41.92 s | 1,935.0 |
| 32,768 | 317,904 | 231.51 s | 1,373.2 |
| 100,000 | **967,286** | 1,380.93 s (~23 min) | 700.5 |

The declining tok/s trend across depth is the only thing arguably interesting here,
and it should not be trusted — see limitations.

## Why the actual depths are wildly off target

The "fresh, no-cache" content was generated as `" ".join(uuid.uuid4().hex for _ in
range(...))` — a naive attempt to defeat vLLM's prefix cache with unique content,
sized by a rough chars-per-token estimate taken from *natural-language* text
earlier in the same session. Random hex strings tokenize far less efficiently
than natural language (more tokens per character), so every target was blown out
10-20x. This was not caught before the run because it was launched as a
long-running background task, and the wall-clock cost (23 minutes for the last
row) discouraged a fast iterate-and-fix cycle.

**This is not this repo's `benchmark_prefill.py` token-ID harness.** That harness
(`results/20260825-fabric-fix/harness/benchmark_prefill.py`) generates prompts by
deterministic token ID, hits an exact target length, embeds a `token_pool_sha256`
for reproducibility, and is what every other prefill number in this repo is
measured with. This run used none of that. Treat any comparison to §2/§4a numbers
as invalid.

## Other confounds not controlled for

- **Single run per depth**, not median-of-3 like every other measurement in this
  repo. No way to distinguish signal from a one-off scheduling hiccup, thermal
  state, or JIT compile (§7.2 in `HANDOFF.md` — an uncaught `jit_monitor` compile
  mid-request has previously cost 40%+ on a sweep).
- **Content is meaningless hex, not the token-ID pool** the production tokenizer
  and any downstream reasoning/tool-call parsing would normally see. Unclear if
  DeepSeek-V4's chunked-prefill scheduler behaves identically on pathological
  content vs. natural-language token distributions at this scale.
- **No fabric-gate check was run first** (`make gate-full`, per `HANDOFF.md` §6) —
  the single most-repeated lesson in this repo's own measurement discipline
  (§7.1) is "check the fabric before anything else," and that was skipped here.
- **Approaching the true context ceiling** (`max_model_len=1,048,576`) at
  967,286 tokens — KV-cache pressure, eviction behavior, or scheduling near the
  ceiling could plausibly explain the falling tok/s independent of anything
  communication-related. Not investigated.

## Needs investigation: live `NCCL_IB_HCA` contradicts §4b's falsified finding

While capturing the exact config for this write-up, the **currently running
production container** (started 2026-08-26T20:26:22Z, same container that served
the 967K-token request above without incident) has:

```
NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
```

`HANDOFF.md` §4b says explicitly: *"Do not set `NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,
rocep1s0f1,roceP2p1s0f1`. It wedges the cluster. Tried, failed, rolled back."* — and
gives a specific failure signature (`IBV_WC_RETRY_EXC_ERR` on link-local `roceP2p`
GIDs). Yet this is the live value, and the cluster served a 23-minute, near-
context-ceiling request cleanly with it. Possibilities, not investigated here:

1. §4b's finding predates the fabric fix and no longer applies (the `roceP2p` pair
   may since have gotten the "stable IPv4 addressing and routing first" treatment
   §4b calls out as the correct fix direction) — in which case §4b itself is now
   stale, same shape as the §4a/#18 correction.
2. The wedge in §4b is intermittent/load-dependent and we simply didn't hit it.
3. Something else changed between when §4b was written and now.

**This should be resolved before anyone trusts either doc.** If (1), §4b needs the
same forward-pointer treatment §4a just got. If (2), that's a live production risk
worth knowing about independent of any throughput question.

## Full config at time of test

**Image:** `ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
(tag `0.1.1`) · **vLLM:** `0.25.2.dev0+g752a3a504.d20260714` · **Container started:**
2026-08-26T20:26:22Z

**vLLM launch flags (rank 0):**

```
vllm serve /models/dsv4-abliterated --served-model-name deepseek-v4-flash-0731 \
  --host 0.0.0.0 --port 8100 --trust-remote-code \
  --tensor-parallel-size 3 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
  --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 192 --gpu-memory-utilization 0.80 \
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --moe-backend flashinfer_b12x --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":false}' \
  --generation-config vllm --enable-flashinfer-autotune \
  --nnodes 3 --node-rank 0 --master-addr 192.168.200.1 --master-port 25000 \
  --jit-monitor-mode warn
```

**Full container environment** (all vars, alphabetized by topic where relevant;
NCCL/DSpark/B12X-relevant ones first):

```
NCCL_CUMEM_ENABLE=0
NCCL_IB_ROCE_VERSION_NUM=2
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=0
NCCL_NET_PLUGIN=none
NCCL_CROSS_NIC=1
NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_ADDR_RANGE=
NCCL_IB_GID_INDEX=
NCCL_NET=IB
GLOO_SOCKET_IFNAME=enp1s0f0np0
TP_SOCKET_IFNAME=enp1s0f0np0
NODE_RANK=0
MASTER_ADDR=192.168.200.1
MASTER_PORT=25000
VLLM_HOST_IP=192.168.200.1
MTP_NUM_TOKENS=5
HEADLESS=

VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0
VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0
VLLM_DSV4_B12X_COMPRESSED_MLA=0
VLLM_TRITON_MLA_SPARSE=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_B12X_W4A16_FORCE_TILE_CONFIG=
VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16
VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0
B12X_W4A16_TC_DECODE=0
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_TRITON_MLA_SPARSE=1
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
VLLM_SKIP_INIT_MEMORY_CHECK=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
DSPARK_SLOT_CLAMP=1
DG_JIT_USE_NVRTC=0
DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
TILELANG_CLEANUP_TEMP_FILES=1
FLASHINFER_DISABLE_VERSION_CHECK=1
FLASHINFER_WORKSPACE_BASE=/cache/huggingface/flashinfer
FLASHINFER_CUDA_ARCH_LIST=12.1a
TORCH_CUDA_ARCH_LIST=12.1a
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_HUB_DISABLE_XET=1
TRANSFORMERS_OFFLINE=0
HF_HUB_OFFLINE=0
HF_HOME=/cache/huggingface
VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
VLLM_ENABLE_CUDA_COMPATIBILITY=0
VLLM_USAGE_SOURCE=production-docker-image
VLLM_BUILD_COMMIT=unknown
VLLM_BUILD_PIPELINE=local
VLLM_BUILD_URL=
VLLM_IMAGE_TAG=local/vllm-openai:dev
```

(Standard CUDA/driver/UV/path env vars omitted — unchanged boilerplate from the
base image, not config-relevant.)

## Harness

Ad hoc, not committed anywhere durable before this write-up:
`sparkmain:/tmp/bench_compare.py` (the corrected version, post the SSE-undercount
fix from the same session — see git history / conversation log if needed, not
itself worth preserving verbatim). If this line of investigation continues, use
the existing `benchmark_prefill.py` token-ID harness instead so results are
directly comparable to §2/§4a.

## Suggested next steps, if anyone picks this up

1. Re-run at true 8K/32K/100K/500K/~1M depths using the actual
   `benchmark_prefill.py` harness (deterministic token IDs, exact-length prompts),
   median-of-3, with `make gate-full` run immediately before.
2. Resolve the `NCCL_IB_HCA` contradiction against §4b before trusting any fabric-
   dependent number from the current live config.
3. If the declining-tok/s-with-depth shape survives a proper re-run, check KV
   cache occupancy and scheduler behavior near `max_model_len`, not just
   communication — this config's KV cache is sized for `MAX_NUM_SEQS=32` at up to
   1,048,576 tokens each, and 967K tokens from one request is a very different
   utilization pattern than anything else measured in this repo.
