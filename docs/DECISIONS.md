# Settled decisions

One row per knob: the value, the measurement that settled it, and what happens if you
change it. If a value is not here, it is not settled.

**Authoritative config:** [`../config/tp3.env.example`](../config/tp3.env.example).
This page is the *reasoning*; that file is the *artifact*.

---

## Parallelism

| Knob | Value | Settled by | If you change it |
|---|---|---|---|
| `TP_SIZE` | **3** | Decode cc=1 76.2 → 89.1 tok/s vs 2 nodes (2026-08-25, healthy fabric, matched) | cc≥16 favours 2 nodes; see the crossover below |
| `PP_SIZE` | **1** | [`PP3-PIPELINE-PARALLEL.md`](PP3-PIPELINE-PARALLEL.md) | ❌ Blocked by MTP + a DSA stride constraint. No PP tok/s exists |
| expert parallel | **off** | [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) | ❌ 2.5x slower — the B12X kernel refuses EP. Also blocks EPLB, whose prerequisite is EP |
| TP=3 padding patch | **required** | Correctness 14/14; [`patch.md`](patch.md) | ☠️ **Silently serves fluent nonsense.** Stock vLLM computes `8 // 3 == 2` and drops 6 of 8 attention groups |

**The node-count answer is conditional.** The 3-node advantage decays monotonically with
concurrency and crosses over near cc=16: three nodes win per-stream latency, two win batch
aggregate. Single-user interactive coding is per-stream-bound → three nodes.

## Engine shape

| Knob | Value | Settled by | If you change it |
|---|---|---|---|
| `MAX_MODEL_LEN` | **1048576** | 1M is free: memory-bound, not comms-bound | Nothing gained by lowering |
| `MAX_NUM_SEQS` | **16** | Sweep; `32` rejected — but against a budget now known 6.6x too small ([#10](../../issues/10)) | Worth re-testing |
| `GPU_MEMORY_UTILIZATION` | **0.80** | KV pool 4.46M tokens, **0 preemptions in every test ever run** | 0.85 leaves 2–4 GB free per node |
| `MTP_NUM_TOKENS` | **5** | Matched control 2026-08-24, beats 4 | `0` is **invalid** (vLLM rejects it); `1` collapses decode to ~47 tok/s |
| `MAX_NUM_BATCHED_TOKENS` | **8192** | A/B: 16384 cost **43% of the KV pool for zero gain** | ⚠️ vLLM's own log suggests 16384. That advice assumes intra-node NVLink. It is a trap here |
| `--kv-cache-dtype` | `nvfp4_ds_mla` | Speed-identical to `fp8_ds_mla` | ⚠️ **Open** — memory-identical too (shared 584-byte envelope), quality unvalidated ([#16](../../issues/16)) |
| `JIT_MONITOR_MODE` | **warn** | Surfaces compiles landing inside requests — one measured at 5 s | Leave on. It is how you know a benchmark is contaminated |

**MTP is quality-neutral.** Speculative decoding is lossless by construction; it is a
speed knob only. Raising it cannot buy accuracy.

## Fabric

| Knob | Value | Settled by | If you change it |
|---|---|---|---|
| `NCCL_IB_HCA` | **all four** (`rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`) | Upper mesh addressed 2026-08-25: **2.0x** bandwidth, live gate 21/21 with `rdma:*` **0 errors** | ⚠️ Prerequisite is IPv4 + routing + persistence on the upper pair. Without it NCCL picks the pair anyway and wedges under load while every container stays `running`. Soak pending — [#17](../../issues/17) |
| `NCCL_NET` | `IB` | — | ⚠️ A **request, not a guarantee.** On failure NCCL falls back to sockets and reports a plausible number. We measured `NET/Socket` at 0.44 GB/s and it looked real. Always confirm `via NET/IB/x` |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | Required on a switchless ring | Undocumented in NVIDIA's public env reference, but present in the NCCL 2.30.7 binary |
| subnet masks | **`/30` on all six** | Consistency | Mixed masks on a fabric are a latent trap even when they cannot overlap |
| MTU | **9000** | Persisted via netplan | ⚠️ netplan **owns** the config; NetworkManager is only a renderer |

## Measured constants

| Quantity | Value | Note |
|---|---|---|
| Healthy pair busbw @64MiB | **~4.6 GB/s** | ~0.7 means a degraded node — reboot it |
| Healthy 3-rank busbw | **2.85–3.25 GB/s** | Two HCAs. Supersedes the 0.49 figure, which was 6.6x pessimistic |
| 3-rank busbw, **four HCAs** | **5.80 GB/s** | Upper mesh addressed. Validated under a live engine, `rdma:*` clean. **This is allgather busbw** — see the note below |
| KV envelope, DeepSeek-V4 | **584 B/token** | 448 NoPE + 128 RoPE + 8 fp8 scale. **Identical for `fp8_ds_mla` and `nvfp4_ds_mla`** |
| Tokens per word, filler prompt | **1.2056** | Measured against `/tokenize`, flat 150K–240K. **Do not estimate this** |
| Idle TTFT penalty | **~22 ms** | Why a keep-alive ping is not worth it |
| New-shape JIT spike | **5–8 s, on request 2** | Per-shape, not per-idle-period. Warm at startup |

---

## Things that cannot be done — do not re-propose

| Idea | Why not |
|---|---|
| Pick which experts live on which node | No static expert→rank map exists in vLLM. EPLB requires EP (2.5x slower), and **under TP there is no imbalance to fix** — TP shards every expert across all ranks |
| BF16 weights for quality | ~570–610 GiB against 363 GiB available. And the model is QAT — upcasting yields bit-identical values in a wider container |
| Raise MTP for quality | Speculative decoding is lossless. Speed knob only |
| Lower concurrency for accuracy | Batch size changes **reproducibility**, not average quality. `VLLM_BATCH_INVARIANT=1` is the real fix but does not support MTP yet |
| KV offload to use spare RAM | Extends the **prefix cache**, not single-request capacity. We have 4x headroom and have never preempted |
| Keep-alive ping for TTFT | Buys ~22 ms and adds a permanent load source that contaminates benchmarks |

---

## A note on comparing bandwidth numbers

Our figures are **`all_gather` busbw**: `nbytes * (world-1)/world / dt`, where `nbytes` is
the **per-rank input**. That convention matters enormously when comparing against a
published number, because the same physical wire speed can be quoted three ways:

| convention | world=2 | world=3 |
|---|---:|---:|
| `all_gather` **busbw** (what we report) | 9.70 | 5.80 |
| `all_gather` **algbw** (`= busbw * w/(w-1)`) | **19.40** | 8.70 |
| `all_reduce` **busbw** (`= 2x` allgather busbw) | **19.40** | 11.60 |

A **3.2x spread from bookkeeping alone.** Always state collective, message size, rank
count, and algbw-vs-busbw before comparing.
