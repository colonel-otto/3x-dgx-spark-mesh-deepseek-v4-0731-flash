# Configuration decisions

One row per operational knob: the current value, the evidence behind it, and what happens
if it changes. This is not a claim that every strategic question is settled; notably, the
performance case for two versus three nodes remains open.

> [!IMPORTANT]
> The four-HCA **throughput** question ([#17](../../issues/17)) was settled 2026-08-26:
> **there is none.** Decode is flat against a matched 2-HCA arm. Four-HCA is kept for
> redundancy and headroom, which is a different justification than the one it was
> originally adopted under.
>
> ⚠️ marks a row whose *value* is settled but which carries an operational trap — read
> the "What breaks" column before changing it.
>
> **`MAX_NUM_SEQS` moved 16 → 32 on 2026-08-26** ([#10](../../issues/10)) — the first
> value in this table changed by a retest rather than a first measurement. The prior
> `_ALLGATHER_BASE` timeout is retained as a degraded-fabric signature in the
> [`degraded-data catalogue`](DEGRADED-DATA-CATALOGUE.md), with its raw bundle preserved.

**Authoritative config:** [`../config/tp3.env.example`](../config/tp3.env.example).
This page is the *reasoning*; that file is the *artifact*.

---

## Parallelism

| Knob | Value | Settled by | If you change it |
|---|---|---|---|
| `TP_SIZE` | **3** operational default | TP=3 passes correctness and quality through 131K, retains the B12X kernel and MTP, and is the currently deployed shape. **Its performance advantage over TP=2 is not settled.** | TP=2 frees one node and is the required comparison arm; use the cluster launcher and re-run the same corrected harness |
| `PP_SIZE` | **1** | [`PP3-PIPELINE-PARALLEL.md`](PP3-PIPELINE-PARALLEL.md) | ❌ Blocked by MTP + a DSA stride constraint. No PP tok/s exists |
| expert parallel | **off** | [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) | ❌ ~2.5x slower — the B12X kernel refuses EP by an explicit source-code check. Also blocks EPLB, whose prerequisite is EP. ⚠️ The **mechanism** is settled (source check + `ValueError`); the **2.5x** came from a degraded-fabric run over TCP fallback and has never been re-measured on RDMA |
| TP=3 padding patch | **required** | Correctness 14/14; [`patch.md`](patch.md) | ☠️ **Silently serves fluent nonsense.** Stock vLLM computes `8 // 3 == 2` and drops 6 of 8 attention groups |

**The node-count performance answer is open.** The former depth comparison returned only
25–26 completion tokens per request and is now
[`VOID-25-token-window`](../results/20260826-decode-depth-2v3/). The corrected TP=3 arm
measures 50.7–56.0 tok/s from 2K–262K, but there is no corrected TP=2 arm. Do not infer a
winner until that matched run exists. The short-prompt concurrency result remains
supporting evidence only because its gate artifact and one headline cell are not committed.

## Engine shape

| Knob | Value | Settled by | If you change it |
|---|---|---|---|
| `MAX_MODEL_LEN` | **1048576** | 1M is free: memory-bound, not comms-bound | Nothing gained by lowering |
| `MAX_NUM_SEQS` | **32** | Retest 2026-08-26 on healthy fabric: **+46.3% at cc=32** (685.9 vs 468.8), parity below it, 70/70 requests OK, zero crash signatures. Supersedes the 08-24 rejection, which died on an `_ALLGATHER_BASE` timeout at a 6.6x-too-small budget with a 600 s watchdog ([#10](../../issues/10)) | Costs 0.92 GiB more graph capture and 1.8% of KV. Nothing below cc=32 |
| `GPU_MEMORY_UTILIZATION` | **0.835** | Winning Profile B ([#25](../../issues/25)): expands KV pool to ~2.49M tokens with 0 preemptions, zero OOMs, and improves starvation TTFT by -10.7% | 0.85 leaves minimal headroom; 0.80 leaves 35% KV capacity on the table |
| `MTP_NUM_TOKENS` | **5** | Matched control 2026-08-24, beats 4 | `0` is **invalid** (vLLM rejects it); `1` collapses decode to ~47 tok/s |
| `MAX_NUM_BATCHED_TOKENS` | **8192** | Evaluated in Issue [#28](../../issues/28) ([`20260827-issue28-speed-bt16384`](../results/20260827-issue28-speed-bt16384/)): 16384 yields +11.5% decode at 262K (51.4 tok/s), but **degrades deep TTFT by +22% at 131K (92.5s vs 75.7s)** and **+29% at 262K (228.6s vs 177.3s)** due to 235 MB activation tensor saturation on the GB10 unified memory bus (273 GB/s). 8192 is the measured sweet spot | ⚠️ vLLM's log suggests 16384. That advice assumes intra-node NVLink. On unified memory and multi-node RoCE it slows prefill |
| `--kv-cache-dtype` | `nvfp4_ds_mla` | Tested against `fp8_ds_mla`: speed and memory equivalent; 23/24 matched quality cells byte-identical | No demonstrated benefit from changing; see [`20260826-kv-dtype-ab`](../results/20260826-kv-dtype-ab/) |
| `JIT_MONITOR_MODE` | **warn** | Surfaces compiles landing inside requests — one measured at 5 s | Leave on. It is how you know a benchmark is contaminated |

**MTP is quality-neutral.** Speculative decoding is lossless by construction; it is a
speed knob only. Raising it cannot buy accuracy.

## Fabric

| Knob | Value | Settled by | If you change it |
|---|---|---|---|
| `NCCL_IB_HCA` | **all four** (`rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`) | Upper mesh addressed 2026-08-25: **2.0x** bandwidth, live gate 21/21 with `rdma:*` **0 errors** | ⚠️ Prerequisite is IPv4 + routing + persistence on the upper pair. Without it NCCL picks the pair anyway and wedges under load while every container stays `running`. **Soak PASSED** 2026-08-26 (408 req, 0 RDMA deltas, 0 log events) — [#17](../../issues/17). **Throughput benefit measured 2026-08-26: none.** Decode is flat cc=1–16 against a matched 2-HCA arm; every apparent gain sits inside the other arm's spread. Kept for redundancy and headroom, not speed — [`../results/20260826-four-hca-throughput/`](../results/20260826-four-hca-throughput) |
| `NCCL_NET` | `IB` | — | ⚠️ A **request, not a guarantee.** On failure NCCL falls back to sockets and reports a plausible number. We measured `NET/Socket` at 0.44 GB/s and it looked real. Always confirm `via NET/IB/x` |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | Required on a switchless ring | Undocumented in NVIDIA's public env reference, but present in the NCCL 2.30.7 binary |
| subnet masks | **`/30` on all six** | Consistency | Mixed masks on a fabric are a latent trap even when they cannot overlap |
| MTU | **9000** | Persisted via netplan | ⚠️ netplan **owns** the config; NetworkManager is only a renderer |
| **peer egress (RDMA/model data)** | **fabric only** | Fabric RTT 0.47-0.93 ms vs Wi-Fi 3-135 ms | ☠️ **No RDMA or model-data traffic over Wi-Fi, ever.** A missing fabric route falls back to Wi-Fi silently — everything still pings. Gated as `egress:*` ([#13](../../issues/13)) |
| **bootstrap / control plane** | shared management interface **permitted** | NVIDIA's launcher uses one common non-fabric interface on every node, and **explicitly supports Wi-Fi** for it | Rendezvous is a few KB, once, at startup. It is not the data path — payload still moves over `NCCL_IB_HCA`. **Confirm `via NET/IB/*`, never `NET/Socket`,** or data has fallen onto TCP and the run is void |

## Measured constants

| Quantity | Value | Note |
|---|---|---|
| Healthy pair busbw @64MiB | **~4.6 GB/s** (2 HCA) / **~9.7** (4 HCA) | ~0.7 means a degraded node — reboot it |
| 3-rank busbw, **official harness** | **23.92 GB/s** @16GiB | The real number. Exceeds the 20.84 published reference |
| 3-rank busbw, custom harness | 5.80 GB/s | ⚠️ **Do not quote as fabric speed.** Workload-shaped harness at 67 MB; official binary reads 23.92 on the same config |
| KV envelope, DeepSeek-V4 | **584 B/token** | 448 NoPE + 128 RoPE + 8 fp8 scale. **Identical for `fp8_ds_mla` and `nvfp4_ds_mla`** |
| Tokens per word, filler prompt | **1.2056** | Measured against `/tokenize`, flat 150K–240K. **Do not estimate this** |
| Idle TTFT penalty | **~22 ms** | Why a keep-alive ping is not worth it |
| New-shape JIT spike | **5–8 s, on request 2** | Per-shape, not per-idle-period. Warm at startup |
| `roceP2p` sysfs error counters | **192 / 64 / 32 / 128 / 96**, frozen | ⚠️ **Pre-existing residue, not a live fault.** Cumulative since boot, left by the earlier failed enable. Verified frozen under active load 2026-08-25. **Judge these by DELTA, never absolute value.** The lower `rocep1s0f*` pair reads zero |

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

## Bandwidth — SETTLED 2026-08-26

Our figures use the [nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
AllGather definition, `busbw = algbw * (n-1)/n` — the same convention published DGX Spark
figures use. Always state **collective, message size, rank count, and algbw-vs-busbw**
when quoting a number.

**There is no fabric deficit.** Measured with official `all_gather_perf`, NCCL 2.30.7,
`-n 20`, engine stopped:

| | 32 MiB | 16 GiB |
|---|---:|---:|
| **ours, 3-rank** | 16.85 | **23.92 GB/s** |
| published reference | 18.70 | 20.84 |

**We exceed the reference at 16 GiB.** The earlier 5.80 GB/s figure came from a
workload-shaped custom harness used outside its purpose — not from the hardware.

Bootstrap topology, NIC merging and HCA discovery each moved the number by **<0.5%**;
all three hypotheses are falsified. Full account:
[`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md).

**Ceiling: ~24 GB/s, confirmed.** Both 200G ports share two PCIe Gen5 x4 lanes
(~252 Gb/s); every 3-rank variant pins to it. Line-rate arithmetic (400 Gb/s → 48.5 GB/s)
**overstates it** and must not be used.
