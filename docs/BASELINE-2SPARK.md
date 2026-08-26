# 2-Spark Baseline — 2026-08-21

> [!WARNING]
> **The throughput numbers on this page are degraded-fabric signatures, not a clean
> 2-node reference — and this page previously carried no warning at all.** It is a
> **2-node** measurement, so the "spark1 was in the 3-node arm" reasoning that softens the
> other degraded pages does **not** apply here. It is worse: **spark1 was the only worker**.
>
> **The evidence chain, each link from a file in this repo:**
>
> | Step | Evidence |
> |---|---|
> | This run used nodes `node0` ↔ `node1` over `192.168.100.1 ↔ .2` | [`../results/20260821T001024Z-2spark-baseline/environment/`](../results/20260821T001024Z-2spark-baseline/environment/) — `ip` sections; and the Fabric table below marks it the **active TP/NCCL path** |
> | `node1` is `spark1` | [`HANDOFF.md`](HANDOFF.md) §1 node table: rank 1 = `spark1` (`node1`), "**the node that had degraded fabric**" |
> | That exact link measured **0.69 GB/s** against 4.6 healthy | [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) §2 — `sparkmain <-> spark1`, both cables |
> | The 2-node arm was measurably slowed, not just theoretically | TP=2 prefill **1,105–1,112 tok/s degraded** → **1,913–2,081 healthy**, same harness ([`../results/20260825-prefill-2v3/`](../results/20260825-prefill-2v3/)) |
>
> **So: yes, affected — 100% of this run's inter-node traffic crossed the degraded link.**
> That last row is the load-bearing one: it is a matched before/after on the 2-node
> configuration itself, so this is a measurement, not an inference.
>
> **The one thing we cannot establish: when the degradation began.** Root cause was never
> determined and the node "had been up a long time"
> ([`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) §2,
> [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md) §6.3). There is no pre-08-21
> fabric measurement in this repo, so **we cannot prove spark1 was already degraded on
> 2026-08-21** — only that it was by 08-24, on the same link, with no intervening reboot
> recorded. Treat these numbers as degraded; do not claim a specific onset date.
>
> **If your own 2-node reproduction lands near ~49–55 tok/s single-stream or ~1,100 tok/s
> prefill, check your fabric before concluding anything** — those are the degraded
> signatures. Diagnostic flow:
> [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md).
>
> **What survives:** everything that is not a timer.
> - **KV accounting** — 19.52 GiB / 1,771,152 tokens / 11,834 B per token / "KV is
>   partitioned across nodes, not replicated". Engine-reported capacity, transport-independent.
> - **Correctness** — 0 request failures, 0 needle failures, 100% retrieval at every
>   context and concurrency.
> - **The V2 Model Runner / `VLLM_PP_LAYER_PARTITION` source findings** in "Interpreting"
>   — code facts.
> - **The environment notes** (`dash` vs `bash`, CRLF, `SSH_USER_MAP`) — reproduction
>   mechanics.
>
> **What is void:** every tok/s and every latency in the two results tables — decode,
> aggregate, TTFT and e2e alike, at all concurrencies.
>
> **Healthy 2-node numbers instead:** decode cc=1 **76.2 tok/s**, cc=16 **481.3**
> ([`../results/20260825-decode-2v3/`](../results/20260825-decode-2v3/)); prefill
> **1,913 / 2,081 / 2,066** at 1K/8K/32K
> ([`../results/20260825-prefill-2v3/`](../results/20260825-prefill-2v3/)). Note those use
> the settled 1M/seqs=16/MTP=5/0.80 profile, not this page's 460800/0.85 profile, so they
> supersede rather than correct these figures.

Frozen `2spark-baseline` measurement of the working 2-node DeepSeek-V4-Flash-0731
deployment, captured with this repo's harness before any 3-node change.

Raw data: [`results/20260821T001024Z-2spark-baseline/`](../results/20260821T001024Z-2spark-baseline/)
(`benchmark.jsonl`, `manifest.env`, per-node `environment/`).

## Deployment under test

| Item | Value |
|---|---|
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| Model runner | **V2 Model Runner** |
| Parallelism | TP=2, PP=1, nnodes=2 |
| Model | 43 hidden layers, 256 routed experts, top-6, FP8 E4M3/UE8M0 128x128, 156 GB on disk |
| Context | `max-model-len` 460800 |
| Batching | `max-num-seqs` 16, `max-num-batched-tokens` 8192 |
| Memory | `gpu-memory-utilization` 0.85 |
| KV | `kv-cache-dtype nvfp4_ds_mla`, `block-size` 256 |
| Speculative | DSpark MTP, `num_speculative_tokens` 5 |
| Scheduling | async ON, chunked prefill ON, prefix caching ON |

**Note:** this baseline runs *with* speculative decoding enabled, because that is the
working production configuration. The PR's candidate disables it. That is a deliberate
and material difference — see "Interpreting" below.

## Fabric

All three CX-7 links verified **UP at 200000 Mb/s**, full triangle, RoCE devices
`rocep1s0f0` / `rocep1s0f1` present on each node.

| Link | Subnet | Endpoints |
|---|---|---|
| sparkmain <-> spark1 | `192.168.100.0/24` | `.1` <-> `.2` — **active TP/NCCL path** |
| sparkmain <-> spark3 | `192.168.101.0/30` | `.1` <-> `.2` |
| spark1 <-> spark3 | `192.168.102.0/30` | `.1` <-> `.2` |

The 3-node cabling is already in place; only software configuration is missing.

## Harness results (90 requests, `MAX_TOKENS=256`) — degraded-fabric signatures

> [!CAUTION]
> **Every timing column below is a degraded-fabric number.** TTFT, e2e, decode tok/s and
> aggregate tok/s all crossed the 0.69 GB/s `sparkmain ↔ spark1` link. The **needle
> column is not** — correctness is transport-independent, and 100% at every cell is a real
> result.

**0 request failures, 0 needle failures — 100% retrieval at every context and concurrency.**

| ctx | C | n | needle | TTFT (s) | e2e (s) | decode tok/s | aggregate tok/s | prompt tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1 | 3 | 100% | 0.807 | 0.98 | 53.61 | 10.17 | 2077 |
| 2048 | 3 | 9 | 100% | 0.633 | 0.86 | 38.87 | 34.64 | 2077 |
| 2048 | 6 | 18 | 100% | 0.975 | 1.27 | 30.11 | 47.13 | 2077 |
| 8192 | 1 | 3 | 100% | 3.200 | 3.34 | 65.33 | 3.00 | 8093 |
| 8192 | 3 | 9 | 100% | 0.480 | 0.69 | 38.81 | 41.03 | 8093 |
| 8192 | 6 | 18 | 100% | 0.912 | 1.11 | 26.72 | 47.96 | 8093 |
| 32768 | 1 | 3 | 100% | 12.378 | 12.56 | 50.86 | 0.80 | 32156 |
| 32768 | 3 | 9 | 100% | 0.753 | 1.01 | 33.62 | 29.49 | 32156 |
| 32768 | 6 | 18 | 100% | 0.842 | 1.54 | 12.84 | 33.75 | 32156 |

### Reading these numbers carefully

- **`aggregate tok/s` is not a throughput ceiling here.** The needle task answers in
  ~10 output tokens, so a wave finishes almost entirely in prefill. The C=1 rows are
  the extreme case: at ctx=32768, 0.80 aggregate tok/s reflects a 12.4 s prefill for a
  10-token answer, not slow decoding. Use `decode tok/s` for generation speed and treat
  `aggregate` as valid only for **candidate-vs-baseline comparison at identical settings**.
- **TTFT at C=1 is cold-ish per wave; C=3/C=6 benefit from prefix caching** (the shared
  instruction preamble is reused), which is why TTFT *drops* as concurrency rises.
- **decode tok/s falls with concurrency** (53.6 -> 30.1 at ctx=2048) — expected on a
  memory-bandwidth-bound GB10.

## Generation-focused supplement (long outputs, 256 tokens each)

The needle task under-measures sustained decode, so this was measured separately against
the same endpoint with a short prompt and a full 256-token generation:

| Concurrency | Wall (s) | Total tokens | Aggregate tok/s | Per-stream tok/s | TTFT p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.69 | 256 | 54.53 | 56.92 | 214 ms |
| 3 | 9.26 | 768 | 82.90 | 30.51 | 223 ms |
| 6 | 14.52 | 1536 | 105.80 | 21.03 | 268 ms |
| 12 | 21.87 | 3072 | 140.48 | 14.69 | 312 ms |
| 16 | 23.53 | 4096 | 174.06 | 12.93 | 343 ms |

**Single-stream ~49-55 tok/s. Aggregate saturates near 174 tok/s at the `max-num-seqs`
16 ceiling** — 3.2x aggregate scaling for a 4.4x per-stream cost.

> [!CAUTION]
> **This whole table is a degraded-fabric signature.** The healthy 2-node figures at the
> settled profile are **76.2 tok/s at cc=1** and **481.3 aggregate at cc=16**
> ([`../results/20260825-decode-2v3/`](../results/20260825-decode-2v3/)). The *shape* the
> table describes — aggregate scales sublinearly, per-stream falls with concurrency — is
> still the right shape; the values are not. **`49-55 tok/s` must not be quoted as this
> deployment's 2-node speed.**

## KV cache accounting (the actual constraint) — SURVIVES

From the engine startup log:

| Item | Value |
|---|---|
| Available KV cache | **19.52 GiB** |
| GPU KV cache size | **1,771,152 tokens** |
| Max concurrency @ 460800 | **3.84x** |
| Engine init | 100.15 s |

Derived — and this is the load-bearing finding for the 3-node decision:

- Full-model KV is `584 B/layer x 43 layers` = **25,112 B/token**.
- Observed pool is `19.52 GiB / 1,771,152` = **11,834 B/token** ~= **20.3 layers**.
- Therefore **KV is partitioned across nodes, not replicated.** Adding a node adds KV
  pool roughly linearly.

Per-node budget at `gpu-memory-utilization 0.85` of 121 GiB unified = 102.8 GiB:

| Nodes | Weights/node | Headroom for KV + activations |
|---:|---:|---:|
| 2 (today) | 78.0 GiB | 24.8 GiB |
| 3 | 52.0 GiB | **50.8 GiB** |

Concurrency the current pool supports at various context lengths:

| Context | Full-length concurrent requests |
|---:|---:|
| 460,800 | 3.84x |
| 262,144 | 6.76x |
| 131,072 | 13.51x |
| 65,536 | 27.03x |
| 32,768 | 54.05x |

## Interpreting this baseline against the PP=3 candidate

1. **PP does not accelerate a single request.** One request still traverses all 43 layers
   in sequence, with 2 network hops added. PP raises *aggregate* throughput by filling
   pipeline stages; per-request decode should be expected to regress.
2. **The candidate disables speculative decoding; this baseline has it on** (MTP, gamma=5).
   That alone is a large single-stream delta, independent of node count. The comparison is
   only apples-to-apples once speculation is matched, which the PR correctly defers to a
   separate experiment.
3. **The genuine 3-node win is KV capacity**, per the accounting above — roughly 2x the
   per-node headroom, freeing weight memory into the KV pool.
4. `VLLM_PP_LAYER_PARTITION` is honoured by this build (`vllm/distributed/utils.py:143`),
   and it validates `len(partitions) == pp_size` and `sum(partitions) == num_hidden_layers`.
   The proposed `14,15,14` sums to 43 and is therefore accepted.
5. This build logs **`Using V2 Model Runner`**, so `max_concurrent_batches` becomes
   `pp_size + 1` under async scheduling (`vllm/config/vllm.py:493-500`). The V1 restriction
   that async scheduling is unsupported with PP does **not** apply here.

**Recommendation:** judge the PP=3 candidate on KV pool size and aggregate throughput at
C=12-16, not on single-stream decode. Expect single-stream to regress and treat that as a
known, priced-in cost rather than a failure.

## Reproducing

```bash
cp configs/2spark.env.example configs/2spark.env
# edit NODE_IPS, SSH_USER, API_BASE; set SSH_USER_MAP if usernames differ per node
bash scripts/run_experiment.sh configs/2spark.env
```

### Environment notes found while running this

- **`make` runs recipes under `/bin/sh`.** On Ubuntu that is `dash`, which rejects
  `set -o pipefail`. Invoke the scripts with `bash` directly, or add `SHELL := /bin/bash`
  handling for recipe lines that call these scripts.
- **CRLF line endings break the scripts.** If the repo is checked out on Windows and
  copied to a node, `#!/usr/bin/env bash\r` fails with
  `set: pipefail: invalid option name`. Use a `.gitattributes` with `*.sh text eol=lf`,
  or run `sed -i 's/\r$//'` after transfer.
- **Per-node usernames.** `SSH_USER` was a single global; DGX Sparks are provisioned with
  a distinct account per box. `SSH_USER_MAP` (added in this branch) handles that without
  renaming accounts or creating a shared admin user.
