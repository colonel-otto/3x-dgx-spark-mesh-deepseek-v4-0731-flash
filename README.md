# DGX Spark Three-Node Lab

Reproducible before/after validation for running `deepseek-ai/DeepSeek-V4-Flash-0731` on DGX Spark.

The repository records the full experiment sequence rather than only the winning
configuration: a frozen 2-Spark baseline, unsuccessful 3-Spark EP and PP approaches,
and the working 3-Spark TP configuration. See
[`docs/EXPERIMENT-LOG.md`](docs/EXPERIMENT-LOG.md) for the decision trail and PR map.

The repository separates **fabric validation** from **model validation** so an NCCL/network problem is not confused with a vLLM/model problem.

## Results so far

| Document | Finding |
|---|---|
| [`docs/BASELINE-2SPARK.md`](docs/BASELINE-2SPARK.md) | Frozen 2-Spark baseline: 90 requests, 0 failures, 100% needle retrieval, ~49–55 tok/s single-stream |
| [`docs/EP3-EXPERT-PARALLEL.md`](docs/EP3-EXPERT-PARALLEL.md) | **3-node sharding works** via expert parallelism (86/85/85 experts, 2.3x KV) but is **2.5x slower** — and the cause is the MoE kernel, not the node count |
| [`docs/TP3-TUNING.md`](docs/TP3-TUNING.md) | TP=3 plus the attention-group padding patch is correct; canonical-ring profiles measured 53.95–56.63 tok/s and the earlier cable rotation reached a historical 57.73 tok/s |

Findings that changed the plan in this README. **Two earlier conclusions here have since
been disproven by measurement** — corrected below:

1. **`PP=3` is not the only route.** Expert parallelism shards 256 experts 86/85/85
   across three nodes.
2. ~~`TP=3` is genuinely impossible~~ — **wrong.** Stock vLLM in the tested image first
   rejects 64 attention heads divided across TP=3. If only that validation is bypassed,
   later floor divisions such as `8 // 3` represent six of the eight global attention
   groups, losing two. The padding patch changes 8 → 9 groups (64 → 72 heads while
   holding eight heads/group) and makes TP=3 boot and pass correctness checks. See
   [`docs/TP3-TUNING.md`](docs/TP3-TUNING.md).
3. ~~RoCE does not work across all three nodes~~ — **wrong; it works.** The point-to-point
   triangle means each node uses a different device per peer, so NCCL's default
   index-based pairing can select NIC pairs with no direct cable
   (`ibv_modify_qp ... INIT -> RTR` timeout). NVIDIA's ring launcher sets
   **`NCCL_IB_SUBNET_AWARE_ROUTING=1`** and `NCCL_NET_PLUGIN=none`; those settings made
   RDMA work on this switchless ring.
4. **24.59 tok/s is the TP=3 TCP/Socket control, not the RoCE result.** On RoCE the
   measured range was 53.95–57.73 tok/s, depending on run and physical cable rotation.

## What is measured

- exact software/environment snapshot per node
- CX-7 link state and addresses
- NCCL 3-node ring correctness and bandwidth
- Ray node/GPU visibility
- OpenAI-compatible endpoint health
- time to first token (TTFT)
- end-to-end latency
- decode tokens/second per request
- aggregate output tokens/second at concurrency 1 / 3 / 6
- actual prompt/output token counts reported by the server
- deterministic long-context needle retrieval
- before/after delta report

## Upstream facts this experiment relies on

As of August 2026:

- NVIDIA's [three-Spark connection playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks)
  documents the physical ring.
- NVIDIA's [NCCL playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nccl)
  and [ring launcher](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nccl/assets/launch.sh)
  set `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_NET_PLUGIN=none` for three nodes.
- vLLM supports multi-node pipeline parallel serving and uneven layer splits.
- DeepSeek-V4-Flash-0731 declares 43 hidden layers.

The example configs pin NVIDIA `dgx-spark-playbooks` to commit
`1fb66f059ee427c5a3678b3117ef73aab042b458` so a rerun can distinguish the version
used in the experiment from later upstream changes.

## Safety rule

**Do not alter the working 2-Spark deployment while collecting the baseline.** Record it first. The candidate launch should use the same model revision, vLLM build/container, tokenizer, KV-cache settings, max model length and sampling settings wherever possible.

## Quick start

Copy the examples and fill in your real node addresses / endpoint:

```bash
cp configs/2spark.env.example configs/2spark.env
cp configs/3spark.env.example configs/3spark.env
```

Collect the current 2-Spark baseline while its API is running:

```bash
make baseline CONFIG=configs/2spark.env
```

After cabling/configuring the third Spark, capture the complete fabric test:

```bash
make fabric CONFIG=configs/3spark.env
```

If NCCL/nccl-tests are not installed yet, first run `make nccl-bootstrap CONFIG=configs/3spark.env` from Spark 1.

Start the 3-node vLLM deployment using the documented multi-node `mp` path (or Ray if you are preserving an existing Ray baseline), then measure it:

```bash
make candidate CONFIG=configs/3spark.env
```

Compare the latest baseline and candidate:

```bash
make compare
```

For a longer context sweep:

```bash
CONTEXTS=2048,8192,32768,65536,131072 make candidate CONFIG=configs/3spark.env
```

## Acceptance order

1. `preflight` passes on every node.
2. NVIDIA NCCL ring test completes with `#wrong = 0`.
3. Ray reports 3 alive nodes and 3 GPUs.
4. `NCCL_DEBUG=INFO` identifies `NET/IB` rather than `NET/Socket` for the model run.
5. vLLM starts with the intended parallel and speculative-decoding settings on every rank.
6. API smoke and correctness tests succeed at the chosen context sizes.
7. Repeated-run medians are compared with the frozen 2-Spark baseline and the raw
   artifacts are retained.

Do not optimize until all seven are reproducible.
