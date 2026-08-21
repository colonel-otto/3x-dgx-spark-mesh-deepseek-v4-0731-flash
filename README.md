# DGX Spark Three-Node Lab

Reproducible before/after validation for running `deepseek-ai/DeepSeek-V4-Flash-0731` on DGX Spark.

The primary experiment compares:

- **Baseline:** current working 2-Spark deployment
- **Candidate:** 3 directly connected DGX Sparks, `TP=1`, `PP=3`, speculative decoding disabled initially

The repository separates **fabric validation** from **model validation** so an NCCL/network problem is not confused with a vLLM/model problem.

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

- NVIDIA's DGX Spark NCCL playbook supports a three-node `ring` topology.
- The NVIDIA ring launcher sets `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_NET_PLUGIN=none`.
- vLLM supports multi-node pipeline parallel serving and uneven layer splits.
- DeepSeek-V4-Flash-0731 declares 43 hidden layers.

The candidate config therefore begins with `TP=1`, `PP=3` and an explicit `14,15,14` layer partition. The example configs pin NVIDIA `dgx-spark-playbooks` to commit `1fb66f059ee427c5a3678b3117ef73aab042b458` so the NCCL helper does not silently change between baseline and candidate.

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
4. vLLM starts with `TP=1`, `PP=3`, speculation disabled.
5. API smoke test succeeds.
6. Needle-retrieval tests succeed at the chosen context sizes.
7. Benchmark results are compared with the frozen 2-Spark baseline.

Do not optimize until all seven are reproducible.
