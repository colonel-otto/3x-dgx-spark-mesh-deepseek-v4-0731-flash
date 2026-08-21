# Three-Spark implementation

## 0. Freeze the baseline

Before changing cabling or software, copy `configs/2spark.env.example` to `configs/2spark.env`, populate it, and run:

```bash
make baseline CONFIG=configs/2spark.env
```

Keep the exact model revision, vLLM/container build, KV cache format, max context, tokenizer and non-parallel launch flags recorded.

## 1. Cable and configure the three-node ring

Use one direct QSFP link for each pair:

```text
Spark 1 <----> Spark 2
   ^             ^
    \           /
     \         /
      Spark 3
```

Use NVIDIA Cluster Assistant / the official three-Spark connection playbook to configure the CX-7 interfaces and SSH. Do not mix direct and switched topology.

## 2. Preflight

```bash
cp configs/3spark.env.example configs/3spark.env
# edit NODE_IPS, SSH_USER, API_BASE
make preflight CONFIG=configs/3spark.env
```

The first hard gate is simply that all three machines are reachable and the expected CX-7 interfaces are up.

## 3. NCCL ring gate

The official NVIDIA `launch.sh` expects management IPs for MPI/SSH bootstrap and auto-discovers the CX-7 RoCE data path. For a ring it sets:

```bash
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_NET_PLUGIN=none
```

If NCCL and `nccl-tests` are not already built on all three nodes, follow NVIDIA's current DGX Spark NCCL playbook first. Then:

```bash
make nccl CONFIG=configs/3spark.env
```

Do not proceed to vLLM until the test completes without NCCL errors and reports `#wrong = 0`.

## 4. Start the three-node distributed runtime

### Recommended first proof: vLLM multi-node multiprocessing

Current vLLM supports multi-node `mp` directly, so the first proof does not need Ray. Set `MASTER_ADDR` in `configs/3spark.env` to Spark 1's management IP. In the identical vLLM environment on all three machines, run:

Spark 1:

```bash
bash scripts/launch_vllm_mp_node.sh configs/3spark.env 0
```

Spark 2:

```bash
bash scripts/launch_vllm_mp_node.sh configs/3spark.env 1
```

Spark 3:

```bash
bash scripts/launch_vllm_mp_node.sh configs/3spark.env 2
```

Ranks 1 and 2 automatically use `--headless`; rank 0 hosts the API.

### Alternate: Ray

If the working two-Spark deployment already uses Ray and you want to keep that variable fixed, create the same three-node Ray cluster and verify:

```bash
bash scripts/verify_ray.sh 3
```

Then use `scripts/launch_vllm_candidate.sh`.

## 5. First vLLM candidate

DeepSeek-V4-Flash-0731 has 43 hidden layers. Start with:

```text
TP = 1
PP = 3
layers = 14,15,14
speculative decoding = OFF
```

Put all existing non-parallel vLLM flags in `EXTRA_VLLM_ARGS` in your local `configs/3spark.env`. Keep them identical to the baseline.

For the `mp` path, the three commands in step 4 are the launch. For Ray, use:

```bash
bash scripts/launch_vllm_candidate.sh configs/3spark.env 2>&1 | tee vllm-3spark.log
```

Verify the startup logs show three distributed workers/pipeline ranks and no network fallback/error storm.

## 6. Measure candidate

When `/v1/models` and `/v1/chat/completions` are healthy:

```bash
make candidate CONFIG=configs/3spark.env
```

Start with the default contexts. If those pass, extend:

```bash
CONTEXTS=2048,8192,32768,65536,131072 make candidate CONFIG=configs/3spark.env
```

## 7. Compare

```bash
make compare
```

The comparison report intentionally treats correctness as a gate. Do not accept a throughput gain if long-context retrieval becomes unreliable.

## 8. Only then test speculative decoding

Keep the first 3-Spark proof free of speculative decoding. After PP=3 is stable and benchmarked, create a separate experiment label/config for speculation so its effect is independently measurable.
