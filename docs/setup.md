# Reproduction setup

This guide assumes one GB10 GPU per Spark, the same username on every node, local copies
of the same model checkpoint, and passwordless SSH over the management network.

## 1. Match software and firmware

Update through the NVIDIA-supported DGX Dashboard path. Record, compare and retain:

```bash
cat /etc/dgx-release
uname -a
nvidia-smi
nvcc --version
ibv_devinfo
ethtool -i enp1s0f0np0
docker version
```

Do not upgrade one rank in isolation. A benchmark is useful only when all three nodes
have matching driver, kernel, firmware, container and checkpoint revisions.

## 2. Configure and prove the ring

Complete [`topology.md`](topology.md), including physical state, addresses, MTU, direct
edge tests and the routed master identity.

Then follow NVIDIA's
[`NCCL for Multiple Sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nccl/README.md)
procedure. Capture both small-message `all_reduce_perf` results and NVIDIA's large
`all_gather_perf` test.

## 3. Obtain the TP=3 patch at a pinned revision

```bash
git clone https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark.git
cd DeepSeek-V4-Flash-DSpark-3x-DGX-Spark
git checkout 496c6a146a383f1b7c3f5991f4f1930091420720
sha256sum patches/tp3/apply_tp3_patch.py
```

Follow that repository's patch instructions against the same immutable container image
on all three nodes. Save the image digest and patch checksum in the run manifest.

## 4. Create per-rank environment files

Copy the templates in [`../config`](../config). Substitute only site-specific values:

- local checkpoint path
- master identity
- node rank
- socket interface facing Node 1
- HCA names if `ibdev2netdev` differs

All performance and model settings must remain identical across ranks. A mismatch in
`MAX_MODEL_LEN`, `MAX_NUM_SEQS` or `GPU_MEMORY_UTILIZATION` can hang startup without a
useful error.

## 5. Forward variables through Compose

Use [`../config/compose.tp3.fragment.yml`](../config/compose.tp3.fragment.yml) as the
minimum forwarding list to merge into the serving project's Compose service. It is a
fragment, not a standalone serving definition.

Render each rank before launch:

```bash
docker compose --env-file config/node0.env -f docker-compose.yml config > rendered.yml
grep -E 'SUBNET_AWARE|NCCL_NET|NCCL_IB_HCA|MAX_MODEL_LEN|MAX_NUM_SEQS' rendered.yml
```

## 6. Start workers before the head

Start rank 2 and rank 1, then rank 0. Use the serving project's launch command while
passing the corresponding env file. Cold startup can take approximately six minutes.

The common model arguments must resolve to:

```text
--tensor-parallel-size 3
--pipeline-parallel-size 1
--nnodes 3
--moe-backend flashinfer_b12x
--kv-cache-dtype nvfp4_ds_mla
--block-size 256
--max-model-len 460800
--max-num-seqs 16
--gpu-memory-utilization 0.85
--speculative-config {method: dspark, num_speculative_tokens: 5, ...}
```

## 7. Validate before benchmarking

Require all of the following:

- all three ranks joined;
- startup reports `B12X_MXFP4`;
- startup reports the DSpark MTP speculator with five speculative tokens;
- NCCL INFO log reports `NET/IB`, not only `NET/Socket`;
- `/v1/models` reports a 460,800-token model limit;
- the correctness suite passes;
- RDMA hardware counters increase during inference.

Only then run the matched benchmark protocol.
