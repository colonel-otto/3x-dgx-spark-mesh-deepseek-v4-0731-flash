# Reproduction setup

This guide assumes one GB10 GPU per Spark, local copies of the same model checkpoint, and
passwordless SSH over the management network. Different login names are supported through
`SSH_USER_MAP` in the workstation harness config.

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

When building `nccl-tests`, use the headers and library from the same NCCL package that
the engine maps at runtime; the container can also contain an older system copy. Run the
reference test inside a container with the serving RDMA devices, memory-lock limit, shared
memory size, host network, and NCCL environment. Stop the engine before a 16 GiB test.
Record the runtime NCCL version, `#wrong=0`, and `via NET/IB/*` with the result.

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

The current production model arguments must resolve to the following. Read them from the
live process after every start; do not infer them from an environment file. The evidence
capture is in
[`20260827-decode-3node-fixed/engine-config.txt`](../results/20260827-decode-3node-fixed/engine-config.txt).

```text
--tensor-parallel-size 3
--pipeline-parallel-size 1
--nnodes 3
--moe-backend flashinfer_b12x
--kv-cache-dtype nvfp4_ds_mla
--block-size 256
--max-model-len 1048576
--max-num-seqs 32
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.80
--speculative-config {method: dspark, num_speculative_tokens: 5, ...}
```

The older `460800` / `seqs=16` / `gpu-memory-utilization=0.85` profile appears in frozen
results and historical docs. It is evidence provenance, not the current default.

## 7. Validate before benchmarking

Require all of the following:

- all three ranks joined;
- startup reports `B12X_MXFP4`;
- startup reports the DSpark MTP speculator with five speculative tokens;
- NCCL INFO log reports `NET/IB`, not only `NET/Socket`;
- `/v1/models` reports a 1,048,576-token model limit;
- the correctness suite passes;
- RDMA hardware counters increase during inference.

Only then run the matched benchmark protocol.

## 8. Switch to the two-node comparison arm

The repository's `scripts/cluster_tp2.sh` is the cluster-specific TP=2 launcher. It checks
that engine-shaping values match, starts the worker before the head, waits for health, and
reads the applied process arguments back. Run `status` before changing shapes and stop the
three-node service before `up` so rank 2 is not still holding its GPU.

```bash
bash scripts/cluster_tp2.sh status
bash scripts/cluster_tp2.sh up
bash scripts/cluster_tp2.sh down
```

Live engine env files remain on the Sparks and are not committed. Workstation gate targets
come from [`../configs/`](../configs/README.md); copy the tracked example to a gitignored
live file before running a gate.
