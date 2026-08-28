# Troubleshooting

## NCCL reports `NET/Socket`

The run is using TCP, even if RDMA link state and ping are healthy.

Check that:

- `/dev/infiniband` is visible inside the container;
- `NCCL_IB_DISABLE=0` and `NCCL_NET=IB` reached the container;
- both peer-facing HCAs are listed in `NCCL_IB_HCA`;
- `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_NET_PLUGIN=none` reached the container;
- the loaded NCCL library supports subnet-aware routing.

`torch.cuda.nccl.version()` reports the NCCL version PyTorch was compiled against, not
necessarily the library mapped by the live process. Verify the live mapping through
`/proc/PID/maps` and query `ncclGetVersion` from that library.

## `ibv_modify_qp` fails during INIT to RTR

This commonly means the chosen HCA/GID cannot reach the remote rank. Confirm the GID is
RoCEv2/IPv4 and that subnet-aware routing can choose a peer-facing HCA. Do not begin by
moving cables when LLDP and the official ring already match.

## Ping succeeds but distributed startup times out

Ping does not validate new TCP flows or RDMA. Check the firewall on every rank, routes
to the master identity, container `/etc/hosts`, and a real NCCL test.

## Startup hangs after weight loading

Verify identical `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, MTP and model
arguments on every rank. Mismatched memory profiles can hang without a useful error.

## Correct-looking but wrong answers

Stop performance testing. Confirm that the TP=3 padding patch applied at every expected
site and every rank uses the same image. Run the correctness suite with enough output
budget to distinguish wrong answers from truncated reasoning.

## Node accepts TCP but SSH never sends a banner

This can be severe unified-memory exhaustion rather than a network problem. A single
unsharded copy of this checkpoint does not fit safely on one Spark. Calculate weights
per node before changing TP/PP settings; do not launch TP=1 for this checkpoint.

## Relaunch after a crash driver-OOMs at boot (external report, unverified)

A community TP=4 recipe ([tonyd2wild](https://github.com/tonyd2wild/Deepseek-V4-Flash-TP4-4x-DGX-Spark),
TROUBLESHOOTING §6) reports the GB10 driver can hold ~100 GiB of unified memory after a
crash, so an immediate relaunch driver-OOMs at boot despite `nvidia-smi` looking idle.
Their fix: `sync; echo 3 > /proc/sys/vm/drop_caches` and a fresh `docker run` (never
`docker restart`, never `--rm` — a crash then leaves no logs). **We have not reproduced
this**; it is recorded because it rhymes with our wedge patterns. If a relaunch OOMs
where the first launch fit, try `drop_caches` before concluding the config regressed.

## Avoid interpreting experimental dead ends as requirements

- A 200 GbE switch is not required for the proven three-node ring.
- A host-built NCCL is not automatically the runtime used by the Python process.
- The uppercase second CX-7 interface pair is optional multi-rail headroom, not a
  prerequisite for the historical result.
