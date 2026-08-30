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

## Startup fails instantly with "Free memory on device ... is less than desired GPU memory utilization"

```
ValueError: Free memory on device cuda:0 (100.94/121.69 GiB) on startup is less
than desired GPU memory utilization (0.835, 101.61 GiB).
```

`GPU_MEMORY_UTILIZATION` is a fraction of **total** memory, but the check compares
it against **free** memory. On a GB10 node the OS, dockerd, `open-webui` (~1 GiB on
rank 0) and the exporters are already resident, so a node never frees more than
~100.9-101.5 GiB of its 121.69 GiB. Any value above ~0.828 therefore fails on a
warm box even though nothing is wrong with the cluster.

Fix by lowering the value on **all three ranks** (a mismatch hangs startup forever
with no error) or by freeing resident host RAM. See `DECISIONS.md` for the current
value and why it moved.

**`VLLM_SKIP_INIT_MEMORY_CHECK` does not work.** The engine logs it as
`Unknown vLLM environment variable detected` and `request_memory()` in
`vllm/v1/worker/utils.py` raises unconditionally — there is no skip branch in this
build. Do not rely on it to let a tight value through.

**Read the container log, not the journal.** `journalctl -u dsv4` shows only the
outer wrapper's traceback ending in `Engine core initialization failed. See root
cause above.` — the actual `ValueError` is upstream of it, in
`docker logs dspark-vllm-gx10-vllm-dspark-1`.

**Free is not available.** These are unified-memory nodes: `nvidia-smi` reports
`N/A` and the check reads *free*. A node showing 74 GiB free / 116 GiB available
has 42 GiB of reclaimable page cache — but vLLM tests the free number, so judge
headroom by that.

**Dropping caches is not the fix on its own.** `sysctl vm.drop_caches=3` makes the
*pre-start* free number look healthy, but the shortfall appears *during* startup:
measured on 2026-08-30, rank 0 had **110.59 GiB free** immediately before a launch
— 9 GiB more than the 101.61 GiB required, with no cache drop needed — and the
run still failed with vLLM reporting only **100.94 GiB**. The ~10 GiB goes to the
CUDA context and NCCL communicator buffers allocated before `request_memory()`
runs. Drop caches as an *extra* lever when a start is tight; do not rely on it to
carry a value the node cannot otherwise sustain.

**Leave a deliberate reserve.** The headroom left by `GPU_MEMORY_UTILIZATION` is
what keeps `sshd` and `open-webui` alive, and SSH is the only way back into a node
that goes bad. Per node (121.69 GiB total):

| util | engine | reserved for OS + sshd + open-webui |
|---|---|---|
| 0.835 | 101.61 GiB | 20.08 GiB |
| **0.82** | **99.79 GiB** | **21.90 GiB** |
| 0.80 | 97.35 GiB | 24.34 GiB |

Do not tune this purely for KV pool size. Recovery access is worth more than the
last GiB of cache.

**A cold start takes ~30 minutes, not 6-8.** Weight loading is only the first
~3 minutes of it (155 GiB main model ~139 s, then the MTP draft ~26 s); the rest
is KV profiling, `torch.compile` and CUDA graph capture. `dsv4-service-start`
budgets 45 minutes for exactly this reason — an earlier 15-minute budget aborted a
startup that was progressing normally. Do not conclude a start has hung because it
has been running 10 minutes; check that the container log is still advancing, and
see "Startup hangs after weight loading" above for what a real hang looks like.
