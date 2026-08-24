# DeepSeek-V4-Flash on three DGX Sparks

A reproducible recipe and controlled benchmark for serving DeepSeek-V4-Flash on three
NVIDIA DGX Spark systems with tensor parallelism (`TP=3`) over a switchless 200 GbE
RoCE ring.

The useful result is not merely that TP=3 starts. With the attention-group padding
patch and subnet-aware RoCE, TP=3 retains the B12X MXFP4 MoE kernel and MTP, passes the
correctness suite, and outperforms our matched two-Spark baseline. The complete decision
trail—including the unsuccessful EP and PP paths—is retained in
[`docs/EXPERIMENT-LOG.md`](docs/EXPERIMENT-LOG.md).

> **Picking this work up?** Start with **[`docs/HANDOFF.md`](docs/HANDOFF.md)** — what is
> running, how to operate it, what is settled, and step-by-step instructions for the next
> three measurements.
>
> **The table below predates 2026-08-24 and describes the 460,800-context, MTP=4 era.**
> The cluster now runs **1M context with `MTP_NUM_TOKENS=5`**, measuring 93.8 / 92.3 /
> 86.1 tok/s single-stream (count / JSON / code) with a 5,444,869-token KV pool. See
> [`docs/MTP5-1M-AND-UPSTREAM-COMPARISON.md`](docs/MTP5-1M-AND-UPSTREAM-COMPARISON.md).

## Measured result

| Configuration | Decode median | TTFT | KV cache | Concurrency @ 460,800 | Correctness |
|---|---:|---:|---:|---:|---:|
| 2 Spark, TP=2, RoCE | 48.23 tok/s | 154 ms | 1,855,255 tokens | ~3.9x | 7/7 |
| 3 Spark, TP=3, TCP control | 24.59 tok/s | 323 ms | 3,579,619 tokens | 7.77x | 7/7 |
| 3 Spark, TP=3, RoCE, canonical ring, prose prompt | 53.95–56.63 tok/s | — | ~3.6M tokens | ~7.8x | 7/7 |
| 3 Spark, TP=3, RoCE, upstream/code prompt | **79.0–79.3 tok/s** | ~105–115 ms | same engine | same engine | 7/7 deployment |
| 3 Spark, TP=3, RoCE, earlier cable rotation | 57.73 tok/s | 186 ms | 3,598,182 tokens | 7.81x | 7/7 |

On the matched prose workload, TP=3 RoCE delivered:

- **11.9–17.4% higher single-stream decode throughput**
- **1.94x KV-cache capacity**
- approximately **2x concurrency** at the configured maximum context
- **about 2.2–2.3x the throughput of the TP=3 TCP control**, isolating transport as the
  important performance difference

Prompt shape materially changes MTP acceptance: the same live engine reached about
49 tok/s on difficult prose and 79–82 tok/s on code-shaped prompts. The upstream-harness
result closes the apparent 75–79 tok/s comparison gap; it was a workload mismatch, not
a hardware shortfall. See
[`docs/BENCHMARK-METHODOLOGY.md`](docs/BENCHMARK-METHODOLOGY.md).

These are measurements from one cluster, not universal product specifications. The
raw samples from the original exploratory run were not all retained, so the historical
summary is published separately from future evidence bundles. See
[`docs/results.md`](docs/results.md) and [`benchmarks/summary.csv`](benchmarks/summary.csv).

## Why a patch is required

DeepSeek-V4-Flash has eight output attention groups. Eight is not divisible by three.
The TP=3 patch pads the group count from 8 to 9 and the corresponding head count from
64 to 72 while keeping eight heads per group unchanged. The added group is masked out
of the model result.

That last invariant matters: the checkpoint's output projection expects eight heads
per group. A TP=3 speed number is not credible without correctness validation.

This repository pins the upstream implementation rather than silently copying a moving
target:

- Patch project: [`localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark`](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark)
- Pinned publication revision: `496c6a146a383f1b7c3f5991f4f1930091420720`

See [`docs/patch.md`](docs/patch.md).

## Hardware topology

Use NVIDIA's canonical cross-connected ring:

```text
Node 1 port 0  <---->  Node 2 port 1
Node 2 port 0  <---->  Node 3 port 1
Node 3 port 0  <---->  Node 1 port 1
```

Port 0 is the CX-7 connector nearest the ordinary Ethernet connector. Port 1 is the
CX-7 connector farther away. Double-check with LLDP before assigning addresses.

Our example address plan gives each point-to-point cable its own subnet:

| Cable | Node-side addresses |
|---|---|
| Node 1 p0 <-> Node 2 p1 | `192.168.100.1` <-> `192.168.100.2` |
| Node 1 p1 <-> Node 3 p0 | `192.168.101.1` <-> `192.168.101.2` |
| Node 2 p0 <-> Node 3 p1 | `192.168.102.1` <-> `192.168.102.2` |

Follow [`docs/topology.md`](docs/topology.md) and NVIDIA's
[`Connect Three DGX Spark in a Ring Topology`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/README.md).

## Working configuration

The essential TP=3/RoCE settings are:

```bash
TP_SIZE=3
PP_SIZE=1
NNODES=3
MOE_BACKEND=flashinfer_b12x

NCCL_IB_DISABLE=0
NCCL_NET=IB
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_NET_PLUGIN=none
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
NCCL_NVLS_ENABLE=0
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_IB_ROCE_VERSION_NUM=2
```

The measured serving profile used:

```bash
MAX_MODEL_LEN=460800
MAX_NUM_SEQS=8
GPU_MEMORY_UTILIZATION=0.85
MTP_NUM_TOKENS=4
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

Use `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` for the first validation
launch. The log must show `NET/IB`; `NET/Socket` is a TCP fallback. Return to
`NCCL_DEBUG=WARN` for normal serving.

Environment-file values do nothing unless Compose forwards them into the container.
Always inspect the rendered configuration before launch:

```bash
docker compose --env-file config/node0.env -f docker-compose.yml config \
  | grep -E 'SUBNET_AWARE|NCCL_NET|NCCL_IB_HCA|tensor-parallel'
```

Start workers first and the HTTP-serving head last. See [`docs/setup.md`](docs/setup.md)
and the sanitized templates in [`config/`](config/).

## Reproducing the comparison

1. Update all three systems through the NVIDIA-supported update path and verify that
   OS, driver, kernel and CX-7 firmware versions match.
2. Cable and address the official ring; prove each direct edge before involving vLLM.
3. Run NVIDIA's NCCL tests and capture a log showing `NET/IB`.
4. Pin the container image, checkpoint revision and TP=3 patch revision.
5. Launch three identical ranks, changing only rank-specific identity and interfaces.
6. Run correctness before performance.
7. Benchmark TP=2 RoCE, TP=3 TCP and TP=3 RoCE with the same prompt, sampling settings,
   context profile and harness revision.
8. Save a complete artifact bundle; do not publish only the best run.

Detailed reproduction protocol:
[`docs/reproduction-methodology.md`](docs/reproduction-methodology.md).

## Repository map

```text
config/       sanitized env and Compose-forwarding examples
docs/         setup, topology, results, patch explanation and troubleshooting
scripts/      environment/fabric collection and benchmark helpers
benchmarks/   machine-readable historical summary
artifacts/    schema for complete future run bundles
```

## Privacy and safety

The examples deliberately use generic node names and RFC1918 fabric addresses. Before
publishing artifacts, remove management-network addresses, MAC addresses, usernames,
SSH material, registry credentials, absolute model paths and container environment
secrets. See [`SECURITY.md`](SECURITY.md).

## Primary references

- [NVIDIA: Connect Three DGX Spark in a Ring Topology](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/README.md)
- [NVIDIA: NCCL for Multiple Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nccl/README.md)
- [NVIDIA: DGX Spark OS and Component Update Guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html)
- [NVIDIA: DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
