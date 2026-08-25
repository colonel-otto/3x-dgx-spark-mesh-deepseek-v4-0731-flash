# DeepSeek-V4-Flash on three DGX Sparks

A reproducible recipe and controlled benchmark for serving DeepSeek-V4-Flash on three
NVIDIA DGX Spark systems with tensor parallelism (`TP=3`) over a switchless 200 GbE
RoCE ring.

The useful result is not merely that TP=3 starts. With the attention-group padding
patch and subnet-aware RoCE, TP=3 retains the B12X MXFP4 MoE kernel and MTP, passes the
correctness suite, and outperforms our matched two-Spark baseline. The complete decision
trail—including the unsuccessful EP and PP paths—is retained in
[`docs/EXPERIMENT-LOG.md`](docs/EXPERIMENT-LOG.md).

> **Picking this work up?** Four pages, in order:
>
> 1. **[`docs/HANDOFF.md`](docs/HANDOFF.md)** — what is running, how to operate it, what is open.
> 2. **[`docs/DECISIONS.md`](docs/DECISIONS.md)** — every settled value and the measurement behind it.
> 3. **[`docs/POSTMORTEM-2026-08-25.md`](docs/POSTMORTEM-2026-08-25.md)** — four classes of silent
>    failure that produced plausible-but-wrong numbers here. **Read before adding a benchmark.**
> 4. **[`docs/DEGRADED-DATA-CATALOGUE.md`](docs/DEGRADED-DATA-CATALOGUE.md)** — what bad data
>    looked like, itemized, so you can recognise the shape.
>
> Full map: **[`docs/README.md`](docs/README.md)**.

## Is the third node worth it?

**Yes for single-stream interactive work; no for batch throughput.** Measured
2026-08-25 on healthy fabric, identical config on both arms (`1M` / `seqs 16` / `MTP=5` /
`0.80`), same harness, node count the only variable:

| measurement | 2 Spark | 3 Spark | winner |
|---|---:|---:|---|
| **decode cc=1** | 76.2 tok/s | **89.1 tok/s** | **3-node, +17%** |
| decode cc=4 | 192.8 | **208.8** | 3-node, +8% |
| decode cc=8 | 302.7 | **322.7** | 3-node, +7% |
| decode cc=16 | **481.3** | 474.8 | 2-node, +1.4% |
| prefill 1K/8K/32K | 1913 / 2081 / 2066 | 2023 / 2070 / 2095 | **parity** (±2%) |
| deep concurrency (4×200K) TTFT | **293,987 ms** | 396,804 ms | 2-node, 1.35x |
| KV cache | 1,711,307 tok | **4,457,627 tok** | 3-node 2.6x — *never binds* |

The 3-node advantage **decays monotonically with concurrency and crosses over near
cc=16**. Three nodes win per-stream latency; two win batch aggregate.

**Choose by workload, not by node count.** Single-user interactive coding is
per-stream-latency bound → three nodes. Multi-user batch serving → two nodes, and the
third is free for another model.

Two things the third node does **not** buy: prefill throughput (parity), and usable KV
capacity (`vllm:num_preemptions_total` has read **0 in every test ever run here**,
including a 4×200K test designed specifically to make KV bite).

Evidence: [`results/20260825-decode-2v3/`](results/20260825-decode-2v3) ·
[`results/20260825-prefill-2v3/`](results/20260825-prefill-2v3) ·
[`results/20260825-deep-concurrency/`](results/20260825-deep-concurrency)

> Historical numbers from before 2026-08-25 were taken while one node ran at ~15% of its
> collective bandwidth and are **provisional**. Every affected measurement is itemized in
> [`docs/DEGRADED-DATA-CATALOGUE.md`](docs/DEGRADED-DATA-CATALOGUE.md) — kept deliberately,
> because knowing what bad data looked like is how you spot the next batch. See also
> [issue #14](../../issues/14).

## Pitfalls — read before you deploy

Each of these cost real time here, and none announced itself.

1. **TP=3 silently serves nonsense without the padding patch.** Stock vLLM computes
   `n_local_groups = 8 // 3 == 2`, dropping 6 of 8 attention groups. Output stays
   *fluent*. Always run a correctness check (17×23 → 391) after any restart.
2. **A parallelism-flag mismatch between ranks hangs startup forever with no error.**
   Verify a checksum across all ranks before starting; `scripts/cluster_tp2.sh` refuses
   to start rather than let you watch a silent hang.
3. **NetworkManager here is only a renderer — netplan owns the config.** NM connections
   live in `/run/…` (tmpfs, wiped on reboot). An `nmcli` change that does not reach
   `/etc/netplan/` looks applied and reverts. This nearly made the cluster unrecoverable.
4. **A degraded RDMA link has zero error indicators.** Ports ACTIVE, 200,000 Mb/s, all
   counters 0 — while running at 15% speed. **TCP throughput will not find it** (it hid a
   6.8x RDMA deficit behind a 1.19x TCP one). Only an NCCL collective does.
5. **Init success is not health.** All ranks can complete NCCL init and every container
   stay `running` while live RDMA completions fail. There is no container health check.
   Check `IBV_WC_*_ERR` in the engine log, not `docker ps`.
6. **The `roceP2p` HCAs double bandwidth — but only once they are addressed.** Each QSFP
   port is two ~100G controllers; using all four takes pairs 4.6 → 9.7 GB/s and the 3-rank
   collective 2.85 → 5.80. Enabling them *before* giving the upper pair IPv4, routes and
   netplan persistence **wedges the cluster** with `IBV_WC_RETRY_EXC_ERR` while every
   container still reports `running`. See
   [`results/20260825-upper-mesh/`](results/20260825-upper-mesh).
7. **JIT compiles land *during* inference** — one measured at 5 s inside a request. Warm
   every shape you intend to measure, and discard sweeps containing a `jit_monitor`
   warning.
8. **`MAX_NUM_BATCHED_TOKENS=16384` is a trap** despite vLLM's own log suggesting it.
   That advice assumes intra-node NVLink; here it cost 43% of the KV pool for zero gain.

**Before any benchmark:** `scripts/fabric_gate.sh configs/<your>.env --nccl=full` with the
engine stopped. It gates on liveness, mesh, latency, subnets, ARP-port correctness, config
persistence, transport, RDMA errors, and collective bandwidth — and exits non-zero so a
bad run cannot silently happen.

These are measurements from one cluster, not universal product specifications. See
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

The settled serving profile (2026-08-25) is:

```bash
MAX_MODEL_LEN=1048576          # 1M is free here: memory-bound, not comms-bound
MAX_NUM_SEQS=16
GPU_MEMORY_UTILIZATION=0.80
MTP_NUM_TOKENS=5               # beats 4; matched control 2026-08-24
MAX_NUM_BATCHED_TOKENS=8192    # do NOT raise to 16384 -- see pitfall 8
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

> Earlier documents describe `460800` / `seqs=8` / `0.85` / `MTP=4`. That profile is
> **superseded**; it survives only inside dated result pages, which are frozen to the
> configuration they were measured under. [`config/tp3.env.example`](config/tp3.env.example)
> is authoritative and carries the reasoning for every value.

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
config/       engine env -- runs ON the Sparks         (config/README.md)
configs/      harness targets -- runs on your WORKSTATION (configs/README.md)
docs/         all documentation, indexed and status-tagged (docs/README.md)
scripts/      fabric gate, launchers, benchmark helpers
results/      dated raw run bundles -- frozen, never edited
benchmarks/   machine-readable summary + CHANGELOG
tests/        schema validation for benchmark artifacts
artifacts/    schema for complete future run bundles
```

> [!IMPORTANT]
> **`config/` and `configs/` are different things.** `config/` holds the vLLM engine
> environment that lives on each Spark; `configs/` holds benchmark harness targets that
> live on your workstation. A file from one will not work in the other — each directory
> has a README stating which is which.

Dated directories under `results/` are **frozen**: they record one experiment at the
configuration it was measured under. Supersede them with a new dated directory; never
edit one in place.

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
