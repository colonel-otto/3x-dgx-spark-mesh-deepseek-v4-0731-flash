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

## Where this got to — the honest scoreboard

**Against our own matched two-Spark baseline, on healthy fabric, node count the only
variable.** Every row links to raw per-run data.

| | 2 Spark | 3 Spark | verdict |
|---|---:|---:|---|
| 🟢 **Decode @131K context** | 54.4 tok/s | **72.6** | **+33.6%** — the strongest case for the third node |
| 🟢 **Decode @262K context** | 71.5 | **84.4** | **+17.9%** |
| 🟢 **Aggregate @cc=32** | — | **685.9 tok/s** | above the 618 published upstream figure |
| 🟢 **Max verified context** | 131K | **967,286 tok** | 92% of the 1M ceiling, served without incident |
| 🟡 **Decode below 32K** | 70.8–75.8 | 70.2–76.3 | **parity** — the third node earns nothing here |
| 🟡 **Prefill ≤32K** | 1,913–2,081 | 2,023–2,095 | **parity** (±2%); TTFT slightly favours 3 nodes |
| 🔴 **TTFT @262K** | **158.4 s** | 181.6 s | **2 nodes 13% sooner** |
| 🔴 **Aggregate @cc=16** | **481.3** | 474.8 | 2 nodes win batch throughput |
| 🔴 **Hardware freed** | **1 whole GB10** | none | two nodes leave a spare machine |

**The one-line answer:** *a third Spark buys per-stream speed at long context and costs
you time-to-first-token and batch throughput.* If your prompts are short or your load is
concurrent, **two nodes are the better buy** — and we publish that as plainly as the wins.

### The bad and the ugly, stated up front

- **A node was silently broken for days** at ~15% network speed with every indicator
  green. It distorted every measurement taken before 2026-08-25, and it sat in the 3-node
  arm — the worst possible place. → [`docs/DEGRADED-DATA-CATALOGUE.md`](docs/DEGRADED-DATA-CATALOGUE.md)
- **We chased a bandwidth deficit that never existed** and produced four wrong
  explanations before running the reference tool. → [`docs/BANDWIDTH-COMPARISON.md`](docs/BANDWIDTH-COMPARISON.md)
- **We published a headline that was wrong in both directions** (`+8–17% from 2K upward`)
  and have retracted it in place rather than quietly editing it. → [`docs/WHY-THREE-NODES.md`](docs/WHY-THREE-NODES.md)
- **Two of our own rejections were wrong.** `MAX_NUM_SEQS=32` and the four-HCA fabric were
  both rejected against a fabric that was lying. Both are now production.
- **Expert parallelism and pipeline parallelism do not work here** — dead ends kept so
  nobody re-proposes them. → [`docs/EP3-EXPERT-PARALLEL.md`](docs/EP3-EXPERT-PARALLEL.md) ·
  [`docs/PP3-PIPELINE-PARALLEL.md`](docs/PP3-PIPELINE-PARALLEL.md)

## How the results progressed

Linear, with every reversal visible. **Read down the "what changed" column to see how much
of the improvement came from fixing our own measurements rather than the machines.**

| date | change | decode cc=1 | prefill @32K | aggregate | what it taught |
|---|---|---:|---:|---:|---|
| 08-21 | 2-Spark baseline (`TP=2`) | 48.2 | — | — | the reference to beat |
| 08-21 | 3-Spark `EP=3` | 19–20 | — | 138 @cc16 | ❌ B12X kernel refuses EP — **2.5x slower** |
| 08-21 | 3-Spark `PP=3` | — | — | — | ❌ blocked by MTP + DSA stride |
| 08-21 | 3-Spark `TP=3` + padding patch | 53.9–57.7 | — | — | ✅ TP=3 works and beats 2-node |
| 08-22 | `MAX_NUM_BATCHED_TOKENS=16384` | — | — | — | ❌ trap: 43% of KV for zero gain, reverted |
| 08-24 | `MTP=5` + 1M context | ~80 | — | 374 @cc16 † | ✅ 1M context is free; MTP=5 beats MTP=4 |
| 08-24 | `MAX_NUM_SEQS=32` | — | — | crash | ❌ *rejected — later proved wrong* |
| **08-25** | **degraded node found + rebooted** | **85.6** | **1,034 → 2,095** | **491 @cc16** | ☠️ **+103% prefill from a reboot.** Everything above is suspect |
| 08-25 | four-HCA upper mesh addressed | — | — | — | ✅ 2x fabric bandwidth, soak-validated |
| 08-26 | official `nccl-tests` run | — | — | — | ✅ **no bandwidth gap ever existed** (23.92 GB/s) |
| 08-26 | KV dtype A/B | — | — | — | ✅ free choice — 23/24 cells byte-identical |
| 08-26 | depth sweep, matched 2v3 | 76.3 | — | — | ✅ **the real curve**: parity <32K, +33.6% @131K |
| 08-26 | **`MAX_NUM_SEQS=32` retest** | 87.6 | — | **685.9 @cc32** | ✅ **+46.3%** — the 08-24 rejection was the fabric |
| 08-26 | four-HCA throughput | — | — | flat | ✅ 2x bandwidth buys **no speed** — a null result, published |

† The 374 aggregate is the `MTP=4`/`seqs=16` profile measured in that window, and it was
taken on the **degraded** fabric — the same configuration read **491** after the reboot.
Rows above the 08-25 line are kept because they are the decision trail, not because their
numbers are trustworthy; each is labelled in
[`docs/DEGRADED-DATA-CATALOGUE.md`](docs/DEGRADED-DATA-CATALOGUE.md).

**Two of the three biggest wins came from re-testing our own rejections**, not from new
ideas. That is the most transferable lesson here.
Full trail: [`docs/EXPERIMENT-LOG.md`](docs/EXPERIMENT-LOG.md).

## Standing on other people's work

This repo is a **measurement** project layered on other people's engineering. Where a
number, a patch, or a harness came from someone else, it is named:

| what | whose | how we used it |
|---|---|---|
| TP=3 attention-group padding (`o_groups` 8→9) | [localaiguyy](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark) | **The patch that makes TP=3 correct at all.** Without it vLLM serves fluent nonsense |
| `benchmark_tp3.py`, the 618 tok/s `seqs=32` figure | [localaiguyy](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark) | Our matched harness — and the target that prompted the retest we had wrongly rejected |
| 2-node recipe, `bench-miaai.py`, dual-controller HCA insight | [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | The comparison baseline, and the reason we looked at the upper HCA pair |
| `dspark-vllm-gx10` runtime image | [Anemll](https://github.com/anemll) | The engine everything runs on; `benchmark_prefill.py` is theirs |
| Ring topology, switchless NCCL settings | [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) | Physical layout and the published bandwidth figures we benchmarked against |

**Our contribution is the controlled comparison, not the stack**: matched arms, a fabric
gate that catches silent degradation, and a documented trail of what we got wrong.
Third-party rows in [`benchmarks/measurements.csv`](benchmarks/measurements.csv) carry
`source=external-published` and are never silently mixed with ours.
Full detail: [`CREDITS.md`](CREDITS.md).

## What this project found, in plain terms

Three findings, in order of how much time they cost:

**1. A machine was silently broken for days.** One Spark ran at ~15% of its network speed
with **every status indicator reading healthy** — link up at 200 Gb/s, all error counters
zero. A reboot fixed it. It had been distorting every prior measurement, and because it
sat in the *3-node* arm, the handicap fell on the configuration under test. The re-runs
found it had **flattened** the comparison in both directions: a strongly depth-dependent
per-stream advantage was compressed into a uniform ~13% band, so the third node looked
mildly useful everywhere when in fact it is useless below 32K and worth +33.6% at 131K.
→ [`docs/DEGRADED-DATA-CATALOGUE.md`](docs/DEGRADED-DATA-CATALOGUE.md)

**2. We chased a network deficit that never existed.** Our own tool measured 5.80 GB/s
where published results showed ~20. We produced **four** plausible explanations. All four
were wrong. The official NVIDIA benchmark, on the same unchanged machines, reads
**23.92 GB/s** — *above* the 20.84 we were envying. Our harness was built to time our
workload, not to measure peak bandwidth; we used it outside its purpose.
→ [`docs/BANDWIDTH-COMPARISON.md`](docs/BANDWIDTH-COMPARISON.md)

**3. Two settings we worried about turned out not to matter.** The KV cache dtype
(`nvfp4_ds_mla` vs `fp8_ds_mla`) is identical on every axis — same memory, same speed, and
**23 of 24 matched tests produced byte-identical output**. It was a free choice all along.
→ [`docs/KV-QUALITY-LONG-CONTEXT.md`](docs/KV-QUALITY-LONG-CONTEXT.md)

> **The through-line:** most of what we found was about **how we measure**, not what we
> own. The hardware was largely fine; our instruments misled us in both directions — a
> healthy-looking signal hid a broken machine, and later a broken-looking signal (a frozen
> counter) hid a perfectly healthy one. Both classes are catalogued so the next person
> recognises them.

## Can the extra memory buy a better model? No.

A reasonable expectation is that three nodes (~363 GiB) allow a higher-quality checkpoint
than two (~242 GiB). **It does not**, and the reason is specific:

| option | size | verdict |
|---|---:|---|
| what we run (FP4 experts) | **167 GB** | leaves ~195 GiB for KV → **1M context** |
| FP8 "upcast" | 307 GB | **same numbers, bigger container** — a documented lossless cast |
| BF16 | 1,137 GB | 3.1x over budget |

The FP8 and BF16 files on Hugging Face are conversions *of the FP4 weights*, not
higher-fidelity originals — the full-precision master was never released. Loading the
307 GB version would cost you 1M context and the fast MoE kernel **for an identical
model**.

The third node buys **latency at depth**, not quality or capacity.

## Is the third node worth it?

**It depends on context depth, and it depends on concurrency.** Those are two separate
axes and we measured them separately. A single percentage cannot answer this question —
the honest answer is a workload description.

### Axis 1: context depth, single stream

Measured 2026-08-26 on healthy fabric, matched arms, node count the only variable, 7 reps
per cell, median, prefill excluded. Per-stream decode tok/s:

| context | 2 Spark | 3 Spark | 3-node gain |
|---:|---:|---:|---:|
| 2,036 | 75.8 | 76.3 | +0.8% |
| 8,081 | 72.4 | 72.6 | +0.3% |
| 32,268 | **70.8** | 70.2 | −0.9% |
| 129,006 | 54.4 | **72.6** | **+33.6%** |
| 257,993 | 71.5 | **84.4** | **+17.9%** |

**Below 32K the third node buys nothing** — three cells inside run-to-run noise, one
negative. **The crossover is between 32K and 131K**, and past it the gain is large. The
mechanism is KV pressure: 1,844,001 tokens on two nodes against ~4.5M on three. Below 32K
neither pool is under pressure and decode is bound by per-token compute, which a third rank
does not improve.

Two findings from the same run that complicate the story:

- **Time to first token favours TWO nodes at depth** — 158.4 s vs 181.6 s at 262K (13%
  sooner), 72.4 s vs 77.1 s at 131K. Deep prompts with short answers are a two-node
  workload; deep prompts with substantial output are a three-node one.
- **The depth curve is a U, not a decay.** At 262K this cluster decodes *faster* than at
  8K (84.4 vs 72.6), on both node counts. Likely MTP acceptance rising with context.

### Axis 2: concurrency, short prompt

Measured 2026-08-25 on healthy fabric, 18-token prompt — this measures request
concurrency, not context depth:

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

Both axes are true at once, and they are not in conflict: three nodes win per-stream **at
depth**, two nodes win aggregate **under concurrency**.

### Choose by workload

| Your workload | Node count |
|---|---|
| Long context (>100K) with substantial generation | **3** — +18–34% per stream |
| Interactive coding under ~32K | **either** — measured parity |
| Deep one-shot prompt, short answer | **2** — first token 6–13% sooner |
| Several concurrent users, agent swarm, batch | **2** — 12–19% more aggregate, third GB10 freed |

Two things the third node does **not** buy: prefill throughput (parity at every depth
tested), and usable KV capacity (`vllm:num_preemptions_total` has read **0 in every test
ever run here**, including a 4×200K test designed specifically to make KV bite — the KV
advantage shows up as *decode speed at depth*, not as capacity headroom that binds).

> **Retracted:** an earlier headline claimed **"+8–17% per-stream from 2K context upward."**
> That was degraded-fabric data ([#14](../../issues/14)) and it is wrong in both
> directions — no benefit below 32K, more than double the claim above 100K. Kept and
> labelled in [`docs/WHY-THREE-NODES.md`](docs/WHY-THREE-NODES.md).

Evidence: [`results/20260826-decode-depth-2v3/`](results/20260826-decode-depth-2v3) ·
[`results/20260825-decode-2v3/`](results/20260825-decode-2v3) ·
[`results/20260825-prefill-2v3/`](results/20260825-prefill-2v3) ·
[`results/20260825-deep-concurrency/`](results/20260825-deep-concurrency)

> Historical numbers from before 2026-08-25 were taken while one node ran at ~15% of its
> collective bandwidth and are **provisional**. Every affected measurement is itemized in
> [`docs/DEGRADED-DATA-CATALOGUE.md`](docs/DEGRADED-DATA-CATALOGUE.md) — kept deliberately,
> because knowing what bad data looked like is how you spot the next batch. See also
> [issue #14](../../issues/14).

## Status: what is settled, what is open

| Question | Status | Where |
|---|---|---|
| Is the 3rd node worth it? | ✅ **Depends on depth** — parity below 32K, **+33.6% at 131K**; no for batch | tables above |
| Is our fabric slow vs published rings? | ✅ **No — 23.92 GB/s, above reference** | [#18](../../issues/18) |
| Does the 4-HCA fabric survive load? | ✅ **Yes** — 408/408 requests, 0 RDMA errors | [#17](../../issues/17) |
| Is `nvfp4_ds_mla` costing us quality? | ✅ **No** — 23/24 cells byte-identical vs `fp8` | [#16](../../issues/16) |
| Can extra RAM buy a better model? | ✅ **No** — no higher-fidelity checkpoint exists | section above |
| Was prefill behind the 2-node recipe? | ✅ **Resolved** — it was the degraded node | [#11](../../issues/11) |
| Does 4-HCA improve *throughput*? | ✅ **No** — decode flat vs a matched 2-HCA arm; kept for redundancy | [#17](../../issues/17) |
| Is `MAX_NUM_SEQS=32` viable? | ✅ **Yes — +46.3% at cc=32**, now the production value | [#10](../../issues/10) |
| Is prefill degraded by the 3rd node? | ✅ **No** — *faster* below 32K, 6–15% slower past 100K | [`results/20260826-decode-depth-2v3/`](results/20260826-decode-depth-2v3) |
| Do pre-08-25 benchmarks need re-running? | ⏳ **Mostly done** — decode, prefill, deep-concurrency, seqs=32 redone | [#14](../../issues/14) |
| Matched `benchmark_prefill.py` above 32K? | ⏳ **Missing** — TTFT corroborates, but no designed run | [`results/20260825-prefill-2v3/`](results/20260825-prefill-2v3) |

**Every ✅ above was measured on healthy fabric with matched arms.** Every ⏳ is named
rather than quietly assumed.

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

9. **No Spark-to-Spark data may cross Wi-Fi.** Wi-Fi is operator access and API responses
   only. If a fabric route disappears the kernel falls back to the management path
   silently — everything still pings while inter-node traffic crawls (fabric 0.47–0.93 ms
   vs Wi-Fi 3–135 ms). Gated as `egress:*`; it bit us once already
   ([issue #13](../../issues/13)).

**Before any benchmark:** `scripts/fabric_gate.sh configs/<your>.env --nccl=full` with the
engine stopped. It gates on liveness, mesh, latency, subnets, ARP-port correctness, config
persistence, **peer egress device**, transport, RDMA errors, and collective bandwidth —
and exits non-zero so a bad run cannot silently happen.

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
