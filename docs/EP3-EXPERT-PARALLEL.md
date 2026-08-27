# 3-Spark expert parallelism (TP=1 / DP=3 / EP=3)

> [!WARNING]
> **Every tok/s on this page is a diagnostic signature, not one of our results.** This
> experiment ran on **both** known bad conditions at once, and is the worst-provenance
> page in the repo:
> 1. **Degraded fabric** ([#14](../../issues/14)) — spark1 was at ~15% of its collective
>    bandwidth, with zero error indicators. Not found until 2026-08-25.
> 2. **TCP fallback** — the configs in
>    [`../results/20260821T031300Z-3spark-ep3/`](../results/20260821T031300Z-3spark-ep3/)
>    set `NCCL_IB_DISABLE=1` and `NCCL_NET=Socket` on all three ranks. **No RDMA byte was
>    carried.** See §3, which explains why we believed at the time that this was forced.
>
> **If you are reproducing this and see numbers like these, that is the point of the
> page** — match against [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md), which
> maps each symptom to its cause and fix. These are not the numbers to beat.
>
> **What survives:** the kernel finding in §2 — `flashinfer_b12x` refuses expert
> parallelism by an explicit source-code check, so enabling EP forces a fallback to a
> generic kernel. That rests on a **control** (EP=2 vs TP=2 on the *same two nodes*), so
> it is independent of node count, and on a source-code check and a `ValueError` that are
> independent of fabric entirely. The expert split (86/85/85) also survives — it came from
> the engine log, not a timer.
>
> **What is void:** every absolute tok/s figure, including EP=3's 19–20, EP=2's 21–22, the
> baseline's 49–55, the aggregate sweep, and the "~2.5x" and "~10%" ratios derived from
> them. The direction (EP is much slower) is not in doubt; the magnitude has never been
> re-measured on healthy fabric with RDMA.
>
> **RETRACTED:** §3's claim that RoCE cannot work across three nodes without a switch.
> See the correction in that section.
>
> **Current healthy-fabric numbers instead:** [`../README.md`](../README.md#is-the-third-node-worth-it)
> for 2v3 decode, [`DECISIONS.md`](DECISIONS.md) for the settled config.

Captured 2026-08-21. Raw data and the exact configs that ran:
[`../results/20260821T031300Z-3spark-ep3/`](../results/20260821T031300Z-3spark-ep3/).

**Summary: it runs, it shards correctly, and it is ~2.5x slower. The slowdown is the
MoE kernel, not the third node — and that is proven by a control experiment, not
argued.** (The *mechanism* is settled; the 2.5x *magnitude* is a degraded-fabric,
TCP-transport figure — see the banner above and §2.)

This supersedes the design assumption in PR #1 that 3-way sharding was impossible.
It is possible. It is just not currently fast.

---

## 1. Results — degraded-fabric, TCP-transport signatures

> [!CAUTION]
> **Do not quote any tok/s in this section.** All three arms ran on the degraded fabric,
> and the two EP arms additionally ran over `NCCL_NET=Socket`. The KV-token and
> concurrency columns are engine-reported capacity figures and are not affected by
> transport; the tok/s columns are.

| Config | Nodes | MoE kernel | Decode tok/s | KV tokens | Concurrency @460800 |
|---|---:|---|---:|---:|---:|
| TP=2 baseline | 2 | `flashinfer_b12x` | **49–55** | 1,771,152 | 3.84x |
| DP=2 / EP=2 | 2 | `auto` | 21–22 | 1,632,510 | 3.54x |
| DP=3 / EP=3 | 3 | `auto` | 19–20 | **4,065,871** | **8.82x** |

EP=3 aggregate throughput:

| Concurrent | Aggregate tok/s | Per stream |
|---:|---:|---:|
| 1 | 17.23 | 17.23 |
| 4 | 49.37 | 12.34 |
| 8 | 95.57 | 11.95 |
| 16 | 138.42 | 8.65 |

The 2-Spark baseline reaches 174 tok/s at C=16, so EP=3 trails on aggregate
throughput as well as on latency.

### Sharding works exactly as predicted

```
[EP Rank 0/3] Expert parallelism is enabled. Expert placement strategy: linear.
Local/global number of experts: 86/256
```

Ranks 1 and 2 report 85/256. **86/85/85**, matching `expert_map_manager.py`:

```python
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
```

Expert parallelism assigns *whole experts*, so a power-of-two expert count never
needs to divide by the node count. TP=3 remains genuinely impossible (64 heads,
4096 hidden, 256 experts are all indivisible by 3) — EP is the way around it.

---

## 2. The control experiment (read this before optimizing anything) — SURVIVES

> [!NOTE]
> **This is the finding that outlived the fabric.** The control holds node count constant
> and changes only the kernel, so it is unaffected by how many nodes the fabric had to
> reach. The **root cause** below is stronger still: it is a source-code check and a
> `ValueError` at model init, which no amount of bandwidth changes.
>
> **One asymmetry to be honest about.** The baseline arm ran RoCE; both EP arms ran
> `NCCL_NET=Socket` (`results.json`, `transport` field on each configuration). So the
> **EP=2 vs EP=3** comparison is transport-matched and the ~10% node cost derived from it
> is internally consistent, but the **EP vs TP=2 baseline** comparison is not — it varies
> kernel *and* transport together. The ~2.5x ratio therefore bounds nothing precisely; it
> mixes two effects. What is certain is the direction and the mechanism, not the number.

The obvious reading of the table — "the third node halved throughput" — is **wrong**.

Running **EP=2 on the same two nodes as the baseline** holds node count constant and
changes only the kernel:

- EP=2, 2 nodes: **21–22 tok/s**
- EP=3, 3 nodes: **19–20 tok/s**

Adding the third node costs roughly **10%** (21 -> 19). The remaining **2.5x** is
entirely the MoE kernel.

### Root cause

`flashinfer_b12x` — the Spark-tuned kernel the baseline uses — refuses expert
parallelism by declaration, in
`vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py:597-603`:

```python
not moe_parallel_config.use_ep
and moe_parallel_config.ep_size <= 1
and not moe_parallel_config.use_all2all_kernels
and not moe_parallel_config.enable_eplb
```

Attempting EP with it fails at model init:

```
ValueError: Mxfp4 MoE backend 'B12X_MXFP4' does not support the deployment
configuration since kernel does not support parallel config
FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=3, ep_size=3, use_ep=True, ...)
```

**B12X and expert parallelism are mutually exclusive by design.** Enabling EP forces
`--moe-backend auto` plus `VLLM_USE_B12X_MOE=0`, and that fallback is the cost.

So the benchmark above is *not* a clean node-count comparison. It is
**b12x on 2 nodes vs a generic kernel on 3**.

---

## 3. ~~RoCE does not work across three nodes on this cluster~~ — RETRACTED

> [!WARNING]
> **This section's conclusion is wrong, and it is the most consequential error on this
> page.** RoCE **does** work across all three nodes on this cluster, without a switch and
> without recabling.
>
> | | Claimed here (2026-08-21) | Measured since |
> |---|---|---:|
> | 3-node RDMA | "requires a switch — a hardware constraint" | **works**, `via NET/IB/4` + `NET/IB/5` |
> | 3-rank collective busbw @16 GiB | n/a — believed impossible | **23.92 GB/s** |
> | vs the published NVIDIA 3-Spark reference | n/a | 20.84 — **we are above it** |
>
> Evidence: [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md) and
> [`../results/20260826-nccl-controlled/`](../results/20260826-nccl-controlled) (12 runs,
> official `all_gather_perf`, NCCL 2.30.7, **zero `NET/Socket` occurrences in any run**).
>
> **What we missed.** Two things, neither of them hardware:
> 1. **`NCCL_IB_SUBNET_AWARE_ROUTING=1`** — this makes NCCL select the HCA whose subnet
>    reaches each peer instead of pairing by device index, which is exactly the failure
>    diagnosed below. It is the whole fix for the triangle, and it is undocumented in
>    NVIDIA's public env reference. See [`WHY-THREE-NODES.md`](WHY-THREE-NODES.md) §5 and
>    [`DECISIONS.md`](DECISIONS.md).
> 2. **The upper `roceP2p` HCA pair had no IPv4.** Giving all four controllers persistent
>    `/30` addressing doubled fabric bandwidth —
>    [`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh).
>
> **The falsified workarounds list below is still accurate** — none of those flags fixes
> it. `NCCL_IB_SUBNET_AWARE_ROUTING` simply was not on the list, because we did not know
> it existed.
>
> **Kept as a diagnostic signature.** The `ibv_modify_qp ... Connection timed out` failure
> below is exactly what index-based HCA pairing looks like on a switchless triangle. If
> you hit it, you need subnet-aware routing — not a switch.

The ConnectX-7 cabling is a **triangle of point-to-point links, not a switched
fabric**:

```
              sparkmain
           f0 /         \ f1
  109.25 Gb/s /           \ 109.08 Gb/s
             /             \
       spark1 ------------- spark-sep
          f1   109.07 Gb/s   f1
```

All 3 cables live, all 6 ports LinkUp @200 Gb/s, zero port errors. Every node
reaches both peers — but **on a different port per peer**.

NCCL's RDMA path assumes any rank's NIC can reach any other rank's NIC. Here
`sparkmain.f0` has no cable to `spark-sep`, so NCCL pairs the wrong device and stalls:

```
ibv_modify_qp failed with 110 Connection timed out, on dev rocep1s0f0:1,
curr state INIT, next state RTR, local GID index 7,
local GID ::ffff:192.168.99.1, remote GID ::ffff:192.168.99.6
```

Falsified workarounds — none of these fix it:
`NCCL_IB_HCA` ordering, `NCCL_IB_GID_INDEX`, `NCCL_IB_ADDR_RANGE`,
`NCCL_IB_MERGE_NICS=0`, `NCCL_CROSS_NIC=0`, distinct per-cable subnets, single-HCA.

**Only working transport *found on 2026-08-21*: `NCCL_IB_DISABLE=1` + `NCCL_NET=Socket`**
(TCP over the fabric). The 200 Gb/s RoCE was therefore **not carrying collectives** in
this EP=3 run — which is why every tok/s on this page is a TCP-fallback signature.

> ~~Using RDMA across all three nodes requires a **switch**, or a topology where one
> NIC reaches every peer. This is a hardware constraint, not a config bug.~~
>
> **RETRACTED.** It was a config gap, not a hardware constraint. The missing piece was
> `NCCL_IB_SUBNET_AWARE_ROUTING=1`. No switch was ever needed; the same triangle now
> carries 3-rank RDMA at 23.92 GB/s.

---

## 4. Gotchas that cost real time

### Internal vs external DP load balancing

`--data-parallel-rank` silently forces **external-LB** mode
(`vllm/engine/arg_utils.py:1975`):

```python
data_parallel_external_lb = (
    self.data_parallel_external_lb or self.data_parallel_rank is not None)
```

External LB then **forbids** `--headless` (`vllm/v1/engine/utils.py:1243,1297`):

```
RuntimeError: Remote engine 1 must not use --headless in external or hybrid dp lb mode
```

For **one endpoint** with vLLM balancing internally — which is what matches the
baseline harness — workers use `--data-parallel-start-rank N` together with
`--headless`, and the head passes **no rank flag at all**.

### ufw, asymmetric across nodes

`ufw` was active on **spark1 only** (`INPUT policy DROP`); the other two nodes had it
inactive. ICMP was permitted and `ct state ESTABLISHED` kept the running 2-Spark
deployment alive, so **`ping` reported every path healthy** while every *new* inbound
TCP flow — exactly what a 3-node bring-up creates — was dropped silently, with no log
and no RST.

**When a multi-node job times out on connect but ping succeeds, check the firewall on
every node before theorizing about topology or NCCL device selection.**

### Backgrounded listeners over SSH

`nc -l` / `nohup ... &` listeners started through a one-shot SSH command die on
disconnect and produce a **false "connection refused"**. Confirm a listener is
actually bound with `ss -lnt` before trusting a negative connectivity result.

### Restarting the 2-Spark stack

The 2-Spark deployment has **no `.env`** in its directory; it must be started with
its script, or `NODE_RANK` interpolates empty:

```bash
# correct
sudo bash scripts/start-node.sh config/worker.env   # on the worker, FIRST
sudo bash scripts/start-node.sh config/head.env     # then the head

# wrong -> vllm serve: error: argument --node-rank/-r: expected one argument
sudo docker compose up -d
```

---

## 5. Reproducing

Configs are in `results/20260821T031300Z-3spark-ep3/`. They deploy to a **separate**
directory (`~/localai/dsv4-ep3`) so the working 2-Spark stack is never modified.

```bash
--tensor-parallel-size 1
--data-parallel-size 3
--data-parallel-size-local 1
--enable-expert-parallel
--moe-backend auto            # b12x refuses EP
```

Environment, all three nodes **as this experiment ran it**:

```
VLLM_USE_B12X_MOE=0
NCCL_IB_DISABLE=1     # <-- do NOT copy: forces TCP, see below
NCCL_NET=Socket       # <-- do NOT copy: forces TCP, see below
```

> [!CAUTION]
> **Do not copy the last two lines.** They disable RDMA and were only needed because §3's
> diagnosis was wrong. Reproducing this experiment as written reproduces the TCP-fallback
> signature, not an EP measurement. For an EP retry, use the fabric settings in
> [`../config/tp3.env.example`](../config/tp3.env.example) — `NCCL_NET=IB` with
> `NCCL_IB_SUBNET_AWARE_ROUTING=1` — and confirm `via NET/IB/*` in `NCCL_DEBUG=INFO`
> before recording a number.

Per node: head gets `DP_START_RANK_ARG=` (empty); workers get
`DP_START_RANK_ARG=--data-parallel-start-rank N` and `HEADLESS=1`.

Fabric addressing used `192.168.99.x/30` per cable with the DP leader on a loopback
`192.168.99.100/32` routed over each peer's own direct cable. **These addresses and
routes are runtime-only and do not survive a reboot** — the original
`.100`/`.101`/`.102` addressing was restored afterwards and the baseline verified
back at 41–44 tok/s / 3.86x.

---

## 6. Not tested

Stated explicitly so nobody assumes these were covered:

- **Multi-QP RDMA** (`ib_write_bw -q 4/8`). The 109 Gb/s figures are single-QP
  defaults, ~55% of line rate. This may be a harness ceiling, not a hardware one.
- **Actual TCP throughput of the `NCCL_NET=Socket` path** — the transport EP=3
  actually runs on, and it was never measured.
- **Alternative EP-capable kernels**: `triton`, `deep_gemm`, `machete`. Only `auto`
  was run, and which kernel it selected was never confirmed.
- **EP=3 with speculation disabled.**
- **Long-context workloads.** The entire case for EP=3 is 8.82x concurrency at
  460,800 tokens, and the benchmarks above generate only 128/256 tokens. The regime
  EP is supposed to win was not measured.
- **`--max-num-seqs` above 16.** That cap was tuned for 2 nodes at 3.84x KV; with
  8.82x headroom it is likely leaving aggregate throughput unclaimed.

## 7. Recommendation

> [!NOTE]
> **Superseded in part.** The recommendation to stay off EP still stands — it rests on the
> kernel finding in §2, which survived. But the deployment recommendation moved: this
> repo now runs **3 nodes on TP=3**, not the 2-Spark TP=2 baseline. See
> [`../README.md`](../README.md#is-the-third-node-worth-it) and
> [`DECISIONS.md`](DECISIONS.md). And item 3 below is retracted with §3 — no switch is
> needed.

Stay off expert parallelism for interactive use — it loses the B12X kernel, and the
tuning is already settled without it.

Revisit 3-node EP when any of these changes:

1. A B12X build gains expert-parallelism support.
2. Long-context or high-concurrency capacity outweighs latency.
3. ~~A switch is added so RoCE carries collectives instead of TCP.~~ **Retracted** — RoCE
   already carries 3-node collectives; see §3. If EP is retried, it should be retried on
   RDMA, which no EP run here ever used.

The highest-value next experiment is **item 6.3** — trying `triton` / `deep_gemm`
explicitly. If any EP-capable kernel closes the gap, the whole verdict changes. Any such
retry must be run on healthy fabric with `via NET/IB/*` confirmed, since **no EP number on
this page was.**
