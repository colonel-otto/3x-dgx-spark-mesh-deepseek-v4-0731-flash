# Handoff — DeepSeek-V4-Flash on 3×DGX Spark, as of 2026-08-25

Read this first. It states what is running, what is settled, **what is now suspect**, and
what to do next.

> ## ⚠ Read this before trusting any number in this repo
>
> On 2026-08-25 we found that **spark1 had silently degraded RDMA fabric** — every NCCL
> collective involving it ran at **~0.7 GB/s against 4.6 GB/s** for the healthy pair, a
> **6.8x deficit with zero error indicators**. A reboot cleared it, and prefill roughly
> doubled with no configuration change.
>
> **Every multi-node measurement recorded before 2026-08-25 was taken on that degraded
> fabric.** They are provisional until re-run —
> [issue #14](https://github.com/colonel-otto/3spark-dsv4/issues/14).
>
> Full detail: [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md).

---

## 1. What is running right now

**Verified live 2026-08-25 08:2x UTC.** Healthy and serving.

| | value |
|---|---|
| Endpoint | `http://192.168.1.223:8100` (LAN), `http://localhost:8100` on the head |
| Model id | `deepseek-v4-flash-0731` |
| Weights | `/models/dsv4-abliterated` — DeepSeek-V4-Flash-0731, abliterated, 156 GiB / 48 FP8 shards |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| Topology | TP=3, PP=1, NNODES=3, `flashinfer_b12x` MoE |
| `MAX_MODEL_LEN` | 1,048,576 |
| `MTP_NUM_TOKENS` | 5 |
| `MAX_NUM_SEQS` | 16 |
| `GPU_MEMORY_UTILIZATION` | **0.80** (was 0.85; see §3) |
| `kv-cache-dtype` | `nvfp4_ds_mla` |
| CUDA-graph capture | 96 (`16 × (5+1)`) |
| GPU KV cache | **4,493,602 tokens** |

### The three nodes

| rank | host | note |
|---:|---|---|
| 0 | `sparkmain` | head — **the only node that serves**; workers emit nothing |
| 1 | `spark1` (`gx10-e146`) | worker. **This is the node that had degraded fabric.** Rebooted 2026-08-25. |
| 2 | `spark2` (`gx10-6b41`) | worker. **`spark2` and `spark-sep` are the same machine** — two aliases, one host. Not four nodes. |

## 2. Current performance

Measured with upstream anemll's `benchmark_prefill.py` **unmodified**, server-side timer,
fresh seed so prefix caching cannot hit, client agreeing within 1%.

**Prefill — at parity with the upstream reference:**

| input tokens | ours | anemll reference (TP=2) | % of reference |
|---:|---:|---:|---:|
| 1,024 | **2,022.6** | 2,033.0 | **99.5%** |
| 8,192 | **2,069.5** | 2,184.2 | **94.8%** |
| 32,768 | **2,094.9** | 2,176.1 | **96.3%** |

> **The third node does not buy prefill.** Measured 2026-08-25 on our own hardware, both
> arms on this same production profile and the same unmodified harness:
>
> | input tokens | our TP=2 | our TP=3 | 3-node gain |
> |---:|---:|---:|---:|
> | 1,024 | 1,913.1 | 2,022.6 | +5.7% |
> | 8,192 | 2,080.6 | 2,069.5 | **−0.5%** |
> | 32,768 | 2,065.8 | 2,094.9 | +1.4% |
>
> Parity within ±2% at realistic depths. This is the expected shape: TP allreduce volume
> in prefill scales with batched-tokens × hidden-size, so a third rank adds collective
> cost that the extra compute must earn back, and at these depths it does not. vLLM
> maintainers say the same — PP is preferred with *"more communication volume from
> prefills, cross-node"* ([#10118](https://github.com/vllm-project/vllm/discussions/10118)).
>
> **The case for the third node therefore does not rest on prefill**, and the
> deep-concurrency re-run has 2-node reaching first token 1.35x sooner. What is still
> untested is **decode**: the numbers below are TP=3 only, and the "+8–17% per-stream at
> long context" claim that justifies three nodes remains a *degraded-fabric* measurement.
> See [`../results/20260825-prefill-2v3/`](../results/20260825-prefill-2v3).

**Decode** (warm, idle engine, 3 reps, `bench_tp3.py`):

| | value | vs pre-fix baseline |
|---|---:|---:|
| cc=1 median | **85.6** tok/s | 80.4 (+6%) |
| cc=16 median | **491.0** tok/s | 374.2 (+31%) |
| cc=16 peak | **593.1** tok/s | 374.2 (+59%) |

Both numbers are post-fabric-fix and are the only ones in this repo not taken on the
degraded fabric.

## 3. What is settled

| Finding | Status |
|---|---|
| **Prefill parity** | **Reached 2026-08-25.** The ~2x gap was one degraded node, not software. |
| `GPU_MEMORY_UTILIZATION=0.80` | Shipped. Was +14% prefill at 78K on the old fabric; **re-confirm on the healthy one**. |
| `MTP_NUM_TOKENS=5` beats 4 | Settled 2026-08-24 via matched control. Acceptance counters are compute-local, so probably unaffected by the fabric. |
| 1M context is free | Settled. Memory-bound, not communication-bound, so likely still valid. |
| `MTP_NUM_TOKENS=0` is invalid | vLLM rejects `num_speculative_tokens: 0`; the service fails to start. **Use 1 as the floor.** MTP=1 leaves prefill flat but collapses decode to ~47 tok/s. |
| KV dtype is not a speed lever | fp8 vs nvfp4 measured identically. Do not switch for throughput. |
| NVFP4 KV quality to 464K | Clean at every depth, single-request. Concurrency half still open (#12). |
| Patch 4 (shared expert) | Does not apply to vLLM 0.25.2 — already handled by generic substring mapping. |
| `NCCL_IB_MERGE_NICS` | ~~No-op~~ **RE-OPENED 2026-08-25** — measured on the degraded fabric, and published 3-Spark results are 5-6x above ours. See §4a. |
| GB10 GPU health | 93.3 TFLOP/s bf16, 236 GB/s memory, 2,470 MHz under load vs a **2,418 MHz application-clock spec** (3,003 MHz is a hardware ceiling, not a target). All healthy. |

## 4. What is now SUSPECT — issue #14

Re-run these on the healthy fabric. Priority order:

1. **Decode baselines** (374.2 cc=16, ~80 cc=1). These anchor most other conclusions in
   `docs/`. Already partly superseded by §2, but the full concurrency sweep needs redoing.
2. **`MAX_NUM_SEQS=32` rejection.** It died on an `_ALLGATHER_BASE` timeout with KV at
   **2.8%** — a degraded link is a plausible cause of exactly that crash. **seqs=32 may
   well be viable now.** See [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md).
3. **2-node vs 3-node comparison.** spark1 was in the 3-node arm, so the comparison was
   unfair to three nodes.
4. **EP=3 and PP=3.** Communication-heavy, disproportionately penalised.
5. **`GPU_MEMORY_UTILIZATION` 0.80 vs 0.85.** The +14% was measured on the bad fabric.
6. **MTP=4 vs 5 aggregate throughput.** Acceptance counters are fine; aggregates are not.

## 4b. ✗ FALSIFIED 2026-08-25: adding the `roceP2p` HCAs to `NCCL_IB_HCA`

**Do not set `NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`.** It wedges
the cluster. Tried, failed, rolled back.

**Why it looked right.** Upstream MiaAI-Lab's `.env.dspark.example` documents that a GB10
QSFP port enumerates as **two virtual NICs** (2× PCIe Gen5 x4 controllers, ~100G each) and
that *"with only one HCA in `NCCL_IB_HCA` the link runs at half the port."* LLDP confirms
the pairing on our hardware — `enP2p1s0f0np0` sees the same neighbour as `enp1s0f0np0`.
A direct pairwise A/B measured a real gain:

| sparkmain↔spark1, per-node HCA | 16 MiB | 64 MiB |
|---|---:|---:|
| one controller (current) | 5.56 / 5.53 | 5.72 / 5.12 |
| both controllers + MTU 9000 | 9.07 / 9.50 | 8.34 / 8.45 |

**Why it fails anyway — the anti-pattern.** The `roceP2p` pair has **no IPv4 address at
all**, only link-local IPv6, while the routed mesh runs IPv4 on the `rocep1` pair. Adding
them makes NCCL select all four devices, emit repeated `GID table changed` warnings, and
then fail live traffic:

```
NCCL WARN NET/IB: Got completion from peer 192.168.101.2 with status=IBV_WC_RETRY_EXC_ERR(12)
  localGid fe80::4ebb:47ff:fe2e:5fa6  remoteGids fe80::32c5:99ff:febe:6b46  hca roceP2p1s0f0
```

Both GIDs are `fe80::` link-local — RDMA over an unaddressed, unrouted path. Failures hit
both sparkmain↔spark2 and spark1↔spark2.

**The trap that makes this dangerous.** All three ranks **completed NCCL init** and the
container stayed `running` while RDMA completions were failing. Init proves initial
connectivity, not a healthy fabric, and the container has **no health check**, so Docker
cannot flag the degraded state. The engine simply never finished loading — 10+ minutes of
`shm_broadcast` "No available shared memory broadcast block" with no error surfaced.

**Correct direction if this is revisited:** give the `roceP2p` pair stable IPv4 addressing
and routing *first*, validate it independently, and only then widen `NCCL_IB_HCA`. Until
then keep the fully-configured mesh pair, `=rocep1s0f0,rocep1s0f1`.

Note `NCCL_IB_HCA==...` is **not** a typo: the first `=` separates the variable, the
second requests exact HCA-name matching.

**Kept from this attempt:** MTU 9000 on all four controllers (persisted via netplan on all
three nodes). Harmless with the narrow HCA list, and prerequisite if the `roceP2p` pair is
ever addressed properly.

## 4a. ⚠ Our fabric may STILL be ~5x slower than this hardware does

**This outranks every tuning question below it.** Independent measurements on the
**identical 3-Spark ring topology** report far higher NCCL bandwidth than we get:

| source | topology | busbw |
|---|---|---:|
| [NVIDIA forum (Turtle7777)](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) | 3-Spark ring, allgather 16 GB | **20.84 GB/s** |
| same thread, NVIDIA staff expectation | 3-Spark ring | ~24 GB/s |
| [route179.dev](https://route179.dev/2026/07/21/dgx-spark-nccl-roce-benchmarking/) | 2-Spark allgather | 22.1 GB/s |
| **ours (2026-08-25, post-fix)** | 3-Spark ring | **3.25 GB/s** 3-rank, ~4.6 pairwise |

The detail that makes this worth chasing: **that same forum user first measured 2.86
GB/s** — very close to ours — and traced it to NCCL binding the wrong interface
addresses. Fixing it took them to 18–21 GB/s.

**Prime suspect: `NCCL_IB_MERGE_NICS`.** GB10 is PCIe Gen5 x4 per device, so NVIDIA uses
ConnectX-7 multi-host mode in which each physical QSFP port appears as **two logical
interfaces capped at 100G each**. Without merging, NCCL uses only one.

**The honest caveat:** every published merge result is from a **2-node** setup where both
ports face the same peer. In our ring the two ports face **different neighbours**, and no
source we found says whether merging is valid or even coherent there. Strong lead, not a
known fix.

Two things that ARE settled and shape any answer:

- **GPUDirect is architecturally impossible on GB10**, not merely absent. NVIDIA staff:
  memory from `cudaMalloc` *"cannot be coherently accessed by the CPU complex nor by I/O
  peripherals"*; `nvidia-peermem`, `dma-buf` and GDRCopy are all non-functional
  ([source](https://forums.developer.nvidia.com/t/dgx-spark-gpudirect-rdma/348787)).
  Every TP collective bounces through system memory over PCIe. That is a fixed tax no
  flag removes.
- **A 4-node Spark benchmarker found RoCE 2x SLOWER than TCP/socket** on their setup
  ([source](https://forums.developer.nvidia.com/t/multi-node-dgx-spark-cluster-4x-k3s-sglang-vllm-connectx-7-sr-iov-full-benchmark-matrix/365555)).
  This contradicts route179's RDMA results, so treat it as setup-dependent — but it is a
  cheap A/B given GPUDirect is off regardless.

**Do not tune vLLM prefill/parallelism flags before resolving this number.** If 3.25
becomes ~18 GB/s, every node-count conclusion in this repo changes again — including the
prefill parity in §2 and the EP=3 / PP=3 / seqs=32 rejections, all of which were decided
against a communication budget that may be ~5x too small.

**The encouraging read:** we were beating the 2-node reference on decode *while one node
ran at 15% of its collective bandwidth*. Every tuning conclusion here was reached under a
communication handicap — including the 0.49 GB/s figure that anchored the "GB10 has a
~0.5 GB/s ceiling" analysis, which was itself measured on the degraded fabric. Re-running
the matrix should find **better** settings, not merely confirm the old ones.

## 5. Operating the cluster

### Restart

`sudo systemctl restart dsv4.service` on sparkmain. It verifies config across all three
ranks, starts workers → 15 s → head, and waits for `/health`. **Cold start is ~7 min;
budget ~12 min for a full cycle.** Warn the user first — VS Code Remote-SSH forwards
these ports and drops when they go away.

Config lives at `~/localai/dspark-vllm-gx10/config/tp3.env` on **each** of the three
nodes. **A mismatch in any parallelism flag hangs startup forever with no error.** Verify
before restarting:

```bash
KEYS='MAX_MODEL_LEN|MAX_NUM_SEQS|MTP_NUM_TOKENS|GPU_MEMORY_UTILIZATION|TP_SIZE|NNODES'
for h in sparkmain spark1 spark2; do printf "%-10s " $h
  ssh $h "grep -E '^($KEYS)=' ~/localai/dspark-vllm-gx10/config/tp3.env|sort|md5sum"; done
```
The rank-specific header (NODE_RANK, VLLM_HOST_IP, IFNAME, home paths) **must** differ —
the files are not byte-identical and should not be.

Always read the applied config back from the engine; never assume:
```bash
ps aux | grep -oE '\-\-tensor-parallel-size [0-9]+|--max-num-seqs [0-9]+|--gpu-memory-utilization [0-9.]+'
docker logs dspark-vllm-gx10-vllm-dspark-1 2>&1 | grep -oE 'num_spec_tokens=[0-9]+|GPU KV cache size: [0-9,]+'
```

### Networking — persisted, but understand it

Gloo needs a **full mesh**: every rank must reach every other rank's advertised
`VLLM_HOST_IP`. spark1's reboot lost runtime-only state and **the cluster would not
start** — spark1 routed the master address over **WiFi**, and spark2 had no route to
spark1 at all.

Now persisted via NetworkManager (verified 2026-08-25):

```bash
# sparkmain — the master address lives on loopback and was NOT persisted before
sudo nmcli con mod lo +ipv4.addresses '192.168.200.1/32'
# spark1
sudo nmcli con mod dac-link        +ipv4.routes '192.168.200.1/32 192.168.100.1'
sudo nmcli con mod dac-link-spark2 +ipv4.routes '192.168.101.2/32 192.168.102.2'
# spark2
sudo nmcli con mod dac-link-spark1 +ipv4.routes '192.168.100.2/32 192.168.102.1'
```

Verify all six directions before starting:
```bash
declare -A A=( [sparkmain]=192.168.200.1 [spark1]=192.168.100.2 [spark2]=192.168.101.2 )
for s in sparkmain spark1 spark2; do for d in sparkmain spark1 spark2; do
  [ "$s" = "$d" ] && continue
  ssh $s "ping -c1 -W2 ${A[$d]} >/dev/null 2>&1 && echo $s->$d OK || echo $s->$d FAIL"; done; done
```

Physical topology, confirmed by MAC: sparkmain-f0 ↔ spark1-f1, spark1-f0 ↔ spark2-f1,
sparkmain-f1 ↔ spark2-f0. A clean ring. Each node also has **two unused `roceP2p1s0f*`
ports**, cabled and ACTIVE.

## 6. ⚠ Pre-benchmark fabric check — do this before trusting any number

This is the single most important addition to this document.

**Do not hand-roll this check.** It is one script, and every benchmark should call it:

```bash
make gate CONFIG=configs/3spark-live.env        # engine up: liveness + mesh + latency
make gate-full CONFIG=configs/3spark-live.env   # engine STOPPED: + every NCCL pair & N-rank
```

`scripts/fabric_gate.sh` **exits non-zero when a link is degraded**, so it gates rather
than merely reports. `scripts/run_experiment.sh` now calls it automatically and refuses to
benchmark if it fails, archiving the verdict as `fabric-gate.json` beside the results.
(Bypass with `FABRIC_GATE=0` — rarely, and deliberately.)

It checks four things in ascending cost:

| # | check | catches |
|---|---|---|
| 1 | SSH liveness (reads the banner) | a wedged node — **an open port 22 is not proof of life**, which misled us twice |
| 2 | Full directed mesh over the **fabric** addresses | a missing route or silent WiFi fallback; Gloo needs all N×(N−1) directions |
| 3 | Per-pair RTT with a ceiling | a link that is up and routable but pathological |
| 4 | NCCL collective bandwidth | **the 2026-08-25 degradation** — nothing else caught it |

Check 2 uses `FABRIC_ADDRS` from the config, **not** the management IPs: the management
LAN passed cleanly all the way through the degradation.

| result @64 MiB busbw | meaning |
|---|---|
| **~4.6 GB/s** | healthy GB10 pair |
| **~3.25 GB/s** | healthy 3-rank collective (measured 2026-08-25, post-fix) |
| **~0.7 GB/s** | **degraded node — reboot it before measuring anything** |

The raw harness underneath is still `results/20260824-seqs32-nccl/agbench.py`, which the
gate deploys to each node itself; vLLM must be STOPPED for it (live vLLM holds ~119/121
GiB and there is no room for a second CUDA context).

**TCP throughput is NOT a valid check.** It showed the degraded link at 858 MB/s vs 1,019
for a healthy one — 1.19x — while RDMA was 6.8x down. TCP does not exercise the RDMA
verbs path. Use the NCCL collective, which is what vLLM actually uses.

Nothing else revealed the fault: ports were ACTIVE at 200,000 Mb/s, every error counter
was 0, NIC firmware and PCIe link were identical across nodes, and NCCL selected the same
merged `NET/IB/2` transport on fast and slow paths alike.

## 7. Measurement discipline

Each of these was learned by getting a wrong answer first.

1. **Check the fabric (§6) before anything else.** New, and it invalidated months of work.
2. **Warm up until two sweeps agree.** The engine JIT-compiles *during inference* — a
   fresh shape can fire a compile several sweeps in and cost 40%+ on that sweep. One
   observed cc=1 request took 187 s against a steady 3 s. **Discard any sweep containing
   a `jit_monitor` warning.**
3. **Confirm the endpoint is idle before every run:**
   `curl -s localhost:8100/metrics | grep -E '^vllm:num_requests_running\{'`
   A single stray request skews a level badly, and one orphaned run blocked us for an hour.
4. **Defeat the prefix cache.** Re-running identical prompts returns **105,167 tok/s** at
   78K — that is the cache, not prefill. Use unique prefixes or upstream's token-ID
   harness, and verify zero cache hits.
5. **Change one variable at a time.**
6. **Measure at the head node only.** Under TP the workers serve nothing.
7. **Run upstream harnesses unmodified.** Their `benchmark_prefill.py` embeds a
   `token_pool_sha256`; ours matched theirs byte-for-byte, which is what made the
   comparison exact.
8. **`pgrep -f <script>` matches its own SSH command string** and will report a finished
   run as RUNNING forever. Use `pgrep -f '[b]ench_tp3.py'`.
9. **Run long remote jobs detached with output to a file** (`nohup … > file 2>&1 &`), then
   poll. A dead local wrapper otherwise orphans the remote process *and* loses its output.
10. **An ad-hoc `docker run` is not the production environment.** Omitting
    `--device /dev/infiniband` silently measures socket fallback. Mirror the compose
    service's devices, ulimits and shm_size, and confirm the transport in
    `NCCL_DEBUG=INFO` before trusting a fabric number.

### The harnesses

| script | source | measures |
|---|---|---|
| `results/20260825-fabric-fix/harness/benchmark_prefill.py` (also `~/results/harness/` on sparkmain) | `Anemll/dspark-vllm-gx10` | **prefill, server-side, token IDs — the authoritative one** |
| `results/20260824-prefill/pf3–pf8.py` | written here | prefill variants (cache-defeated, TTFT, content types) |
| `~/results/seqs32/bench_tp3.py` | `localaiguyy/…-3x-DGX-Spark` | decode concurrency sweep |
| `results/20260824-seqs32-nccl/agbench.py` | written here | **NCCL allgather — the fabric check** |
| `results/20260824-mtp5-1m/accept.py` | written here | MTP acceptance counters |

`/tmp` is not durable. Copies of the key ones live under `results/`.

## 8. Next steps

1. **Work issue #14** — re-run the suspect matrix (§4) on the healthy fabric. Start with
   `MAX_NUM_SEQS=32`, which is the most likely to have been wrongly rejected.
2. **Issue #12** — NVFP4 KV quality under *concurrency*. The single-request half is done
   and clean to 464K; the concurrent half is untested and needs no restart.
3. **Close #11** — the prefill gap is resolved by the fabric fix; the issue's original
   leads (NIC merge, runtime version) were both falsified.
4. **Close #13** if it duplicates the persistence work already done (§5).
5. **Investigate why spark1 degraded.** It had been up a long time. If this recurs, a
   periodic fabric check or a documented reboot cadence is warranted.
6. **Separately: assistant-prefill correctness.** The API advertises
   `add_generation_prompt` / `continue_final_message` but the DeepSeek-V4 tokenizer
   adapter ignores both, so an assistant prefix is EOS-terminated instead of continued.
   Fix is to port [vLLM PR #46257](https://github.com/vllm-project/vllm/pull/46257) into
   the anemll overlay. **This is a correctness bug, not a throughput one** — it does not
   affect any number in this document.

## 9. Where things live

| what | where |
|---|---|
| Repo | `github.com/colonel-otto/3spark-dsv4` |
| Open PR | [#9](https://github.com/colonel-otto/3spark-dsv4/pull/9) — open, not merged |
| Open issues | [#10](../../issues/10) seqs=32 · [#11](../../issues/11) prefill (resolved, close) · [#12](../../issues/12) KV quality · [#13](../../issues/13) mesh persistence (done) · [#14](../../issues/14) **re-run suspect benchmarks** |
| Parity writeup | [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) |
| Prefill investigation | [`PREFILL-MEASURED.md`](PREFILL-MEASURED.md) — 5 addenda, several self-corrections |
| seqs=32 / NCCL | [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) |
| Post-fix results | `results/20260825-fabric-fix/` |
| Live config | `~/localai/dspark-vllm-gx10/config/tp3.env` on each of the 3 nodes |
| Config backups | `tp3.env.bak-*` beside it, timestamped |
| Grafana / Prometheus | `sparkmain:3001` / `sparkmain:9090` |

### Before committing

The repo has a pre-commit hook (`.githooks/pre-commit` → `scripts/check_no_sensitive.py`)
that blocks serials and sensitive data. Run `make install-hooks` after cloning. Docs here
contain internal IPs and hostnames, so this matters.

## 10. Open questions

- **Why did spark1's fabric degrade?** No error indicator of any kind. Unknown.
- **Is `MAX_NUM_SEQS=32` viable on a healthy fabric?** The rejection is now suspect.
- ~~**What is the real 3-rank allgather ceiling?**~~ **ANSWERED 2026-08-25: 3.25 GB/s.**
  Measured with `make gate-full` on the healthy fabric (all three pairs read 4.58–4.60
  GB/s). The old **0.49 GB/s** figure was taken on the degraded fabric and is **6.6x
  pessimistic**. That number anchored the "GB10 has no GPUDirect, ~0.5 GB/s is the
  ceiling" analysis, so **that analysis needs revisiting** — the communication budget is
  far larger than it assumed, which bears directly on the EP=3 / PP=3 / seqs=32
  rejections.
- **Does prefill exceed the reference on a healthy fabric?** We are at 95–99%; nobody has
  tuned *for* prefill since the fix.
- Does `max_num_seqs=64` continue the trend past 32? Untested by anyone.
- Is MTP=5 optimal, or would k=10 go further? Untested on 0.25.2.
- What does abliteration cost against the stock checkpoint? No A/B exists anywhere.
