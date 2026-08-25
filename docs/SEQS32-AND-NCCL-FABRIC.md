# `MAX_NUM_SEQS=32` and the NCCL fabric ceiling — 2026-08-24

Answers issue #10 (seqs=32) and falsifies the leading hypothesis of issue #11
(prefill gap). Both answers come from the same root cause.

**Bottom line:** `MAX_NUM_SEQS=32` **crashes the cluster** under sustained load, and
`NCCL_IB_MERGE_NICS=1` is a **no-op on this hardware**. The binding constraint on all
inter-node collectives is that **GB10 has no GPUDirect RDMA**, which caps effective
allgather bandwidth at **~0.5 GB/s**.

Rolled back to `MAX_NUM_SEQS=16`. Cluster verified healthy and serving.

---

## 1. seqs=32 — issue #10: **rejected**

It boots, serves, and then dies. Config applied cleanly and CUDA-graph capture succeeded:

| | seqs=16 | seqs=32 | delta |
|---|---:|---:|---:|
| `max_cudagraph_capture_size` | 96 | 192 | 2x |
| capture cost | 1.24 GiB | 2.08 GiB | +0.84 GiB |
| capture time | 15 s | 24 s | +9 s |
| GPU KV cache | 5,444,869 | 5,382,503 | **-62,366 (-1.1%)** |

**The KV cost was a non-issue** — 1.1%, not the collapse we budgeted for. The memory
risk flagged in the handoff did not materialise. The failure was elsewhere.

### Measured before the crash

| cc | median tok/s | spread |
|---:|---:|---:|
| 1 | 84.6 | 2% |
| 8 | 245.5 | 16% |
| 16 | 325.1 | 14% |
| 24 | 373.6 | 36% |
| 32 | 395.2 | 15% |
| 40 | **ALL REQUESTS FAILED** | — |

Two things to read here. **cc=16 got *worse*** — 325.1 against the seqs=16 baseline of
374.2. And cc=32 peaked at ~395–426 against the **618 tok/s** the 3-node repo reports at
seqs=32. We did not reproduce their gain; we lost ground at our own operating point.

### The crash

At sustained cc=32 the engine died. Twice.

```
TimeoutError: RPC call to sample_tokens timed out.
EngineCore encountered a fatal error.  ->  EngineDeadError
```

All three ranks then hit the same NCCL watchdog, within 79 ms of each other:

```
[Rank 0] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=3474, OpType=_ALLGATHER_BASE,
         NumelIn=8282112, NumelOut=24846336, Timeout(ms)=600000)
         ran for 600033 ms before timing out
```

**Same SeqNum on all three ranks** means the ranks stayed in lockstep and the data never
moved. This is a transport stall, not a scheduler bug, not desync, and not OOM:

- `kv_cache_usage=0.028` — KV was **2.8% full**
- `num_preemptions_total=0`
- **Zero** link_downed / rcv_errors / symbol_err across all 12 fabric ports
- **Zero** GPU Xid errors
- `last enqueued 3475, last completed 3473` — a clean mid-stream stall on 3474

### Why seqs=32 specifically

The hung collective is MTP's draft-token allgather, and **it scales with
`max_num_seqs x vocab_size`**:

```
8,282,112 / 129,280 (vocab) = 64.06 = 2 x 32 seqs
```

Raising seqs 16 -> 32 quadrupled that allgather onto a transport that (see §2) sustains
~0.5 GB/s. vLLM PR #46448 exists specifically to shrink this collective from
O(vocab) to O(TP).

> Note the engine's own startup warning, which points the same way:
> `max_num_scheduled_tokens is set to 8064 based on the speculative decoding settings...
> consider increasing max_num_batched_tokens or decrease num_speculative_tokens or
> max_num_seqs`. At MTP=5, seqs=32 wants a batched-token budget we do not have — and
> `MAX_NUM_BATCHED_TOKENS=16384` is a documented trap here (43% of KV for zero gain).

**Verdict: keep `MAX_NUM_SEQS=16`.** Raising it is not a tuning question on this
hardware; it is a stability regression. It might become viable after PR #46448 lands, or
at MTP=1, but both change a second variable.

---

## 2. `NCCL_IB_MERGE_NICS` — issue #11 lead #1: **falsified**

Measured directly with `results/20260824-seqs32-nccl/agbench.py`, a 3-rank
`all_gather_into_tensor` benchmark run **inside the production vLLM image** with the
same device passthrough, ulimits, and NCCL env the service uses. It reproduces the exact
shape that hung (8,282,112 elements) plus a size sweep.

busbw, GB/s:

| shape | bytes | MERGE_NICS=0 | MERGE_NICS=1 |
|---|---:|---:|---:|
| seqs8 x vocab | 2.07 MB | 0.48 | 0.48 |
| seqs16 x vocab | 4.14 MB | 0.51 | 0.52 |
| seqs32 x vocab | 8.27 MB | 0.50 | 0.51 |
| **seqs64 x vocab (the hung shape)** | 16.55 MB | 0.50 | 0.52 |
| 16 MiB | 16.78 MB | 0.50 | 0.52 |
| 64 MiB | 67.11 MB | 0.47 | 0.50 |

**No effect at any size.** The reason is visible in `NCCL_DEBUG=INFO`:

```
NET/IB : Made virtual device [2] name=rocep1s0f0+rocep1s0f1 speed=400000 ndevs=2
Channel 00/0 : 2[0] -> 0[0] [receive] via NET/IB/2
```

**NCCL already merges both HCAs by default** and routes every channel over the merged
400 Gb/s virtual device. Setting the flag changes nothing because the merge is
already on. The upstream +64% claim targets a configuration where it was off.

**Do not re-open this lead.** It is measured, on our hardware, both ways, same day.

## 3. The real ceiling: no GPUDirect RDMA

```
NET/IB : GPU Direct RDMA Disabled for HCA 0 'rocep1s0f0'
NET/IB : GPU Direct RDMA Disabled for HCA 1 'rocep1s0f1'
Symmetric memory is not supported. cuMemGdrSupport 0
```

Confirmed on the host: `nvidia_peermem` is **not loaded and not available** on GB10.
Every inter-node byte is host-staged — GPU -> system RAM -> NIC -> system RAM -> GPU,
with CPU-side progress. This is architectural, not a misconfiguration
(vllm-project/vllm#46253).

**Measured cost:** ~0.5 GB/s effective allgather busbw against a 200 Gb/s (25 GB/s)
link — roughly **2% of line rate**. Published DGX Spark figures of 13.5–22 GB/s are
`ib_write_bw` / `all_gather_perf` point-to-point numbers on 2 nodes, which are not the
same measurement as a 3-rank collective through host staging.

### This explains the prefill gap

Prefill is large-message and allgather-dominated; decode is small-message. A transport
that adds a fixed host-staging penalty per byte punishes prefill and barely touches
decode — which is **exactly our profile**: we lead the 2-node recipe on every decode
metric and trail it only on prefill.

**Issue #11 is not a tuning problem.** It is the cost of a third node on a fabric with no
GPUDirect. The remaining leads (runtime version, `MAX_NUM_BATCHED_TOKENS`) are unlikely
to move a 30x transport gap.

---

## 4. Method notes for whoever repeats this

**Warm-up took six sweeps, not two.** The engine JIT-compiles kernels lazily *during
inference*, and a fresh shape can fire a compile several sweeps in:

```
WARNING [jit_monitor.py:129] Triton kernel JIT compilation during inference:
  _topk_topp_kernel. This causes a latency spike; consider extending warmup...
```

Sweeps 1, 3 and 4 were each corrupted by a JIT spike (one cc=1 request took **187 s**,
1.4 tok/s against a steady 85). Sweeps 5 and 6 agreed to 1.6% and were used. **Discard
any sweep containing a JIT warning** — do not average it in.

**`pgrep -f <script>` matches its own SSH command string** and will report a finished
run as still RUNNING forever. Use `pgrep -f '[b]ench_tp3.py'`.

**Run remote benchmarks detached with output to a file** (`nohup ... > file 2>&1 &`),
then poll. A local wrapper that dies otherwise orphans the remote process *and* loses
its output — that cost us a 700K/900K KV run earlier the same day.

**An ad-hoc `docker run` is not the production environment.** The first A/B ran without
`--device /dev/infiniband` and measured `NET/Socket` fallback at 0.44 GB/s — a plausible
number that was measuring the wrong thing. Always mirror the compose service's devices,
ulimits and shm_size, and confirm the transport in `NCCL_DEBUG=INFO` before trusting a
fabric number.

**MPI is not usable for ad-hoc cluster tests here.** `mpirun` fails to route to the
workers (`ORTE does not know how to route a message`) across three subnets, and the
bundled `nccl-tests` binaries are MPI-linked. `torch.distributed` with a TCP
`init_method` on the management LAN works and uses the same NCCL library.

## 5. Config state

Rolled back on all three ranks (`config/tp3.env`), backups at
`tp3.env.bak-preRollback-20260824`:

| var | value | note |
|---|---|---|
| `MAX_NUM_SEQS` | **16** | rolled back from 32 |
| `NCCL_DEBUG` | **INFO** | was WARN — transport selection was invisible |
| `NCCL_TIMEOUT` | **3600** | added; 600 s default turned a stall into a hard crash |

Verified live: seqs=16, capture 96, KV 5,424,080, 1M context, `/health` 200.

## 6. What we did NOT test

- **seqs=24** — an intermediate value might be stable. Untested; the crash at 32 made
  further seqs exploration a stability risk rather than a tuning question.
- **seqs=32 at MTP=1** — would shrink the allgather 6x and might well be stable, but it
  changes two variables at once and answers a different question.
- **GPUDirect via a driver/module change** — `nvidia_peermem` is absent on GB10; we did
  not attempt to build or force-load it, and public sources treat its absence as
  architectural.
- **Whether the crash reproduces at cc=32 with `NCCL_TIMEOUT=3600`** — the longer timeout
  is now set, so a future stall would stall rather than kill the engine. Not re-tested.

## 7. Raw data

`results/20260824-seqs32-nccl/` — allgather A/B JSON (both arms, with and without RDMA
passthrough), the seqs=32 sweep, warm-up logs 5/6, post-rollback verification,
`agbench.py`, and `crash/` with all three ranks' full logs from the crash window.
