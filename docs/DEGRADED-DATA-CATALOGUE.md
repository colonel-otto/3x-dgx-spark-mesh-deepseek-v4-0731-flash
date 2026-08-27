# Troubleshooting guide: your numbers don't match ours

> [!IMPORTANT]
> **This page is a diagnostic reference, not a results page.**
>
> Every number on it is a **symptom to match against**, captured from a cluster with a
> real, specific fault. None of it is what this deployment achieves — for that, see
> [`../README.md`](../README.md#is-the-third-node-worth-it) and
> [`DECISIONS.md`](DECISIONS.md).
>
> We keep these numbers because they are the hardest part of reproducing this work. Both
> faults below produce **plausible results with zero error indicators**: the engine
> serves, the containers stay `running`, `ibstat` reads ACTIVE, every counter reads 0, and
> the tok/s figure looks like a legitimate finding. If your reproduction lands on one of
> these numbers, you have not measured our cluster differently — you have measured a
> broken one.
>
> Start with the flow below.

---

## Start here: your numbers don't match ours

| Symptom you are seeing | Likely cause | How to confirm | Fix |
|---|---|---|---|
| 3-node TP=3 decode near **~25 tok/s** at cc=1, against a healthy range of **54–89** depending on prompt shape † | **D2** — NCCL fell back to TCP | `NCCL_DEBUG=INFO` shows `via NET/Socket` instead of `via NET/IB/*` | Set `NCCL_IB_SUBNET_AWARE_ROUTING=1`, confirm it reaches the container (`docker compose config \| grep SUBNET_AWARE`), and verify `/dev/infiniband` is passed through |
| Pairwise NCCL busbw near **~0.7 GB/s** where a healthy pair reads ~4.6 (2 HCA) / ~9.7 (4 HCA) | **D1** — one node's fabric has silently degraded | Run the pairwise collective on all three pairs. The bad node is the one present in every slow pair | **Reboot the degraded node.** A cable swap does not fix it — we tested the alternate cable and got the same 0.68 GB/s |
| 3-rank collective reads **below** your worst pair (e.g. 0.49 vs 0.69) | **D1** — a collective is paced by its slowest member | Compare the 3-rank figure against each pair individually | Reboot the node common to the slow pairs |
| Two configurations you expect to differ measure suspiciously **equal** | **D1** — both throttled to a common floor | Gate the fabric before benchmarking, then re-run both arms | Fix the fabric first; the comparison is void until then |
| A gap between two arms is suspiciously **uniform across a swept variable** (e.g. a flat ~13% at 2K, 8K, 32K and 131K alike) | **D1** — a shared floor flattens a real curve into a constant | Re-run the sweep on gated fabric. Ours turned out to be **parity below 32K and +33.6% at 131K** | Fix the fabric. Note that consistency across levels reads as *robustness* and is the reason this went unquestioned for five days |
| Benchmark is slow but **no counter anywhere is failing** | **D1 or D2** — this whole family is invisible to status checks | `make gate-full CONFIG=configs/3spark-live.env`, engine stopped | Follow whichever gate check fails |
| Containers `running`, ranks completed NCCL init, engine never finishes loading | **Init success ≠ health** (see below) | Look for `IBV_WC_RETRY_EXC_ERR` with both GIDs `fe80::` in the engine log | The HCA pair you enabled has no IPv4. Address and route it, or roll back to the pair that has one |
| A single stream is fast but wildly variable run to run | JIT compilation landing inside a request — not a fabric fault | `jit_monitor` warning in the log: `JIT compilation during inference` | Warm every shape you intend to measure, discard contaminated sweeps, take median of ≥7 |

† **Never compare a tok/s figure to one taken on a different prompt.** On this deployment
the prompt alone moves single-stream decode **1.65x** (81.8 code-shaped vs 49.4 dense
prose, same engine, minutes apart) because MTP acceptance is content-dependent. The
healthy range above spans 53.95–57.73 on a dense-prose prompt and 85.6–89.1 on an
18-token code brief. See [`BENCHMARK-METHODOLOGY.md`](BENCHMARK-METHODOLOGY.md).

**Before anything else, check the date on the data you are comparing against.** Anything
in this repo measured before **2026-08-25** was taken on the degraded fabric in D1, and
carries a banner saying so.

**A healthy TCP number proves nothing.** TCP never touches the RDMA verbs path. During D1
it showed a 1.19x deficit while the real RDMA deficit was 6.8x. Use an NCCL collective.

---

## What these numbers are, and are not

| Tier | Meaning | Where it lives |
|---|---|---|
| **Our results** | Healthy fabric, RDMA confirmed `via NET/IB/*`, matched arms | [`../README.md`](../README.md#is-the-third-node-worth-it) · [`DECISIONS.md`](DECISIONS.md) · [`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) |
| **Diagnostic signatures** | Numbers from a known-broken setup, kept so you can recognise the fault | **This page** |
| **Falsified claims** | Assertions we made that later proved wrong — neither result nor symptom | Listed per-page, and in the "What it corrupted" table below |

Nothing here is deleted. It is itemized, bounded, and labelled with what specifically was
wrong and how far off it was.

---

## The two causes

| # | Cause | Window | Signature |
|---|---|---|---|
| **D1** | **Silent RDMA degradation** — spark1 ran at ~15% of collective bandwidth | until 2026-08-25 reboot | Port `ACTIVE`, link 200,000 Mb/s, **every error counter 0**, firmware and PCIe width identical, NCCL selecting the correct transport |
| **D2** | **TCP fallback** — NCCL silently used sockets instead of RDMA | intermittent, pre-fix | `via NET/Socket` in the log. Reports a plausible number: **0.44 GB/s** |

Both are **invisible to every ordinary health check.** `docker ps` said running. `ping`
said fine. `ibstat` said ACTIVE. Only an NCCL collective found either one.

---

## D1 — the degraded fabric

### What it measured vs the truth

| Path | Degraded | Healthy | Error |
|---|---:|---:|---:|
| sparkmain ↔ spark2 (never degraded) | 4.60–4.64 GB/s | 4.60–4.75 | ✅ correct |
| sparkmain ↔ spark1 | **0.69 GB/s** | 4.78 | **6.9x low** |
| sparkmain ↔ spark1, *alternate cable* | **0.68 GB/s** | — | cable swap did **not** fix it |
| spark1 ↔ spark2 | **0.71 GB/s** | ~4.6 | **6.5x low** |
| all three ranks | **0.49 GB/s** | **3.25** | **6.6x low** |

Note the 3-rank figure sits *below even the worst pair* — a collective is paced by its
slowest member. That is the tell, and we read it as a hardware ceiling instead.

### What it corrupted

| Claim made | Reality | Status |
|---|---|---|
| "GB10 has a ~0.5 GB/s communication ceiling" | The ceiling is **3.25 GB/s** | ❌ **Retracted.** This was the most expensive error — it anchored three rejections below |
| `MAX_NUM_SEQS=32` rejected | Rejected against a budget **6.6x too small** | ⚠️ Re-open, [#10](../../issues/10) |
| EP=3 rejected | 2.5x slower — but partly re-measure | ⚠️ Kernel finding stands; the margin does not |
| PP=3 rejected | Blocked by MTP + DSA stride | ✅ **Survives** — a hard block, not a perf number |
| "2 and 3 nodes are equally unusable at 4×200K" | 1.025x apart then; **1.35x** apart now | ❌ **Retracted** |
| "KV capacity is not the binding constraint" | 0 preemptions then and now | ✅ **Survives** |
| ~2x prefill gap vs upstream | Was **one degraded node**, not architecture | ❌ Cause was wrong |

**Why the 2v3 comparison was hit hardest:** spark1 sat in the **3-node arm**. The handicap
fell disproportionately on the arm under test, which is the worst possible place for it —
it made two configurations look equivalent by throttling one to the other's floor.

### Pages whose numbers are diagnostic signatures, not results

Each carries a banner naming what survives, what is void, and where the healthy number
lives instead. Read the banner before quoting anything from these pages.

| Page | What its numbers are | Contaminated by |
|---|---|---|
| [`WHY-THREE-NODES.md`](WHY-THREE-NODES.md) | Degraded-fabric 2v3 decode table. **Its "+8–17% from 2K upward" headline is retracted** — re-measured 2026-08-26 as parity below 32K and +33.6% at 131K | D1 — and spark1 sat in the **3-node** arm |
| [`EP3-EXPERT-PARALLEL.md`](EP3-EXPERT-PARALLEL.md) | Degraded fabric **and** TCP fallback | D1 + D2 — the only transport that ran was `NCCL_NET=Socket` |
| [`TP3-TUNING.md`](TP3-TUNING.md) | Degraded-fabric tuning sweep at the superseded profile | D1 |
| [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) | Degraded-fabric collective budget (0.49 GB/s) | D1 |
| [`results.md`](results.md) | Earliest TP=2/TP=3 comparison, pre-rewire cabling | D1 + D2 (the 24.59 arm) |
| [`BASELINE-2SPARK.md`](BASELINE-2SPARK.md) | 2-node baseline over the **sparkmain↔spark1** link | D1 — see that page's banner for the timing caveat |

---

## D2 — TCP fallback masquerading as RDMA

`NCCL_NET=IB` is a **request, not a guarantee.** When IB init fails, NCCL falls back to
sockets, logs it once, and proceeds to report a number that looks like a real result.

| Transport | Decode | Reads as |
|---|---:|---|
| `NET/Socket` (TCP fallback) | **24.59 tok/s** | a plausible "TP=3 is slow" result |
| `NET/IB` (actual RDMA) | 53.95–57.73 tok/s | the truth |

**2.2x error.** The 24.59 figure is retained as a legitimate *transport control* — it is
the correct answer to "what does TCP cost?" — but it must never be presented as RoCE
performance.

### Why TCP throughput cannot be used as a fabric check

This is the trap that hid D1 for days:

| Test | Apparent deficit | Real deficit |
|---|---:|---:|
| TCP throughput (`iperf`-style) | **1.19x** | — |
| NCCL collective (RDMA verbs) | — | **6.8x** |

TCP never touches the RDMA verbs path, so it barely noticed. **A healthy TCP number is
not evidence of a healthy fabric.**

---

## The third silent class — init success ≠ health

Not a data-corruption bug, but the same family, and it wedged the cluster on 2026-08-25.

Widening `NCCL_IB_HCA` to the `roceP2p` pair: **all three ranks completed NCCL init** and
every container stayed `running` while live RDMA completions failed with
`IBV_WC_RETRY_EXC_ERR` (both GIDs `fe80::`). The engine simply never finished loading.
The container has **no health check**, so Docker could not flag it.

**Initialisation proves connectivity, not health.** Check live error counters, not
container state.

---

## How to tell if data you are looking at is degraded

1. **Check the date.** Anything before **2026-08-25** is suspect by default.
2. **Check for a banner.** Superseded pages carry one at the top.
3. **Look for the shape, not the value:**
   - A 3-rank collective **below** the worst pair → one node is dragging
   - Two configurations suspiciously *equal* → both throttled to a common floor
   - `~0.7 GB/s` pairwise → degraded node, **reboot it**
   - `via NET/Socket` anywhere in the log → not an RDMA measurement
   - A benchmark that is slow but has **no failing counter anywhere** → this family
4. **Run the gate.** `make gate-full CONFIG=configs/3spark-live.env`, engine stopped. It
   exits non-zero on every one of the faults above — each verified by injecting the real
   fault.

---

## The lesson worth keeping

> A green status is the weakest evidence there is. Check the **working** path, not the
> **startup** path.

Every failure catalogued here passed every status check it had. What found them was
measuring the actual thing, on the actual path, against a known-good reference.

**Related:** [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md) ·
[`FABRIC-FIX-PARITY.md`](FABRIC-FIX-PARITY.md) · [#14](../../issues/14)
