# Catalogue of degraded data — what bad numbers looked like

**Kept deliberately.** Every measurement below is wrong, and every one of them looked
completely reasonable at the time. That is the point: this page is a reference for
recognising bad data *before* you build on it.

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

### Superseded result pages

Frozen, banner-marked, kept for the record:
[`WHY-THREE-NODES.md`](WHY-THREE-NODES.md) ·
[`TP3-TUNING.md`](TP3-TUNING.md) ·
[`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) ·
[`results.md`](results.md) ·
[`BASELINE-2SPARK.md`](BASELINE-2SPARK.md)

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
