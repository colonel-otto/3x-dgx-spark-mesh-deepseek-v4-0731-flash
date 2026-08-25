# Post-mortem — 2026-08-25: the fabric was lying, and so were our benchmarks

**Scope.** One working day. Started as "re-run the deep-concurrency test from issue #15."
Ended having overturned or qualified most of what this repo believed about node count,
and having found four classes of failure that were invisible to every check we had.

**Outcome in one line.** Three nodes ARE superior for this deployment (+17–18% decode at
cc=1), but not for the reasons previously claimed, and the evidence for it did not exist
until today.

---

## 1. What we set out to do

Issue #15 flagged that the "KV capacity is not the bottleneck" conclusion had been
measured on the degraded spark1 fabric documented in #14. The ask was narrow: re-run the
4×200K deep-concurrency test on healthy fabric and re-state or retract the conclusion.

## 2. What we found

### 2.1 The original conclusion split in two

| | Aug-21 (degraded) | Aug-25 (healthy) | |
|---|---:|---:|---|
| TP=3 TTFT | 553,113 ms | 396,804 ms | 1.39x faster |
| TP=2 TTFT | 539,666 ms | 293,987 ms | 1.84x faster |
| preemptions | 0 | 0 | unchanged |

- **"KV capacity is not the binding constraint" SURVIVED.** Preemptions were 0 then and
  are 0 now; the pool never fills. That half was an accurate observation.
- **"Equally unusable on both" DID NOT.** The arms were 1.025x apart and are now 1.35x
  apart, in 2-node's favour. The old parity was two configurations being throttled to a
  common floor by one degraded link — and spark1 sat in the *3-node* arm, so the handicap
  fell disproportionately there. Removing it separated them.

### 2.2 The answer to "are 3 nodes better?" is conditional, and the condition matters

| measurement | 2-node | 3-node | winner |
|---|---:|---:|---|
| **decode cc=1** | 76.2 | **89.1** | **3-node, +17%** |
| decode cc=4 | 192.8 | **208.8** | 3-node, +8% |
| decode cc=8 | 302.7 | **322.7** | 3-node, +7% |
| decode cc=16 | **481.3** | 474.8 | 2-node, +1.4% |
| prefill 1K/8K/32K | 1913/2081/2066 | 2023/2070/2095 | parity (±2%) |
| deep-concurrency TTFT | **293,987 ms** | 396,804 ms | 2-node, 1.35x |
| KV capacity | 1,711,307 | **4,457,627** | 3-node 2.6x — never binds |

The 3-node advantage **decays monotonically with concurrency and crosses over near
cc=16**. Three nodes win per-stream latency; two win batch aggregate. Because this
deployment is single-user interactive coding — per-stream-latency bound — cc=1 is the
operative number and three nodes are correct. **A multi-user batch deployment should run
two nodes and free the third.**

This vindicates the old "+8–17% per-stream at long context" claim, which #14 had marked
suspect. It reproduces in the same range on healthy fabric.

---

## 3. Four failures that were invisible to every check we had

This is the part worth remembering. Each was silent, each produced plausible numbers, and
each is now gated.

### 3.1 A degraded link with zero error indicators (pre-existing, #14)

spark1 ran at ~0.7 GB/s against 4.6 for a healthy pair — a 6.8x deficit — with port state
ACTIVE, link 200,000 Mb/s, every error counter 0, identical firmware and PCIe width, and
NCCL selecting the same transport. **TCP throughput hid it** (1.19x apparent vs 6.8x real)
because TCP never touches the RDMA verbs path.

### 3.2 Config that exists only in tmpfs

NetworkManager on these boxes is only a *renderer*; **netplan owns the config**. Every NM
connection lives in `/run/NetworkManager/system-connections/` — wiped on reboot — and
`/etc/NetworkManager/system-connections/` is **empty**. An `nmcli con mod` that does not
write through to `/etc/netplan/` looks perfectly applied and silently reverts.

Precedent: spark1's reboot lost `192.168.200.1`, which existed only on sparkmain's
loopback at runtime. Had sparkmain rebooted, the cluster would have been unrecoverable
without knowing that.

### 3.3 Init success is not health

Widening `NCCL_IB_HCA` to the `roceP2p` controllers wedged the cluster. **All three ranks
completed NCCL init and every container stayed `running`** while live RDMA completions
failed with `IBV_WC_RETRY_EXC_ERR`. The engine simply never finished loading, emitting
`shm_broadcast` "no available block" for 10+ minutes with no error surfaced. The container
has **no health check**, so Docker could not flag it, and `docker ps` looked normal.

Cause: the `roceP2p` pair has **no IPv4 at all**, only link-local IPv6, while the routed
mesh runs IPv4 on `rocep1`. Both GIDs in the failure were `fe80::` — RDMA over an
unaddressed, unrouted path.

### 3.4 JIT compilation during inference

A ~5 s TileLang/CuTeDSL compile can land **inside** a request. The engine warns about it
itself (`jit_monitor`) but the request just looks slow. This produced a stall tail we
initially mis-diagnosed (see §4.2) and is the mechanism behind the "wide spread" warnings
that made decode numbers hard to trust.

---

## 4. Where WE went wrong (process, not hardware)

### 4.1 Estimating instead of measuring

The deep-concurrency prompt builder used an *estimated* tokens-per-word ratio. Two runs
missed the 200K target (185.5K, then 172.2K), and the first "correction" moved the
constant the **wrong way** — it divides the word target, so raising it shortens the
prompt. Asking the engine's `/tokenize` gave 1.2056 flat across 150K–240K and landed all
four streams within 0.013% of target.

This was not cosmetic: TTFT rises **+44% for +8% depth**, so an off-depth run answers a
different question. **Fix: measure the thing, do not model it.**

### 4.2 Calling a distribution from too little data

We described the decode noise as "bimodal" from n=15 and published it. At n=30 there are
**three gaps, not one** — a fast mode (85.1–91.0, 80%) plus a *tail* of stall severities
(77.4/79.1, and 68.8/69.8/72.1). It is variable-length compiles, not two discrete states.
**Fix: state n, and look at the gap structure before naming a shape.**

### 4.3 Chasing a real number down a wrong premise

Upstream MiaAI-Lab documents that a GB10 QSFP port enumerates as two ~100G controllers and
that listing one "runs the link at half the port." A direct A/B confirmed it:
**5.72 → 8.45 GB/s at 64 MiB (+56%)** with MTU 9000. The measurement was real. The premise
was not — that path has no IPv4 and cannot carry production traffic. **Fix: before acting
on a measured gain, verify the path it was measured on is one production can use.**

### 4.4 Arguing against a correctness fix

The `.100.x` fabric link ran `/24` while the other two ran `/30`. This was argued down
twice as "cosmetic, the subnets cannot overlap" — technically true and beside the point.
Inconsistent masks on a fabric are a latent trap; the user was right to insist. Now `/30`
on all six addresses. **Fix: "correct in isolation" is not the standard for shared
infrastructure.**

### 4.5 Comparing across a boundary the harness was not built for

`agbench.py` was built to mirror the vLLM MTP allgather shape (4–8 MB) — the right size
for "does our workload run well". We then used its top point (67 MB) as a **peak
bandwidth** figure and compared it against a forum number measured at **16 GB**, and
treated the 3-6x difference as a hardware defect for days.

Both variables that differed — message size and algbw-vs-busbw convention — push the
number the same way. Our own `w=2` algbw (19.40 GB/s) already sits inside the band we
thought we were missing.

**Fix: match the independent variable before calling a gap a defect.** And state
collective, size, rank count and convention on every bandwidth number, ours or theirs.

### 4.6 Stale pointers outliving the data they point at

Three summaries were quietly lying while the underlying measurements were fine:
- `generate_summary.py` matched `prompt_tokens == "200000"` exactly and hardcoded
  "~14.5 min wall" — a re-run at 200,046 would have been **silently skipped** while the
  generator kept publishing the retracted conclusion.
- The changelog's "current production configuration" block still advertised
  `460800`/`MTP=4`/`0.85` after 1M/`MTP=5`/`0.80` was settled.
- The deep-concurrency harness never existed as a file, which is part of why its result
  could not be checked.

---

## 5. What is now gated

`scripts/fabric_gate.sh`, run before every benchmark and wired into `run_experiment.sh`
(which refuses to benchmark on failure). Each check was verified by **injecting the real
fault** and confirming exit 1.

| check | catches |
|---|---|
| SSH liveness (reads banner) | wedged node; open port 22 is not proof of life |
| directed mesh over **fabric** addrs | missing route / silent WiFi fallback |
| per-pair RTT ceiling | link up but pathological |
| duplicate/overlapping subnets | the same address on two fabric NICs |
| **ARP peer-on-cabled-port** | wrong-port entries — flushes them |
| **config persistence in netplan** | addresses that vanish on reboot |
| **transport assertion** | NCCL silently falling back to sockets |
| **RDMA completion errors in a running engine** | §3.3 — degraded despite `running` |
| NCCL collective bandwidth | the §3.1 degradation |

---

## 6. Open, and honestly unresolved

1. ~~**Our 3-rank collective is 3.25 GB/s where published 3-Spark rings report 18–21.**~~
   **Largely explained, and it was our error.** Two things:
   - The `roceP2p` path was indeed the candidate. Addressed 2026-08-25 → **2.0x**
     (3-rank 2.85 → 5.80 GB/s), live gate clean.
   - The *remaining* gap is mostly a bad comparison. We measured `all_gather` **busbw** at
     a **67 MB** input; the forum figure is `all_gather` at **16 GB** — 238x larger, and
     far closer to the asymptote. Our own `w=2` **algbw is 19.40 GB/s**, inside the
     18–21 band. See [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md).

   **We never ran the collective at the size the comparison used.** Settle it with
   `scripts/bwsweep.py`, which prints all three conventions per line.
2. **The JIT stall tail** — 13% of runs still stall after warming. Warm-up does not cover
   every shape.
3. **Root cause of spark1's original degradation** — never determined. It had been up a
   long time. If it recurs, a periodic fabric check or reboot cadence is warranted.
4. **The 0.49 GB/s "GB10 ceiling" analysis** was measured on the degraded fabric. True
   3-rank is 3.25 GB/s — **6.6x higher** — so the EP=3 / PP=3 / seqs=32 rejections were all
   decided against a communication budget that was far too small. Worth revisiting.

---

## 7. The one-sentence lessons

1. A green status is the weakest evidence there is — check the working path, not the
   startup path.
2. Measure the quantity you are varying; do not model it.
3. State `n` before naming a distribution.
4. A real measurement on an unusable path is still unusable.
5. Summaries rot faster than data — pin them to the data or delete them.
