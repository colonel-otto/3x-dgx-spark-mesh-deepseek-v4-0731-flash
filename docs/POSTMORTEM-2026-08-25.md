# Post-mortem — 2026-08-25: the fabric was lying, and so were our benchmarks

> [!CAUTION]
> **The fabric diagnosis and failure lessons survive; the node-count performance verdict
> does not.** The later depth comparison cited in §2.2 returned only 25–26 output tokens
> per request and is now `VOID-25-token-window`. A corrected 3-node arm exists, but no
> matching corrected 2-node arm does. Use the [current handoff](HANDOFF-2026-08-27.md),
> not this dated post-mortem, for the present comparison status.

**Scope.** One working day. Started as "re-run the deep-concurrency test from issue #15."
Ended having overturned or qualified most of what this repo believed about node count,
and having found four classes of failure that were invisible to every check we had.

**Outcome in one line.** A silent RDMA fault invalidated the earlier performance corpus
and established that fabric gating is mandatory; the node-count conclusion drawn that day
was later withdrawn.

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

This vindicates the **direction** of the old "+8–17% per-stream" claim, which #14 had
marked suspect. At cc=1 it reproduces in the same range.

> **Correction, 2026-08-26.** The words "at long context" did not belong in that sentence,
> and the vindication is narrower than it reads. **This run used an 18-token prompt** — it
> measured concurrency, not context depth. The matched depth sweep landed the next day and
> the old claim fails on the axis it actually named:
>
> | context (cc=1) | 2-node | 3-node | gain |
> |---:|---:|---:|---:|
> | 2,036 | 75.8 | 76.3 | +0.8% |
> | 8,081 | 72.4 | 72.6 | +0.3% |
> | 32,268 | **70.8** | 70.2 | −0.9% |
> | 129,006 | 54.4 | **72.6** | **+33.6%** |
> | 257,993 | 71.5 | **84.4** | **+17.9%** |
>
> **No per-stream benefit below 32K, and past 100K more than double what was claimed.** The
> table in §2.2 stands as measured — it is simply about a different axis. Both are true at
> once: three nodes win per-stream **at depth**, two nodes win aggregate **under
> concurrency**.
>
> **The methodological lesson, and it belongs in this postmortem:** a re-run that confirms
> a claim's *direction* on a different workload is not a re-run of the claim. Restating it
> as "vindicated" carried the original's depth range along with it for free, unmeasured.
> [`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3).

> **Further correction, 2026-08-27.** That depth table is also void. Its prompt requested
> 256 tokens but instructed the model to answer in one sentence, and all 70 responses
> stopped at 25–26 tokens. The apparent depth-dependent advantage measured draft-acceptance
> variance over roughly five MTP steps. It does not establish a 2-vs-3 winner.

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

### 4.5 Reasoning from resemblance instead of checking the source

Faced with a 3-6x bandwidth gap, we produced a tidy explanation — metric convention plus
message size — that fit the numbers and required no further work. **It was wrong.**
Checking `nccl-tests/doc/PERFORMANCE.md` takes minutes and shows the published DGX Spark
figures use the same collective and the same formula we do. There was never a factor of 2
to find.

Two supporting errors rode along: a 48.5 GB/s ceiling estimated from line rate (PCIe caps
it near 24), and message size credited with ~3x when the published curve shows ~29% — and
when the closest comparison was taken at a *smaller* size than ours.

**Fix: when an explanation would let you stop investigating, verify it against a primary
source before acting on it.** The convention discipline it recommended was still correct;
the conclusion it reached was not.

### 4.6 Four hypotheses before one control run

We produced four successive explanations for a bandwidth deficit that did not exist:
metric convention, a real ~3.2x hardware gap, cross-port NIC merging, and bootstrap
topology. Each was plausible, mechanistically coherent, and consistent with the numbers in
hand. **All four were wrong, and one matched run against the official binary settled it.**

The merging hypothesis is the instructive one. The mechanism was **real** — NCCL genuinely
merges `rocep1s0f0+rocep1s0f1`, two ports facing different neighbours on this ring — and it
**predicted the observed pattern**, including a deficit that grew with rank count. It was
still irrelevant: disabling merging moved the number 0.3%. A correct mechanism that
predicts your data is not thereby the cause.

What we never did was run the reference implementation. `agbench.py` was built to mirror
the vLLM MTP allgather shape and was the right instrument for its purpose; we used its top
point as a peak-bandwidth figure and compared it against someone else's peak-bandwidth
number.

**Fix: when your measurement disagrees with a published one, suspect the measurement
first, and reach for the standard tool before the third hypothesis — let alone the
fourth.**

### 4.7 Stale pointers outliving the data they point at

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
| **peer egress device** | peer traffic that would leave via Wi-Fi/mgmt instead of the fabric |
| per-pair RTT ceiling | link up but pathological |
| duplicate/overlapping subnets | the same address on two fabric NICs |
| **ARP peer-on-cabled-port** | wrong-port entries — flushes them |
| **config persistence in netplan** | addresses that vanish on reboot |
| **transport assertion** | NCCL silently falling back to sockets |
| **RDMA completion errors in a running engine** | §3.3 — degraded despite `running` |
| NCCL collective bandwidth | the §3.1 degradation |

---

## 6. Open, and honestly unresolved

1. ~~**Our 3-rank collective is far below published 3-Spark rings.**~~ **RESOLVED
   2026-08-26 — there was never a gap.** Official `all_gather_perf` reads **23.92 GB/s**
   on the same config our custom harness read 5.80 on, *exceeding* the 20.84 published
   reference. Bootstrap, NIC merging and HCA discovery each moved it <0.5%; all our
   hypotheses were falsified. See [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md).

2. **The JIT stall tail** — 13% of runs still stall after warming. Warm-up does not cover
   every shape.
3. **Root cause of spark1's original degradation** — never determined. It had been up a
   long time. If it recurs, a periodic fabric check or reboot cadence is warranted.
4. ~~**The 0.49 GB/s "GB10 ceiling" analysis.**~~ **RESOLVED.** The same fabric reaches
   23.92 GB/s in official `all_gather_perf`; 3.25 and 5.80 GB/s are results from the
   workload-shaped custom harness, not hardware ceilings. The healthy retest overturned
   the `seqs=32` rejection. PP=3 remains source-blocked; EP=3's historical slowdown margin
   remains unmeasured on healthy RDMA.

---

## 7. The one-sentence lessons

1. A green status is the weakest evidence there is — check the working path, not the
   startup path.
2. Measure the quantity you are varying; do not model it.
3. State `n` before naming a distribution.
4. A real measurement on an unusable path is still unusable.
5. Summaries rot faster than data — pin them to the data or delete them.
