# Four-HCA fabric: 2x the bandwidth buys **no measurable decode throughput** — 2026-08-26

Closes the last open branch of [#17](../../issues/17). The soak and correctness gates
passed on 2026-08-26; this is the throughput question that remained.

**Answer: decode is flat. Doubling fabric bandwidth does not make this workload faster.**
Per [#17](../../issues/17)'s own decision rule, that means: *document that fabric
bandwidth is not the decode bottleneck and leave the config alone.* Four-HCA stays
adopted — it was adopted for **headroom and redundancy**, and it costs nothing.

## The measurement

Same harness (`bench_tp3.py`), same 18-token code-brief prompt, 256 tokens out,
temperature 0, **median of 7 after 3 warm-up sweeps** on both arms, TP=3.

| cc | 2 HCA (2026-08-25) | 4 HCA (2026-08-26) | apparent change |
|---:|---:|---:|---:|
| 1 | 89.1 | 88.6 | −0.6% |
| 4 | 208.8 | 218.7 | +4.7% |
| 8 | 322.7 | 350.0 | +8.5% |
| 16 | 474.8 | 488.8 | +2.9% |

**Ignore the apparent gains — they are inside run-to-run spread.** The per-run
distributions on the 4-HCA arm contain the 2-HCA median at every level where a gain
appears:

```
cc=4   [188.0 203.2 218.5 [218.7] 225.4 228.9 252.0]   2-HCA median 208.8 -> inside
cc=8   [283.5 307.8 349.0 [350.0] 350.1 350.6 366.6]   2-HCA median 322.7 -> inside
cc=16  [437.8 454.6 467.8 [488.8] 502.9 600.6 608.8]   2-HCA median 474.8 -> inside
```

A difference that sits inside the other arm's own spread is not a difference. The honest
reading of all four rows is **flat**.

## The fabric really did double — that part is not in doubt

| busbw @64MiB | 2 HCA | 4 HCA | gain |
|---|---:|---:|---:|
| pair 0↔1 | 4.63 | **7.78** | 1.68x |
| pair 0↔2 | 4.73 | **9.19** | 1.94x |
| pair 1↔2 | 4.60 | **9.33** | 2.03x |

Measured today with the engine stopped, gate 33 passed / 0 failed
([`../20260826-decode-depth-2v3/fabric-gate-pre-tp2.txt`](../20260826-decode-depth-2v3/fabric-gate-pre-tp2.txt)). NCCL confirms all four
controllers in use:

```
NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE
                [2]roceP2p1s0f0:1/RoCE [3]roceP2p1s0f1:1/RoCE
```

with traffic split `via NET/IB/4` and `via NET/IB/5`.

**So this is not "the change did not take."** The bandwidth doubled and the workload did
not care.

## Why this is the expected answer, in hindsight

Three independent results already pointed here, and [#17](../../issues/17) said so before
the measurement:

1. **Prefill measures at parity between 2 and 3 nodes.** Prefill is the communication-heavy
   phase. If the fabric were the constraint, that is where it would show first.
2. **Decode at cc=1 is not bandwidth-bound.** A single stream exchanges small activations
   per token; latency and per-token compute dominate, not throughput.
3. **The 3-rank collective already sat at the PCIe ceiling** — ~24 GB/s under the official
   `all_gather_perf`, because both 200G ports share two PCIe Gen5 x4 lanes
   ([`../../docs/BANDWIDTH-COMPARISON.md`](../../docs/BANDWIDTH-COMPARISON.md)).

**Adding capacity to a resource that is not the bottleneck does not make anything faster.**
That is the finding, and it is worth stating plainly because the intuition runs the other
way.

## Why four-HCA stays adopted anyway

It costs nothing measurable and it buys:

- **Redundancy.** Four addressed controllers instead of two.
- **Headroom** for workloads that *are* fabric-bound — larger tensor-parallel degrees,
  higher concurrency, or a future model with heavier all-reduce traffic.
- **A validated configuration**: soak passed 408 requests with zero RDMA counter deltas
  across 132 counters, correctness 391, gate 24 pass / 0 fail.

The alternative — reverting to two HCAs — would save nothing and lose the redundancy.

## A config note worth knowing

The live `config/tp3.env` contains what looks like a typo:

```
NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
             ^^ double equals
```

NCCL receives the literal value `=rocep1s0f0,...`. **This is harmless, and arguably
correct**: a leading `=` in `NCCL_IB_HCA` means *exact device match* rather than prefix
match. Verified from the engine log — all four devices are selected and both merged
devices carry traffic.

It is recorded here because it is **indistinguishable from a mistake** on inspection, and
a future reader "fixing" it would silently switch from exact to prefix matching.

## Caveat on scope

This measures **decode throughput on an 18-token prompt at cc=1–16.** It does not measure:

- long-context decode (see [`../20260826-decode-depth-2v3/`](../20260826-decode-depth-2v3),
  which was measured on four HCAs throughout and has no two-HCA arm),
- prefill,
- concurrency above 16,
- any workload with a larger all-reduce footprint.

The claim is narrow on purpose: **for this workload, fabric bandwidth is not the
constraint.** It is not a claim that fabric never matters.

**Related:** [#17](../../issues/17) · [`../20260825-upper-mesh/`](../20260825-upper-mesh) ·
[`../20260825-decode-2v3/`](../20260825-decode-2v3) (the 2-HCA arm)
