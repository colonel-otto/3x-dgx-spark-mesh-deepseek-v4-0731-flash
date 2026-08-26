# Why three DGX Sparks instead of two

> [!WARNING]
> **The numbers on this page are degraded-fabric signatures, not our results.** Measured
> 2026-08-21, when one node was running at ~15% of its collective bandwidth with zero
> error indicators ([#14](../../issues/14)) — and that node sat in the **3-node** arm, so
> the handicap fell disproportionately on the configuration under test.
>
> **If you are reproducing this and your 2-node and 3-node arms come out near-equal, that
> is the symptom, not a finding.** Two configurations landing suspiciously level is what a
> shared bandwidth floor looks like. Gate the fabric before believing either arm —
> [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md) maps the symptom to the fix.
>
> **What survives:** the direction and the rough range. A matched healthy-fabric re-run on
> 2026-08-25 measured **+16.9% at cc=1** ([`../results/20260825-decode-2v3/`](../results/20260825-decode-2v3)),
> landing inside the +8–17% band claimed here. The *shape* of the argument — three nodes
> win per-stream, two win aggregate — survives too, and was strengthened.
>
> **What is void:** every absolute tok/s in the tables below. Also **incomplete**: the
> healthy re-run found a **crossover near cc=16** that this page could not see, because
> the degraded fabric compressed the arms together.
>
> **What has not been matched yet:** the healthy re-run used an **18-token prompt**, so it
> confirms the direction at cc=1 but does **not** re-measure the 2K/8K/32K/131K
> long-context decode table below. That gap is tracked and a matched long-context re-run is
> in progress; until it lands, the per-context magnitudes here have no healthy-fabric
> counterpart.
>
> **Quote instead:** the table in [`../README.md`](../README.md#is-the-third-node-worth-it).

The case for the third node, with the measurements behind it — and the cases where it
is **not** the right answer.

Measured 2026-08-21 on DeepSeek-V4-Flash-0731, three GB10 DGX Sparks over a switchless
100G RoCE triangle. Raw data: [`../benchmarks/measurements.csv`](../benchmarks/measurements.csv).

---

## The one-sentence version

**A third Spark makes every response 8–17% faster for the person waiting on it, from 2K
context upward, and serves contexts up to 409,600 tokens with ~2x the KV headroom — at the cost
of total throughput when many requests run at once.**

---

## Side by side: what each configuration gives you

Every figure measured on the same cluster, same day, same harness
(`bench-miaai`), same prompt shape, `MTP=4` on both, warm-up discarded,
median-of-3. Node count is the only variable.

| | **2 Sparks (TP=2)** | **3 Sparks (TP=3)** | Difference |
|---|---:|---:|---|
| **Per-stream decode @2K ctx** | 69.2 tok/s | **79.2 tok/s** | **+14%** |
| **Per-stream decode @8K ctx** | 67.9 tok/s | **73.5 tok/s** | **+8%** |
| **Per-stream decode @32K ctx** | 70.8 tok/s | **82.5 tok/s** | **+17%** |
| **Per-stream decode @131K ctx** | 74.0 tok/s | **83.5 tok/s** | **+13%** |
| Per-stream decode @262K ctx | not measured * | **99.1 tok/s** | 3 nodes verified |
| Per-stream decode @409K ctx | not measured * | **83.9 tok/s** | 3 nodes verified |
| Max context **verified serving** | 131K | **409,600** | **~3x** |
| **KV cache** | 20.12 GiB | **37.36 GiB** | **1.87x** |
| **KV cache tokens** | 1,832,675 | **3,565,267** | **1.95x** |
| **Max concurrency @460K ctx** | 3.98x | **7.74x** | **1.95x** |
| Aggregate tok/s @c=8 | **161.0** | 143.6 | 2 nodes +12% |
| Aggregate tok/s @c=16 | **191.2** | 161.0 | **2 nodes +19%** |
| Draft acceptance | **82.6%** | 76% | 2 nodes better |
| 4x200K concurrent | 0.5 tok/s | 0.6 tok/s | tie - both unusable |
| Hardware freed for a 2nd model | **1 whole GB10** | none | 2 nodes |
| Correctness (17x23) | pass | pass | tie |

\* **Honest caveat on the two "not measured" rows.** We measured 3-node up to 409,600
tokens and 2-node up to 131,072. We did **not** test 2-node at 262K+, so "two nodes
cannot do this" is an inference from its 1.83M-token KV pool, not a measurement. Two
nodes may well serve a *single* 262K request; what they demonstrably cannot do is hold
anywhere near the concurrent depth three nodes can. Treat the top four bolded rows as the
proven case and these two as unverified.

**How to read this table.** The rows in bold at the top are what one person
waiting on a response experiences. The aggregate rows are the sum across many
simultaneous requests. If you are one user, the top rows are your reality and
the aggregate rows are not. If you are serving a team, invert that.

### The trade in one line each

- **Buy the third Spark for:** faster responses on real coding contexts (+8-17%),
  contexts beyond ~131K that two nodes simply cannot hold, and ~2x KV headroom.
- **Keep two Sparks for:** 12-19% more total throughput under concurrency, and a
  spare GB10 you can point at a second model.

---

## The five talking points

### 1. It is faster where a user actually feels it — 8–17%

Per-stream decode is the rate one caller experiences. It is what makes an assistant feel
responsive.

| Context | 2-node TP=2 | 3-node TP=3 | Gain |
|---:|---:|---:|---:|
| 2,048 | 69.2 | **79.2** | **+14%** |
| 8,192 | 67.9 | **73.5** | **+8%** |
| 32,768 | 70.8 | **82.5** | **+17%** |
| 131,072 | 74.0 | **83.5** | **+13%** |

Consistent at every length from 2K up. Coding contexts live in exactly this range.

### 2. It is verified serving contexts up to 409,600 tokens

| Context | 2-node | 3-node |
|---:|---|---:|
| 262,144 | not measured | **99.1 tok/s** |
| 409,600 | not measured | **83.9 tok/s** |

Decode stays **flat from 256 to 409,600 tokens** — no long-context collapse, which is
itself worth noting: throughput does not degrade as the context grows, only
time-to-first-token does.

⚠️ We verified 3-node to 409,600 and 2-node to 131,072. We did not test 2-node above
131K, so this is a *demonstrated* three-node capability rather than a proven two-node
limitation.

### 3. It nearly doubles KV capacity — 1.95x

| | 2-node | 3-node | Ratio |
|---|---:|---:|---:|
| KV cache | 20.12 GiB | 37.36 GiB | 1.87x |
| KV cache tokens | 1,832,675 | 3,565,267 | **1.95x** |
| Max concurrency @460,800 | 3.98x | 7.74x | **1.95x** |

⚠️ **Do not oversell this one.** See "What the third node does *not* buy" below. The
capacity is real and enables point 2, but on its own it did not convert into measurable
capability in our tests.

### 4. It holds the fast kernel and speculative decoding together

Three-node DeepSeek-V4 has three possible shardings. Only one keeps everything:

| Sharding | B12X MoE kernel | MTP speculation | Result |
|---|---|---|---|
| **TP=3** | ✅ | ✅ | **works, 8–17% faster per stream** |
| EP=3 | ❌ refuses EP | ✅ | 2.5x slower |
| PP=3 | ✅ | ❌ no `SupportsPP` | blocked, never served a token |

TP=3 was widely believed impossible — 64 heads, 4096 hidden and 256 experts are all
indivisible by 3. The real blocker was narrower: `o_groups = 8`. Stock vLLM computes
`8 // 3 == 2` **silently**, dropping six of eight attention groups and serving fluent
nonsense. A padding patch (groups 8→9, heads/group held at 8) fixes it — 14/14 anchors,
correctness verified.

### 5. RoCE works switchlessly — no extra hardware

`NCCL_IB_SUBNET_AWARE_ROUTING=1` is the whole fix. It makes NCCL select the HCA whose
subnet reaches each peer instead of pairing by device index. Three point-to-point cables
in a triangle, no switch, no rebuild.

### Transport matters more than node count — and 24.59 tok/s is a diagnostic signature

> [!IMPORTANT]
> **This subsection is a troubleshooting reference for people reproducing our work, and
> it is the most reusable thing on this page.** The 24.59 figure is not a result of ours
> in any tier — it is the number you get when NCCL has silently fallen back to TCP. Match
> against it; do not chase it.

The same TP=3 configuration measures:

| Transport | decode tok/s | What it means |
|---|---:|---|
| **Socket / TCP** | **24.59** | NCCL could not bring up RDMA and **fell back to TCP** |
| **RoCE / RDMA** | **57.73** | NCCL RDMA working as intended |

**The 24.59 figure is not a TP=3 result and must never be quoted as one.** It is the
control measurement for a *failed* NCCL RDMA bring-up. When subnet-aware routing was not
set, NCCL paired HCAs by device index, could not reach the peer, and silently degraded
to the TCP socket path. That degradation is expected and well understood - TCP carries
the collectives correctly, just slowly, so the run *succeeds* and only the token rate
betrays it.

**Diagnostic value:** if three-node DSv4 lands near ~25 tok/s rather than ~55-58, you are
almost certainly on the TCP fallback, not suffering a node-count penalty. Check
`NCCL_IB_SUBNET_AWARE_ROUTING=1` is actually reaching the container
(`docker compose config | grep SUBNET_AWARE`) before drawing any conclusion about
scaling.

### Cabling: the NVIDIA mesh layout is a convention, not a requirement

Both cable layouts were benchmarked on identical software, checkpoint, harness and
prompts:

| | Prior layout (same-port) | NVIDIA cross-connected mesh |
|---|---:|---:|
| decode tok/s (median) | **57.73** (5 reps) | 53.95 (5 and 7 reps, twice) |
| KV cache tokens | 3,598,182 | **3,606,027** |
| Max concurrency | 7.81x | **7.83x** |
| Correctness | pass | pass |

**Deviating from NVIDIA's reference mesh works, with no quality loss.** Each layout
needs its own per-cable subnet assignment - moving a lane means changing the addresses on
both ends, and getting that wrong makes the node unreachable rather than slow. But once
`NCCL_IB_SUBNET_AWARE_ROUTING=1` is set, NCCL selects the HCA whose subnet reaches each
peer, so **the physical pairing convention stops mattering**.

The cross-connected layout exists so that *index-based* device pairing works without
subnet-aware selection. It is not a prerequisite once you set that variable. On this
cluster the reference mesh actually measured **~6.5% slower** (reproduced twice) with a
wider spread, and was retained anyway for topological conformance - correctness and KV
capacity were unaffected either way.

**Takeaway for anyone cabling a triangle:** use whichever lane arrangement your DACs
reach, give every cable its own subnet, set `NCCL_IB_SUBNET_AWARE_ROUTING=1`, and verify
any-to-any reachability **including worker-to-worker** before launching.

---

## What the third node does *not* buy — state this up front

Leading with these makes the rest credible.

### ❌ It does not increase aggregate throughput. It reduces it.

| Concurrency | 2-node | 3-node | |
|---:|---:|---:|---|
| 8 | **161.0** | 143.6 | 2-node +12% |
| 16 | **191.2** | 161.0 | **2-node +19%** |

Trials at c=16 were 190.8 / 192.6 / 191.2 — a 1% spread, so this is a real effect, not
noise. TP=3 adds a communication hop on every layer; with 8–16 requests in flight that
coordination cost exceeds what the extra compute returns.

**Aggregate is the right metric for a multi-tenant server. It is not what one user
experiences.**

### ❌ The doubled KV did not pay off at deep concurrency

We tested the case it was supposed to win — four concurrent 200K-token prompts:

| | 2-node | 3-node |
|---|---:|---:|
| decode | 0.5 tok/s | 0.6 tok/s |
| TTFT | 539.7 s | 553.1 s |
| wall clock | 14.5 min | 14.4 min |
| preemptions | **0** | **0** |

Identical, and both unusable. Zero preemptions on both means neither engine ever ran out
of KV — they **serialized the prefills**. The bottleneck is long prefill, not capacity,
and more KV does not accelerate prefill.

`num_preemptions_total` has been **0 in every test** on this cluster. **Check that metric
before accepting any KV-capacity argument, including this one.**

### ❌ Draft acceptance is slightly worse

82.6% on two nodes versus 76% on three, at equal MTP depth. Consistent with 2-node also
winning aggregate.

---

## Decision guide

| Your situation | Recommendation |
|---|---|
| **One user, interactive coding, long contexts** | **3 nodes.** Points 1 and 2 are exactly your workload. |
| Contexts beyond ~200K | **3 nodes** - verified to 409,600. Two nodes are untested above 131K. |
| Several concurrent users / agent swarm | **2 nodes** — 12–19% more aggregate throughput, and it frees a whole GB10 for a second model. |
| Batch jobs where total tokens/hour is the goal | **2 nodes**, same reason. |
| You have a third Spark sitting idle anyway | **3 nodes.** The per-stream win is free; the aggregate cost only materialises under concurrency you are not generating. |

---

## How to argue this honestly

**Lead with the limitation.** "It costs 12–19% aggregate throughput, and the extra KV
didn't help at deep concurrency" earns the credibility to then say "and it makes every
response 8–17% faster for the person waiting."

**Never quote a tok/s number without its prompt.** On this deployment the benchmark
prompt alone moves single-stream decode **1.65x** (81.8 code-shaped vs 49.4 dense prose,
same script and engine, minutes apart) because MTP acceptance is content-dependent.
Cross-repo comparisons are meaningless unless the prompt matches. See
[`BENCHMARK-METHODOLOGY.md`](BENCHMARK-METHODOLOGY.md).

**Distinguish per-stream from aggregate every time.** Most arguments about this go wrong
because the two sides are quoting different metrics at each other. Both of ours are in
[`../benchmarks/summary.csv`](../benchmarks/summary.csv), each tagged with its
comparability.

**Expect the "19.7% faster" claim — it is wrong.** An earlier version of this comparison
circulated that figure. It compared a code-shaped prompt against a dense-prose one at
different MTP depths: three confounds. The *direction* was right; the number was not.
The measured per-stream advantage is **8–17%**.

---

## Methodology

Node count was the only variable. Same cluster, same afternoon, same harness
(`bench-miaai`, MiaAI-Lab methodology), same prompt shape, warm-up sweep discarded,
median-of-3.

| Setting | 2-node | 3-node |
|---|---|---|
| `TP_SIZE` / `NNODES` | 2 / 2 | 3 / 3 |
| `MTP_NUM_TOKENS` | 4 | 4 |
| `MAX_NUM_SEQS` | 16 | 16 |
| `MAX_MODEL_LEN` | 460,800 | 460,800 |
| `MAX_NUM_BATCHED_TOKENS` | 8,192 | 8,192 |
| `GPU_MEMORY_UTILIZATION` | 0.85 | 0.85 |
| checkpoint / image | official 0731 / `anemll:0.1.1` | same |

The 2-node baseline normally runs MTP=5; it was set to 4 for this test so speculation
depth matched. Both configurations passed correctness (`17x23 → 391`).

### Caveats a sceptic will raise, and should

- **Single-stream decode is noisy on this cluster.** 8 reps at c=1 spanned 66.6–88.5
  (median 80.4). Any single-stream difference under ~20% needs median-of-N with N ≥ 5.
  The 8–17% gains hold because they are consistent across *four independent context
  lengths*, not because any one reading is decisive.
- **Deep concurrency is n=1 per configuration.** Both runs were unambiguous and cost
  ~15 minutes each, but it is one data point.
- **Quality is unmeasured.** Everything here is speed. Long-context retrieval accuracy,
  tool calling, and garbling have not been tested on either configuration.
- **One model, one hardware generation.** These numbers are DeepSeek-V4-Flash on GB10.
  Do not extrapolate to other models or other Sparks.
