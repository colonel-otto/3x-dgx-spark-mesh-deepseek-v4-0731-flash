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
> **What survives:** the *shape* of the argument — three nodes win per-stream, two win
> aggregate. Nothing else.
>
> **What is void:** every absolute tok/s in the tables below, **and the depth range**. The
> matched long-context re-run landed 2026-08-26
> ([`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3)) and the
> "+8–17% from 2K upward" headline is wrong in **both** directions: below 32K the advantage
> **does not exist** (+0.8% / +0.3% / −0.9%), and at 131K it is **+33.6%**, more than double
> what was claimed. The degradation had compressed a strongly depth-dependent effect into a
> flat ~13% band. See **§1 below**.
>
> **Also incomplete:** the 2026-08-25 healthy re-run found a **crossover near cc=16** that
> this page could not see, because the degraded fabric compressed the arms together.
>
> **Quote instead:** the table in [`../README.md`](../README.md#is-the-third-node-worth-it).

The case for the third node, with the measurements behind it — and the cases where it
is **not** the right answer.

Measured 2026-08-21 on DeepSeek-V4-Flash-0731, three GB10 DGX Sparks over a switchless
100G RoCE triangle. Raw data: [`../benchmarks/measurements.csv`](../benchmarks/measurements.csv).

---

## The one-sentence version

> **Superseded 2026-08-26.** The sentence that stood here — *"8–17% faster from 2K context
> upward"* — is degraded-fabric data and is retracted. The matched healthy-fabric
> replacement:

**A third Spark buys nothing per-stream below 32K context and a great deal above it —
+33.6% at 131K and +17.9% at 262K — plus ~2.4x the KV headroom, at the cost of total
throughput under concurrency and of time-to-first-token on deep one-shot prompts.**

Measured 2026-08-26, matched arms, five depths, 7 reps each:
[`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3).

---

## Side by side: what each configuration gives you

> [!CAUTION]
> **The four bolded per-stream decode rows in this table are void.** They were measured
> 2026-08-21 on degraded fabric and re-measured 2026-08-26 on healthy fabric with matched
> arms. Both the values and the *pattern* changed:
>
> | context | this table (degraded) | measured healthy 2026-08-26 |
> |---:|---:|---:|
> | 2K | +14% | **+0.8%** |
> | 8K | +8% | **+0.3%** |
> | 32K | +17% | **−0.9%** |
> | 131K | +13% | **+33.6%** |
> | 262K | not measured on 2 nodes | **+17.9%** |
>
> The rows are kept as a **diagnostic signature**: a flat ~13% band across every depth is
> what a shared bandwidth floor looks like when the real effect is depth-dependent.
> Healthy numbers: [`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3).

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

> **Updated 2026-08-26.** The inference was right: two nodes **do** serve a single 262K
> request, at 71.5 tok/s. The 409K row is still unverified on two nodes. The four "proven"
> rows this footnote points at are the ones that turned out **not** to be proven — see the
> CAUTION above.

**How to read this table.** The rows in bold at the top are what one person
waiting on a response experiences. The aggregate rows are the sum across many
simultaneous requests. If you are one user, the top rows are your reality and
the aggregate rows are not. If you are serving a team, invert that.

### The trade in one line each

> **Rewritten 2026-08-26.** The old version said *"faster responses on real coding
> contexts (+8-17%)"*. That is retracted — on healthy fabric there is **no per-stream
> decode benefit below 32K**, which is where most coding contexts live. The honest guidance
> is workload-dependent and does not compress into one percentage:

- **Buy the third Spark for:** decode at depth — **+33.6% at 131K, +17.9% at 262K** — and
  ~2.4x KV headroom. The win arrives somewhere between 32K and 131K and it is large.
- **Keep two Sparks for:** short and mid-length prompts (**parity to 32K** — the third node
  changes nothing you can feel), 12–19% more aggregate throughput under concurrency,
  **first token 6–13% sooner on deep one-shot prompts**, and a spare GB10 for a second model.

The dividing question is not context length alone but **how much you generate at that
length**. A 200K prompt answered in twenty tokens is a two-node workload; a 200K prompt
answered in two thousand is a three-node one.

---

## The five talking points

### 1. The per-stream win is real — but only past 32K

Per-stream decode is the rate one caller experiences. It is what makes an assistant feel
responsive. **It is where the third node pays, and it pays only at depth.**

Measured 2026-08-26 on healthy fabric, matched arms, node count the only variable, 7 reps
per cell, median, prefill and queueing excluded:

| Context | 2-node TP=2 | 3-node TP=3 | Gain |
|---:|---:|---:|---:|
| 2,036 | 75.8 | 76.3 | +0.8% |
| 8,081 | 72.4 | 72.6 | +0.3% |
| 32,268 | **70.8** | 70.2 | −0.9% |
| 129,006 | 54.4 | **72.6** | **+33.6%** |
| 257,993 | 71.5 | **84.4** | **+17.9%** |

**Below 32K there is no advantage.** Three cells sit inside run-to-run noise and one is
negative. **The crossover is between 32K and 131K**, and past it the gain is more than
double what this page used to claim.

The mechanism is KV pressure, not compute. The 2-node pool is **1,844,001 tokens** against
the 3-node **~4.5M**. Below 32K neither is under pressure and decode is bound by per-token
compute, which a third rank does not improve — consistent with prefill measuring at parity
([`PREFILL-MEASURED.md`](PREFILL-MEASURED.md)). Past ~100K the smaller pool costs real work
per decode step.

At 131K, the load-bearing cell, the distributions barely overlap and six of seven TP=3 reps
beat the TP=2 median:

```
TP=2 @131K:  37.0  47.1  54.2 [54.4] 54.7  64.0  64.3
TP=3 @131K:  53.8  64.7  72.3 [72.6] 74.0  76.0  79.3
```

> **Retracted from this section:** *"Consistent at every length from 2K up. Coding contexts
> live in exactly this range."* Coding contexts do live in that range, and that is exactly
> where the third node now measures at parity. The degraded-fabric table it rested on
> (69.2 / 67.9 / 70.8 / 74.0 vs 79.2 / 73.5 / 82.5 / 83.5) is a diagnostic signature.

Full methodology, including the three traps the harness is built to avoid:
[`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3).

### 1a. Decode gets *faster* past 131K, and TTFT favours two nodes

Two findings that arrived with the same run and complicate the simple story.

**The depth curve is a U, not a decay.** On three nodes: 76.3 → 72.6 → 70.2 → 72.6 →
**84.4**. At 262K this cluster decodes faster than at 8K. Every one of the seven 262K reps
beat the *median* at 32K, so it is not a stall artifact. The likely mechanism is MTP
speculative decoding — a longer context gives the draft model more signal, acceptance
rises, and that offsets the growing attention cost. Two nodes show the same upturn
(54.4 → 71.5), so it is **not node-count-specific**.

**Time to first token favours two nodes at depth:**

| Context | 2-node TTFT | 3-node TTFT | |
|---:|---:|---:|---|
| 129,006 | **72.4 s** | 77.1 s | 2-node 6% sooner |
| 257,993 | **158.4 s** | 181.6 s | 2-node **13% sooner** |

Consistent with prefill parity plus a third rank's added collective cost. So the two halves
of a deep request pull in opposite directions: **two nodes start sooner, three nodes finish
sooner** — and which wins depends on how many tokens you generate.

### 2. It is verified serving contexts up to 409,600 tokens

| Context | 2-node | 3-node |
|---:|---|---:|
| 262,144 | not measured *(see below)* | **99.1 tok/s** |
| 409,600 | not measured | **83.9 tok/s** |

> **Partly answered 2026-08-26.** Two nodes **do** serve 262K: measured at
> **71.5 tok/s** against three nodes' 84.4 on healthy fabric, 7 reps each
> ([`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3)). So 262K
> is a **speed** difference (+17.9% for three nodes), not a two-node capability wall. 409,600
> remains untested on two nodes.

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
| **TP=3** | ✅ | ✅ | **works; parity to 32K, +18–34% per stream past 100K** |
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

Rewritten 2026-08-26 against the matched depth sweep. **Context depth is now the first
question, not node count.**

| Your situation | Recommendation |
|---|---|
| **One user, long context (>100K), generating substantial output** | **3 nodes.** This is the case the third node wins outright: +33.6% at 131K, +17.9% at 262K. |
| One user, interactive coding under ~32K | **Either.** Measured **parity** (+0.8% / +0.3% / −0.9%). If you have a third Spark it costs nothing; if you do not, do not buy one for this. |
| One-shot deep prompts with **short** answers (summarise, extract, classify) | **2 nodes** — first token 6–13% sooner, and there is not enough decode to recover it. |
| Contexts beyond ~200K | **3 nodes** for decode. 2-node served 262K in this run at 71.5 tok/s, so it is no longer a capability wall — it is a **17.9% speed difference** plus KV headroom under concurrency. |
| Several concurrent users / agent swarm | **2 nodes** — 12–19% more aggregate throughput, and it frees a whole GB10 for a second model. |
| Batch jobs where total tokens/hour is the goal | **2 nodes**, same reason. |
| You have a third Spark sitting idle anyway | **3 nodes.** Free at every depth, and large past 100K. The aggregate cost only materialises under concurrency you are not generating. |

---

## How to argue this honestly

**Lead with the limitation.** "It costs 12–19% aggregate throughput, it does nothing
per-stream below 32K, and it reaches first token later on deep prompts" earns the
credibility to then say "and past 100K it decodes 18–34% faster for the person waiting."

**Never state the gain without the depth.** This is the specific mistake this page made
for five days. "+8–17%" sounded like a property of the cluster; it is a property of a
*context length*, and the number at 8K (+0.3%) and the number at 131K (+33.6%) are two
orders of magnitude apart in what they justify.

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

**Expect the "8–17% from 2K upward" claim too — it is also wrong**, and it came from this
page. It was degraded-fabric data. On healthy fabric the per-stream advantage is **parity
below 32K and +17.9–33.6% past 100K**.

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
  The 8–17% gains were claimed to hold because they were consistent across *four
  independent context lengths*. **That reasoning was wrong**: a consistent band across
  depths was the signature of a shared bandwidth floor, not of a robust effect. The
  2026-08-26 replacement takes median-of-**7** per depth and reports the spread.
- **Deep concurrency is n=1 per configuration.** Both runs were unambiguous and cost
  ~15 minutes each, but it is one data point.
- **Quality is unmeasured.** Everything here is speed. Long-context retrieval accuracy,
  tool calling, and garbling have not been tested on either configuration.
- **One model, one hardware generation.** These numbers are DeepSeek-V4-Flash on GB10.
  Do not extrapolate to other models or other Sparks.
