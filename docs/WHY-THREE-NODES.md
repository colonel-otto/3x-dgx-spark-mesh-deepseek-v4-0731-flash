# Why three DGX Sparks instead of two

The case for the third node, with the measurements behind it — and the cases where it
is **not** the right answer.

Measured 2026-08-21 on DeepSeek-V4-Flash-0731, three GB10 DGX Sparks over a switchless
100G RoCE triangle. Raw data: [`../benchmarks/measurements.csv`](../benchmarks/measurements.csv).

---

## The one-sentence version

**A third Spark makes every response 8–17% faster for the person waiting on it, from 2K
context upward, and unlocks context lengths two nodes cannot serve at all — at the cost
of total throughput when many requests run at once.**

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

### 2. It reaches context lengths two nodes cannot serve

| Context | 2-node | 3-node |
|---:|---|---:|
| 262,144 | out of reach | **99.1 tok/s** |
| 409,600 | out of reach | **83.9 tok/s** |

Decode stays **flat from 256 to 409,600 tokens** — no long-context collapse. Two nodes
cannot hold these contexts at all, so this is a capability difference, not a speed
difference.

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

Transport matters enormously: the same TP=3 config runs **24.59 tok/s over TCP** versus
**57.73 over RoCE**. If someone reports three-node DSv4 being slow, check the transport
before concluding anything about node count.

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
| Contexts beyond ~200K | **3 nodes.** Two cannot serve them. |
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
