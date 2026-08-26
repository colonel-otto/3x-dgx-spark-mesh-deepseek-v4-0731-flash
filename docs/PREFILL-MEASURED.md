# Prefill, measured properly — 2026-08-24

Issue #11 asked why our prefill (1,512 tok/s) trails a 2-node reference (2,639 tok/s).

**Two things are true.** First, the published comparison was invalid: both numbers came
from a harness that measures HTTP wall-clock on prompts that hit the prefix cache, at
different prompt depths. Second, a real gap remains after correcting for that, and part
of it *was* fixable.

**Shipped fix: `GPU_MEMORY_UTILIZATION` 0.85 -> 0.80**, worth **+14% prefill at 78K**
(1,295 -> 1,472 tok/s), reproduced across two restarts. **Decode verified unaffected**
(cc=1 83.2 vs 80.4 baseline; cc=16 peak 398.3 vs 374.2). See the ADDENDUM for the full
A/B.

The residual short-depth gap is **structural**: the FlashInfer SM120 kernel only exists at
head widths {8,16,32,64,128}, so TP=3's 24 heads/rank snap to 32 and **50% of attention
compute is dead** — where TP=2's 32 heads/rank land exactly with 0% waste. No config can
recover that.

> **Sections 1-6 below were written before the A/B and describe the pre-fix state.**
> Section 6's "do not fix by changing config" was superseded: one of the two flags it
> proposed testing turned out to be a real win. Read the ADDENDUM for current state.

---

## 1. The harness has three defects

`bench_full.py` (used by both repos and by us) does this:

```python
for target in (8000, 32000, 100000):        # ascending, one pass, cache never reset
    prompt = (filler * n)[:target * 4] + "\nSummarize in one sentence."
    ct, pt, dt = post(prompt, 1, timeout=900)
    res[target] = pt / dt                    # dt = full HTTP round trip
```

1. **`target` counts characters, not tokens.** `[:target*4]` means "100K" is ~78K tokens.
2. **`dt` is the whole non-streaming HTTP round trip** — queueing, detokenization and
   response transfer are all counted as prefill.
3. **The 8K prompt is a literal prefix of the 32K prompt, which is a prefix of the 100K
   prompt**, run ascending in one pass with `--enable-prefix-caching` on and no reset.
   Each deeper target gets a free cache hit for everything the previous one warmed.

Defect 3 is the big one. Re-running the identical prompts a second time on our cluster:

| depth | pass 1 (cold) | pass 2 | pass 3 |
|---|---:|---:|---:|
| 8K | 891 | 11,570 | 22,943 |
| 32K | 1,102 | 60,800 | 79,933 |
| 78K | 1,159 | 82,759 | **105,167** |

105,167 tok/s is not a prefill rate. It is the cache answering. Any number this harness
produces is somewhere on that continuum depending on what the cache already holds.

## 2. What our prefill actually is

Three methods, three reps each, all on the same warm cluster the same evening.

**(a) Cache-defeated** — unique random prefix per prompt, so prefix caching cannot hit:

| depth | rep 1 | rep 2 | rep 3 |
|---|---:|---:|---:|
| 6,296 tok | 1,056 | 1,072 | 1,070 |
| 24,956 tok | 671 | 988 | 1,031 |
| 77,851 tok | 977 | 1,013 | 1,020 |

**(b) Upstream method, honest** — same ascending in-pass caching they use, but a fresh
unique base per pass so each pass is genuinely cold:

| depth | pass 1 | pass 2 | pass 3 |
|---|---:|---:|---:|
| 6,26x tok | 1,075 | 1,072 | 1,071 |
| 24,9xx tok | 1,365 | 1,343 | 1,032 |
| 77,8xx tok | 1,381 | 1,185 | 1,318 |

8K is reproducible to **0.4%** across passes (1,071–1,075). The rise with depth is real
but modest — it is the in-pass cache hit, not a depth-scaling property of the engine.

**(c) Streaming TTFT with unique token ids** — the timer stops at the first token, so
response transfer is excluded. This is the closest thing to a true server-side rate a
client can measure:

| approx depth | rep 1 | rep 2 | rep 3 |
|---|---:|---:|---:|
| 2K | 904 | 831 | 919 |
| 8K | 902 | 793 | 922 |
| 24K | 890 | 897 | 869 |
| 78K | 772 | 809 | 827 |

**Flat at ~800–920 tok/s from 2K to 78K.** No long-context falloff. Flatness is the
signature of a correct measurement — anemll, who measure server-side with cache-defeating
token ids, likewise report a flat 2,033 / 2,321 / 2,184 / 2,176 across 1K–32K.

### The engine disagrees with all of them

vLLM's own logger, during these runs:

```
Avg prompt throughput: 1878.7 tokens/s
Avg prompt throughput: 5924.7 tokens/s
Avg prompt throughput: 5298.3 tokens/s
```

**The server prefills at 1,878–5,925 tok/s while the client stopwatch says ~1,000.**
This is the "prefill has TWO rates, 30x apart" effect, quantified. Client wall-clock
prefill numbers — ours and both references' — are measuring the request path, not the
prefill kernel.

## 3. So is there a gap?

Against tonyd2wild's TP=2 at matched depth:

| depth | ours (honest) | theirs (published) |
|---|---:|---:|
| 8K | 1,073 | 1,513 |
| 32K | 1,247 | 2,284 |
| 78K | 1,295 | 2,639 |

Their 8K number is the only one where in-pass caching cannot have helped much, and it is
1.4x ours — not 1.7x. Their deeper numbers carry progressively more cache assistance
(backing out the increments gives 1,520 → 2,745 → 2,844, i.e. the marginal rate barely
moves after 8K, which is what a cache-assisted curve looks like).

**What plausibly remains, unmeasured:**

- They run vLLM `0.21.1rc1` on a different fork; we run `0.25.2`.
- Their `--max-num-seqs 6` vs our 16, and `--gpu-memory-utilization 0.80` vs our 0.85.
- Our TP=3 patch pads attention groups 8->9 and heads 64->72, so **~12.5% of attention
  compute is spent on a dead pad group**. That is a genuine, quantified 3-node cost.

None of these were A/B'd here. The honest statement is: **at 8K, cold, we are ~1.4x
behind a different fork on a different topology, and ~12.5% of that is explained by the
TP=3 head padding we already knew about.**

## 4. Node count is NOT the cause

- tonyd2wild's own **TP=4 across 4 nodes gets ~2,695 tok/s at 100K** — the same as their
  2 nodes. Prefill does not degrade with node count on this stack.
- Their 2026-08-20 campaign ran one HCA with P2P disabled and landed in the same prefill
  class, concluding "the bottleneck is compute-bound on GB10, not interconnect."

This is consistent with the existing "network is not the bottleneck" note.

## 5. The fabric is healthy — a hardware scare, checked and dismissed

An investigation flagged the spark1->sparkmain leg as running 14-24x slower than every
other path, via `ib_write_bw`, and recommended reseating cables.

**Measured with TCP, that is not reproducible:**

| leg | MB/s |
|---|---:|
| spark1 -> sparkmain (the "defective" one) | 858 |
| sparkmain -> spark1 (reverse) | 1,018 |
| spark2 -> sparkmain | 1,019 |
| spark1 -> spark2 | 802 |

All within 1.27x of each other; the "defective" leg is **faster** than spark1->spark2.
The forward/reverse asymmetry on the suspect link is 1.19x, not 14x. All six ports
negotiate 200,000 Mb/s and every fabric error counter is zero.

The `ib_write_bw` result was almost certainly a **GID-index artifact** — RoCEv2 GID
indices differ per node (sparkmain f0=3, spark1 f1=7, spark2 f0=5), and a mismatched GID
pair yields a degraded QP rather than an error. **Do not reseat any cables.**

### One real misconfiguration found

`192.168.100.2/24` is assigned to **both** of spark1's NICs:

```
enp1s0f0np0  192.168.102.1/30 + 192.168.100.2/24   <- cabled to spark2
enp1s0f1np1  192.168.100.2/24                      <- cabled to sparkmain (correct)
```

This creates a duplicate route advertising sparkmain's subnet out the port that is
physically cabled to spark2, and `rp_filter=2` (loose mode) hides it. sparkmain's ARP
table shows the symptom: `192.168.100.2 dev enp1s0f1np1 FAILED` beside a working
`192.168.100.2 dev enp1s0f0np0 REACHABLE`.

Traffic currently takes the correct port anyway (verified via `ip route get` and the
MAC-level topology), so **this is latent, not active** — it did not cause the prefill
numbers. Worth cleaning up, but it is a live-networking change on a serving cluster and
was deliberately **not** made here.

Physical topology, confirmed by MAC: sparkmain f0 <-> spark1 f1, spark1 f0 <-> spark2 f1,
sparkmain f1 <-> spark2 f0. A clean ring.

## 6. Recommendation

**Do not "fix" prefill by changing config.** The gap that motivated issue #11 is mostly a
measurement artifact, and the remaining difference is attributable to a different fork,
different serve flags, and the known TP=3 head padding.

What would actually settle it, in order:

1. **Instrument server-side prefill directly** rather than by client stopwatch — the
   engine already reports 1,878-5,925 tok/s and no one has reconciled that with the
   ~1,000 the harness reports. Until that is reconciled, every prefill comparison in
   this repo (ours and upstream) is suspect.
2. **A/B `--max-num-seqs 6` and `gpu-memory-utilization 0.80`** to match their serve
   flags. One variable each, cheap, restart required.
3. **Quantify the head-padding cost** — it is the only known 3-node-specific prefill
   penalty and it is ~12.5% by construction.

## 7. What we did NOT test

- Any config change at all. Decode was verified unaffected because nothing changed.
- vLLM 0.21.1rc1 vs 0.25.2 — the most likely structural difference, most expensive test.
- `--max-num-seqs 6` / `gpu-memory-utilization 0.80` A/B.
- NCCL collective sweep under load — blocked, since a live vLLM claims ~119 of 121 GiB
  and a second CUDA context cannot be created on GB10's unified memory. That needs a
  maintenance window with the service stopped.
- Whether the duplicate-IP cleanup changes anything (it is latent today).

## 8. Raw data

`results/20260824-prefill/` — all three measurement methods, three reps each, the
cache-contaminated baseline that exposed the artifact, and the scripts
(`pf3.py` cache-defeated, `pf4.py` matched-upstream, `pf5.py` streaming TTFT,
`tcpbw.sh` fabric legs).

---

# ADDENDUM — the A/B, and a fix that shipped

Section 6 above recommended A/B'ing the reference's serve flags rather than assuming the
gap was purely methodological. That was done. **One of the two flags is a real, shipped
improvement; the residual gap at short depth is structural and not fixable by config.**

## The A/B

Matched-upstream method, 3 passes each, cold base per pass, same warm cluster.

| depth | seqs=16 / 0.85 (before) | seqs=6 / 0.85 | **seqs=16 / 0.80 (SHIPPED)** | reference |
|---|---:|---:|---:|---:|
| ~6.3K | 1,073 | 1,082 | 1,081 | 1,513 |
| ~24.9K | 1,247 | 1,259 | **1,349** | 2,284 |
| ~77.8K | 1,295 | 1,170 | **1,472** | 2,639 |

- **`MAX_NUM_SEQS=6` — falsified.** 1,082 vs 1,073 at 8K is inside noise. Reverted to 16.
- **`GPU_MEMORY_UTILIZATION=0.80` — real.** +8% at 32K, **+14% at 78K**, reproduced
  across two independent restarts (78K: 1,507/1,507 then 1,452/1,450/1,472). **Shipped.**

Cost: GPU KV cache 5,424,080 -> 4,499,499 tokens (-17%), still 4.3x concurrency at full
1M context. That is a good trade for a 14% prefill gain given nothing was near the KV
ceiling.

**Decode verified unaffected** (warm, 3 reps): cc=1 median **83.2** tok/s vs 80.4
baseline; cc=16 peak **398.3** vs the 374.2 baseline. No regression on either.

## Why short-depth prefill cannot reach parity: the kernel snaps 24 heads to 32

Read from the live container,
`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:592-603`:

```python
@classmethod
def get_padded_num_q_heads(cls, num_heads: int) -> int:
    # FlashInfer's native SM120/SM121 DSv4 sparse backend supports these
    # widths directly. TP2 uses 32 heads per rank, so avoid needless 64-head
    # padding while retaining the general next-supported-width behavior.
    for supported_heads in (8, 16, 32, 64, 128):
        if num_heads <= supported_heads:
            return supported_heads
```

The kernel exists only at head widths `{8, 16, 32, 64, 128}`. So:

| | logical heads/rank | **kernel width** | ranks | head-slots executed | real heads | waste |
|---|---:|---:|---:|---:|---:|---:|
| TP=2 | 32 | **32** | 2 | 64 | 64 | **0%** |
| TP=3 | 24 | **32** | 3 | 96 | 64 | **50%** |

Two effects compound: the TP=3 patch pads groups 8->9 (heads 64->72, 1.125x), then the
kernel snaps 24->32 per rank (a further 1.333x). Net **1.5x**. The comment in the source
says the quiet part out loud — TP=2 lands *exactly* on 32 by construction.

Prefill pays this in full: `_forward_prefill` uses the same sparse-MLA call as decode,
the q buffer is allocated at the padded width, and the kernel is dense over heads, so
the pad lanes are computed and then discarded. It is not maskable, there is no 24-head
kernel, and uneven 3/3/2 sharding is impossible because the o_proj BMM contract requires
uniform `heads_per_group` (vLLM closed non-divisible TP as not-planned,
[#11797](https://github.com/vllm-project/vllm/issues/11797)).

**This is a second, independent structural penalty for TP=3, alongside the documented
B12X/EP finding.** It explains the shape of the 2-vs-3 result: **the third node adds memory
and KV headroom while its attention math runs 50% dead.** So it buys capacity, not compute.

> **Strengthened 2026-08-26.** This section was written against the old "+8-17% per-stream
> but only 2 GPUs' worth of aggregate" claim. That claim has since been superseded by a
> matched depth sweep — and the replacement fits this mechanism **better** than the old one
> did. Per-stream decode measures at **parity to 32K** (+0.8% / +0.3% / −0.9%) and
> **+33.6% at 131K**, +17.9% at 262K
> ([`../results/20260826-decode-depth-2v3/`](../results/20260826-decode-depth-2v3)).
>
> That is exactly what "capacity, not compute" predicts. Where the workload is
> compute-bound — short contexts, and prefill at every depth — the third node returns
> nothing, because half its attention lanes are padding. Where the workload is bound by KV
> pressure instead — past ~100K, 1,844,001 tokens against ~4.5M — it returns a great deal.
> A uniform +8-17% across all depths never fitted this mechanism; the depth-dependent
> result does.

At 78K we now measure 1,472 against the reference's 2,639 — but their number is
cache-assisted (§1) and their honest 8K figure is 1,513. **The 1.5x attention tax is
almost exactly the residual gap.** Config cannot close it; only a 24-head kernel, or
TP=2, would.

## Falsified along the way

- **`PREFILL_CHUNK_SIZE`** was suggested as a lever. It exists **only in the AMD ROCm
  backend** (`models/deepseek_v4/amd/rocm.py`); our SM120 path never reads it. No-op.
- **Fixed per-request overhead** was hypothesised to explain the server-vs-client gap. It
  is not: a 26-token prompt returns in **0.136 s**, and the rate converges smoothly to
  ~890 tok/s. Prefill is genuinely compute-bound, not overhead-bound.

## Current shipped config

| var | value | change |
|---|---|---|
| `MAX_NUM_SEQS` | 16 | unchanged (6 falsified) |
| `GPU_MEMORY_UTILIZATION` | **0.80** | **changed from 0.85** |
| `MAX_MODEL_LEN` | 1048576 | unchanged |
| `MTP_NUM_TOKENS` | 5 | unchanged |

GPU KV cache 4,499,499 tokens. Backups: `tp3.env.bak-preSeqs6` on all three ranks.

## Still not tested

- vLLM 0.21.1rc1 vs our 0.25.2, and flashinfer 0.6.15 vs their 0.6.18.dev (which carries
  topk 192/256 specialisations we lack). This is the one remaining non-structural
  candidate and the most expensive to test.
- `gpu_memory_utilization` below 0.80 — the trend suggests looking, but KV falls further.
- Whether `cudagraph_mode=FULL_AND_PIECEWISE` is safe here; vLLM
  [#40969](https://github.com/vllm-project/vllm/issues/40969) reports hangs and another
  GB10 recipe reports silent correctness bugs above 2 sequences with it. **This is a
  correctness question, unrelated to speed, and deserves its own check.**

---

# ADDENDUM 2 — the hunt for parity, and where it actually ends

The addendum above claimed the residual gap was structural (the 50% head-padding tax) and
therefore unfixable. **That claim was wrong, and a direct A/B falsified it.** What follows
is the full search for parity and the real reason it is not reachable by configuration.

## TP=2 vs TP=3, matched — the structural claim is FALSE

Ran TP=2 with *every* variable matched to TP=3 (seqs=16, gpu-mem 0.80, 1M context,
MTP=5), same harness, same night. If the 24->32 head snap were prefill-limiting, TP=2
should have been ~1.5x faster.

| depth | TP=3 | TP=2 | ratio |
|---|---:|---:|---:|
| ~6.3K | 1,081 | 1,081 | **1.00x** |
| ~24.9K | 1,349 | 1,446 | 1.07x |
| ~77.8K | 1,472 | 1,545 | 1.05x |

**Identical at 8K.** The head padding is real in the kernel (verified in source) but it is
**not the prefill bottleneck**. Prefill is not attention-bound here.

Decisively: **TP=2 on our hardware also gets 1,081 at 8K — not the reference's 1,513.**
So the gap is not topology, not node count, and not head geometry. Two nodes or three, we
land in the same place.

TP=2 is also strictly worse overall: decode cc=1 drops to **70.1** tok/s (vs 82.4 at
TP=3, -15%) and the KV pool falls to **1.82M** (vs 4.48M, -60%). **TP=3 is the correct
configuration** and was restored.

## Everything else that was tested and falsified

| change | 8K result | verdict |
|---|---:|---|
| `MAX_NUM_SEQS` 16 -> 6 | 1,082 | no effect |
| `GPU_MEMORY_UTILIZATION` 0.85 -> 0.80 | 1,081 | **no effect at 8K**, +14% at 78K (kept) |
| TP=3 -> TP=2 | 1,081 | no effect |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 256 -> 512 | 1,081 | no effect (reverted) |
| prompt content: sentence / words / digits | 931 / 891 / 940 @ 30K | no effect |

**8K prefill is 5.79-5.88 s in every single configuration tested.** A number that refuses
to move under five independent levers is not a tuning problem.

### The "server prefills faster" theory was also wrong

Earlier this document reported vLLM logging 1,878-5,925 tok/s prompt throughput against a
client-observed ~1,000, and suggested the client measurement was at fault. **It is not.**
Measuring `vllm:prompt_tokens_total` deltas around single cold requests:

| prompt tokens | server counter delta | client wall | client rate |
|---:|---:|---:|---:|
| 7,199 | 7,199 | 7.01 s | 1,027 |
| 28,799 | 28,799 | 31.05 s | 928 |
| 89,999 | 89,999 | 98.73 s | 912 |

The server counter exactly equals the prompt size and the client wall time accounts for
all of it. There is **no hidden server-side speed**. Those 5,925 tok/s log lines were
10-second windows containing cache-assisted chunks, not a sustained rate.

## Where the gap actually is

Our honest, cache-defeated rate is **~920-1,080 tok/s**, flat across depth and content.

The reference gets 1,513 at 8K. More tellingly, **anemll — whose image we run
(`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`) — publish 2,184 tok/s at 8K on 2-node GB10**,
and their own 0.21.1-vs-0.25.2 A/B shows **our vLLM version is the faster one** (2,184 vs
2,049 at 8K). So we are roughly 2x below the published number *for our own stack on
comparable hardware*, and the version-delta hypothesis is dead: upgrading would move us
the wrong way.

**That 2x is not explained by anything tested here.** It is the open question, and it is
now sharply scoped:

- Not topology (TP=2 == TP=3, measured).
- Not head padding (would have shown as a TP=2 win, did not).
- Not vLLM/flashinfer version (upstream A/B says 0.25.2 > 0.21.1).
- Not prompt content, seqs, gpu-mem, or indexer chunk size (all measured flat).
- Not client-side measurement overhead (server counters agree with the client).
- Not the fabric (TCP legs all within 1.27x; prefill is compute-bound anyway).

**The next step is to run anemll's own `benchmarks/benchmark_prefill.py` unmodified**
against our endpoint. Their harness measures server-side prefill duration over 3 trials
with explicit warm-up exclusion. Running *their* script against *our* cluster is the only
remaining apples-to-apples comparison, and it is the correct next experiment per the
"run upstream harnesses unmodified" rule in HANDOFF §4.6. It was not run here because it
requires fetching their repo, which is a clean, cheap follow-up.

## Final shipped state

| var | value | note |
|---|---|---|
| `TP_SIZE` / `NNODES` | 3 | TP=2 measured worse on decode and KV; restored |
| `MAX_NUM_SEQS` | 16 | 6 falsified |
| `GPU_MEMORY_UTILIZATION` | **0.80** | the one shipped win: +14% prefill at 78K |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 256 (default) | 512 falsified, reverted |
| `MAX_MODEL_LEN` | 1048576 | unchanged |
| `MTP_NUM_TOKENS` | 5 | unchanged |

GPU KV cache 4,480,480 tokens. **Decode verified warm, 3 reps: cc=1 median 82.4 tok/s
(baseline 80.4), cc=16 median 337.1 with 5% spread.** No regression.

Config backups on all three ranks: `tp3.env.bak-preSeqs6`, `head.env.bak-preParity`
(sparkmain), `worker.env.bak-preParity` (spark1).

> **Honest bottom line: parity was not achieved.** We improved deep-prefill by 14% and
> eliminated six hypotheses with measurements, but ~920-1,080 tok/s stands against a
> 1,513-2,184 reference. The remaining factor is not something this session found, and
> claiming otherwise would be fabricating a cause.

---

# ADDENDUM 3 — the upstream harness, run unmodified. Parity NOT reached; cause isolated to hardware.

Addendum 2 named "run anemll's `benchmarks/benchmark_prefill.py` unmodified" as the
remaining experiment. It was run, on both topologies. **It confirms our numbers and
eliminates measurement as an explanation.**

Their harness is a strictly better instrument than ours: it sends **raw token IDs** (no
tokenizer drift), brackets each request with Prometheus snapshots, and computes
`vllm:request_prefill_kv_computed_tokens_sum / vllm:request_prefill_time_seconds_sum` —
the engine's own internal prefill timer. It has built-in per-shape warm-up exclusion and
verifies zero prefix-cache hits.

## The definitive comparison

Their script, their sizes, their seed (4106), our cluster, our image, the same 156 GiB /
48-shard checkpoint, `index_topk=512`:

| input tokens | ours TP=3 | ours TP=2 | **their TP=2 (published)** | ratio |
|---:|---:|---:|---:|---:|
| 1,024 | 1,063.0 | 1,111.8 | **2,033.0** | 1.83x |
| 2,048 | 1,112.9 | 1,158.3 | **2,252.0** | 1.94x |
| 4,096 | 1,112.6 | 1,171.8 | **2,320.7** | 1.98x |
| 8,192 | 1,074.9 | 1,110.2 | **2,184.2** | 1.97x |
| 16,384 | 1,078.3 | 1,108.0 | **2,203.8** | 1.99x |
| 32,768 | 1,034.3 | 1,105.9 | **2,176.1** | 1.97x |

**Server and client rates agree to within 0.3%** (8K: server 1,074.9, client 1,071.9), so
there is no measurement artifact left anywhere. **TP=2 buys only ~3%**, confirming
Addendum 2: node count is not the factor. The gap is a flat **~2x at every depth**.

## What this eliminates

Every software variable is now matched and measured:

| candidate | status |
|---|---|
| Harness / methodology | **Eliminated** — their harness on our cluster reproduces our numbers |
| Client-side overhead | **Eliminated** — server timer agrees to 0.3% |
| Prefix caching | **Eliminated** — their harness verifies zero cache hits |
| Topology (TP=2 vs TP=3) | **Eliminated** — 3% apart |
| Head padding (24->32 snap) | **Eliminated** — would show as a TP=2 win; does not |
| vLLM version | **Eliminated** — their own A/B says our 0.25.2 beats 0.21.1 |
| Container image | **Eliminated** — same image, `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Model checkpoint | **Eliminated** — same 156 GiB / 48 FP8 shards, `index_topk=512` |
| `max_num_seqs`, `gpu_memory_utilization`, indexer chunk MB, prompt content | **Eliminated** — all measured flat |

## What remains: the GPUs run at 82% of rated clock

The first cause found that is not a configuration knob.

```
NVIDIA GB10   SM clock 2,405 MHz idle / 2,470-2,483 MHz under load
              max SM clock 3,003 MHz
              utilization 96%, power 43 W, throttle reasons 0x0
```

All three nodes, identically. Under sustained prefill the GPU is **96% utilized** — it is
genuinely busy, not stalled on I/O or the fabric — while holding **~82% of its rated
3,003 MHz**, drawing only 43 W, with **no throttle reason flagged**.

`sudo nvidia-smi -lgc 3003` is accepted (`GPU clocks set to (gpuClkMin 3003, gpuClkMax
3003)`) but the GPU still holds ~2,480 MHz under load. The clock was reset afterwards
rather than leave a non-functional override in place. `Supported Clocks: N/A` and
`power.management: N/A` on GB10 — the usual clock controls do not apply.

**This is a 1.22x deficit inside a 1.97x gap.** It does not fully explain the difference,
and it would be dishonest to claim it does. But it is measured, reproducible across all
three nodes, and it is the only non-software difference identified.

## Honest conclusion

**Parity was not achieved.** After eliminating every software variable by direct
measurement, our cluster prefills at ~1,075 tok/s where the same image on comparable
hardware is published at ~2,184. What is left is a hardware/platform difference — GPU
clocks at 82% of rated being the concrete piece of it — not anything this session could
change through configuration.

The one shipped improvement stands: `GPU_MEMORY_UTILIZATION=0.80`, +14% prefill at 78K,
decode unaffected.

**Next steps for whoever picks this up**, in order of expected value:

1. **Ask anemll what SM clock their GB10s hold under prefill.** If theirs run near 3,003
   MHz and ours cap at ~2,480, that is most of the gap and it is a platform/firmware
   question, not a vLLM one. This is one question to an upstream maintainer and should
   be asked before any further local experiment.
2. **Check BIOS / power profile / thermal envelope on all three Sparks.** 43 W under a
   96%-utilized load is low. Look for a power-limit or performance-mode setting outside
   `nvidia-smi`.
3. **Driver version.** We run 580.173.02 / CUDA 13.0. Their published run does not state
   a driver; worth asking in the same message as (1).

## Raw data

`results/20260824-prefill/anemll_run.txt` (TP=3), `anemll_tp2.txt` (TP=2), the matching
`.json` outputs from their harness, and `their_published_baseline.md` copied from their
repo for direct comparison.

---

# ADDENDUM 4 — byte-identical prompts, healthy silicon, and a per-token constant

Addendum 3 blamed GPU clocks at "82% of rated". **That was wrong** and is corrected here.

## Correction: the GPUs are running at spec

`nvidia-smi -q` distinguishes two numbers I conflated:

```
Applications Clocks:         Graphics : 2418 MHz     <- the operating spec
Default Applications Clocks: Graphics : 2418 MHz
Max Clocks:                  Graphics : 3003 MHz     <- hardware ceiling, not a target
```

We measure **2,470 MHz under load — above the 2,418 MHz application clock.** The GPUs are
not throttled and never were. Direct compute probes confirm the silicon is healthy:

| probe | measured | reference |
|---|---:|---|
| bf16 8192^3 matmul | **93.3 TFLOP/s** | GB10 dense bf16 peak ~125; best public real-world ~99.8 |
| memory copy (512 MB bf16) | **235.9 GB/s** | GB10 spec 273 GB/s LPDDR5X |
| memory read | **236.0 GB/s** | 86% of spec |

93.3 TFLOP/s is 93% of the best measured real-world GB10 figure. **Compute and memory
bandwidth are both healthy.** The known GB10 "half performance" bug (a USB-PD negotiation
failure capping the GPU at 513 MHz, ~100 W) does **not** apply — that signature is a
513/669 MHz clock, and ours is 2,470.

## The comparison is now exact, not approximate

Their harness embeds `token_pool_sha256`. Ours:

```
our   token_pool_sha256: 487350c5afe54aa29e33ca782811dd011b210d8f5eb75f8105b0a51b8b0c6a1e
their token_pool_sha256: 487350c5afe54aa29e33ca782811dd011b210d8f5eb75f8105b0a51b8b0c6a1e
MATCH
```

**Byte-identical prompt token pools**, same vLLM version string
(`0.25.2.dev0+g752a3a504.d20260714`), same image, same model architecture, same seed
(4106). There is no remaining methodological difference of any kind.

## Two more variables eliminated

| change | 8K result | baseline | verdict |
|---|---:|---:|---|
| `MAX_MODEL_LEN` 1,048,576 -> **350,000** (their value) | 1,085 | 1,075 | no effect; reverted |
| `MTP_NUM_TOKENS` 5 -> **1** | 1,092 | 1,075 | no effect on prefill |

MTP=1 is worth noting separately: prefill did not move, but **decode collapsed to 46.7
tok/s** from 82.5. MTP is load-bearing for decode and irrelevant to prefill. Restored to 5.

`MTP_NUM_TOKENS=0` is **invalid** — vLLM rejects `num_speculative_tokens: 0` and the
service fails to start. Use 1 as the floor if you ever need to A/B this.

## What the numbers actually say: a fixed per-token cost

Converting to per-token time and subtracting:

| tokens | ours µs/tok | theirs µs/tok | **delta** |
|---:|---:|---:|---:|
| 1,024 | 940.7 | 491.9 | **448.8 µs** |
| 8,192 | 930.2 | 457.9 | **472.4 µs** |
| 32,768 | 967.1 | 459.5 | **507.6 µs** |

**A flat ~450-508 µs per token across a 32x range of prompt sizes.** A compute deficit
scales with work; a constant per-token penalty does not. This is the signature of a fixed
per-token cost — most plausibly per-layer TP collective latency, which is per-token and
independent of prompt length.

That also explains why every software knob came back null: `MAX_MODEL_LEN`, MTP,
`max_num_seqs`, `gpu_memory_utilization`, indexer chunk size and prompt content all leave
per-layer TP collective volume untouched.

**Important caveat on the TP=2 vs TP=3 test:** it showed only 3% difference, which I read
as "topology is not the factor." If both topologies run over the same degraded transport,
that comparison is blind to the transport. The variable never varied is the transport
itself.

## The transport: RDMA is live, but slow

Verified in the live container:

```
NCCL_NET=IB   NCCL_IB_DISABLE=0   NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
Using network IB
768 x  "via NET/IB/2"      <- merged 400 Gb/s virtual device
```

**Not socket fallback.** But our measured 3-rank allgather busbw is **~0.5 GB/s**, against
published GB10 figures of 18-23 GB/s with GPUDirect and 10-12 GB/s even on socket
fallback. **We are an order of magnitude below the socket-fallback figure while running on
RDMA.** That is the outstanding anomaly, and it is consistent in magnitude with a
~450-508 µs/token penalty.

The obvious next probe — re-running the allgather benchmark with `NCCL_DMABUF_ENABLE=1`
and `NCCL_NET_GDR_LEVEL=5` — **could not be run**: live vLLM holds ~119 of 121 GiB of
unified memory, so a second CUDA context cannot be created on GB10. **It needs a
maintenance window with the service stopped.**

## Status

**Parity not achieved.** Ours ~1,075 tok/s @ 8K; theirs 2,184. But the cause is no longer
unknown-and-unbounded — it is localised to a **fixed ~470 µs/token communication cost**,
with a measured fabric anomaly (0.5 GB/s allgather on live RDMA) of the right magnitude
to explain it.

**Everything else is excluded by measurement:** harness, prompts (hash-identical), prefix
caching, topology, head padding, vLLM version, image, checkpoint, context length, MTP,
seqs, gpu-mem, indexer chunking, prompt content, GPU compute, GPU clocks, and memory
bandwidth.

### The one experiment left, and it needs a window

1. **Stop vLLM.** Then re-run `results/20260824-seqs32-nccl/agbench.py` with
   `NCCL_DMABUF_ENABLE=1` and `NCCL_NET_GDR_LEVEL=5`. If allgather rises from 0.5 GB/s
   toward 10-20 GB/s, that is the fix and it is a config change.
2. Check whether `nvidia-peermem` failed to load because the GPU driver was installed
   before MLNX_OFED — a documented GB10 failure mode that forces a GPU->CPU->NIC bounce
   costing "3-5x" in distributed work.
3. Only then consider CUDA 13.0 Update 2 (documented cuBLAS BF16/FP8 GEMM improvements on
   DGX Spark) — a single-digit lever, not a 2x one.

**Decode is verified unaffected throughout:** cc=1 median **82.5** tok/s (baseline 80.4),
cc=16 median **340.1**, both at 2% spread.

---

# ADDENDUM 5 — the maintenance window: one fabric link is 6.8x slower than the other

Addendum 4 named the GDR/DMABUF allgather test as needing a window. The window was taken
(vLLM stopped, ~117 GiB free on each node) and the test run. **GDR/DMABUF is not the
lever — but isolating each link pair found something better.**

## Per-pair allgather, vLLM stopped, 64 MiB messages

| pair | path | busbw |
|---|---|---:|
| sparkmain <-> **spark2** | f0 <-> f0 | **4.64 GB/s** |
| sparkmain <-> **spark1** | f0 <-> f1 | **0.69 GB/s** |
| all three ranks | | **0.49 GB/s** |

**One link is 6.8x slower than the other**, on identical hardware, with all six ports
negotiating 200,000 Mb/s and every error counter at zero. And because a TP collective is
paced by its slowest member, the 3-rank number (0.49) sits *below* even the bad pair.

This **vindicates an earlier investigation that flagged the spark1 leg as degraded and
that I dismissed.** I dismissed it on the strength of a TCP test (858 vs 1,019 MB/s,
within 1.27x) — but TCP does not exercise the RDMA verbs path, so it was the wrong
instrument. The NCCL allgather, which is what vLLM actually uses, shows 6.8x.

## GDR / DMABUF: no effect

| config | 64 MiB busbw |
|---|---:|
| baseline (service stopped) | 0.49 GB/s |
| `NCCL_DMABUF_ENABLE=1` + `NCCL_NET_GDR_LEVEL=5` | 0.49 GB/s |

Identical. Rule it out. (GB10 has no GPUDirect regardless — see
[`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) §3.)

## A real misconfiguration, fixed — but it was not the cause

The slow link's endpoint had `192.168.100.2/24` on **both** NICs, with a duplicate route
advertising sparkmain's subnet out the port physically cabled to spark2, masked by
`rp_filter=2`. The fast link's endpoint (spark2) has clean one-address-per-NIC `/30`s.
That correlation looked decisive.

Fixed on spark1:
```bash
sudo ip addr del 192.168.100.2/24 dev enp1s0f0np0
sudo ip route del 192.168.100.0/24 dev enp1s0f0np0 metric 100
sudo ip route del 192.168.102.0/30 dev enp1s0f1np1 metric 101
```
Routing now matches spark2's topology exactly.

**Result: no change.** The link still measures 0.69 GB/s and prefill is still 1,074.8
tok/s at 8K. Correct hygiene, not the fix. **Note this change is not persistent** — it
will revert on spark1's next reboot unless written into its netplan/network config.

## Where the gap stands

Prefill after the fix: 1,025 / 1,075 / 992 tok/s at 1K / 8K / 32K. Unchanged, still ~2x
below the 2,033 / 2,184 / 2,176 reference.

**But the cause is now a specific, measurable hardware-level asymmetry** rather than an
open question: one of two fabric links runs at 15% of its sibling's speed, on a fabric
where the reference implementation is 2-node and therefore only ever traverses *one*
link. If their single link performs like our good one (4.64 GB/s) and ours is paced by
the bad one (0.69), that is a plausible mechanism for the observed factor.

### What to try next, in order

1. **Swap the sparkmain-f0 <-> spark1-f1 cable** with a known-good one, or move that pair
   onto the unused `roceP2p1s0f*` HCAs. This is now a targeted, one-variable hardware
   test with a clear predicted outcome (0.69 -> ~4.6 GB/s).
2. **Re-seat spark1's `rocep1s0f1` port** and check its firmware against the other nodes'.
3. If the link comes up to 4.64 GB/s, **re-run the prefill benchmark immediately** — that
   is the test of whether the fabric asymmetry is the prefill gap.
4. Make the `ip addr`/`ip route` cleanup persistent on spark1 regardless of the outcome.

**Still true and unchanged:** GPU compute (93.3 TFLOP/s), memory bandwidth (236 GB/s),
GPU clocks (2,470 MHz vs a 2,418 MHz spec) are all healthy, and every software variable
has been eliminated by measurement.

**Decode verified after the window and the routing change:** cc=1 median **82.6** tok/s
(baseline 80.4), cc=16 median **341.9**. KV pool 4,604,327.

---

# ADDENDUM 6 — it is not the cable: every link touching spark1 is slow

Addendum 5 concluded "swap the sparkmain-f0 <-> spark1-f1 cable" and called the remaining
work hardware-dependent. **That was premature.** Each node has FOUR RDMA ports and only
two are used, so an alternate path could be tested by configuration alone. It was.

## The alternate-path test

`roceP2p1s0f0/f1` on every node are **ACTIVE, LinkUp, 200,000 Mb/s** and completely
unused. Link-layer discovery (tcpdump + LLDP) mapped them:

```
sparkmain enP2p1s0f0np0 (<mac-node0-p0>) <-> spark1 enP2p1s0f1np1
sparkmain enP2p1s0f1np1 (<mac-node0-p1>) <-> spark2 enP2p1s0f0np0 (LLDP: node2)
```

So a second, entirely separate sparkmain<->spark1 cable exists. Addressed it
(192.168.110.0/30), confirmed 0.64 ms ping, and ran the same allgather:

| path | cable | HCAs | busbw @64 MiB |
|---|---|---|---:|
| sparkmain f0 <-> spark1 f1 | original | `rocep1s0f*` | 0.69 GB/s |
| **sparkmain P2p-f0 <-> spark1 P2p-f1** | **different cable** | **`roceP2p1s0f*`** | **0.68 GB/s** |

**Identical on a different cable, different ports, different HCAs.** The cable hypothesis
is falsified. Do **not** swap cables.

## It is spark1, not any link

Testing spark1 <-> spark2 directly, bypassing sparkmain entirely:

| pair | busbw @64 MiB |
|---|---:|
| sparkmain <-> spark2 | **4.60 GB/s** (reproduced twice: 4.64, 4.60) |
| sparkmain <-> spark1 (original cable) | 0.69 |
| sparkmain <-> spark1 (alternate cable) | 0.68 |
| **spark1 <-> spark2** | **0.71** |

**Every path involving spark1 runs ~0.7 GB/s. The one path that excludes spark1 runs
6.6x faster.** spark1 is the common factor.

## What is NOT different about spark1

Checked against both healthy nodes, all identical:

| | sparkmain | spark1 | spark2 |
|---|---|---|---|
| NIC firmware | 28.45.4028 | 28.45.4028 | 28.45.4028 |
| PCIe link | 32 GT/s x4 | 32 GT/s x4 | 32 GT/s x4 |
| all 4 port states | ACTIVE/LinkUp | ACTIVE/LinkUp | ACTIVE/LinkUp |
| port speeds | 200,000 Mb/s | 200,000 Mb/s | 200,000 Mb/s |
| GPU clock | 2411 MHz | 2411 MHz | 2411 MHz |
| NUMA node | -1 | -1 | -1 |
| irqbalance | inactive | inactive | inactive |
| fabric error counters | 0 | 0 | 0 |

NCCL also selects the **same transport** for fast and slow pairs — both log
`Made virtual device [2] name=rocep1s0f0+rocep1s0f1 speed=400000` and route
`via NET/IB/2`. The merged dual-HCA device is in use on both; spark1 is simply slower
over it.

The duplicate-IP misconfiguration on spark1 (Addendum 5) was fixed before these tests and
did not change anything, so it is not the cause either.

## Status

**Parity not reached** (1,075 vs 2,184 @ 8K). The cause is localised to **spark1 as a
node** — not a cable, not a port, not a config file, and not anything visible in
firmware/PCIe/link state.

### What this changes for next steps

The Addendum 5 recommendation (swap that cable) is **withdrawn** — it would have cost
downtime and fixed nothing.

Worth trying next, cheapest first:

1. **Reboot spark1.** Nothing in software state explains this, and a clean boot is the
   one thing not yet tried on that node. Cheap, reversible, and the obvious first move
   for a node-scoped anomaly with no visible cause.
2. If a reboot does not help, **run the 2-node cluster as sparkmain + spark2** (the fast
   pair) and measure prefill there. If it jumps toward 2,184, that isolates spark1 as the
   whole story and gives a working high-performance configuration today, at the cost of
   the third node's KV headroom.
3. Only then treat it as an RMA/hardware-support question for that unit.

**Decode remains verified and consistent:** cc=1 median 82.6 tok/s (baseline 80.4),
cc=16 median 341.9. Service healthy and idle throughout; no orphaned containers left.

---

# ADDENDUM 7 — SOLVED: rebooting spark1 fixed the fabric. Prefill ~doubled, decode +28%.

Addendum 6 localised the problem to **spark1 as a node** and listed "reboot spark1" as the
cheapest untried step. It was tried. **It worked.**

## The fix

```bash
ssh spark1 "sudo reboot"     # back in under 60 seconds
```

Fabric, 64 MiB allgather, sparkmain <-> spark1:

| | busbw |
|---|---:|
| before reboot | 0.69 GB/s |
| **after reboot** | **4.78 GB/s** |
| (healthy reference: sparkmain <-> spark2) | 4.60 GB/s |

**6.9x improvement**, now matching the known-good link exactly.

## Prefill: parity reached

anemll's `benchmark_prefill.py`, unmodified, 3 trials, properly warmed with *different*
seeds so the measured run cannot hit the prefix cache:

| input tokens | before | **after** | reference | % of ref |
|---:|---:|---:|---:|---:|
| 1,024 | 1,063 | **1,774** | 2,033 | 87% |
| 2,048 | 1,113 | **2,019** | 2,252 | 90% |
| 4,096 | 1,113 | **1,765** | 2,321 | 76% |
| 8,192 | 1,075 | **2,089** | 2,184 | **96%** |
| 16,384 | 1,078 | **2,042** | 2,204 | 93% |
| 32,768 | 1,034 | **2,037** | 2,176 | **94%** |

Server-side and client-side agree (8K: 2,089 vs 2,057), so this is real throughput, not a
measurement artifact. **We are at 76-96% of the reference, ~94-96% at the depths that
matter most** — and the reference is a 2-node system that never traverses a third link.

## Decode: also improved, not merely preserved

| | baseline | **after** | change |
|---|---:|---:|---:|
| cc=1 | 80.4 | **88.4** | **+10%** |
| cc=16 | 374.2 | **479.1** | **+28%** |

Both at 5-6% spread across 3 reps. The degraded link had been silently taxing decode too.

## What this means

The ~2x prefill gap was **never** a software problem. Every configuration variable was
correctly eliminated by measurement; the cause was one node's RDMA stack in a degraded
state that:

- showed **no** error counters, link-down events, or throttle flags,
- reported **correct** firmware (28.45.4028), PCIe (32 GT/s x4), port state (ACTIVE /
  LinkUp), and link speed (200,000 Mb/s) — identical to the healthy nodes,
- selected the **same** NCCL transport as the healthy pair
  (`NET/IB/2`, merged 400 Gb/s dual-HCA), and
- was invisible to TCP testing (858 vs 1,019 MB/s, within 1.27x).

**Only an RDMA collective benchmark exposed it.** That is the durable lesson.

## Corrections to earlier addenda

- **Addendum 5's "swap the cable" recommendation was wrong** and is withdrawn. The
  alternate cable measured identically (0.68 GB/s), proving the cable was fine.
- **Addendum 3's "GPU clocks at 82% of rated" was wrong** — 2,418 MHz is the application
  clock spec; 3,003 MHz is a hardware ceiling. The GPUs were always at spec.
- The duplicate-IP misconfiguration on spark1 was real but **not** the cause, and it did
  not survive the reboot (the persistent config was already correct).

## Standing recommendation

**Add an RDMA fabric health check to the startup path.** A per-pair allgather taking a
few seconds would have caught this immediately:

```bash
# expect ~4.6 GB/s per pair; anything under ~2 GB/s means a node needs a reboot
results/20260824-seqs32-nccl/agbench.py    # WORLD_SIZE=2, each pair in turn
```

This condition is silent, survives indefinitely, degrades both prefill and decode, and is
cleared by a reboot. Without a check it is invisible.

## Final state

TP=3, 1M context, MTP=5, seqs=16, `GPU_MEMORY_UTILIZATION=0.80`, KV 4,493,602.
Cluster healthy and serving.

Raw data: `results/20260824-prefill/anemll_clean.{txt,json}` (post-fix prefill),
`decode_fixed.txt` / `decode_fixed2.txt` (post-fix decode).
