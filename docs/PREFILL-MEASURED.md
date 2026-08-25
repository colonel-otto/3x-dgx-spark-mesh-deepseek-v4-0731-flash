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
B12X/EP finding.** It also explains the shape of the existing "+8-17% per-stream but only
2 GPUs' worth of aggregate" result: the third node adds memory and KV headroom while its
attention math runs 50% dead.

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
