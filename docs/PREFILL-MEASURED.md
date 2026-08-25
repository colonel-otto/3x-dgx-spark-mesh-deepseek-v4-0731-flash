# Prefill, measured properly — 2026-08-24

Issue #11 asked why our prefill (1,512 tok/s) trails a 2-node reference (2,639 tok/s).

**The comparison was invalid.** Both numbers were produced by a harness that measures
HTTP wall-clock on prompts that hit the prefix cache, at different prompt depths. Once
the measurement is corrected, most of the gap disappears and what remains is a
*methodology* difference, not a hardware one.

**No config change was needed and none was made.** Decode is untouched.

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
