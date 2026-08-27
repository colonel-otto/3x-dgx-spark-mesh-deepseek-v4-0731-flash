# KV-cache dtype A/B: `nvfp4_ds_mla` vs `fp8_ds_mla`

**Status:** `CURRENT` within the provenance caveats in [`../index.yaml`](../index.yaml).

Issue [#16]. Experiment date 2026-08-26 (UTC). 3-node DGX Spark cluster, TP=3.

Answers the gap left by [`docs/KV-QUALITY-LONG-CONTEXT.md`](../../docs/KV-QUALITY-LONG-CONTEXT.md),
which measured `nvfp4_ds_mla` clean to 464K **with no comparison arm**. This run supplies
the missing arm.

## Headline

**The two dtypes produce the same output.** In the main sweep, **11 of 12 cells returned
byte-identical replies** across the arms — including the single failing cell, where both
arms emitted the *same wrong answer*. Speed and KV pool size are also equal within noise.

## Configuration (identical across arms except the dtype)

Both arms: `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=8192`,
`MTP_NUM_TOKENS=5`, `GPU_MEMORY_UTILIZATION=0.80`, TP=3 / PP=1 / NNODES=3,
`MOE_BACKEND=flashinfer_b12x`, `VLLM_DSV4_TP_PAD=1`, block-size 256, prefix caching on.
Image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM `0.25.2.dev0+g752a3a504.d20260714`.

Engine-shaping keys were md5-compared across all three ranks before each start
(`9e53666aa533ec074dcfd0b7a2f8f268`), and `docker-compose.yml` was verified byte-identical
on all three (`82a36e44ac896478bbd883e423a839c4`). A mismatch here hangs startup forever
with no error, so this check is not optional.

### The dtype is now a variable

`--kv-cache-dtype` was hardcoded at line 139 of `docker-compose.yml` on all three nodes. It
is now:

```yaml
--kv-cache-dtype ${KV_CACHE_DTYPE:-nvfp4_ds_mla}
```

Behaviour is unchanged when the variable is unset. Arms are selected by setting
`KV_CACHE_DTYPE` in `config/tp3.env` on **every** rank. Backups:
`docker-compose.yml.bak-kvab-20260826` and `config/tp3.env.bak-kvab-20260826`.

Substitution was verified with `docker compose config` in both directions before any
restart, rather than assumed.

## Engine facts (read from the log, not assumed)

| | `nvfp4_ds_mla` (boot 1) | `fp8_ds_mla` | `nvfp4_ds_mla` (boot 2) |
|---|---|---|---|
| `kv_cache_dtype` in engine config | `nvfp4_ds_mla` | `fp8_ds_mla` | `nvfp4_ds_mla` |
| Available KV cache memory | 31.24 GiB | 31.62 GiB | 30.96 GiB |
| **GPU KV cache size** | **4,451,877 tok** | **4,504,137 tok** | **4,483,281 tok** |
| Max concurrency @ 1,048,576 tok | 4.25x | 4.30x | 4.28x |
| Correctness gate (17x23 = 391) | PASS | PASS | PASS |

**KV pool size is equal; the differences are restart noise, not dtype.** The control that
settles it: nvfp4 was booted **twice**, and the two nvfp4 boots differ by **31,404 tokens**
— comparable to the 52,260-token gap between nvfp4 and fp8. Both dtypes land inside the
same restart-to-restart band, exactly as the shared 584 B/token sparse-MLA envelope
predicts. **Switching dtype costs no context.**

> The earlier doc's 5,444,869-token pool was measured at `GPU_MEMORY_UTILIZATION=0.85`;
> the cluster now runs 0.80, which accounts for the smaller pool in all three boots here.

## Method

Both arms received identical treatment on the same day with the same harness
([`kvab.py`](kvab.py)):

1. Set dtype on all three ranks, verify md5 agreement, restart, wait for the API.
2. **Verify from the engine log which dtype actually loaded** (both confirmed above).
3. Record the KV cache size the engine reports.
4. Correctness gate — a broken config serves fluent nonsense, not an error.
5. **Warm every shape** to be measured (see [`docs/TTFT-AND-WARMUP.md`](../../docs/TTFT-AND-WARMUP.md)).
   A ~5s JIT compile can otherwise land inside a request. On both arms all
   `JIT compilation during inference` events were absorbed during warmup; the counter did
   not move during any measurement.
6. Quality: needle retrieval, 4K/32K/128K/256K x 3 depths (10%/50%/90%) x 3 trials.
7. Speed: decode tok/s at concurrency 1/4/8, median of 5 runs, plus TTFT.

Needle codes are generated deterministically per (length, depth, trial), so both arms see
**byte-identical prompts** — confirmed by matching `prompt_tokens` in every cell. Filler is
varied technical prose, never repeated text, so the sparse-attention indexer cannot exploit
compressibility. Temperature 0, fixed seed.

### Trap: prefix caching silently fakes a pass

`--enable-prefix-caching` is on. The needles sit at 10%/50%/90%, so **everything before the
first needle is shared prefix**. Re-running a (length, trial) an engine has already served
returned in **1.5 s instead of ~135 s** — a cache hit that re-exercises none of the KV path,
and would have counted as a clean pass.

The main sweep is unaffected: each arm saw every prompt for the first time on a freshly
restarted engine, and the ~135 s latencies confirm real prefills. But the extended
high-`n` runs repeat cells, so `kvab.py` grew a `--salt` flag that prepends a per-trial
reference line, forcing the prompt to diverge at token ~0 and re-prefill in full. The salt
is identical across arms, so the A/B stays matched.

**Any future rerun of this harness against a warm engine must pass `--salt`**, or it is
measuring the cache rather than the model.

### Trap: a long prefill looks exactly like a hang

Twice during the extended runs I read these and concluded the engine was wedged:

```
vllm:num_requests_running     1.0     (steady)
vllm:prompt_tokens_total      FROZEN
vllm:generation_tokens_total  FROZEN
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs
```

**That reading was wrong.** Those are *completion-time* counters — they increment when a
request finishes, not while it runs, so during a single ~160 s chunked prefill they are
supposed to sit still. The first time, I killed the client and destroyed a trial that was
very likely fine. The next identical-looking request completed in 166.4 s with all three
needles.

To distinguish a real hang from a long prefill, use a counter that advances *within* a
request — GPU KV cache usage %, per-step scheduler lines, or `nvidia-smi` on a rank —
and only suspect a hang well past the known ~150 s/246K budget. Details in
[`stall-note.txt`](stall-note.txt). No confirmed hang occurred in this experiment, on
either dtype.

## Results 1 — Quality, main sweep (n=3 trials per cell, 36 needles per arm)

Exact pass counts, not percentages. Each cell is 3 trials; each trial retrieves 3 needles.

| Requested | Actual ptok | nvfp4 early/mid/late | fp8 early/mid/late | replies identical |
|---:|---:|:---|:---|:---:|
| 4,000 | 3,938 | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 2/3 |
| 32,000 | 30,855 | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 3/3 |
| 128,000 | 123,169 | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 3/3 |
| 256,000 | 246,241 | **2/3** · 3/3 · 3/3 | **2/3** · 3/3 · 3/3 | 3/3 |
| **Total** | | **35/36**, garble 0 | **35/36**, garble 0 | **11/12** |

### The one failure is identical on both dtypes

Both arms missed the same needle, in the same cell, **with the same wrong answer**:

```
truth : SLATE-HARROW-4676      (early / 10% depth, 246,241 prompt tokens)
nvfp4 : SLATE-HARROW-4677      <- off by one in the last digit
fp8   : SLATE-HARROW-4677      <- byte-identical to nvfp4
```

A KV-precision defect would not reproduce byte-for-byte across two different KV dtypes.
This is a property of the model on this prompt at this depth, **not** KV-cache damage, and
it is therefore not evidence for or against either dtype. (It is a mild long-context
retrieval wobble worth knowing about on its own.)

More broadly, **11 of 12 cells produced byte-identical replies** at temperature 0. The
twelfth (4K trial 0) differed only in the *order* the three codes were listed — both arms
retrieved all three correctly. The dtypes are not merely scoring the same; they are largely
computing the same token stream.

## Results 2 — Quality, extended run at 256K (n=12 trials per arm, cache-busted)

256K is the only depth where anything failed, so it got the high-`n` treatment, with
`--salt KVAB26` so every trial is a genuine uncached prefill.

| arm | trials | early | mid | late | needles | garble | median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nvfp4_ds_mla` | 12 | 12/12 | 12/12 | 12/12 | **36/36** | 0 | 158.5 s |
| `fp8_ds_mla` | 12 | 12/12 | 12/12 | 12/12 | **36/36** | 0 | 159.4 s |

**Both arms complete: 36/36 needles each, zero garbling.** And the decisive result —
**all 12 matched pairs returned byte-identical replies.** Two different KV dtypes emitted
the same token stream on every one of twelve uncached 246K-token prefills. A KV-precision
defect cannot do that.

Combined with the main sweep (11 of 12 identical; the 12th differed only in the *order*
of three correctly-retrieved codes), that is **23 of 24 matched cells byte-identical**.

## Results 3 — Speed (median of 5 runs per cell, streaming, 512 max tokens)

| conc | nvfp4 per-stream | fp8 per-stream | delta | nvfp4 agg | fp8 agg | nvfp4 TTFT | fp8 TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.47 tok/s | 51.98 tok/s | −1.50 (−2.9%) | 50.5 | 52.0 | 233 ms | 222 ms |
| 4 | 28.60 tok/s | 27.91 tok/s | +0.69 (+2.5%) | 114.5 | 112.1 | 499 ms | 366 ms |
| 8 | 20.42 tok/s | 20.09 tok/s | +0.33 (+1.6%) | 163.0 | 162.2 | 371 ms | 356 ms |

**Speed is a tie, and the data says so twice over:**

1. Every delta is **smaller than one pooled standard deviation** of the run-to-run spread
   (conc=1: 1.50 vs sd 2.69; conc=4: 0.69 vs 0.75; conc=8: 0.33 vs 0.46).
2. **The sign flips.** fp8 leads at concurrency 1; nvfp4 leads at 4 and 8. A real effect
   does not change direction with concurrency — noise does.

This independently reproduces the upstream A/B (41.4 vs 41.5 tok/s, *"a context lever, not
a speed lever"*) on our own hardware, same day, same harness.

TTFT is likewise indistinguishable except conc=4, where nvfp4's 499 ms vs fp8's 366 ms sits
inside a per-run TTFT spread of 329–507 ms on nvfp4 alone — i.e. noise, not signal.

## What this sample size can and cannot support

Being explicit, because the honest limit matters more than the headline:

- **The main sweep alone proves nothing about a difference.** 35/36 vs 35/36 is a tie with
  nothing to test; and had it been 35/36 vs 36/36, Fisher's exact gives **p = 1.00**. n=36
  per arm cannot separate these arms on pass-counts.
- **Even with the extended run**, the design detects only a fairly gross gap: at 36 needles
  per arm at 256K, 36/36 vs 32/36 is p = 0.115, and 36/36 vs 34/36 is p = 0.49. So this
  experiment can rule out a **~11-point-or-worse** accuracy gap at 256K. **It cannot rule
  out a difference of a few percent.**
- The result that actually carries the weight is not the pass-count at all — it is that
  **11/12 replies were byte-identical**. That is a far tighter constraint than any
  pass-rate comparison at this n, because it shows the two dtypes are producing the same
  tokens, not merely scoring alike.
- **Untested:** concurrency *combined with* long context (the specific pairing in the
  upstream warning), agentic/tool-call/JSON context structure, depths above ~246K, and
  temperature > 0. See the caveats in
  [`docs/KV-QUALITY-LONG-CONTEXT.md`](../../docs/KV-QUALITY-LONG-CONTEXT.md), which this
  run does not fully close.
