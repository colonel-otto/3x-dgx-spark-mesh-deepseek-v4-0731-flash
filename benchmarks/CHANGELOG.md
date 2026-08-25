# Configuration changelog

Every `config_id` in [`measurements.csv`](measurements.csv) is defined here. If you are
reading a CSV row and want to know what settings produced it, look up its `config_id`
below. See [`README.md`](README.md) for the three file schemas and which comparisons
between them are valid.

Entries are newest first. **When a setting changes, record it here in the same commit as
the measurements** — a throughput number whose configuration is undocumented cannot be
reproduced or trusted.

---

## ✅ Current production configuration

3 nodes, `TP=3`, `PP=1`, RoCE, `moe-backend flashinfer_b12x`, official
DeepSeek-V4-Flash-0731 checkpoint, image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`.

```
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192
MTP_NUM_TOKENS=5
GPU_MEMORY_UTILIZATION=0.80
--kv-cache-dtype nvfp4_ds_mla
--block-size 256
```

Live KV on this config: **4,493,602 tokens @1M context**. Correctness verified
(17×23 → 391).

> **This block was stale until 2026-08-25** — it still described `460800` / `MTP=4` /
> `0.85`, the profile superseded when 1M context and `MTP=5` were settled on 2026-08-24.
> The measurement rows carrying those older values are correct and stay as they are; it
> was this summary that had drifted. Two separate pieces of work have now been misled by
> a stale pointer, so: **when the live config changes, update this block in the same
> commit.**

> **Note on KV variance.** Reported KV cache has read 38.49 / 38.47 / 37.72 GiB across
> restarts with *identical* settings. That spread is normal run-to-run profiling
> variance, not a configuration difference or a leak. Do not go hunting for it.

---

## `tp2-matched-mtp4` — the matched 2-node vs 3-node comparison — 2026-08-21

**The first like-for-like test of node count.** Everything before this compared configs
that also differed in prompt, harness, or engine settings. This one does not: same
cluster, same afternoon, same harness (`bench-miaai`), same prompt shape
(`synthetic-numbered-words`), warm-up sweep discarded, median-of-3. Both sides passed
correctness (17×23 → 391).

2 nodes, `tp_size 2`, `pp_size 1`, `max_model_len 460800`, `max_num_seqs 16`,
`gpu_mem_util 0.85`. KV cache **20.12 GiB / 1,832,675 tokens / 3.98x max concurrency**,
against the 3-node's 37.36 GiB / 3,565,267 tokens / 7.74x — a **1.95x** KV advantage for
3 nodes.

### MTP was temporarily changed 5 → 4 for parity

The frozen `ours-2spark-tp2-baseline` ran `MTP_NUM_TOKENS=5`; the 3-node production
config runs `4`. Comparing them directly would have confounded node count with MTP depth,
so MTP was set to **4 on both** for this test. **Node count is the only variable.**

The originals are backed up on the nodes and must be restored to return the 2-node config
to its frozen baseline:

| node | backup |
|---|---|
| sparkmain | `config/head.env.bak-mtp5-20260821-155903` |
| spark1 | `config/worker.env.bak-mtp5-20260821-155903` |

### Result — read all three lines, not just the first

**1. 2-node WINS aggregate throughput.** At the shared `c=16` cap: **191.2 vs 161.0
tok/s, +19%.** Three trials read 190.8 / 192.6 / 191.2 — a **1% spread, so this is not
noise**. 2-node also leads at `c=8` (161.0 vs 143.6, **+12%**).

**2. 3-node WINS per-stream decode at long context**, consistently **+8–17% from 2K to
131K**:

| prompt tokens | 2-node (`tp2-matched-mtp4`) | 3-node | 3-node gain |
|---:|---:|---:|---:|
| 2,048 | 69.2 | 79.2 | +14% |
| 8,192 | 67.9 | 73.5 | +8% |
| 32,768 | 70.8 | 82.5 | +17% |
| 131,072 | 74.0 | 83.5 | +13% |

> The 3-node context sweep rows sit under `config_id` **`tp3-seqs8`** in
> `measurements.csv`, not `tp3-seqs16`. Same 3-node TP=3 shape; the two differ only in
> `max_num_seqs` (8 vs 16), which is documented below as not moving single-stream decode.
> `summary.csv` states this sourcing explicitly in the row's notes rather than relabelling
> the config.

**3. The 1.95x KV advantage did NOT convert at deep concurrency.** Four concurrent
**200,000-token** prompts were run on each config:

| | 2-node | 3-node |
|---|---:|---:|
| decode tok/s | 0.5 | 0.6 |
| aggregate tok/s | 0.6 | 0.6 |
| TTFT | 539,666 ms | 553,113 ms |
| wall | 14.5 min | 14.4 min |
| preemptions | **0** | **0** |

**Equally unusable on both.** Nearly double the KV cache bought essentially nothing. With
preemptions at 0 on both sides, the engine never had to evict a block on either config —
the bottleneck is **serialized long prefill**, not KV capacity.

> ### ⚠️ RE-RUN 2026-08-25 on healthy fabric — the two halves of this fare differently
>
> The rows above were taken on the degraded spark1 fabric ([issue #15](../../issues/15)).
> Re-run at matched depth with byte-identical prompts on both arms
> (`results/20260825-deep-concurrency/`, harness `deepconc.py`):
>
> | | 2-node | 3-node |
> |---|---:|---:|
> | TTFT | **293,987 ms** (was 539,666) | **396,804 ms** (was 553,113) |
> | wall | 7.8 min (was 14.5) | 10.2 min (was 14.4) |
> | improvement | **1.84x** | **1.39x** |
> | preemptions | **0** | **0** |
>
> **"KV capacity is not the binding constraint" SURVIVES.** Preemptions are still 0 on
> both arms with the pool never filling. That half was an accurate observation and stands.
>
> **"Equally unusable on both" does NOT survive.** They are no longer equal: 2-node now
> reaches first token **1.35x sooner**. The old parity was two different configurations
> being throttled to a similar floor by the same degraded link — and spark1 was in the
> 3-node arm, so the handicap fell disproportionately there. Removing it did **not**
> rescue the 3-node arm; it separated them, in 2-node's favour.
>
> **Both remain unusable at this depth.** ~5 and ~6.6 minutes to first token are not
> workable numbers, so the practical guidance is unchanged.
>
> **Depth is steeply non-linear here** — measured on the 3-node arm, TTFT went 275 s at
> 185.5K to 397 s at 200K: **+44% for +8% depth**. Any comparison at this depth must
> verify prompt tokens from the engine rather than trusting an estimator; ours are
> recorded per-run (200,045–200,048 on both arms).
>
> Not re-tested: whether the "serialized prefill" mechanism itself is unchanged. Prefill
> got materially faster, so the *magnitude* attributed to serialization was inflated by
> the fabric; the mechanism claim was never measured directly and should not be restated
> as settled without one.

### ⚠️ This CORRECTS an earlier claim

Earlier write-ups described the doubled KV cache as *"the third node's real structural
payoff."* **That framing was wrong and is retracted.** The doubling is real and is still
recorded as a capacity metric — but it has **never been the binding constraint**.
`vllm:num_preemptions_total` has read **0 in every test to date**, including concurrency
32 and including the 4×200K deep-concurrency test above, which was designed specifically
to make KV bite and did not. Capacity that is never reached is not a payoff. The 3-node
payoff that is actually measurable is the long-context per-stream decode win in point 2.

### Which win matters here

This deployment's workload is **single-user interactive coding**. That is
**per-stream-latency-bound**, not aggregate-throughput-bound — one caller waiting on one
stream, not sixteen streams sharing the cluster. So the **3-node long-context per-stream
win is the one that matters for this deployment**, and the 2-node aggregate win, while
larger in percentage terms, is throughput this workload does not consume.

**A multi-user workload inverts that conclusion.** With several concurrent callers, 2-node
is the better configuration: +19% aggregate at the cap, and it frees **spark-sep** to host
a second model. Anyone re-deciding this should decide it on their own workload shape, not
on these percentages in isolation.

---

## `tp3-seqs8` → `tp3-seqs16` — 2026-08-21

**Changed: exactly one setting.**

```
MAX_NUM_SEQS=8   →   MAX_NUM_SEQS=16
```

In `~/localai/dspark-vllm-gx10/config/tp3.env`, applied **identically on all three
ranks** (sparkmain, spark1, spark-sep). Backups saved on each node as
`config/tp3.env.bak-seqs8`.

> ⚠️ **This value must match on every rank.** A mismatch between nodes hangs startup
> forever with no error message — the ranks never finish negotiating and nothing is
> logged to explain why.

### Side effect — IMPORTANT, and not obvious from the setting name

`docker-compose.yml` **derives** the CUDA graph capture size:

```
--max-cudagraph-capture-size = MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)
```

So changing `MAX_NUM_SEQS` silently changed capture size too:

| | before | after |
|---|---:|---:|
| `--max-cudagraph-capture-size` | 8 × 5 = **40** | 16 × 5 = **80** |
| graph capture memory | 0.44 GiB | **1.11 GiB** |
| graph capture time | 5 s | **13 s** |

Nothing was edited to cause this. It follows from the derivation, and it is the reason
startup got slower and used more memory. Anyone changing `MAX_NUM_SEQS` or
`MTP_NUM_TOKENS` in future is changing capture size as well.

### Cost

**None measurable.** KV cache was essentially unchanged — marginally *more* tokens:

| | `tp3-seqs8` | `tp3-seqs16` |
|---|---:|---:|
| KV cache | 38.49 GiB | 38.47 GiB |
| KV cache tokens | 3,591,962 | **3,624,398** |
| max concurrency | 7.80x | 7.87x |

### Benefit

Aggregate throughput at the useful concurrency cap **146.9 → 161.0 tok/s, about +10%**.
Both figures are each config's peak at its own cap (c=8 for `seqs8`, c=16 for
`seqs16`), which makes this the like-for-like comparison.

`seqs16` does read 174.9 tok/s at c=32, but that is **oversubscribed — queueing, not
added capacity**: c=32 is double the sequence cap, and per-stream decode falls from 18.4
to 14.8 tok/s across that range while the aggregate creeps up. Requests are waiting, not
being served faster. **Do not quote 174.9 as the result, and do not describe this change
as +19%.**

**Single-stream decode is UNCHANGED: 70.7 → 71.6 tok/s, within noise.** This change buys
batch capacity, not per-stream speed. Do not quote it as a latency improvement.

### Rollback

On each of the three nodes:

```bash
cp config/tp3.env.bak-seqs8 config/tp3.env
```

Then restart **workers first, head last**:

```bash
sudo bash scripts/start-node.sh config/tp3.env
```

---

## ❌ NEGATIVE RESULT — `MAX_NUM_BATCHED_TOKENS` 8192 → 16384, TESTED AND REVERTED (2026-08-21)

**This is a trap. It looks like free performance and it is not.** All three nodes are
back to `8192` and verified serving correctly (17×23 → 391).

MiaAI-Lab recommend `16384` in their "long coding / big prompts" profile, and the engine
itself asks for it at startup with `seqs=16` + `MTP=4`:

```
WARNING [vllm.py:1648] max_num_scheduled_tokens is set to 8144 based on the
speculative decoding settings. This may lead to suboptimal performance.
Consider increasing max_num_batched_tokens...
```

The test was therefore well-motivated — **the warning is real** — but raising the value
did not pay off on this deployment.

### Measured cost

| | 8192 | 16384 |
|---|---:|---:|
| KV cache | 38.47 GiB | 35.45 GiB |
| KV cache tokens | 3,624,398 | **2,053,893** |
| Max concurrency @460800 | 7.87x | **4.46x** |
| c=16 aggregate tok/s | 161.0 | 162.0 |
| c=8 aggregate tok/s | 143.6 | **130.2** |
| c=1 decode tok/s | 71.6 | 87.3 |

### Verdict

It **cost 43% of KV cache tokens** (3.62M → 2.05M) and cut max concurrency by the same
43% (7.87x → 4.46x), while c=16 aggregate was statistically unchanged (161.0 vs 162.0)
and c=8 aggregate got *worse* (143.6 → 130.2).

Single-stream decode did read higher (87.3 vs 71.6). **That is measured noise, not a
gain.** An 8-rep noise-floor study on the *unchanged* 8192 production engine (warm-up
discarded, endpoint verified idle) gives:

```
88.3  67.5  77.0  88.5  66.6  83.7  88.1  75.1     median 80.4
```

Range **66.6 – 88.5 tok/s, a 33% spread with nothing changed at all.** The 16384 config's
87.3 sits comfortably inside that band — below the 88.5 maximum the 8192 config reached
on its own. It is therefore not evidence of a single-stream improvement, and it **does
not justify halving KV capacity** — the capacity the third node exists to provide.

This is also why the original 71.6 figure should not be read as 16384 "winning" by 22%:
both readings are single samples drawn from the same wide distribution.

### Why it costs KV

`max_num_batched_tokens` sizes the per-step activation and scheduling buffers. Those come
out of the **same memory pool as the KV cache** at a fixed
`gpu_memory_utilization=0.85`, so every byte given to batching buffers is taken from KV.

### Do not re-run this

The only scenario where it might pay is heavy **concurrent long prefill**, which is not
this deployment's workload. Anyone revisiting it must weigh it against losing half the KV
capacity, and must re-measure rather than trusting the upstream profile.

**Corollary:** the `max_num_scheduled_tokens=8144` startup warning is **known and
accepted**. Raising `max_num_batched_tokens` to silence it *is* the regression documented
above. Leave the warning alone.

---

## Settings NOT changed / verified

Recorded so they are not re-litigated. Each was considered and deliberately left alone.

### `VLLM_USE_BREAKABLE_CUDAGRAPH=0` — kept

Measured to gain nothing on this hardware.

### `MTP_NUM_TOKENS=4` — kept

4 is the measured knee. **3 is 29% slower.** Note this also feeds the capture-size
derivation above.

### `max_model_len=460800` — kept

It is a per-request **CEILING, not a reservation.** Lowering it *would* free KV cache —
but **KV is not our binding constraint**: `vllm:num_preemptions_total` stayed at **0**
through every test, including concurrency 32. The engine never once had to evict a
sequence to reclaim blocks. Lowering `max_model_len` would therefore buy nothing while
costing long-context capability.

The binding constraint is the **sequence cap** (`max_num_seqs`), which is exactly why the
change above was the one worth making.

### `kv-cache-dtype=nvfp4_ds_mla` — kept

MiaAI-Lab's issue #22 reports that `flashmla_sparse.py:880` routes `nvfp4_ds_mla` to a
slow BF16 kernel, collapsing long-context decode to ~1 tok/s. **That code path is present
verbatim in our image** (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`).

**It does not reproduce here.** Measured decode is flat **75–99 tok/s from 256 up to
409,600 tokens** — no collapse anywhere in our range. Their symptom appears at 600K+
context; our `max_model_len` ceiling is 460,800, so we never reach the regime where it
bites.

> ⚠️ **Do not apply that hotfix without re-measuring.** The bug being present in the
> source is not evidence that it affects us, and the long-context sweep in `measurements.csv`
> is the evidence that it does not.

---

## `tp3-seqs8` — baseline for the 3-node TP=3 work

3 nodes, `tp_size 3`, `pp_size 1`, `max_model_len 460800`, `max_num_seqs 8`,
`mtp_num_tokens 4`, `gpu_mem_util 0.85`, RoCE, `moe-backend flashinfer_b12x`,
`kv-cache-dtype nvfp4_ds_mla`, `block-size 256`, official DeepSeek-V4-Flash-0731
checkpoint, image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`.

KV cache 38.49 GiB / 3,591,962 tokens / 7.80x max concurrency.

## `ours-2spark-tp2-baseline` — frozen 2-node reference

2 nodes, `tp_size 2`, `max_model_len 460800`, `max_num_seqs 16`, `mtp_num_tokens 5`.
KV cache 19.52 GiB / 1,771,152 tokens / 3.84x max concurrency. See PR #2.

## `miaai-2spark-tp2` — EXTERNAL, not our hardware

MiaAI-Lab's published 2-node TP=2 figures (2026-08-14): `max_model_len 1048576`,
`max_num_seqs 6`, `mtp_num_tokens 5`. Recorded for comparison only; we did not measure
these and cannot vouch for the conditions beyond the harness they documented.
