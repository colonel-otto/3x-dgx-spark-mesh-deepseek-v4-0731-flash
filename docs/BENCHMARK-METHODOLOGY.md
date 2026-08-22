# Benchmark methodology: why a single tok/s number is not a measurement

Measured 2026-08-21 on the live 3-node TP=3 deployment. No configuration was changed to
produce these numbers — same engine and uptime as the tuning sweep in
[`TP3-TUNING.md`](TP3-TUNING.md).

---

## Summary

Running the **upstream harness** from
`localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark` against our deployment gives
**79.0–79.3 tok/s single-stream**, matching their published 75–79 band.

Our own harness reported ~49–57 tok/s for the same engine. Both are correct. The
difference is **the benchmark prompt**, because MTP speculative decoding accepts drafts
at a rate that depends on the content being generated.

**Therefore: every tok/s figure in this repository must be read together with the prompt
that produced it.** Cross-repo comparisons are invalid unless the prompt matches.

---

## The upstream-harness measurement

`scripts/benchmark_tp3.py` (theirs), run on the head node, endpoint verified idle
(`num_requests_running 0`), one warm-up sweep discarded, then repeated until two sweeps
agreed.

Config under test: `TP=3`, `MTP_NUM_TOKENS=4`, `MAX_NUM_SEQS=8`,
`MAX_MODEL_LEN=460800`, RoCE, `flashinfer_b12x`.

| concurrency | sweep 1 median | sweep 2 median | agreement |
|---:|---:|---:|---|
| 1 | **79.3** | **79.0** | 0.4% |
| 2 | 110.0 | 115.1 | 5% |
| 4 | 175.5 | 169.6 | 3% |
| 8 | 266.4 | 242.2 | 10% |

Within-sweep spread at `cc=1` was 2% and 1%.

`cc≥2` is **not** comparable to their published table: we run `MAX_NUM_SEQS=8` against
their 16/32, and their own result is that aggregate peak is bounded by `max_num_seqs`.
Only `cc=1` is a like-for-like comparison.

Raw data: [`../results/20260821T142000Z-3spark-tp3-upstream-harness/`](../results/20260821T142000Z-3spark-tp3-upstream-harness/).

---

## Root cause: MTP acceptance is content-dependent

The only substantive difference between the two harnesses is the prompt string. Measured
back-to-back with the *same* script on the *same* engine, 4 repetitions each:

| prompt | decode tok/s (median) | samples |
|---|---:|---|
| `Write a Python function that merges two sorted lists. Explain briefly.` | **81.8** | 81.7 / 82.1 / 81.9 / 81.7 |
| `Write a detailed technical explanation of how pipeline parallelism differs from tensor parallelism in large language model inference.` | **49.4** | 49.8 / 51.6 / 45.6 / 49.1 |

**1.65x from the prompt alone.**

The engine's `SpecDecoding metrics` log lines confirm the mechanism directly:

| prompt | mean acceptance length | avg draft acceptance |
|---|---:|---:|
| code, "explain briefly" | **4.44 – 4.67** | **86 – 92%** |
| dense technical prose | **2.89 – 3.25** | **47 – 56%** |

Acceptance-length ratio ≈ 1.50; throughput ratio ≈ 1.66. Throughput tracks acceptance
length, which is exactly how speculative decoding is supposed to behave — the MTP
drafter predicts routine code and short explanations well and novel dense prose poorly.

**The defensible statement about this deployment is a range: ~49 tok/s on hard prose to
~82 tok/s on routine code**, both real, measured minutes apart on one engine.

---

## Two plausible explanations that are FALSE

Recorded so they are not re-proposed.

### ❌ "The upstream metric includes prefill/TTFT and ours does not"

Their `aggregate_tok_s = total_out / wall` does include prefill; our decode-only metric
excludes it. This is a real definitional difference and the two should not be divided by
one another — but it is **not** the cause, because their prompt is 18 tokens:

```
TTFT                                  0.105 – 0.115 s
wall                                  3.25  – 3.31  s     (TTFT ≈ 3% of wall)
aggregate  (total_out / wall)         77.2  – 78.7 tok/s
decode-only (TTFT excluded)           79.8  – 80.2 tok/s
```

The definitions differ by ~3%, not by 40%.

### ❌ "Insufficient warm-up — everything was measured cold"

Upstream warns that one warm-up run is not enough after a change that alters graph
capture, citing a `cc=16` reading of 248 against a steady state of 454.

**That warning does not apply at our settings.** The *discarded warm-up sweep* — the
coldest sample available — already read `cc=1` at **78.6**, statistically identical to
the two warm sweeps (79.3, 79.0). Capture size is
`max_num_seqs × (MTP+1) = 8 × 5 = 40`, versus up to 192 upstream, so capture warms
almost immediately.

The warning becomes relevant again if `MAX_NUM_SEQS` is raised substantially.

---

## Implications for the results already in this repo

- The `48.23` TP=2 baseline and the `53.95 – 57.73` TP=3 figures were **all** measured
  with the same prose-shaped prompt. They are internally consistent, so **the relative
  deltas between configs remain valid** — the TP=3 improvement over TP=2 stands.
- Only **absolute cross-repo comparisons** were unsound. Upstream's 75–79 was never a
  target we were missing.
- Any comparison against MiaAI-Lab's 2-Spark figures needs their prompt before it means
  anything.

---

## Rules for future measurements

1. **Publish the prompt with every tok/s figure.** A number without its workload is not
   a measurement.
2. Keep **both** prompt shapes in the suite and report the range — code-shaped (best
   case) and prose-shaped (worst case).
3. Run on the **head node**; verify `num_requests_running 0` first.
4. Discard one sweep; repeat until two agree. Mandatory if `MAX_NUM_SEQS` is raised.
5. Compare configurations only against the **same prompt**.
6. **State the denominator on every prefill figure.** Two different rates exist and they
   differ by ~30x on this cluster — see below.

---

## A prefill number without its denominator is not a measurement

The same rule that governs tok/s and prompt shape applies to prefill, for a different
reason. Two rates are both correct and answer different questions:

| Rate | Formula | Observed (2026-08-22) | Answers |
|---|---|---:|---|
| **End-to-end** | `prompt_tokens_total ÷ wall-clock` | ~855–960 tok/s | How long will my job take? |
| **True engine** | `request_prefill_kv_computed_tokens_sum ÷ request_prefill_time_seconds_sum` | **9,000–33,440 tok/s** | What can the hardware do? |

The gap is idle time between scheduler passes, not compute. It is directly visible in
`docker logs` as `Running: 1` with `Avg prompt throughput: 0.0` while GPU utilisation
reads 96%: at `MAX_NUM_BATCHED_TOKENS=8192` a 108K-token prompt takes ~13 sequential
passes, and the batch is not filled between them.

**Why this matters.** Quoting the end-to-end figure as engine capability understates the
hardware by ~30x and makes prefill look like a hard wall rather than a scheduling
problem. Quoting the true rate as job throughput understates completion time by the same
factor. Both errors are easy to make from the same `/metrics` scrape.

The engine's own log line (`Avg prompt throughput`) reports the **true** rate, sampled
only while a batch is in flight — it is not comparable to a rate you compute by dividing
a counter delta by a wall-clock window.

**Rule:** report both, label which is which, and never compare one to the other.
