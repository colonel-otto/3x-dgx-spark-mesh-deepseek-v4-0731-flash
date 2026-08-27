# Benchmark policy — what a number must carry before it is published

Every invalidated result in this repository failed one of the rules below, and
each rule exists because we published something wrong. This page is the standard
for any new benchmark, ours or a contributor's.

## The four hard requirements

### 1. A fabric gate must run, and its artifact must be committed

**Rule:** `scripts/fabric_gate.sh` runs before every measurement arm, with the
engine in the state it will be measured in, and `fabric-gate.json` is committed
next to the results. `scripts/run_experiment.sh` does this automatically and
**refuses to benchmark on gate failure**. `FABRIC_GATE=0` exists but using it
makes the run unpublishable.

**Why:** between roughly 2026-08-21 and 2026-08-25 one node ran at **~15% of
its collective bandwidth** with every error counter reading zero, every
container `running`, and every ping clean. It silently corrupted seven result
directories. A reboot moved prefill **+103%**. Nothing in the engine, the OS, or
the container runtime reported a fault — only an NCCL collective test found it.

**Consequence:** a result without a committed gate artifact is marked
`fabric_gate: ABSENT` in [`../results/INDEX.md`](../results/INDEX.md) and is not
citable as evidence, regardless of how clean it looks.

### 2. The decode window must be long enough, and verified

**Rule:** force the output length (`min_tokens == max_tokens` plus
`ignore_eos`), use **at least 256 tokens**, and **assert
`completion_tokens == max_tokens` per rep**. The run must fail, not warn, when
the window collapses.

**Why:** with MTP=5 speculative decoding, a short window measures draft
acceptance variance rather than throughput.

| run | output tokens | ~MTP steps | effect |
|---|---:|---:|---|
| `20260821T001024Z-2spark-baseline` | **10** | ~2 | the frozen 2-node reference |
| `20260826-decode-depth-2v3` | **25–26** | ~5 | overstated decode **30–39%** |
| corrected (`20260826-harness-window-calibration`) | 256 | ~51 | reproducible to ~1% |

`decode_depth_sweep.py` requested 256 but its prompt ended *"in one sentence"*,
so the model stopped early on **all 70 reps** and nothing checked. Measured
consequence: identical reps at 131K spanned **37.0–64.3 tok/s**.

### 3. Publish the spread, never the median alone

**Rule:** every result publishes sorted per-rep values, not just a median, and
commits the raw per-rep file.

**Why:** both defects above were caught by looking at spreads. A median alone
hides them. Suspiciously repeated values (`85.5, 85.6, 85.6`) are the signature
of a window quantized by draft-acceptance granularity rather than a measured
rate.

### 4. Config comes from the live process

**Rule:** capture engine config with `ps -eo args` on the running engine and the
KV pool size from that boot's log — **not** from a config file, which may not be
what is running.

**Why:** config files drift from reality, and a matched comparison is only
matched if both arms were verified while running.

## Cold vs warm

Cold and warm are different measurements and must never be mixed.

- **Warm** (the default): warm at least 2 requests **per prompt shape** before
  measuring. TileLang/CuTeDSL compile kernels *during* inference on first sight
  of a shape — one compile measured at 5 s landing inside a request. Set
  `JIT_MONITOR_MODE=warn` to surface them.
- **Cold** is legitimate to publish, but only labelled as cold and reported
  separately. A first request after restart carries a 6–10 s autotune TTFT that
  **is not a regression**.

A comparison is invalid if one side is warm and the other cold.

## Why we keep results we know are wrong

**Invalid results are retained deliberately, as diagnostic baselines.** They are
not evidence of performance; they are evidence of what a specific failure looks
like from the outside, so somebody reproducing this work can recognise it.

If your numbers resemble a `VOID-*` row in
[`../results/INDEX.md`](../results/INDEX.md), you are probably hitting the fault
that row records. Each carries a `baseline_value` saying exactly that. Examples:

- **~19–20 tok/s at TP=3** → likely a degraded fabric link. Run
  `scripts/fabric_gate.sh`.
- **Decode 30–40% higher than expected, clustering at repeated values** → check
  that `completion_tokens` equals the `max_tokens` you asked for.
- **A cluster that starts, reports `running`, and never serves** → check for
  `IBV_WC_RETRY_EXC_ERR` and `fe80::` GIDs; an unaddressed HCA wedges silently.

Deleting these would leave the next person to rediscover each failure from
scratch. **Nothing is deleted; it is relabelled.**

## Comparing against other repositories

Numbers from another repository are **not** comparable to ours unless the
harness, output-window length, prompt, warm-up state, and `max_num_seqs` all
match. We got this wrong once: we compared their 512-token-window result against
our 25-token-window result and read a methodology difference as a hardware
deficit.

**Cross-repo material is valuable for config and methodology. It is not
valuable for numbers.** To compare, re-run *their* harness on *our* cluster and
publish both arms.

When citing an external number, state its rep count. If it is `n=1`, say so
rather than calling it a median.
