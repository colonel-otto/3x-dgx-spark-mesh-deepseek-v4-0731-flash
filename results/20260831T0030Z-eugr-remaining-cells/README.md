# eugr engine — the remaining A/B cells (2026-08-31)

**Status: CURRENT** — fabric gated clean before *and* after the run
(24/0 each, `fabric-gate.json` / `fabric-gate-post.json`), engine on the sweep
winner `nst=5 / mnbt=8192`, JIT cache warm and frozen.

Closes the three cells left open by
[20260830T2245Z-eugr-ksweep](../20260830T2245Z-eugr-ksweep/) and by the
"The cells" table in [ENGINE-AB-3NODE.md](../../docs/ENGINE-AB-3NODE.md).

Harness: `eugr-remaining-cells.py`, committed here. Sampling matches
bench-miaai (streamed, temp 0.6 / top_p 0.95, min=max tokens, `ignore_eos`,
thinking off, unique nonce per long prompt to defeat the prefix cache).

## Results

| cell | anemll-v0.25.1 | eugr (this run) | read |
|---|---:|---:|---|
| prompt-effect: code-brief, c=1 | 81.8 | **89.4** | +9% |
| prompt-effect: dense-prose, c=1 | 49.4 | 45.9 | see caveat |
| **prompt effect ratio** | **1.65×** | **1.95×** | effect is *larger* here |
| decode @131,072 ctx, c=1 (cold) | 83.5 | **42.3** | −49%, but not matched — see below |
| deep concurrency 4×~200K | 0.9 (unusable) | 1.4 | still unusable |

## The prompt effect is real and bigger on this engine

Code-brief decodes **1.95× faster than dense prose** on identical hardware,
identical engine, minutes apart (89.4 vs 45.9 tok/s, 5 reps each, tight spreads:
88.0–90.7 and 44.5–48.2). anemll recorded 1.65×.

This is the DSpark/MTP acceptance effect: a short code prompt yields far more
accepted draft tokens per step than continuous prose. It means **any decode
number is meaningless without its prompt shape**, and the gap is wider on the
new engine, not narrower.

### ⚠ Caveat — the dense-prose prompt is a reconstruction

`ours-bench.py` was never committed, and no document in this repo records the
dense-prose prompt text — only that it was ~51 tokens and produced 49.4 tok/s.
The prompt used here is a **reconstruction** to that recorded shape (~49 tokens
of continuous prose, no code, no lists); it is stored verbatim in
`prompt-effect.json` under `_meta` and in the harness source.

So: the **cross-engine dense-prose number (49.4 vs 45.9) is NOT a matched
comparison and must not be quoted as one.** What is sound is the *within-engine*
ratio — both prompts measured here, on one engine, minutes apart, both recorded.
That ratio is what the cell exists to test.

The code-brief prompt *is* exact: recovered verbatim from the committed
`results/20260825-fabric-fix/harness/bench_tp3.py` `--prompt` default, and it
tokenizes to 18 tokens, matching the anemll CSV rows.

## 131K decode: slower, but the configs differ

Median **42.3 tok/s** (reps 41.2 / 42.3 / 59.9) against anemll's 83.5.

**Two things make this not a clean regression claim:**

1. **Context config differs.** The anemll row ran `max_model_len=460800`; this
   engine serves the full **1,048,576**. A larger declared context changes KV
   block accounting and the attention path. Node count, TP size and prompt shape
   match; `max_model_len` does not.
2. **Prefill is much *faster*.** TTFT 53,721 ms here vs **138,076 ms** recorded
   on the anemll row — a **2.6× faster** 131K prefill. The engine is not slower
   at long context in general; it is slower at *decode after* a long prefill.

### A confound found and removed

The first attempt reused one 131K prompt across reps and produced a median of
48.9 tok/s — but TTFT collapsed from 58,742 ms on rep 0 to 1,262 ms on rep 1.
**The prefix cache was serving reps 2–3**, so those reps measured a cache hit,
not a 131K prefill. Since the anemll reference is a cold number, a warmed median
would have been an invalid comparison.

The harness now builds a **unique 131K prompt per rep**; every rep in
`ctx131k.log` shows a cold TTFT (53.6 / 53.7 / 56.4 s). The 48.9 figure is
discarded and appears nowhere in the rows.

## Deep concurrency 4×200K: still unusable, now with numbers

**1.4 tok/s aggregate**, median per-stream decode 1.17 tok/s, median TTFT
**226.6 s**, wall 364.8 s, **0 errors**, 800K prompt tokens against a 2.36M-token
KV pool.

It completes rather than failing — no preemptions, no errors — but at 227 s to
first token it is not a usable interactive shape. anemll measured 0.9 tok/s and
the same verdict holds: **this is a workload-shape limit, not an engine defect.**
The eugr engine is ~1.5× better here, which changes nothing practical.

## Files

- `prompt-effect.json` / `ctx131k.json` / `deep4x200k.json` — results, each with
  a `_meta` block recording the exact prompts used.
- `ctx131k.log`, `deep4x200k.log` — per-rep raw output.
- `fabric-gate.json`, `fabric-gate-post.json` — gates before and after
  (24 passed / 0 failed each; NCCL bandwidth skipped because the engine was up
  by design — the full 30/30 bandwidth gate for this engine config is in the
  ksweep bundle, taken while the engine was down).
- `eugr-remaining-cells.py` — the harness, with both prompts inline.

Gate artifacts have DGX hostnames replaced with generic `node0/1/2` labels
(public repo; `scripts/check_no_sensitive.py` enforces this). Measured values
untouched.
