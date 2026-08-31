# Engine A/B — the remaining cells, MATCHED harness and exact prompts (2026-08-31 05:25Z)

**Status:** `CURRENT`. Supersedes two cells of
[`20260831T0030Z-eugr-remaining-cells`](../20260831T0030Z-eugr-remaining-cells/) (131K decode,
dense-prose); confirms its deep-concurrency cell. Same engine, same service boot
(`eugr.service`, nst=5 / mnbt=8192, persistent caches, port 8100; KV 2,364,598 tokens), same
weights (official 0731 @ 7872f01b), fabric gate inherited from that boot (no NCCL warnings in the
service log). Runs at 05:25–05:43Z, five hours after the superseded run — no overlap.

| cell | harness here | anemll reference | eugr (this bundle) | superseded value |
|---|---|---:|---:|---:|
| decode at 131,072 ctx, c=1 | `bench-miaai --prompt 131072` (byte-identical to the harness behind the reference) | 83.5, TTFT 138.1 s | **90.5** (85.9 / 90.5 / 94.4), TTFT **53.7 s** | 42.3 — driver filler prompt |
| prompt-effect: code-brief | `eugr-remaining-cells-v2.py` at `ours-bench.py` conditions | 81.8 | **91.0** (89.5–91.8) | 89.4 (bench-miaai sampling) |
| prompt-effect: dense-prose | same, **exact original prompt** (git `b078eb4`) | 49.4 | **49.2** (48.0–55.2); ratio **1.85×** | 45.9 (reconstruction; 1.95×) |
| deep 4×~200K, 128 out | `eugr-remaining-cells-v2.py` (deepconc.py replica) | 0.9, ~870 s | 1.26, TTFT 224.0 s, wall 360.8 s | 1.17 / 226.6 s — consistent |

## Why the earlier 131K number was not a regression

The superseded driver built its 131K prompt as `"benchmark context datum "` × ~44,000. DSpark
acceptance depends on the prompt — the prompt-effect cell exists precisely because of that — so a
repetitive filler is a *different measurement* from `bench-miaai`'s numbered-words prompt, which
is what produced the anemll 83.5. Re-run on the same harness: +8 %, with a 2.6× faster 131K
prefill. The one remaining config delta vs the 2026-08-21 anemll row is `max_model_len`
(1,048,576 here vs 460,800 there).

## Why the dense-prose cell is now byte-comparable

The prompt was believed lost (`ours-bench.py` never committed; README shows it elided). It is in
git history: *"Write a detailed technical explanation of how pipeline parallelism differs from
tensor parallelism in large language model inference."* (23 tokens). The driver in this bundle
uses that text and the original `ours-bench.py` conditions (temperature 0, max_tokens 256,
streamed, decode excludes TTFT, no forced length) for both prompts of the pair.

## Files
`bench-131k.log` (bench-miaai trials) · `remaining-cells.log` / `.json` (driver output for the
prompt pair and deep cell) · `eugr-remaining-cells-v2.py` (the driver as run; vendored under
`scripts/eugr-ab/`) · `engine-config.txt` (service boot time, pinned env, KV size, virtual-TP
line, JIT miss count 8).
