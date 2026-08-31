# Engine A/B — the three remaining cells on the eugr serving config (2026-08-31)

**Status:** `CURRENT` · **Engine:** `eugr-spark-vllm-b12x` (digest `sha256:7dc02f16…`), `eugr.service`
on port 8100, nst=5 / mnbt=8192, persistent kernel caches (`/opt/eugrcache-*`, JIT miss count 8 at
service boot and frozen) · **Nodes:** 3 (TP=3), 1M context, kv fp8 · **Weights:** official
`deepseek-ai/DeepSeek-V4-Flash-0731` @ 7872f01b · **Config id:** `eugr-tp3-seqs16-dspark5-cached`
(same as the K-sweep winner in `20260830T2245Z-eugr-ksweep`) · **Fabric gate:** inherited from the
sweep bundle's pre-boot gate (30/30 incl. NCCL bandwidth) on the same service boot; no NCCL warnings
in the service log.

Completes the cell table in `docs/ENGINE-AB-3NODE.md`. Every cell below is a cold, unique-nonce
measurement.

| cell | harness | anemll reference (2026-08-21) | eugr | file |
|---|---|---:|---:|---|
| decode at 131,072-token context, c=1 | `bench-miaai` (byte-identical to arm 1, `--prompt 131072`, unique cold prefix per trial) | 83.5 (TTFT 138.1 s) | **90.5** median of 85.9 / 90.5 / 94.4 (TTFT **53.7 s**) | bench-131k.log |
| prompt-effect: code-brief, c=1 | `eugr-remaining-cells.py` at `ours-bench.py` conditions (temperature 0, max_tokens 256, streamed, decode excludes TTFT), 5 reps | 81.8 | **91.0** (89.5–91.8) | remaining-cells.log / .json |
| prompt-effect: dense-prose, c=1 | same | 49.4 | 49.2 (48.0–55.2); ratio **1.85×** (anemll 1.65×) | same |
| deep concurrency 4 × ~200K, 128 out | `eugr-remaining-cells.py` replicating `deepconc.py` (unique 200,015-token nonces) | 0.9 tok/s, ~870 s to first token, "unusable" | 1.26 tok/s, **TTFT 224.0 s**, wall 360.8 s, 0 errors — still unusable as a workload, 3.9× faster to first token | same |

## Prompt provenance — the dense-prose prompt was NOT lost

An earlier version of the driver reconstructed the dense-prose prompt because
`benchmarks/README.md` shows it with an ellipsis and `ours-bench.py` was never committed. The full
text survives in git history (commit `b078eb4`, the table row that produced the 49.4 number):
*"Write a detailed technical explanation of how pipeline parallelism differs from tensor
parallelism in large language model inference."* The driver in this bundle uses that exact text
(23 prompt tokens) and the exact code-brief text (18 tokens), so both prompt-effect cells are
byte-comparable to the anemll rows. Sampling for the pair follows the original `ours-bench.py`
conditions, not bench-miaai's (`temperature 0`, natural `max_tokens 256`, no forced length).

## Files
`bench-131k.log` (bench-miaai trials), `remaining-cells.log` / `remaining-cells.json` (driver output),
`eugr-remaining-cells.py` (the driver as run; also vendored under `scripts/eugr-ab/`),
`engine-config.txt` (service boot time, pinned env, KV size, virtual-TP line, JIT miss count).
