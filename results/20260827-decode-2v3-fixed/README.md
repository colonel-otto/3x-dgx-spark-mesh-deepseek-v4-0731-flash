# Corrected 256-token decode: two nodes versus three — 2026-08-27

**Status:** `CURRENT` · **Harness:** corrected `decode_depth_sweep.py` · **Output:**
256 tokens asserted on all 70 reps · **Cache:** zero cached tokens on all reps

This bundle imports the raw data behind the fixed-harness node-count result posted to
issue #26. It supersedes the node-count interpretation of
[`20260826-decode-depth-2v3`](../20260826-decode-depth-2v3/), whose requests stopped after
25–26 tokens.

## Result

| target depth | TP=2 decode | TP=3 decode | TP=3 advantage | TP=2 TTFT | TP=3 TTFT |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 46.53 | **54.30** | +16.7% | 1.08 s | **0.94 s** |
| 8,192 | 46.29 | **52.87** | +14.2% | 3.90 s | **3.46 s** |
| 32,768 | 46.81 | **51.98** | +11.0% | 16.20 s | **15.80 s** |
| 131,072 | 44.40 | **47.65** | +7.3% | **70.43 s** | 92.73 s |
| 262,144 | 44.40 | **48.83** | +10.0% | **161.89 s** | 176.00 s |

Medians are from seven measured reps per arm and depth, following two warmups. Full
per-rep distributions are committed in `tp2/*.jsonl` and `tp3/tp3-fixed.jsonl`.

## Caveats

- TP=2 ran with `MAX_NUM_SEQS=16`; the restored production TP=3 engine used 32. At cc=1
  this scheduler ceiling should not bind, but this is not a configuration-identical arm.
- TP=2 carries an engine-stopped passing pairwise fabric gate (15 pass, 0 fail, 2 expected
  engine skips; 9.21 GB/s). TP=3 was checked after restoration with the engine live
  (24 pass, 0 fail, NCCL bandwidth skipped), so its absolute timing lacks its own stopped
  gate artifact.
- Seven matched reps answer the decode-direction question but do not satisfy issue #24's
  stricter 15-rep 131K TTFT criterion. That requirement remains open and is folded into
  the tuning experiment in issue #25.

## Files

- `tp2/`: live configuration, gate artifacts, orchestration log, per-depth JSONL/logs,
  and summaries from the successful TP=2 run.
- `tp3/`: live configuration, combined 35-rep JSONL/log, and summary from the restored
  TP=3 engine.
