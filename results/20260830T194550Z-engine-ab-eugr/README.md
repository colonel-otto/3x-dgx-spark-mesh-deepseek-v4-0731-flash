# Engine A/B, arm 1: eugr/spark-vllm-b12x on 3 nodes (TP=3) — 2026-08-30

**Status:** `CURRENT` · **Engine:** `eugr-spark-vllm-b12x` digest `sha256:7dc02f16…` (image-digest.txt),
vLLM `0.1.dev20133+gb5f995e73.d20260823` (main), FlashInfer 0.6.18, torch 2.13 · **Nodes:** 3 (TP=3) ·
**Checkpoint:** official `deepseek-ai/DeepSeek-V4-Flash-0731` @ 7872f01b (same weights as every anemll row) ·
**Harness:** the 2-node repo's suite, unmodified — checksums in `vendored-SHA256SUMS.txt` matched byte-for-byte
before the run · **Launch:** `recipe.yaml` via eugr's `run-recipe.py` (runbook in docs/ENGINE-AB-3NODE.md) ·
**Fabric gate:** NCCL formed over the 200Gb mesh with the proven `tp3.env` NCCL values; no `GID table changed`.

Protocol: docs/ENGINE-AB-3NODE.md. This is the first new-engine arm; the anemll arm is every row before 2026-08-30.

## Correctness gate — PASSED

| check | anemll baseline (2026-08-27) | eugr arm | file |
|---|---|---|---|
| quick gate (endpoint, virtual-TP active, capital, 17×23, red/blue, needle, degeneration) | — | 7/7 | (stdout) |
| tool battery | 7/7 | **6/7** — `forced_choice` emitted two *valid* `get_weather` JSON calls with finish=stop; tool-choice API semantics, not garble | tool-battery.log |
| deep-context tools 32K + 131K | 8/8 | **8/8** | deepctx-tool-battery.log |
| garble sweep 2K/8K/32K/131K × 2 | ALL CLEAN | **ALL CLEAN** | context-garble-sweep.log, garble.json |
| RULER-lite | 12/12 (4×3) | **16/16** (4 tasks × 4 depths incl. 262K) | ruler-lite.log, ruler-lite.json |

Settles the open question from the plan review: the image's native virtual-TP (heads 64→72, o_groups 8→9,
zero-filled pad slabs, zero-filled pad-head attn_sink) serves correct output at TP=3. Our padding patch is
not applied and must not be.

## Throughput — bench-miaai, synthetic-numbered-words, 256-token prompt

Config deltas vs the anemll arm, stated per protocol: speculative **dspark nst=5 probabilistic** (anemll: MTP K=2);
**kv fp8** (anemll: nvfp4_ds_mla); V2 model runner; AOT compile; `--no-cache-dirs` (see caveat).

| c | anemll tp3-seqs16 decode / agg | eugr decode / agg | Δ agg |
|---|---:|---:|---:|
| 1 | 80.4 / — | **82.1** / — (warm, median-of-8, 57.8–90.4) | parity |
| 4 | 42.8 / 115.2 | **54.4 / 162.7** | **+41%** |
| 8 | 28.2 / 143.6 | **33.3 / 171.7** | **+20%** |
| 16 | 18.4 / 161.0 | 16.0 / 133.9 (4 of 5 trials; best 161.6) | **−17%** |

KV capacity: 2,415,674 tokens at 1M (anemll: 3,588,422) — the fp8-vs-nvfp4 KV dtype delta plus CUDA-graph
memory profiling, not an engine property.

### Two effects to separate before reading the table

1. **JIT kernel compiles contaminate cold runs.** 20 `[b12x cute.compile] … disk-cache-miss` events fired
   after startup (launch log); each stalls the batch it lands in. The first c=1 run read 65.4; the warm rerun
   read 82.1. Cause: `--no-cache-dirs` (needed because the launcher expands the head's `$HOME` on workers)
   leaves no persisted kernel cache. Fix for next boot: mount uniform `/tmp/eugrcache-*` dirs onto the
   container cache paths.
2. **c=16 is a scheduling cliff, not a prefill-rate limit.** TTFT: c=1 0.33s → c=4 0.75s → c=8 1.9–2.1s →
   c=16 **7.0s** (3.5× time for 2× tokens). Compile misses were frozen at 20 during these runs, so it is
   steady-state. The engine warns at startup: `max_num_scheduled_tokens is set to 8128 based on the
   speculative decoding settings … decrease num_speculative_tokens or max_num_seqs`. nst=5 × 16 seqs of
   draft slots is the prime suspect. Boot-time levers to test: `num_speculative_tokens: 2` (also removes the
   speculator delta vs our MTP K=2) and/or a larger batched-token budget (which was a KV-cost trap on the
   anemll engine — re-measure, do not assume).

Bottom line: correctness parity or better; single-stream parity; **+20–41% aggregate at c=4–8**; a fixable
cliff at the c=16 cap. Not yet a matched same-day A/B against a live anemll engine (that engine was down
today) — the anemll numbers above are the 2026-08-21 rows from measurements.csv.

## Files
`engine-config.txt` (engine init line, virtual-TP plan, KV size, load time), `launch-header.txt`,
`bench-c1.log` (cold), `bench-c16.log` (cold), `bench-c16-warm.log`, `battery-run*.log` (the first two
battery runs show the harness URL mistakes — garble/RULER expect a `/v1` base — kept for honesty).
The warm c=1 rerun and the c=4/c=8 probes were captured from terminal output into measurements.csv notes.
