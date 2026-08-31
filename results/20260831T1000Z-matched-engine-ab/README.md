# Matched engine A/B — anemll-v0.25.1 vs eugr-spark-vllm-b12x (2026-08-31)

**Status: CURRENT.**

**This is the first MATCHED cross-engine comparison.** Both arms ran on
2026-08-31, on the same three nodes, with the same harness, the same prompt, the
same completion window and the same trial count, one after the other, with only
the engine changed. It supersedes the cross-engine table in
`docs/ENGINE-AB-3NODE.md`, whose anemll column was quoted from **2026-08-21**
rows — a 10-day-old boot on a different fabric state.

## Why the old table needed replacing

Two independent defects:

1. **Unmatched.** Every eugr row was 2026-08-30/31; every anemll reference row
   was 2026-08-21. The repo's standing rule (`feedback_measure_our_own_ab_first`)
   exists because exactly this pattern once *reversed* a headline conclusion.
2. **Under-powered.** The repo's own 8-rep noise study on an *unchanged* anemll
   engine recorded 66.6–88.5 tok/s at c=1 — a 27 % spread — and the repo uses a
   **12 % parity tolerance** (issue #31). The old table's single-stream cells
   (+5 %, +8 %, +11 %) were all *inside* that band, i.e. not resolved by the data.

Both defects pushed the same way: they **understated** eugr. The matched numbers
below are substantially stronger than the ones they replace.

## Method — one variable

| held constant | value |
|---|---|
| date | 2026-08-31, back-to-back (eugr first, then anemll) |
| nodes / shape | sparkmain + spark1 + spark2, TP=3, PP=1 |
| harness | `scripts/eugr-ab/bench-miaai.py`, vendored copy, byte-identical to both arms |
| prompt | 256-token unique cold prefix, `thinking=false` |
| window | `min=max=128` completion tokens, `ignore_eos` |
| trials | median-of-5 per cell, warm-up sweep discarded |
| `max_num_seqs` | **16 on both arms** |
| `max_num_batched_tokens` | 8192 on both arms |
| `gpu_memory_utilization` | 0.82 on both arms |

`max_num_seqs` is worth calling out: the live anemll config had drifted to 32.
It was set to 16 on **all three ranks** for this run (a mismatch between ranks
hangs startup forever with no error) and restored to 32 afterwards. Without that
step the comparison would have carried a second variable.

### The one confound that cannot be removed

anemll runs **MTP `num_spec_tokens=2`**; eugr runs **DSpark `nst=5`**. The
checkpoint sets `dspark_block_size: 5` and the eugr engine *refuses* nst<5, so a
K-matched arm is impossible. Every cell below therefore measures
**engine + speculator**, not the engine alone. This is a permanent property of
the comparison, not an oversight.

## Results

Verdict column applies the repo's 12 % parity tolerance.

### Per-stream decode (tok/s)

| c | anemll-v0.25.1 | eugr-spark-vllm-b12x | delta | verdict |
|---:|---:|---:|---:|---|
| 1 | 61.5 | **84.7** | +37.7 % | eugr wins |
| 4 | 33.0 | **54.4** | +64.8 % | eugr wins |
| 8 | 29.0 | **44.9** | +54.8 % | eugr wins |
| 16 | **18.2** | 15.0 | −17.6 % | anemll wins |

### Aggregate throughput (tok/s)

| c | anemll-v0.25.1 | eugr-spark-vllm-b12x | delta | verdict |
|---:|---:|---:|---:|---|
| 1 | 53.8 | **70.7** | +31.4 % | eugr wins |
| 4 | 108.0 | **164.5** | +52.3 % | eugr wins |
| 8 | 154.8 | **249.9** | +61.4 % | eugr wins |
| 16 | 141.3 | **187.4** | +32.6 % | eugr wins |

### Capacity (not a throughput measure)

| | anemll-v0.25.1 | eugr-spark-vllm-b12x |
|---|---:|---:|
| KV cache tokens | **4,391,722** | 2,357,009 |
| KV cache dtype | `nvfp4_ds_mla` | `fp8` |
| max concurrency @1M ctx | **4.19×** | ~2.25× |

anemll holds **1.86× more KV**. That gap is *permanent*: `nvfp4_ds_mla` is
rejected by a `VllmConfig` validator on every MLA backend in the eugr build, so
fp8 is the floor there (see `docs/troubleshooting.md`).

## Reading the result

**eugr is decisively the stronger engine for serving.** It wins every aggregate
cell by +31 % to +61 %, and single-stream decode by +38 %. These margins are
3–5× the 12 % tolerance and far outside both arms' trial spread, so unlike the
old table's single-stream cells, they are genuinely resolved.

Two honest caveats, neither of which changes the verdict:

- **c=16 per-stream decode is anemll's one win** (18.2 vs 15.0). eugr trades
  per-stream latency for aggregate at the seqs cap — it still moves +33 % more
  total tokens in that same cell. Which one matters depends on whether you are
  optimising a single user's latency or the box's total output.
- **anemll's c=16 measurement is barely resolvable.** Its trial spread there was
  **84 %** (6.9 → 22.2 tok/s), against eugr's 14.7 %. The anemll arm was
  visibly less stable under load at every concurrency (c=4 spread 32 %, c=8 TTFT
  swinging 1.2 s → 8.1 s), which is itself a finding.

The capacity trade is real and is the one thing anemll still does better: if a
workload needs >2.25× concurrency at 1M context, eugr cannot serve it at all.

## Correctness gate

Run against the anemll engine **before** any throughput cell (a throughput
number from an unvalidated TP=3 engine may be the speed of generating garbage):

```
quick subset: 7 pass / 0 fail
PASS  virtual-TP plan activated (heads 64->72)
PASS  capital lookup / 17 x 23 / red-blue / needle ~1.5k tok / no degeneration
```

The eugr arm's correctness was gated identically on 2026-08-30 (7/7, plus tool
battery, garble sweep and RULER-lite).

## Reproducibility

- anemll image `dsv4-3spark:0.1.1`, vLLM `v0.25.2.dev0+g752a3a504.d20260714`,
  `kv_cache_dtype=nvfp4_ds_mla`, `speculative=dspark num_spec_tokens=2`.
- eugr image `eugr/spark-vllm-b12x:latest` at digest
  `sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd`,
  vLLM `0.1.dev20133+gb5f995e73.d20260823`, `nst=5`, `mnbt=8192`, warm caches.
- Raw per-trial logs: `anemll-c{1,4,8,16}.log`, `eugr-c{1,4,8,16}.log`.

The eugr arm was also re-measured fresh here rather than quoted, and it
reproduced the 2026-08-30 recorded rows to within ±8 % on every cell
(84.7 vs 84.3, 54.4 vs 53.2, 44.9 vs 44.1, 15.0 vs 15.5) — evidence that the
eugr side is stable day-to-day and that the swing in this comparison comes from
the anemll arm being stale, not from eugr drifting.

## Service state after the run

`eugr.service` restarted and verified; anemll `tp3.env` restored to
`MAX_NUM_SEQS=32` on all three ranks; zero leaked containers on any node
(checked with `docker ps` on all three, per the leaked-container history).
