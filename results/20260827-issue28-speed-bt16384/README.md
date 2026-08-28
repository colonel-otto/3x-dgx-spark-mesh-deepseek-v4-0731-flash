# Speed profile sweep (MAX_NUM_BATCHED_TOKENS=16384) — 2026-08-27

**Status:** `CURRENT` · **Target:** Speed optimization by scaling prefill batch tokens · **Output:** 256 tokens asserted on all reps · **Cache:** zero cached tokens

This bundle records the Issue #28 experiment trading unneeded KV cache headroom to investigate whether doubling `MAX_NUM_BATCHED_TOKENS` from 8,192 to 16,384 improves TTFT and decode throughput.

## Configuration
- `MAX_NUM_BATCHED_TOKENS=16384`
- `MAX_MODEL_LEN=460800`
- `NCCL_BUFFSIZE=16777216`
- `GPU_MEMORY_UTILIZATION=0.835`
- `LONG_PREFILL_TOKEN_THRESHOLD=1024`
- `DSPARK_MAX_INFLIGHT_PREFILLS=2`
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`
- **Resulting KV Pool**: **1,954,299 tokens** (33.44 GiB available per GPU).

## Single-Stream Depth Sweep (7 reps per depth)

| Target Depth | Actual Prompt | Median Decode (tok/s) | Min (tok/s) | Max (tok/s) | Spread | Median TTFT (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,037 | 52.90 | 46.77 | 62.09 | 29.0% | 1.09 |
| 8,192 | 8,083 | 55.98 | 49.42 | 58.74 | 16.6% | 4.17 |
| 32,768 | 32,271 | 54.91 | 51.08 | 57.98 | 12.6% | 18.46 |
| 131,072 | 129,007 | 51.18 | 43.21 | 56.85 | 26.6% | 92.46 |
| 262,144 | 257,995 | **51.39** | 37.18 | 53.72 | 32.2% | 228.64 |

## Starvation Probe (5 trials under concurrent prefill)
- **Median Decoder TTFT**: 149.32 s
- **Median Decoder Decode**: 46.28 tok/s
- **Median Max Event Gap**: 0.107 s
- **Max Event Gap Across All Trials**: 0.341 s

## Findings

1. **Decode at extreme depth**: 262K median decode was 51.39 tok/s vs Profile B's 46.08
   (+11.5%). **This sits inside its own run spread** (32.2%: min 37.18, max 53.72, n=7),
   and Profile B's 262K spread is 16.5%. Treat it as suggestive, not established.
2. **TTFT penalty**: 131K TTFT rose from 75.7s to 92.5s (+22.1%) and 262K from 177.3s to
   228.6s (+29.0%). The regression is large, consistent across reps, and **the mechanism is
   unknown**.

   An earlier version of this document attributed it to ~235 MB per-layer activation
   tensors saturating the GB10 unified memory bus (273 GB/s) and causing RoCE
   serialization. **That hypothesis is withdrawn**: streaming 235 MB at 273 GB/s is
   ~0.86 ms per layer-pass, which cannot account for +16.7 s at 131K even allowing for
   many passes per layer across 8 chunks. It also sits badly with this repo's four-HCA
   null result, which found fabric bandwidth is not the binding constraint. It was never
   measured.

   Untested candidates: chunked-prefill scheduler interaction, attention-kernel behaviour
   at large chunk sizes, and all-reduce message sizing.
3. **Conclusion**: `MAX_NUM_BATCHED_TOKENS=8192` remains the choice. 16384 lost deep TTFT,
   lost the starvation probe, and cost 21% of the KV pool; its single win is inside the
   noise. That verdict does not depend on knowing the mechanism.

   `32768` was **not tested**. A previous extrapolation ("~470 MB activations would be
   worse") inherited the withdrawn hypothesis and is removed.

## Configuration correction (recorded 2026-08-27 during review)

The Configuration section above lists this arm as it was *intended*. The captured
artifacts — which govern, per `docs/BENCHMARK-POLICY.md` "config comes from the live
process" — show two of those entries never took effect:

- **`NCCL_BUFFSIZE=16777216` was never applied**, and cannot be by that route.
  `scripts/configure_speed_profile.py` did write it into `config/tp3.env` on all three
  nodes, where it remains (`tp3.env:65`). But `docker-compose.yml` lists NCCL variables
  **individually** in an explicit `environment:` block and never references
  `NCCL_BUFFSIZE`; there is no `env_file:` directive. A value in `tp3.env` reaches the
  container only if compose forwards it by name, and this one is not forwarded.

  Confirmed against the live cluster: `docker exec ... env | grep -i buffsize` returns
  nothing, while `env | grep -c '^NCCL_'` returns 15 — matching the 15 `NCCL_*` entries in
  `container-env.json` for both arms. The setting has never been in effect.
- **`GPU_MEMORY_UTILIZATION` was not a variable.** Both arms ran
  `gpu-memory-utilization 0.835`.

`diff` of the two arms' engine command lines differs by exactly two tokens:

```
--max-model-len            1048576 -> 460800
--max-num-batched-tokens      8192 -> 16384
```

So this A/B was **cleaner than originally written** (one confound, not three), but
`MAX_MODEL_LEN` remains uncontrolled and the TTFT delta cannot be attributed to
`MAX_NUM_BATCHED_TOKENS` alone. `docs/DECISIONS.md` separately states "nothing gained by
lowering" `MAX_MODEL_LEN`, so its effect here is unquantified rather than known-zero.

A single-variable re-run (both arms at `MAX_MODEL_LEN=1048576`) is the prerequisite for
any mechanism work.
