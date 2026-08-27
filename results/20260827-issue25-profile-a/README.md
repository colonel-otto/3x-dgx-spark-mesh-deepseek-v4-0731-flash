# Profile A baseline tuning evaluation - 2026-08-27

**Status:** `SUPERSEDED-BY-20260827-issue25-profile-b` • **Nodes:** 3 • **TP:** 3 • **Fabric gate:** `PRESENT-PASS` (30 pass, 0 fail, 3 expected skips; 9.29 GB/s NCCL)

Baseline TP=3 production configuration before applying the recipe tuning deltas in issue #25.

## Configuration

- `GPU_MEMORY_UTILIZATION`: `0.80` (KV pool: ~1.84M tokens)
- `LONG_PREFILL_TOKEN_THRESHOLD`: `0` (default / unset)
- `MAX_NUM_SEQS`: `32`
- `MAX_NUM_BATCHED_TOKENS`: `8192`
- `MTP_NUM_TOKENS`: `5`
- `KV_CACHE_DTYPE`: `nvfp4_ds_mla`

## Results

### Decode Depth Sweep (256-token asserted window, n=7 reps)

| Depth | Prompt tokens | Median decode | Decode spread | Median TTFT |
|---|---|---|---|---|
| 2,048 | 2,034 | 54.12 tok/s | 15.4% | 0.94 s |
| 8,192 | 8,080 | 57.54 tok/s | 18.0% | 3.48 s |
| 32,768 | 32,268 | 54.96 tok/s | 17.1% | 14.63 s |
| 131,072 | 129,006 | 48.89 tok/s | 17.0% | 75.19 s |
| 262,144 | 257,992 | 44.94 tok/s | 35.6% | 189.57 s |

### Starvation Probe (5 trials, concurrent long prefills)

- Decoder median TTFT: 149.99 s
- Decoder median decode: 47.00 tok/s
- Max event gap: 0.176 s
