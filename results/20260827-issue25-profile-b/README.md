# Profile B recipe tuning evaluation - 2026-08-27

**Status:** `CURRENT` • **Nodes:** 3 • **TP:** 3 • **Fabric gate:** `PRESENT-PASS` (30 pass, 0 fail, 3 expected skips; 9.50 GB/s NCCL)

Evaluates the published recipe deltas on the TP=3 cluster for issue #25.

## Configuration

- `GPU_MEMORY_UTILIZATION`: `0.835` (KV pool: ~2.49M tokens, +35% capacity)
- `LONG_PREFILL_TOKEN_THRESHOLD`: `1024`
- `DSPARK_MAX_INFLIGHT_PREFILLS`: `2`
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: `4096`
- Hotfixes: `hotfix-dsv4-issue26-hybrid-swa-min.py` (v2) and `hotfix-dsv4-issue27-partial-prefill-concurrency.py`
- `MAX_NUM_SEQS`: `32`
- `MAX_NUM_BATCHED_TOKENS`: `8192`
- `MTP_NUM_TOKENS`: `5`
- `KV_CACHE_DTYPE`: `nvfp4_ds_mla`

## Results

### Decode Depth Sweep (256-token asserted window, n=7 reps)

| Depth | Prompt tokens | Median decode | Decode spread | Median TTFT |
|---|---|---|---|---|
| 2,048 | 2,034 | 53.94 tok/s | 7.9% | 1.10 s |
| 8,192 | 8,080 | 59.82 tok/s | 26.8% | 4.37 s |
| 32,768 | 32,268 | 53.52 tok/s | 38.2% | 17.47 s |
| 131,072 | 129,006 | 50.12 tok/s | 11.3% | 75.73 s |
| 262,144 | 257,992 | 46.08 tok/s | 16.5% | 177.29 s |

### Starvation Probe (5 trials, concurrent long prefills)

- Decoder median TTFT: 133.91 s (-16.1 s / 10.7% faster TTFT under load vs Profile A)
- Decoder median decode: 49.18 tok/s (+4.6% vs Profile A)
- Max event gap: 0.178 s
