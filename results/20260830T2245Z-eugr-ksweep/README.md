# eugr engine — K sweep + persistent kernel caches (2026-08-30)

**Status: CURRENT** — healthy fabric (gate 30/30 including NCCL bandwidth) and
sound methodology (correctness 7/7, warm caches, JIT miss counter frozen before
recording). This bundle supersedes the throughput conclusions of
`20260830T194550Z-engine-ab-eugr`.

Engine `eugr-spark-vllm-b12x`, image digest
`sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd`,
3 nodes TP=3, 1M context, kv fp8, `max_num_seqs 16`, `gpu_memory_utilization 0.82`.
Harness `scripts/eugr-ab/bench-miaai.py` (byte-identical to arm 1), 256-token
unique cold prefix, `thinking=false`, min=max=128 tokens, median-of-5 trials.

## Headline 1 — the K sweep is NOT {2,3,5,7}. nst<5 is ILLEGAL on this checkpoint.

Booting nst=2 fails at config validation:

```
DSpark requires num_speculative_tokens >= dspark_block_size (5); got 2.
Smaller values produce incorrect output. Use num_speculative_tokens=5 or larger (e.g. 7).
```

`dspark_block_size: 5` is a property of the CHECKPOINT
(`config.json`: `dspark_block_size 5`, `dspark_markov_rank 256`,
`dspark_target_layer_ids [40,41,42]`), not a tunable. DSpark is a
semi-autoregressive **block** drafter; a speculative length below the block size
feeds the block/Markov-head machinery an unsupported layout and yields
**incorrect (garbled) output**, not merely lower acceptance. The engine refuses
rather than silently serving nonsense.

Consequence: the plan's "match our MTP K=2 to remove the speculator delta" is
**not achievable on this engine**. The cross-engine A/B keeps a permanent
speculator delta (anemll MTP K=2 vs eugr DSpark K>=5) and every row must say so.
The real sweep is nst in {5, 7} plus the `max_num_batched_tokens` lever.

## Headline 2 — arm 1's "c=16 scheduling cliff" was mostly JIT contamination.

Same nst=5, same harness, same node count. The ONLY change is persistent kernel
caches (`/opt/eugrcache-*` mounts replacing `--no-cache-dirs`):

| c | arm 1 (cold caches) agg | this run (warm caches) agg | Δ |
|---|---:|---:|---:|
| 1 | 82.1 warm / 65.4 cold (decode) | **84.3** (decode) | +2.7% vs warm |
| 4 | 162.7 | 152.8 | −6% |
| 8 | 171.7 | **252.9** | **+47%** |
| 16 | 133.9 | **198.8** | **+48%** |

TTFT at c=16: **7000ms -> 1755ms (4x better)**. The c=16 cell is no longer a
regression vs the anemll reference (161.0); it is now the best aggregate cell
measured on this engine. Boot time also halved: ~4.3 min vs ~8 min cold.

Arm 1's c=1 log shows the mechanism directly — decode DECAYED across trials
(83.8 -> 80.3 -> 64.6 -> ... -> 57.8) as new kernel shapes hit the JIT. With warm
caches the same cell rises and holds (78.4 -> 91.4 -> 84.5 -> 78.3, median 84.3).

**Any arm-1 throughput row taken under `--no-cache-dirs` should be treated as a
lower bound, not as engine capability.**

## Headline 3 — the sweep verdict: nst=5 / mnbt=8192 wins. Serve it.

Complete matrix (aggregate tok/s; c=1 column is single-stream decode):

| c | nst=5 mnbt=8192 | nst=7 mnbt=8192 | nst=5 mnbt=16384 |
|---|---:|---:|---:|
| 1 (decode) | **84.3** | 79.5 | 83.5 |
| 4 | 152.8 | 151.2 | **165.0** |
| 8 | **252.9** | 208.8 | 241.8 |
| 16 | 198.8 | 197.2 | **214.3** |
| **KV cache tokens** | **2,415,674** | 2,415,674 | **1,165,679** |
| max concurrency @1M ctx | **2.30x** | 2.30x | 1.11x |

- **nst=7 loses everywhere.** It never wins a cell and costs 21% at c=8. The
  anemll-derived expectation "high K wins single-stream" does NOT transfer:
  nst=5 wins single-stream too. With the block-size floor at 5, the legal range
  is {5,7} and 5 is simply better — the depth question is settled, not open.
- **mnbt=16384 is the anemll KV trap again.** It wins c=4 (+8%) and c=16 (+8%)
  but costs **52% of the KV cache** (2.42M -> 1.17M tokens; max concurrency at 1M
  context falls 2.30x -> 1.11x). It also loses c=8 by 4%. Trading half the KV
  capacity for ~8% on two cells is a bad trade for a 1M-context server, so this
  confirms the anemll finding on a different engine and different KV dtype —
  measured, not assumed, as the plan required.
  (One thing it DOES fix: the `max_num_scheduled_tokens is set to 8128` warning
  disappears at 16384. The warning was real but the cure costs more than the
  disease.)

**`eugr.service` is pinned to EUGR_NST=5 / EUGR_MNBT=8192** — the winner is what
the gateway serves.

## Rows

`nst5-mnbt8192/rows.tsv`, `nst7-mnbt8192/rows.tsv`, `nst5-mnbt16384/rows.tsv`. The c=1 row was re-measured (median-of-7) after the
JIT counter froze; the sweep's own first pass recorded 67.8 while still warming,
and that superseded value is noted in `nst5-mnbt8192/notes.txt`.

## Gates (both required by BENCHMARK-POLICY, both here)

- **Fabric**: `nst5-mnbt8192/fabric-gate-preboot.json` — full gate WITH the NCCL
  bandwidth check (only possible with the engine down): **30 passed, 0 failed**;
  all three pairs 9.08–9.27 GB/s @64MiB over RDMA with correct NIC merging.
- **Correctness**: `eugr-quick-validate.sh` 7/7 at nst=5 on :8100, virtual-TP
  plan confirmed active (heads 64->72, groups 8->9).

While running that gate we found `configs/3spark-live.env` still held the
pre-NVIDIA-Sync fabric addresses (`192.168.100/101/200.x`), which are DEAD. The
gate reported 8 failures against a perfectly healthy fabric. Fixed, and
`scripts/discover_fabric_addrs.sh` now derives the addressing live so it cannot
drift again.

## JIT cache-miss counters

`nst5-mnbt8192/jit-miss-counter.txt`. 12 misses this boot vs 20 on arm 1, and
they now PERSIST to `/opt/eugrcache-*` (identical ~54MB on all three nodes), so
subsequent boots inherit them.

## Note on the committed artifacts

`fabric-gate-preboot.json` has its DGX hostnames replaced with the repo's
generic `node0`/`node1`/`node2` labels (this repo is public; the pre-commit
`scripts/check_no_sensitive.py` enforces it). Measured values are untouched.
