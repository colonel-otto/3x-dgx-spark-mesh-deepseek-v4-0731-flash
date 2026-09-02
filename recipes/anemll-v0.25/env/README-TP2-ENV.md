# READ BEFORE BRINGING UP TP=2

`head.env` and `worker.env` are **NOT** at the production engine profile, and
bringing up a 2-node arm from them as-is produces a benchmark that is silently
invalid.

## What is wrong with them

`dsv4.service` reads `tp3.env`. Nothing reads `head.env`/`worker.env` except a
manual TP=2 bringup, so these files drift and no one notices. As of
2026-08-30 they sit at:

| key | head/worker.env | tp3.env (production) |
|---|---|---|
| MAX_NUM_SEQS | 16 | 32 |
| MTP_NUM_TOKENS | 5 | 2 |
| GPU_MEMORY_UTILIZATION | 0.80 | 0.835 |
| KV_CACHE_DTYPE | *absent* | nvfp4_ds_mla |
| LONG_PREFILL_TOKEN_THRESHOLD | *absent* | 1024 |
| DSPARK_MAX_INFLIGHT_PREFILLS | *absent* | 2 |
| DSPARK_VLLM_IMAGE | ghcr.io/anemll/dspark-vllm-gx10:0.1.1 | **dsv4-3spark:0.1.1** |

That is **seven variables**, including a different container image — a different
engine build. vLLM starts fine either way and reports nothing, so a 2v3
comparison run from these files will look clean and mean nothing.

This is not hypothetical: it was found and fixed mid-flight during the
2026-08-30 llama-benchy run.

## The fix

Do not hand-edit. Run:

    bash ~/bench-repo/scripts/match_env_for_benchy.sh apply

It rewrites only the managed engine keys, backs up to `.prebenchy.<STAMP>`, and
refuses to continue unless both ranks agree. `... restore` puts them back.

Rank-specific keys (`NODE_RANK`, `VLLM_HOST_IP`, `MASTER_ADDR`, `NCCL_IB_HCA`,
`NCCL_SOCKET_IFNAME`) are deliberately left alone — the 2-node arm's direct
192.168.100.x fabric is the topology under test, not a confound.

## Then verify against the LIVE engine, not the file

A config file is a claim; the running process is the fact. Before measuring:

    ps -eo args | grep '[v]llm.*tensor-parallel-size'

and confirm TP, `max-num-seqs`, `gpu-memory-utilization`, `kv-cache-dtype`,
`long-prefill-token-threshold`, `max-num-batched-tokens`, `max-model-len`,
`num_speculative_tokens` and `moe-backend` all match the arm you are comparing
against. `scripts/run_llama_benchy_2v3.sh` does this automatically and dies if
any one is off.
