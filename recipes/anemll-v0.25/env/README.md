# `env/` — engine environment (runs ON the Sparks)

> [!IMPORTANT]
> **There are two config directories and they are not interchangeable.**
>
> | Directory | Contains | Lives on | Consumed by |
> |---|---|---|---|
> | **`recipes/anemll-v0.25/env/`** ← you are here | vLLM engine + NCCL env, per rank | **each Spark**, at `~/localai/dspark-vllm-gx10/config/` | `docker compose` / `dsv4.service` |
> | [`../../../configs/`](../../../configs/) | benchmark harness targets (SSH, fabric addrs, thresholds) | **your workstation** | `scripts/fabric_gate.sh`, `Makefile` |
>
> A file from one will not work in the other.

## Files

| File | Purpose |
|---|---|
| [`tp3.env.example`](tp3.env.example) | **Authoritative.** The full 3-node TP=3 config with every pitfall documented inline. Start here. |
| [`common.env.example`](common.env.example) | The settled shared values, to be merged into each rank file |
| `node0/1/2.env.example` | The per-rank header — the **only** part that may differ between nodes |
| [`compose.tp3.fragment.yml`](compose.tp3.fragment.yml) | Compose forwarding. Env values do nothing unless forwarded |

## The two rules

**1. Everything below the rank header must be byte-identical across all three nodes.**
A mismatch in any parallelism flag **hangs startup forever with no error.** Verify before
every start:

```bash
KEYS='MAX_MODEL_LEN|MAX_NUM_SEQS|MTP_NUM_TOKENS|GPU_MEMORY_UTILIZATION|TP_SIZE|NNODES'
for h in node0 node1 node2; do printf "%-10s " $h
  ssh $h "grep -E '^($KEYS)=' ~/localai/dspark-vllm-gx10/config/tp3.env|sort|md5sum"; done
```

**2. Confirm Compose actually forwards what you set.** An env value that is not forwarded
is silently ignored:

```bash
docker compose --env-file config/node0.env -f docker-compose.yml config \
  | grep -E 'SUBNET_AWARE|NCCL_NET|NCCL_IB_HCA|tensor-parallel'
```

After **any** restart, run a correctness check (17×23 → 391). The TP=3 padding patch means
a broken config serves *fluent nonsense*, not an error.

Reasoning behind every value: [`../../../docs/DECISIONS.md`](../../../docs/DECISIONS.md).
