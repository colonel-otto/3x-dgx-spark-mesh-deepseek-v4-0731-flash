# `configs/` — benchmark harness targets (runs on your WORKSTATION)

> [!IMPORTANT]
> **There are two config directories and they are not interchangeable.**
>
> | Directory | Contains | Lives on | Consumed by |
> |---|---|---|---|
> | [`../config/`](../config/) | vLLM engine + NCCL env, per rank | **each Spark** | `docker compose` / `dsv4.service` |
> | **`configs/`** ← you are here | harness targets: SSH, fabric addrs, gate thresholds | **your workstation** | `scripts/fabric_gate.sh`, `Makefile` |
>
> A file from one will not work in the other.

## Files

| File | Purpose |
|---|---|
| [`3spark-live.env.example`](3spark-live.env.example) | **The live 3-node cluster.** Copy to `3spark-live.env` and edit |
| [`3spark.env.example`](3spark.env.example) | Generic 3-node template |
| [`2spark.env.example`](2spark.env.example) | The 2-node comparison arm |

`*.env` is gitignored; only `*.env.example` is tracked. Your live copy stays local.

## Usage

```bash
make gate      CONFIG=configs/3spark-live.env   # engine UP: skips bandwidth
make gate-full CONFIG=configs/3spark-live.env   # engine STOPPED: full measurement
```

## Why `FABRIC_ADDRS` is the important field

The gate checks the **fabric** addresses, not the management IPs. Checking the management
LAN would have passed cleanly straight through the 2026-08-25 degradation, and a node
silently falling back to routing over Wi-Fi is a failure we have actually hit.

## Thresholds, and why they sit where they do

| Setting | Value | Rationale |
|---|---:|---|
| `FABRIC_GATE_BUSBW_MIN` | 3.5 | Healthy pair ~4.6 GB/s, degraded ~0.7. Sits clear of both, so it fails loudly without flapping |
| `FABRIC_GATE_BUSBW_MIN_ALL` | 2.5 | Healthy 3-rank measures 3.25 GB/s. The old 0.49 figure was degraded and 6.6x pessimistic |
| `FABRIC_GATE_RTT_MAX_MS` | 2.0 | Catches a link that is up but pathological |

See [`../docs/DEGRADED-DATA-CATALOGUE.md`](../docs/DEGRADED-DATA-CATALOGUE.md) for what
these thresholds are protecting you from.
