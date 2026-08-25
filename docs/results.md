# Controlled result and experiment progression

> [!WARNING]
> **FROZEN and PROVISIONAL.** Earliest TP=2/TP=3 comparison, measured on the pre-rewire
> cable rotation and before the 2026-08-25 fabric fix ([#14](../../issues/14)). Retained as
> a historical record only. For current numbers see
> [`../README.md`](../README.md#is-the-third-node-worth-it).
> 
> Itemized with every other bad measurement in [`DEGRADED-DATA-CATALOGUE.md`](DEGRADED-DATA-CATALOGUE.md).

> This page freezes the earlier MTP=5 result on the pre-rewire cable rotation. For the
> canonical-ring tuning results, see [`TP3-TUNING.md`](TP3-TUNING.md). For the later
> prompt-matched 79.0–79.3 tok/s result and raw sweeps, see
> [`BENCHMARK-METHODOLOGY.md`](BENCHMARK-METHODOLOGY.md).

## Summary

| Metric | TP=2 RoCE | TP=3 TCP control | TP=3 RoCE |
|---|---:|---:|---:|
| Decode throughput, median | 48.23 tok/s | 24.59 tok/s | **57.73 tok/s** |
| Observed range | 46.8-50.2 | 24.0-25.5 | 52.6-59.3 |
| Measured repetitions | 3 | 3 | 5 |
| TTFT | 0.154 s | 0.323 s | 0.186 s |
| KV-cache tokens | 1,855,255 | 3,579,619 | **3,598,182** |
| Maximum concurrency at 460,800 | approximately 3.9x | 7.77x | **7.81x** |
| Engine initialization | 92.0 s | 167.9 s | 103.7 s |
| MoE backend | B12X_MXFP4 | B12X_MXFP4 | B12X_MXFP4 |
| MTP | active | active | active |
| Correctness | 7/7 | 7/7 | 7/7 |

The RoCE run was independently repeated: an initial three-repetition run had a 56.8
tok/s median, and the five-repetition confirmation had a 57.73 tok/s median.

## Recorded environment

All three nodes matched for the confirmed run:

| Component | Recorded value |
|---|---|
| DGX OTA | 7.5.0 |
| GPU driver | 580.173.02 |
| Kernel | 6.17.0-1029-nvidia |
| ConnectX-7 firmware | 28.45.4028 |
| Serving image tag | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Live process NCCL | 2.30.7 (`nvidia-nccl-cu13`) |
| PyTorch compile-time NCCL report | 2.28.9 |

The original notes did not retain an immutable container digest or complete package
lockfile. A fresh artifact bundle must capture both; the tag alone is insufficient for
bit-for-bit reproduction.

## Progression

### 1. TP=2 RoCE baseline

Two Sparks established the controlled baseline at 48.23 tok/s. The checkpoint, serving
profile, prompt and sampling configuration were held fixed for the TP=3 comparison.

### 2. TP=3 with group padding over TCP

The TP=3 attention-group patch passed correctness and retained B12X plus MTP, but TCP
delivered only 24.59 tok/s. This was a transport control, not the final TP=3 result.

### 3. TP=3 over RoCE

The working transport configuration added subnet-aware HCA selection:

```bash
NCCL_IB_DISABLE=0
NCCL_NET=IB
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_NET_PLUGIN=none
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
```

The result rose to 57.73 tok/s. Hardware port counters increased on all three physical
edges during inference, establishing that the result was not an unnoticed socket run.

## What the result establishes

- TP=3 is viable for this checkpoint when attention groups are padded correctly.
- TP=3 need not sacrifice the B12X MXFP4 backend or MTP.
- Three-node RoCE works on a direct, switchless ring with subnet-aware routing.
- On this cluster, TP=3 RoCE improved both single-stream speed and capacity over TP=2.

## What it does not establish

- It is not a universal performance guarantee for every image or firmware release.
- It does not prove multi-rail performance; only the two lowercase RoCE devices were
  configured for the measured run.
- Raw per-request samples from every historical exploratory run were not retained.
  Consequently, the summary is marked historical rather than represented as a complete
  evidence bundle.
- The measured RoCE result preceded the final physical rotation into NVIDIA's canonical
  port-number convention. The logical three-edge ring and subnet-aware behavior were
  equivalent. A new canonical-layout run should be published as a fresh artifact bundle.
- Raw KV arithmetic is not the same as active request count. `MAX_NUM_SEQS=16` caps the
  engine at 16 active sequences and queues additional work.

Machine-readable values are in [`../benchmarks/summary.csv`](../benchmarks/summary.csv).
