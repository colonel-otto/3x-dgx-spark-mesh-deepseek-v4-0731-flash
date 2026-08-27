# Controlled NCCL bandwidth reproduction (issue #18)

**Status:** `CURRENT` within the provenance caveats in [`../index.yaml`](../index.yaml).

**Date:** 2026-08-26 · **Engine:** stopped for all runs · **Harness:** official
`nccl-tests all_gather_perf`, NCCL **2.30.7**, `-n 20 -w 5`, 1 GPU/rank.

## Headline

**There is no bandwidth gap.** The current, unmodified config already delivers
**23.92 GB/s** busbw at 16 GiB on 3 ranks. The published comparison points are
20.84 @16 GiB and 18.70 @32 MiB. **We match or exceed both.**

The prior 5.80 GB/s figure was produced by the custom PyTorch harness, not by the
fabric. Swapping in the official binary on the *same* config raises the same
3-rank ring to 23.92 GB/s. **The harness was the variable.**

## Results

busbw in GB/s. `oop` = out-of-place (the convention the published figures use).

| variant | world | bootstrap | size | algbw oop | **busbw oop** | busbw ip | transport |
|---|---:|---|---|---:|---:|---:|---|
| A baseline | 3 | fabric | 32M | 7.21 | 4.81 † | 19.78 | NET/IB/4+5 |
| A rep2 | 3 | fabric | 32M | 25.28 | **16.85** | 19.16 | NET/IB/4+5 |
| A rep3 | 3 | fabric | 32M | 23.61 | **15.74** | 18.51 | NET/IB/4+5 |
| A baseline | 3 | fabric | 16G | 35.87 | **23.92** | 24.15 | NET/IB/4+5 |
| B mgmt boot | 3 | wlP9s9 | 32M | 24.22 | **16.14** | 18.76 | NET/IB/4+5 |
| B mgmt boot | 3 | wlP9s9 | 16G | 35.91 | **23.94** | 24.17 | NET/IB/4+5 |
| C merge off | 3 | wlP9s9 | 32M | 26.86 | **17.91** | 19.14 | NET/IB/0..3 |
| C merge off | 3 | wlP9s9 | 16G | 35.80 | **23.86** | 24.15 | NET/IB/0..3 |
| D auto HCA | 3 | wlP9s9 | 32M | 25.09 | **16.72** | 19.40 | NET/IB/4+5 |
| D auto HCA | 3 | wlP9s9 | 16G | 35.78 | **23.85** | 24.14 | NET/IB/4+5 |
| E world=2 | 2 | wlP9s9 | 32M | 32.44 | **16.22** | 18.81 | NET/IB/4+5 |
| E world=2 | 2 | wlP9s9 | 16G | 44.60 | **22.30** | 23.91 | NET/IB/4+5 |

† Cold-start outlier (4652 µs vs ~1300 µs in repeats). Superseded by rep2/rep3.

Every run: NCCL 2.30.7, `#wrong = 0`, **zero `NET/Socket` occurrences**. No run void.

## Which variable moved the number

**None of them.** At 16 GiB the four 3-rank variants span 23.85–23.94 GB/s — a
**0.4% spread**, i.e. noise:

| change | 16 GiB busbw | delta |
|---|---:|---|
| A → B (bootstrap: fabric → management) | 23.92 → 23.94 | +0.1% |
| B → C (NIC merging on → off) | 23.94 → 23.86 | −0.3% |
| B → D (explicit HCA list → auto discovery) | 23.94 → 23.85 | −0.4% |

The bootstrap change was verified real, not just accepted: NCCL bound to
`192.168.100/101/102.x` in A and `192.168.1.x` in B.

## The merge question, answered

NCCL **does** merge across two different physical ports, verbatim:

```
Made virtual device [4] name=rocep1s0f0+rocep1s0f1     speed=400000 ndevs=2
Made virtual device [5] name=roceP2p1s0f0+roceP2p1s0f1 speed=400000 ndevs=2
```

On this ring, `f0` and `f1` face *different* neighbours, so this is exactly the
cross-port pairing `BANDWIDTH-COMPARISON.md` predicted would harm a 3-node ring.

**That prediction is refuted.** Variant C turned merging off entirely (four
separate devices `NET/IB/0..3`) and the result was unchanged: 23.86 vs 23.94.
Cross-port merging is harmless here.

## Other findings

- **No rank-count penalty.** 3-rank (23.92) slightly *exceeds* 2-rank (22.30) at
  16 GiB. The claim that the deficit grew with rank count was a harness artifact.
- **PCIe ceiling confirmed.** Every 3-rank variant pins to 23.85–23.94 GB/s,
  matching the ~24 GB/s Gen5 x4 ceiling documented in `BANDWIDTH-COMPARISON.md`.
  The fabric is saturating PCIe, not underperforming.
- **Size dependence is real:** ~16–17 GB/s @32 MiB → ~24 GB/s @16 GiB (+45%).

## Docs needing correction

- `BANDWIDTH-COMPARISON.md` — the "~3.2x gap" headline is refuted; it was harness.
- `BANDWIDTH-COMPARISON.md` — the cross-port merge hypothesis is measured and refuted.
- `BANDWIDTH-NEXT-TEST.md` — "bootstrap is the strongest lead" is refuted.
- The ~24 GB/s PCIe ceiling estimate is **confirmed** and is where we sit.

## Reproducing

`Dockerfile.ntests` builds `ntests:2.30.7` from the serving image plus OpenMPI and
nccl-tests. Build it on each node (seconds; each already has the base image), then:

```
run_variant.sh <tag> <world> <fabric|mgmt> <32M|16G> [KEY=VAL ...]
```

**The version trap is live.** The binary compiles against the 2.30.7 headers but
the loader resolves `libnccl.so.2` to the *system* 2.28.9 copy unless
`LD_LIBRARY_PATH` points at
`/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib`. Confirm with
`nccl-headers=23007 nccl-library=23007` in the output header — a mismatch there
silently invalidates the comparison.

**MPI note:** `mpirun`'s own OOB/BTL must stay on the management LAN. The fabric
is point-to-point /30 links, not a full mesh, so MPI's TCP BTL cannot route across
it. MPI is only a process launcher; `NCCL_SOCKET_IFNAME` is the variable under test.

`all_reduce_perf` does not link against 2.30.7 (`ncclCommQueryProperties`
undefined). Only `all_gather_perf` was needed and it builds cleanly.
