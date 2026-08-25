# Why our bandwidth number looked 3-6x too low

**Short answer: we compared two different measurements.** The gap is mostly, possibly
entirely, an artifact of collective convention and message size. Before chasing a fabric
problem, run [`../scripts/bwsweep.py`](../scripts/bwsweep.py) and compare like with like.

---

## What we were comparing

| source | topology | collective | message | reported |
|---|---|---|---|---:|
| [NVIDIA forum (Turtle7777)](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) | 3-Spark ring | allgather | **16 GB** | 20.84 GB/s |
| [route179.dev](https://route179.dev/2026/07/21/dgx-spark-nccl-roce-benchmarking/) | **2**-Spark | allgather | not stated | 22.1 GB/s |
| **ours** | 3-Spark ring | allgather | **67 MB** | **5.80 GB/s** busbw |

Two variables differ, and **both** move the number in the same direction.

## Confound 1 — convention (up to 3.2x)

For all_gather with input `n` bytes per rank across `w` ranks:

```
algbw = n / dt                  # "how fast my input moved"
busbw = n * (w-1)/w / dt        # "wire rate each NIC sustains"
```

`nccl-tests all_reduce_perf` busbw carries an **extra factor of 2**, because ring
all-reduce is reduce-scatter + all-gather. So one wire speed reads three ways:

| convention | our w=2 | our w=3 |
|---|---:|---:|
| all_gather **busbw** ← what we report | 9.70 | **5.80** |
| all_gather **algbw** | **19.40** | 8.70 |
| all_reduce **busbw** equivalent | 19.40 | 11.60 |

**Our 2-rank algbw is 19.40 GB/s — inside the 18–21 band exactly.** Same fabric, same
run, different bookkeeping. We report the most conservative of the three.

## Confound 2 — message size (238x)

Our largest point is a **67 MB** input; the forum figure used **16 GB**.

NCCL bandwidth is `time = latency + bytes/bandwidth`. Small messages are latency-bound;
throughput only asymptotes toward line rate at large sizes. **A 67 MB allgather is nowhere
near the asymptote on a 400 Gb/s fabric**, so its number is structurally low as a measure
of peak.

This was not a sloppy choice — it was the *right* size for the question the harness was
built for. `agbench.py` mirrors the vLLM MTP allgather shape:

| shape | input per rank |
|---|---:|
| MTP allgather, `seqs=16` | **4.14 MB** |
| MTP allgather, `seqs=32` | 8.27 MB |

The error was **reusing a workload-shaped harness as a peak-bandwidth harness**, then
comparing its top point against someone else's peak-bandwidth figure.

## What we did not think of

1. **We never ran the collective at the size the comparison used.** Not once. The gap was
   treated as a hardware mystery for days without matching the independent variable.
2. **We never stated our convention when comparing.** Neither did the sources. `busbw` and
   `algbw` differ by `w/(w-1)`, and all_reduce adds another 2x.
3. **The 2-node sources are not comparable to a 3-node ring at all.** route179 is 2-Spark.
   At `w=2` each rank has one peer and both ports face it; at `w=3` each rank relays for a
   neighbour and the two ports face *different* peers. Different topology, different
   number.
4. **We treated "peak fabric bandwidth" as the goal.** It is not our workload. At 4.14 MB
   the MTP allgather is **latency-bound**, and peak bandwidth at 16 GB says nothing about
   it. Optimizing toward a number that does not govern our performance is wasted effort.
5. **The efficiency figure was flat at ~20% of line in both configs** (2 HCA and 4 HCA).
   A hard ceiling does not scale linearly — and doubling HCAs doubled throughput
   (4.73 → 9.70, 2.05x). That linearity was evidence *against* a wall, and we read it as
   evidence of one.

## What is genuinely lower at w=3, and is not a bug

- **Ring relaying.** At `w=3` each rank forwards for a neighbour; per-rank egress rises.
- **No GPUDirect on GB10** — architecturally impossible, not merely absent. Every byte
  bounces through system memory over PCIe. A fixed tax no flag removes.
- **PCIe Gen5 x4 per ConnectX device** ≈ 14.4 GB/s practical per device.
- **No redundant path in a 3-node ring** — the slowest link paces the whole collective.

## The test that settles it

[`../scripts/bwsweep.py`](../scripts/bwsweep.py) sweeps 4 MiB → 4 GiB and prints **all
three conventions on every line**, so a published figure can be matched rather than
guessed at.

```bash
# engine STOPPED -- needs tens of GiB of buffer
RANK=$i WORLD_SIZE=3 INIT_METHOD=tcp://<rank0-fabric>:29555 \
  TAG=4hca MAX_GIB=4 python3 scripts/bwsweep.py
```

Run at `w=2` and `w=3`. Then compare **only** against a source that states its collective,
size, rank count, and convention.

### What each outcome means

| Result at 1–4 GiB | Reading |
|---|---|
| busbw climbs toward ~15–20 GB/s | ✅ **No fabric problem.** The gap was size + convention. Close [#11](../../issues/11) |
| busbw plateaus near 5.8 GB/s | ⚠️ A real ceiling exists. Now it is worth chasing — and the sweep shows *where* it binds |
| w=2 scales but w=3 does not | ⚠️ Ring-specific: relaying or routing, not raw link speed |

**Whatever it shows, the small-message end is the one that governs our serving.** If
4 MB is latency-bound, peak bandwidth is the wrong target and `NCCL_IB_MERGE_NICS`,
buffer sizes and channel counts matter more than line rate.

## The transferable lesson

> Before treating a gap as a hardware defect, **match the independent variable.** A
> benchmark built to answer "does my workload run well" cannot be reused to answer
> "what is this hardware's peak" — and comparing across that boundary manufactures
> mysteries.

**Related:** [#11](../../issues/11) · [`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh) · [`POSTMORTEM-2026-08-25.md`](POSTMORTEM-2026-08-25.md)
