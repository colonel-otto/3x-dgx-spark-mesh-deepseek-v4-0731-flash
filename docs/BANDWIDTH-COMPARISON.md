# The bandwidth gap: what it is, and what it is not

> [!IMPORTANT]
> **This page was rewritten 2026-08-25 after source verification refuted its first
> version.** That version argued the gap was mostly a metric-convention artifact.
> **It was wrong** — published DGX Spark figures use the *same* collective and the
> *same* formula we do. The gap is real. Corrected analysis below.

**The gap is ~3.2x at comparable size and rank count, and it is not a bookkeeping error.**

> [!WARNING]
> **It is also not yet a controlled comparison.** Four variables still differ from the
> published run -- bootstrap interface, harness, NIC-merge setting, and HCA discovery.
> The one public 3-Spark result began at **2.86 GB/s**, almost exactly our original
> number, and recovered to 18.64 by changing **how the job bootstraps**, not the fabric.
> Before treating 3.2x as a hardware deficit, run
> [`BANDWIDTH-NEXT-TEST.md`](BANDWIDTH-NEXT-TEST.md).

---

## Verified: our formula matches theirs exactly

From [nccl-tests `doc/PERFORMANCE.md`](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md):

```
algbw = S/t
AllGather:  busbw = algbw * (n-1)/n
AllReduce:  busbw = algbw * 2*(n-1)/n     <- the 2x applies ONLY to all_reduce
```

Our harness computes `nbytes * (world-1)/world / dt` — the AllGather definition, exactly.
And the DGX Spark community overwhelmingly publishes **`all_gather_perf` busbw**, not
all_reduce. **There is no factor-of-2 to reconcile.**

## The closest apples-to-apples comparison

[Forum 365160](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) — **3-Spark ring**, `all_gather_perf`, same topology as ours:

| | message | busbw |
|---|---:|---:|
| forum, 3-rank ring | 32 MB | **18.70 GB/s** |
| forum, 3-rank ring | 16 GB | 20.75 |
| **ours, 3-rank ring** | 67 MB | **5.80** |

**~3.2x at comparable size.** Critically, that reporter ran **`NCCL_IB_MERGE_NICS=0`** —
so the deficit is not purely an HCA-count story.

## Corrected: the real ceiling is ~24 GB/s, not 48.5

An earlier estimate here — "4 HCAs = 400 Gb/s line = 48.5 GB/s theoretical" — is **not
supported and is contradicted by the PCIe evidence.** From
[forum 350417 post 61](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417/61):

> "both 200G physical ports are sharing **two PCIe Gen5 x4 lanes**"
> "The theoretical max of a single x4 lane is around 126 Gbit/sec. The theoretical max of
> using both interfaces is **252 Gbit/sec**."

Measured raw RDMA tops out near 196 Gb/s ≈ **24.5 GB/s**, and the best observed
`all_gather` busbw (23.74–24.32) essentially saturates it. **PCIe binds before the wire
does.** The realistic target is ~24 GB/s, which makes 18–21 near-optimal rather than
exotic.

## Where our number sits

| config | ours | published comparison |
|---|---:|---|
| 2-rank | 9.70 | 18.92 @64MiB (2 HCAs), 13.49 @16G (**1 HCA**) |
| 3-rank ring | 5.80 | 18.70 @32MB |

**Our 2-rank sits below even the published single-HCA figure**, and our 3-rank gap (3.2x)
is *worse* than our 2-rank gap (2.0x). A deficit that grows with rank count points at
**ring topology handling**, not raw link speed.

## One hypothesis within the merge question: HCA *pairing*, not *count*

> Note this is **not** the leading explanation. The strongest lead is bootstrap topology
> — see [`BANDWIDTH-NEXT-TEST.md`](BANDWIDTH-NEXT-TEST.md). Pairing is one candidate
> mechanism inside the NIC-merge variable.

Every working published config merges the **two PCIe domains of the same physical port**:

```
NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1        # note: both "f1"
NCCL_IB_MERGE_NICS=1
NCCL_IB_GID_INDEX=3
```

Per forum 350417 the interfaces enumerate as *port0-half1, port1-half1, port0-half2,
port1-half2*, and aggregation should be built **between the halves of the same port.**

Our own `NCCL_DEBUG` (2-HCA era) shows the opposite grouping:

```
NET/IB : Made virtual device [2] name=rocep1s0f0+rocep1s0f1 speed=400000 ndevs=2
```

That merges `f0` with `f1` — **two different physical ports.** On a 2-node setup both
ports face the same peer, so it is harmless. **On a 3-node ring, port 0 faces one
neighbour and port 1 faces the other**, so the merged "400G" virtual device spans two
different peers. NCCL believes it has a pipe to each neighbour that does not exist.

That is a coherent mechanism for a *ring-specific* deficit, and it predicts exactly what
we observe: the 3-rank gap being worse than the 2-rank gap.

**We have never set `NCCL_IB_MERGE_NICS` or `NCCL_IB_GID_INDEX`, and we have never tested
same-port pairing.** Note the forum's *working* ring ran `MERGE_NICS=0` -- merging off
entirely -- which is a third possibility we also never tried.

Our 4-HCA gate recorded `via NET/IB/4 via NET/IB/5`. With four physical HCAs at indices
0-3, **4 and 5 are two merged virtual devices**, so automatic merging was on. Which HCAs
it grouped, the gate did not record. It does now (`vdev:*`).

### On the existing "MERGE_NICS is falsified" note

[`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) §2 records MERGE_NICS as a no-op —
"measured, both ways, same day — do not re-open." **That measurement ran at 0.47–0.52
GB/s**, i.e. entirely on the degraded fabric, where one bad link paced everything and no
flag could have shown a difference. It does not constrain healthy hardware, and it never
tested same-port pairing. **Re-open it.**

## What message size does and does not explain

Size matters, but far less than this page first claimed. route179's curve (2 ranks,
2 HCAs merged):

| message | busbw |
|---|---:|
| 512 MB | 18.39 |
| 1 GB | 19.93 |
| 16 GB | 23.74 |

**+29% from 512 MB to 16 GB** — real, but nowhere near 3x. And the forum's 3-rank ring
reached **18.70 at 32 MB**, a *smaller* message than our 67 MB point. **Size does not
rescue us.**

Still worth fixing the methodology: our sweep tops out at 67 MB because `agbench.py` was
built to mirror the vLLM MTP allgather shape (`seqs16` = 4.14 MB), not to find peak
bandwidth. [`../scripts/bwsweep.py`](../scripts/bwsweep.py) extends to 4 GiB and prints
all three conventions per line.

## Confirmed: no GPUDirect RDMA, and it is a fixed tax

`GPU Direct RDMA Disabled for HCA 0/1` appears in our logs and in every published one.
NVIDIA states it is not implemented on Spark. Every byte bounces through system memory
over PCIe, and independent reporting attributes the gap between raw RDMA (~24.6 GB/s) and
NCCL (~9–10 GB/s) to exactly this. **No flag removes it** — but it applies to the
published numbers too, so it does not explain a gap *against* them.

## What we got wrong

1. **Claimed the gap was a convention artifact.** It is not — same collective, same
   formula, verified against nccl-tests source.
2. **Estimated a 48.5 GB/s ceiling from line rate.** PCIe Gen5 x4 shared across both ports
   caps it near 24.
3. **Over-weighted message size.** Worth ~29%, not 3x. The forum's 3-rank number at a
   *smaller* size than ours refutes the size explanation directly.
4. **Never tested the pairing the working configs use.** We merge across ports; they merge
   within a port. On a ring those are not equivalent.
5. **Treated a degraded-fabric measurement as a permanent falsification.** The MERGE_NICS
   result was taken at 0.5 GB/s, where nothing could have shown an effect.

**The one thing that survives from the first version:** always state collective, message
size, rank count and convention when quoting a bandwidth number. That discipline is what
made this correction possible.

## The test

**Full plan: [`BANDWIDTH-NEXT-TEST.md`](BANDWIDTH-NEXT-TEST.md).** In priority order:

1. **Bootstrap** over a common management interface (the change that took the one public
   3-Spark result from 2.86 to 18.64).
2. **`NCCL_IB_MERGE_NICS=0`** vs default — their working ring ran it *off*.
3. **Official MPI `nccl-tests all_gather_perf`** at 32 MiB and 16 GiB, so the harness stops
   being a variable.
4. **Automatic HCA discovery** vs our explicit four-HCA list.

Read `Made virtual device … name=` from `NCCL_DEBUG=INFO` on every run — the gate now
records it as `vdev:*`. Do not infer the grouping from the env var.

## Sources

- [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) — the formulas
- [Forum 365160](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) — 3-rank ring, 18.70 @32MB, MERGE_NICS=0
- [Forum 350417](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417/61) — PCIe lanes, enumeration, no GPUDirect
- [Forum 368025](https://forums.developer.nvidia.com/t/nccl-all-gather-performance-halved-on-dual-spark-setup-connectx-7-after-msi-firmware-update-solved-via-downgrade/368025) — 13.49 (1 HCA) vs 24.32 (2 HCA)
- [route179.dev](https://route179.dev/2026/07/21/dgx-spark-nccl-roce-benchmarking/) — size curve, merge config
- [note.com tsuru_mitsu](https://note.com/gb10_tsurumitsu/n/n1c5efc62a92e?hl=en) — 18.92 @64MiB, raw RDMA ceiling

**Not verified:** NVIDIA publishes **no** numeric NCCL target for DGX Spark — the
clustering doc and the nccl playbook both contain none. Every 18–21 figure is
community-measured. No published measurement uses all four HCAs.

**Related:** [#18](../../issues/18) · [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) · [`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh)
