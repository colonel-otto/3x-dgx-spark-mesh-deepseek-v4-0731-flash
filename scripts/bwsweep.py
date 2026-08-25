#!/usr/bin/env python3
"""Fabric bandwidth sweep to the asymptote -- for comparing against published numbers.

WHY THIS EXISTS
---------------
`results/20260824-seqs32-nccl/agbench.py` tops out at a 67 MB input because it was
built to mirror the vLLM MTP allgather shape (max_num_seqs x vocab = 4-8 MB). That
is the right size for "does our workload run well". It is the WRONG size for
"what is this fabric's peak bandwidth" -- 67 MB is nowhere near the asymptote, so
its top point systematically understates peak.

We then compared that top point against a forum figure measured with a **16 GB**
allgather (238x larger) and concluded we were 3-6x slow. That comparison was not
valid. This script exists so the comparison can be made at matched size.

CONVENTIONS -- STATE THESE WHENEVER YOU QUOTE A NUMBER
-----------------------------------------------------
For all_gather with input `n` bytes per rank over `w` ranks:

    algbw = n / dt                      # "how fast my input moved"
    busbw = n * (w-1)/w / dt            # "wire rate each NIC sustains"

nccl-tests all_reduce_perf busbw carries an EXTRA factor of 2, because ring
all-reduce is reduce-scatter + all-gather. So the SAME wire speed reads as:

    all_gather busbw  <  all_gather algbw  <=  all_reduce busbw

At w=3 that is a 2x spread; at w=2 the algbw and all_reduce-busbw numbers coincide.
A published figure without its collective and convention cannot be compared to.

USAGE
-----
One process per node, engine STOPPED (needs tens of GiB of buffer):

    RANK=$i WORLD_SIZE=3 INIT_METHOD=tcp://<rank0-fabric>:29555 \
      TAG=4hca python3 scripts/bwsweep.py

Set MAX_GIB to bound the largest message (default 4).
"""
import os, sys, time, json, datetime
import torch, torch.distributed as dist

rank  = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
tag   = os.environ.get("TAG", "run")
outdir = os.environ.get("OUTDIR", ".")
max_gib = float(os.environ.get("MAX_GIB", "4"))

dist.init_process_group("nccl", init_method=os.environ["INIT_METHOD"],
                        rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=600))
torch.cuda.set_device(0)

VOCAB = 129280
# Small end: the shapes vLLM actually issues. Large end: toward the asymptote.
SIZES = [("mtp-seqs16", 16 * VOCAB), ("mtp-seqs32", 32 * VOCAB),
         ("4MiB",   2 * 1024**2), ("16MiB",  8 * 1024**2),
         ("64MiB",  32 * 1024**2), ("256MiB", 128 * 1024**2),
         ("1GiB",   512 * 1024**2), ("2GiB", 1024 * 1024**2),
         ("4GiB",  2048 * 1024**2), ("8GiB", 4096 * 1024**2)]

out = []
for name, numel in SIZES:
    in_bytes = numel * 2                       # bfloat16
    if in_bytes / 1024**3 > max_gib:
        if rank == 0:
            print(f"{name:12s} SKIP (input {in_bytes/1024**3:.1f} GiB > MAX_GIB={max_gib})",
                  flush=True)
        continue
    try:
        src = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
        dst = torch.empty(numel * world, dtype=torch.bfloat16, device="cuda")
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(f"{name:12s} SKIP (OOM allocating {in_bytes*(world+1)/1e9:.1f} GB)", flush=True)
        torch.cuda.empty_cache()
        continue

    # Big messages need fewer reps; small ones need more to beat the timer.
    reps = 20 if in_bytes < 64 * 1024**2 else (10 if in_bytes < 1024**3 else 5)
    for _ in range(3):
        dist.all_gather_into_tensor(dst, src)
    torch.cuda.synchronize(); dist.barrier()

    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_gather_into_tensor(dst, src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps

    algbw = in_bytes / dt / 1e9
    busbw = in_bytes * (world - 1) / world / dt / 1e9
    # What nccl-tests all_reduce_perf would print at this same wire speed.
    ar_busbw_equiv = 2 * busbw

    if rank == 0:
        print(f"{name:12s} in={in_bytes/1e6:9.2f} MB  {dt*1000:9.2f} ms  "
              f"algbw={algbw:6.2f}  busbw={busbw:6.2f}  "
              f"[allreduce-busbw-equiv={ar_busbw_equiv:6.2f}] GB/s", flush=True)
        out.append(dict(name=name, numel=numel, in_bytes=in_bytes, ms=dt * 1000,
                        algbw_GBs=algbw, busbw_GBs=busbw,
                        allreduce_busbw_equiv_GBs=ar_busbw_equiv, reps=reps))
    del src, dst
    torch.cuda.empty_cache()

if rank == 0 and out:
    peak = max(out, key=lambda r: r["busbw_GBs"])
    print(f"\nPEAK busbw {peak['busbw_GBs']:.2f} GB/s at {peak['name']} "
          f"(algbw {peak['algbw_GBs']:.2f}, allreduce-equiv {peak['allreduce_busbw_equiv_GBs']:.2f})",
          flush=True)
    path = os.path.join(outdir, f"bwsweep_{tag}_w{world}.json")
    json.dump(dict(world=world, tag=tag, collective="all_gather",
                   dtype="bfloat16", results=out), open(path, "w"), indent=1)
    print("wrote", path, flush=True)

dist.destroy_process_group()
