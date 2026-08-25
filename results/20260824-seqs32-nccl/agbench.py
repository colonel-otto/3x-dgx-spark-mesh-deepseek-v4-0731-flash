#!/usr/bin/env python3
"""Allgather A/B on the Spark mesh, mirroring the shape that hung vLLM.

Reproduces WorkNCCL(SeqNum=3474, _ALLGATHER_BASE, NumelIn=8282112) --
max_num_seqs * vocab_size for DeepSeek MTP -- plus a size sweep.
Run one process per node; env MERGE tags the output.
"""
import os, sys, time, json
import torch, torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
tag = os.environ.get("TAG", "run")
dist.init_process_group("nccl", init_method=os.environ["INIT_METHOD"],
                        rank=rank, world_size=world,
                        timeout=__import__("datetime").timedelta(seconds=120))
torch.cuda.set_device(0)

VOCAB = 129280
SIZES = [("seqs8",  8 * VOCAB), ("seqs16", 16 * VOCAB),
         ("seqs32", 32 * VOCAB), ("seqs64", 64 * VOCAB),
         ("16MiB", 8 * 1024 * 1024), ("64MiB", 32 * 1024 * 1024)]
out = []
for name, numel in SIZES:
    src = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(numel * world, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):                      # warm up
        dist.all_gather_into_tensor(dst, src)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    REPS = 10
    for _ in range(REPS):
        dist.all_gather_into_tensor(dst, src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / REPS
    nbytes = numel * 2
    busbw = (nbytes * (world - 1) / world) / dt / 1e9
    algbw = nbytes / dt / 1e9
    if rank == 0:
        print(f"{name:8s} numel={numel:>10d} {nbytes/1e6:8.2f} MB  "
              f"{dt*1000:8.2f} ms  algbw={algbw:6.2f} GB/s  busbw={busbw:6.2f} GB/s",
              flush=True)
        out.append(dict(name=name, numel=numel, bytes=nbytes, ms=dt*1000,
                        algbw_GBs=algbw, busbw_GBs=busbw))
if rank == 0:
    json.dump(out, open(f"/home/sparkmain/results/seqs32/ag_{tag}.json", "w"), indent=1)
    print("OK", flush=True)
dist.destroy_process_group()
