#!/usr/bin/env python3
"""Raw GPU matmul throughput - is the silicon itself slow?"""
import torch, time
torch.cuda.set_device(0)
for dtype,name in ((torch.bfloat16,"bf16"),):
    for n in (4096, 8192):
        a=torch.randn(n,n,dtype=dtype,device='cuda'); b=torch.randn(n,n,dtype=dtype,device='cuda')
        for _ in range(3): c=a@b
        torch.cuda.synchronize()
        t0=time.perf_counter()
        REP=20
        for _ in range(REP): c=a@b
        torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/REP
        tflops=2*n**3/dt/1e12
        print(f"{name} {n}x{n} matmul: {dt*1000:7.2f} ms  {tflops:7.2f} TFLOP/s",flush=True)
print("clock:",torch.cuda.clock_rate() if hasattr(torch.cuda,'clock_rate') else 'n/a')
