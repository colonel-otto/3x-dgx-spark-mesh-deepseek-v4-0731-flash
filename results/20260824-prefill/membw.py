#!/usr/bin/env python3
"""GB10 unified memory bandwidth - prefill is bandwidth-bound."""
import torch, time
torch.cuda.set_device(0)
N = 512*1024*1024//2   # 512MB of bf16
a = torch.empty(N, dtype=torch.bfloat16, device='cuda')
b = torch.empty(N, dtype=torch.bfloat16, device='cuda')
a.fill_(1.0)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize()
t0=time.perf_counter(); REP=20
for _ in range(REP): b.copy_(a)
torch.cuda.synchronize()
dt=(time.perf_counter()-t0)/REP
gb = N*2*2/1e9   # read+write
print(f"copy 512MB bf16: {dt*1000:6.2f} ms -> {gb/dt:7.1f} GB/s effective")
# read-only reduction
for _ in range(3): s=a.sum()
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REP): s=a.sum()
torch.cuda.synchronize()
dt2=(time.perf_counter()-t0)/REP
print(f"sum  512MB bf16: {dt2*1000:6.2f} ms -> {N*2/1e9/dt2:7.1f} GB/s read")
