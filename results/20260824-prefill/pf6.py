#!/usr/bin/env python3
"""Isolate fixed overhead: measure TTFT across a depth ladder including TINY prompts.
If a ~5s floor exists regardless of depth, the 'prefill rate' is really overhead."""
import json,time,urllib.request,random
URL="http://localhost:8100/v1/completions"; MODEL="deepseek-v4-flash-0731"
def ttft(p,timeout=900):
    b=json.dumps({"model":MODEL,"prompt":p,"max_tokens":1,"temperature":0,"stream":True}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x:
        for line in x:
            if line.startswith(b"data:") and b"[DONE]" not in line: return time.time()-t0
def uq(n): return " ".join(str(random.randint(100000,999999)) for _ in range(n))
for _ in range(2): ttft(uq(200))
print("warm",flush=True)
for rep in range(2):
    print(f"### REP {rep+1}",flush=True)
    for w in (10, 50, 200, 800, 2400, 9600):
        t=ttft(uq(w)); tok=w*2.6
        print(f"words={w:>6} ~tok={tok:>8.0f} ttft={t:>7.3f}s rate~{tok/t:>8.0f}",flush=True)
print("DONE",flush=True)
