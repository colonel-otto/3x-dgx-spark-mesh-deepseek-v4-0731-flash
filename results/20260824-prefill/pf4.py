#!/usr/bin/env python3
"""Replicate upstream bench_full.py prefill EXACTLY (ascending, prefix-caching on),
but with a fresh unique base per pass so each pass is a genuine cold first run.
This is the apples-to-apples comparison against their published 1513/2284/2639."""
import json, time, urllib.request, random, string

URL="http://localhost:8100/v1/chat/completions"; MODEL="deepseek-v4-flash-0731"
FILLER=("Distributed inference on GB10 schedules prefill and decode in the same step; "
        "long prompts dominate the step budget and delay in-flight decodes. ")

def post(p, mt, timeout=900):
    b=json.dumps({"model":MODEL,"messages":[{"role":"user","content":p}],
                  "max_tokens":mt,"temperature":0}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x: d=json.loads(x.read())
    return d["usage"]["completion_tokens"], d["usage"].get("prompt_tokens",0), time.time()-t0

def uniq(k=32): return "".join(random.choice(string.ascii_letters) for _ in range(k))

print("warming",flush=True)
for _ in range(3): post("Write a python function that adds two numbers. Code only.",100)
print("warm done",flush=True)

for rep in range(3):
    base = uniq(48)   # fresh per pass => pass is cold, but 8K IS a prefix of 32K/100K within the pass
    print(f"### PASS {rep+1} (upstream ascending method, cold base)",flush=True)
    for target in (8000,32000,100000):
        n=max(1,int(target/22))
        prompt = base + " " + (FILLER*n)[:target*4] + "\nSummarize in one sentence."
        ct,pt,dt = post(prompt,1)
        print(f"target={target:>7} prompt_tok={pt:>7} sec={dt:>7.2f} tok/s={pt/dt:>8.0f}",flush=True)
print("DONE",flush=True)
