#!/usr/bin/env python3
"""Server-side prefill: measure vllm:prompt_tokens_total and the engine's own
iteration time around a single cold request. Excludes ALL client-side cost."""
import json,time,urllib.request,random,re

BASE="http://localhost:8100"; MODEL="deepseek-v4-flash-0731"
def metric(name):
    d=urllib.request.urlopen(BASE+"/metrics",timeout=20).read().decode()
    m=re.search(r'^%s\{[^}]*\}\s+([0-9.e+]+)'%re.escape(name),d,re.M)
    return float(m.group(1)) if m else None
def post(p,mt=1,timeout=900):
    b=json.dumps({"model":MODEL,"prompt":p,"max_tokens":mt,"temperature":0}).encode()
    r=urllib.request.Request(BASE+"/v1/completions",data=b,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x: d=json.loads(x.read())
    return d["usage"]["prompt_tokens"], time.time()-t0
def uq(n): return " ".join(str(random.randint(100000,999999)) for _ in range(n))

print("warming (3 shapes, discarded)",flush=True)
for w in (300,1200,3000): post(uq(w))
time.sleep(2)
print("warm done\n",flush=True)

for rep in range(3):
    print(f"### REP {rep+1}",flush=True)
    for w in (2400, 9600, 30000):
        p=uq(w)
        t_before=metric("vllm:prompt_tokens_total")
        pt, wall = post(p)
        time.sleep(1.5)
        t_after=metric("vllm:prompt_tokens_total")
        served = t_after - t_before
        print(f"words={w:>6} prompt_tok={pt:>7} served_delta={served:>8.0f} "
              f"client_wall={wall:>7.2f}s client_rate={pt/wall:>7.0f}",flush=True)
print("DONE",flush=True)
