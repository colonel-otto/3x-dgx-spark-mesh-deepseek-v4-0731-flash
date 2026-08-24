#!/usr/bin/env python3
import json, time, urllib.request, sys

BASE="http://localhost:8100"
MODEL="deepseek-v4-flash-0731"

def metrics():
    txt=urllib.request.urlopen(BASE+"/metrics",timeout=30).read().decode()
    out={}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        for key in ("vllm:spec_decode_num_drafts_total",
                    "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(key):
                out[key]=float(line.rsplit(" ",1)[1])
    return out

PROMPTS={
 "code":"Implement a binary search tree in Python with insert, search, delete and in-order traversal. Include docstrings and two usage examples. Code only.",
 "prose":"Write a 200-word story about an engineer debugging a distributed system at 3am.",
 "json":'Output a JSON array of 60 objects, each exactly {"id":N,"name":"user_N","active":true}. JSON only.',
 "count":"Print the numbers 1 to 300, one per line, exact format N. No commentary.",
}

k=int(sys.argv[1]) if len(sys.argv)>1 else 5
print(f"{'type':<8}{'tok/s':>8}{'accept%':>9}{'len/'+str(k):>8}")
rows={}
for name,p in PROMPTS.items():
    b=metrics()
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":p}],
                     "max_tokens":400,"temperature":0}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=body,
                               headers={"Content-Type":"application/json"})
    t0=time.time()
    d=json.loads(urllib.request.urlopen(req,timeout=600).read())
    el=time.time()-t0
    a=metrics()
    ct=d["usage"]["completion_tokens"]
    dr=a["vllm:spec_decode_num_draft_tokens_total"]-b["vllm:spec_decode_num_draft_tokens_total"]
    ac=a["vllm:spec_decode_num_accepted_tokens_total"]-b["vllm:spec_decode_num_accepted_tokens_total"]
    nd=a["vllm:spec_decode_num_drafts_total"]-b["vllm:spec_decode_num_drafts_total"]
    rows[name]={"tok_s":ct/el,"accept_pct":100*ac/dr if dr else 0,"accept_len":ac/nd if nd else 0}
    print(f"{name:<8}{ct/el:>8.1f}{100*ac/dr if dr else 0:>9.1f}{ac/nd if nd else 0:>8.2f}")
print("JSON "+json.dumps(rows))
