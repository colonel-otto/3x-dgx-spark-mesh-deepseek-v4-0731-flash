#!/usr/bin/env python3
"""Server-side prefill rate, the anemll method: unique token ids to defeat prefix
caching, and TTFT measured from a STREAMING response so the timer stops at the first
token instead of after the whole HTTP round trip."""
import json, time, urllib.request, random

URL="http://localhost:8100/v1/completions"; MODEL="deepseek-v4-flash-0731"

def ttft(prompt, timeout=900):
    b=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":1,"temperature":0,
                  "stream":True}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x:
        for line in x:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                return time.time()-t0
    return None

def uniq_tokens(n):
    # random digit groups -> unique token ids, no prefix sharing at all
    return " ".join(str(random.randint(100000,999999)) for _ in range(n))

print("warming",flush=True)
for _ in range(2): ttft(uniq_tokens(500))
print("warm done",flush=True)

for rep in range(3):
    print(f"### REP {rep+1}",flush=True)
    for approx in (2000, 8000, 24000, 78000):
        words = int(approx/2.6)     # ~2.6 tokens per 6-digit group + space
        p = uniq_tokens(words)
        t = ttft(p)
        # ask server for the real prompt token count via a non-stream 1-token call is
        # costly; approximate from words and verify once below
        print(f"approx_tok={approx:>6} words={words:>6} ttft={t:>7.2f}s rate~{approx/t:>8.0f} tok/s",flush=True)
print("DONE",flush=True)
