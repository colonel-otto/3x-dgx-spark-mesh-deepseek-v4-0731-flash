#!/usr/bin/env python3
"""Prefill, cache-free: every prompt has a unique prefix so prefix caching cannot hit."""
import json, time, urllib.request, random, string

URL = "http://localhost:8100/v1/chat/completions"
MODEL = "deepseek-v4-flash-0731"
FILLER = ("Distributed inference on GB10 schedules prefill and decode in the same step; "
          "long prompts dominate the step budget and delay in-flight decodes. ")

def post(prompt, max_tokens, timeout=900):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    u = d["usage"]
    return u["completion_tokens"], u.get("prompt_tokens", 0), dt

def uniq(k=24):
    return "".join(random.choice(string.ascii_letters) for _ in range(k))

print("warming", flush=True)
for _ in range(3):
    post("Write a python function that adds two numbers. Code only.", 100)
print("warm done", flush=True)

for rep in range(3):
    print(f"### REP {rep+1}", flush=True)
    for target in (8000, 32000, 100000):
        n = max(1, int(target / 22))
        # unique token salad at the FRONT defeats prefix caching
        prompt = uniq(64) + " " + (FILLER * n)[:target * 4] + f"\nRef {uniq()}. Summarize in one sentence."
        try:
            ct, pt, dt = post(prompt, 1)
            print(f"target={target:>7} prompt_tok={pt:>7} sec={dt:>7.2f} tok/s={pt/dt:>8.0f}", flush=True)
        except Exception as e:
            print(f"target={target:>7} FAILED {str(e)[:60]}", flush=True)
print("DONE", flush=True)
