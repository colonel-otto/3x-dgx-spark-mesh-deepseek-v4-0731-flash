#!/usr/bin/env python3
"""Does prompt CONTENT change prefill? Same length, three content types,
each with a unique prefix so nothing is cached."""
import json,time,urllib.request,random,string
BASE="http://localhost:8100"; MODEL="deepseek-v4-flash-0731"
def post(p,timeout=900):
    b=json.dumps({"model":MODEL,"prompt":p,"max_tokens":1,"temperature":0}).encode()
    r=urllib.request.Request(BASE+"/v1/completions",data=b,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x: d=json.loads(x.read())
    return d["usage"]["prompt_tokens"], time.time()-t0
def uq(k=40): return "".join(random.choice(string.ascii_letters) for _ in range(k))
SENT=("Distributed inference on GB10 schedules prefill and decode in the same step; "
      "long prompts dominate the step budget and delay in-flight decodes. ")
WORDS=open('/usr/share/dict/words').read().split() if __import__('os').path.exists('/usr/share/dict/words') else None
def rand_digits(n): return " ".join(str(random.randint(100000,999999)) for _ in range(n))
def rand_words(n):
    pool=WORDS if WORDS else ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"]
    return " ".join(random.choice(pool) for _ in range(n))

for _ in range(2): post(uq()+" "+SENT*100)
print("warm\n",flush=True)
for rep in range(2):
    print(f"### REP {rep+1}",flush=True)
    # target ~24k tokens each
    cases=[("repeated_sentence", uq()+" "+SENT*1100),
           ("random_words",      uq()+" "+rand_words(18000)),
           ("random_digits",     uq()+" "+rand_digits(9200))]
    for name,p in cases:
        pt,dt=post(p)
        print(f"{name:<18} prompt_tok={pt:>7} sec={dt:>7.2f} tok/s={pt/dt:>8.0f}",flush=True)
print("DONE",flush=True)
