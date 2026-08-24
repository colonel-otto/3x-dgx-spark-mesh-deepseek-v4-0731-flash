#!/usr/bin/env python3
"""Probe NVFP4 KV-cache output quality as context depth grows.

This is a CORRECTNESS test, not a benchmark. The upstream 2-node repo warns that
4-bit KV "can collapse into salad under long, heavy agentic context" while fp8 KV
stays clean. Throughput is deliberately not the subject here.

Two independent signals per depth, so a failure is not a matter of taste:

  1. Needle retrieval -- a unique fact is buried at a known fraction of the
     context. The model either reproduces the exact token or it does not. Three
     needles (early/middle/late) catch position-dependent degradation, which is
     the shape KV-cache damage would take.

  2. Garble detection -- CJK characters, replacement chars, stray BOS/special
     tokens, and pathological repetition in the *reply*. This is the specific
     "multilingual / BOS salad" failure mode described upstream.

Usage: kvquality.py <depth_tokens> [<depth_tokens> ...]
"""
import json, re, sys, time, urllib.request

BASE = "http://localhost:8100"
MODEL = "deepseek-v4-flash-0731"

# Filler is ordinary technical prose that does not mention any needle value, so a
# correct answer cannot be guessed from context. Varied sentences avoid the
# highly-compressible repeated-text path that distorts prefill measurements.
FILLER = [
    "The scheduler batches prefill and decode work into a single engine step.",
    "Each attention layer writes its keys and values into the paged cache.",
    "Block tables map logical positions onto physical cache blocks.",
    "Speculative decoding proposes several tokens before the target verifies them.",
    "Tensor parallelism shards attention heads across the participating devices.",
    "The router selects a small subset of experts for every token it processes.",
    "Quantized weights are dequantized inside the kernel just before the matmul.",
    "Prefix caching lets a shared conversation prefix skip recomputation entirely.",
    "Continuous batching admits new requests without draining the current batch.",
    "The sampler applies penalties and temperature after the logits are produced.",
]

NEEDLES = [
    (0.10, "MAGENTA-HERON-4417"),
    (0.50, "OBSIDIAN-LANTERN-8352"),
    (0.90, "CRIMSON-FULCRUM-2096"),
]

CJK = re.compile(r"[　-鿿가-힯]")
SPECIAL = re.compile(r"<\|.*?\|>|�|<pad>|<s>|</s>")


def build(depth_tokens):
    """~1.3 tokens per word for this filler; overshoot then trim by word count."""
    target_words = int(depth_tokens / 1.3)
    lines, n = [], 0
    while n < target_words:
        lines.append(FILLER[len(lines) % len(FILLER)])
        n += len(lines[-1].split())
    for frac, val in NEEDLES:
        idx = min(int(len(lines) * frac), len(lines) - 1)
        lines.insert(idx, f"Note carefully: the access code at this point is {val}.")
    body = "\n".join(lines)
    q = ("\n\nQuestion: three access codes appear above. List all three, exactly as "
         "written, one per line, and nothing else.")
    return body + q


def ask(prompt, max_tokens=200):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return (d["choices"][0]["message"]["content"] or "",
            d["usage"]["prompt_tokens"], time.time() - t0)


def garble(text):
    flags = []
    if CJK.search(text):
        flags.append("CJK")
    if SPECIAL.search(text):
        flags.append("SPECIAL")
    words = text.split()
    if len(words) > 12:
        for w in set(words):
            if len(w) > 3 and words.count(w) > len(words) * 0.35:
                flags.append(f"REPEAT:{w[:16]}")
                break
    return flags


for depth in [int(x) for x in sys.argv[1:]]:
    prompt = build(depth)
    try:
        reply, ptok, el = ask(prompt)
    except Exception as e:
        print(f"depth~{depth}: REQUEST FAILED: {str(e)[:120]}", flush=True)
        continue
    found = [v for _, v in NEEDLES if v in reply]
    missing = [v for _, v in NEEDLES if v not in reply]
    g = garble(reply)
    status = "PASS" if len(found) == 3 and not g else "FAIL"
    print(f"depth~{depth:>7} | {ptok:>7} prompt tok | {el:6.1f}s | "
          f"needles {len(found)}/3 | garble {','.join(g) if g else 'none'} | {status}",
          flush=True)
    if missing:
        print(f"    missing: {', '.join(missing)}", flush=True)
    if status == "FAIL":
        print(f"    reply[:300]: {reply[:300]!r}", flush=True)
