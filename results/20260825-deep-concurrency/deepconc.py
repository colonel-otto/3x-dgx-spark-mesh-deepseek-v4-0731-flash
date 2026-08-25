#!/usr/bin/env python3
"""Deep-concurrency probe: N concurrent very-long prompts against one endpoint.

Re-runs the 2026-08-21 "4 x 200,000-token" test that issue #15 flags as suspect.
That original run was ad-hoc and left no script behind, so this file exists to
make the re-run reproducible. It reconstructs the recorded shape from
`benchmarks/README.md`:

  harness      bench-miaai
  prompt_shape synthetic-numbered-words
  sampling     temperature 0.6, top_p 0.95, min_tokens = max_tokens = 128,
               ignore_eos, thinking off
  decode_tok_s per-stream decode measured AFTER the first token

The question being asked is narrow: at four concurrent 200K prompts, does TTFT
still collapse to ~540 s, and do preemptions stay at 0? On the 2026-08-21 fabric
one node ran at ~15% of its sibling's collective bandwidth, which punishes
exactly the long-prefill path this test exercises.

Three things this harness does that the numbers depend on:

  * **Streams, so decode can be separated from prefill.** A non-streaming call
    only yields total latency, which at these depths is ~99% prefill and would
    report a decode rate that is really a prefill rate. The first SSE chunk
    timestamps TTFT; decode tok/s is the remaining tokens over the remaining
    wall.

  * **Defeats the prefix cache.** Each stream gets a unique nonce prefix AND a
    distinct filler rotation, so no two of the four concurrent prompts share a
    block-aligned prefix either. Re-running identical prompts returns cache hits
    (105,167 tok/s at 78K was measured that way) which is not prefill at all.
    `--verify-cache` reads the engine's own hit-rate counter and fails the run
    if it moved.

  * **Reads preemptions from the engine, not from inference.** The whole point
    of the original test was `vllm:num_preemptions_total`; it is sampled before
    and after and the delta is reported, so "preemptions 0" is a measurement
    rather than an assumption.

Usage:
    deepconc.py --tag tp3-2026xxxx [--concurrency 4] [--prompt-tokens 200000]
                [--host localhost:8100] [--json OUT]
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.request

# Ordinary technical prose, varied per line. Repeated text is highly
# compressible and takes a different prefill path, which would flatter the
# result; this mirrors the filler used by the KV-quality probe for consistency.
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

# The bench-miaai instruction, verbatim from benchmarks/README.md.
INSTRUCTION = ("Return exactly 128 numbered lowercase English words, then stop.")

MAX_TOKENS = 128


# Words-per-token for this filler, calibrated against the engine's own reported
# prompt_tokens: a 1.3 tokens/word estimate produced 185,529 tokens for a
# 200,000 request (0.928x). Measure, don't guess -- the depth is the independent
# variable of this whole test, so an 8% shortfall would not be a re-run of the
# 2026-08-21 rows at all.
TOKENS_PER_WORD = 1.3 / 0.928


def build_prompt(depth_tokens, nonce, rotation):
    """Build a prompt of approximately `depth_tokens` tokens.

    `rotation` offsets the filler cycle so two concurrent streams do not share a
    block-aligned prefix even after their distinct nonces.
    """
    target_words = int(depth_tokens / TOKENS_PER_WORD)
    lines, n = [f"Session nonce {nonce}. Ignore this line."], 0
    i = rotation
    while n < target_words:
        line = FILLER[i % len(FILLER)]
        lines.append(line)
        n += len(line.split())
        i += 1
    return "\n".join(lines) + "\n\n" + INSTRUCTION


def stream_one(base, model, prompt, out, idx):
    """One streaming request. Records TTFT and post-first-token decode rate."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "min_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})

    t0 = time.time()
    ttft = None
    tokens = 0
    usage = {}
    text = []
    try:
        # 3600 s: on the degraded fabric this test took 14.5 min; leave room for
        # a worse outcome rather than truncating the observation into an error.
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        tokens += 1
                        text.append(piece)
        wall = time.time() - t0
        # Decode rate AFTER the first token, matching the bench-miaai definition.
        decode_s = wall - ttft if ttft is not None else 0.0
        out.append({
            "i": idx,
            "wall_s": wall,
            "ttft_s": ttft,
            "chunks": tokens,
            "completion_tokens": usage.get("completion_tokens", tokens),
            "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tok_s": ((usage.get("completion_tokens", tokens) - 1) / decode_s
                             if decode_s > 0 else None),
            "sample": "".join(text)[:200],
        })
    except Exception as exc:                # noqa: BLE001 - report, don't kill the sweep
        out.append({"i": idx, "wall_s": time.time() - t0, "ttft_s": None,
                    "completion_tokens": 0, "decode_tok_s": None,
                    "error": f"{type(exc).__name__}: {exc}"})


def metrics(base):
    """Engine counters we care about, as floats keyed by bare metric name."""
    want = ("vllm:num_preemptions_total",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_queries_total",
            "vllm:gpu_prefix_cache_hits_total")
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:                       # noqa: BLE001
        return {}
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:[a-z_]+)\{[^}]*\}\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in want:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost:8100")
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--prompt-tokens", type=int, default=200000)
    p.add_argument("--tag", required=True,
                   help="label recorded in the JSON, e.g. tp3-postfabricfix")
    p.add_argument("--json", default=None)
    p.add_argument("--allow-busy", action="store_true",
                   help="run even if the endpoint already has traffic")
    a = p.parse_args()

    base = f"http://{a.host}"
    model = a.model
    if model is None:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]

    before = metrics(base)
    busy = (before.get("vllm:num_requests_running", 0)
            + before.get("vllm:num_requests_waiting", 0))
    if busy and not a.allow_busy:
        sys.exit(f"endpoint is not idle ({busy:g} in flight); "
                 f"a stray request skews this badly. Re-run when quiet "
                 f"or pass --allow-busy.")

    # Unique nonce per stream AND per run, so nothing can hit the prefix cache.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prompts = [build_prompt(a.prompt_tokens, f"{a.tag}-{stamp}-s{i}", i * 3)
               for i in range(a.concurrency)]

    print(f"[{stamp}] {a.tag}: {a.concurrency} x ~{a.prompt_tokens} tok, "
          f"model={model}", flush=True)

    out, threads = [], []
    t_start = time.time()
    for i, prompt in enumerate(prompts):
        t = threading.Thread(target=stream_one,
                             args=(base, model, prompt, out, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t_start

    after = metrics(base)
    preempt_delta = (after.get("vllm:num_preemptions_total", 0)
                     - before.get("vllm:num_preemptions_total", 0))

    def cache_delta(prefix):
        q = after.get(f"vllm:{prefix}_queries_total", 0) - \
            before.get(f"vllm:{prefix}_queries_total", 0)
        h = after.get(f"vllm:{prefix}_hits_total", 0) - \
            before.get(f"vllm:{prefix}_hits_total", 0)
        return {"queries": q, "hits": h}

    ok = [r for r in out if r.get("decode_tok_s")]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    decodes = [r["decode_tok_s"] for r in ok]
    total_out = sum(r.get("completion_tokens", 0) for r in out)

    result = {
        "tag": a.tag,
        "timestamp_utc": stamp,
        "model": model,
        "concurrency": a.concurrency,
        "prompt_tokens_requested": a.prompt_tokens,
        "prompt_tokens_actual": [r.get("prompt_tokens") for r in
                                 sorted(out, key=lambda r: r["i"])],
        "wall_s": wall,
        "ttft_s_median": statistics.median(ttfts) if ttfts else None,
        "ttft_ms_median": statistics.median(ttfts) * 1000 if ttfts else None,
        "decode_tok_s_median": statistics.median(decodes) if decodes else None,
        "aggregate_tok_s": total_out / wall if wall else None,
        "preemptions_delta": preempt_delta,
        "prefix_cache": {"cpu": cache_delta("prefix_cache"),
                         "gpu": cache_delta("gpu_prefix_cache")},
        "errors": [r for r in out if "error" in r],
        "streams": sorted(out, key=lambda r: r["i"]),
    }

    print(json.dumps({k: v for k, v in result.items() if k != "streams"},
                     indent=2), flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
