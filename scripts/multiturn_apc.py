#!/usr/bin/env python3
"""Multi-turn prefix-cache (APC) measurement: the warm path. Issue #29.

Every other bundle in this repo asserts cached_tokens == 0 and therefore
describes the COLD path. The production workload -- single-user interactive
coding -- is overwhelmingly multi-turn against a growing shared prefix. This
harness measures what a user actually feels on turns 2..N.

Method (from issue #29):

1. Turn 1: a large unique-salted context + a question. Asserted COLD
   (cached_tokens == 0). This is the cold reference TTFT.
2. Turns 2..N: append the previous answer and a short follow-up, resend the
   whole conversation. Asserted WARM (cached_tokens > 0). A warm turn that
   silently reports 0 cached tokens is a FAILURE, not a slow result -- the
   run records it, flags it, and exits nonzero, the same way the collapsed
   25-token window now fails loudly (issue #26).
3. Per turn: TTFT, cached_tokens, prompt_tokens, hit ratio
   (cached/prompt), decode tok/s over a pinned 256-token window.

--gap-s inserts think-time between turns: a cache that only survives
back-to-back turns does not help a human. Sweep it to find the retention
boundary (VLLM_PREFIX_CACHE_RETENTION_INTERVAL is 4096 in Profile B).

Inherited traps honoured here:
- unique salt per session so sessions do not warm each other;
- decode window pinned with min_tokens + ignore_eos and ASSERTED, so tok/s
  stays comparable with the depth-sweep bundles;
- warm every shape? NO -- deliberately not. Warm-up requests would populate
  the prefix cache and contaminate the cold reference. The JIT-stall trap is
  instead handled by reporting turn-1 TTFT from >= 3 sessions and taking the
  median; a JIT spike inflates one session, not the median of three.
- cache regime is DECLARED AND ASSERTED per record (cache_regime field), per
  the policy change issue #29 proposes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

# Same calibrated filler as decode_depth_sweep.py: repeated English prose
# tokenizes far denser than 4 chars/token on DeepSeek-V4's tokenizer.
_LOREM = (
    "The distributed inference engine partitions attention heads across nodes "
    "and exchanges activations over the fabric on every decode step. "
)
_CHARS_PER_TOKEN = 6.42

_FOLLOWUPS = [
    "Which components does the text say exchange data, and how often?",
    "List every distinct technical noun phrase the text uses.",
    "Does the text describe a single-node or multi-node system? Justify.",
    "Rewrite the first sentence of the text in passive voice, then comment.",
    "What failure modes would you predict for the system described?",
    "Summarise the entire conversation so far in one paragraph.",
    "What question has NOT been asked about this text yet? Ask and answer it.",
    "Contrast the described system with a single-GPU deployment.",
]


def build_context(target_tokens: int, salt: str) -> str:
    header = "[session " + salt + "] Read the following log excerpt.\n\n"
    body_chars = max(int(target_tokens * _CHARS_PER_TOKEN) - len(header), 0)
    reps = body_chars // len(_LOREM) + 1
    body = (_LOREM * reps)[:body_chars]
    return header + body + "\n\nDescribe what this text is, in detail."


def scrape_cache_counters(metrics_url: str) -> tuple:
    """(prefix_cache_queries_total, prefix_cache_hits_total) from /metrics.

    This build's per-request usage reports cached_tokens=0 even on an obvious
    hit (turn-3 TTFT 0.43s vs 10s cold in the smoke run) -- the OpenAI-layer
    plumbing is absent, while the engine-level Prometheus counters do track
    hits. On a single-user cluster with strictly sequential requests, the
    per-turn counter delta attributes exactly to that turn.
    """
    queries = hits = None
    with urllib.request.urlopen(metrics_url, timeout=10) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace")
            if line.startswith("vllm:prefix_cache_queries_total"):
                queries = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:prefix_cache_hits_total"):
                hits = float(line.rsplit(" ", 1)[1])
    if queries is None or hits is None:
        raise RuntimeError("prefix cache counters not found at " + metrics_url)
    return queries, hits


def run_turn(url: str, model: str, messages: list, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content = None
    completion_tokens = None
    prompt_tokens = None
    cached = 0
    content_parts = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached = details.get("cached_tokens", 0) or 0
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    if first_content is None:
                        first_content = time.perf_counter()
                    content_parts.append(piece)
    finished = time.perf_counter()
    if first_content is None:
        first_content = finished
    if completion_tokens is None:
        raise RuntimeError("server did not return completion token usage")
    if completion_tokens != max_tokens:
        raise RuntimeError(
            "decode window is %d tokens, expected %d (issue #26 guard)"
            % (completion_tokens, max_tokens))
    decode_s = max(finished - first_content, 1e-9)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "hit_ratio": round(cached / prompt_tokens, 4) if prompt_tokens else 0.0,
        "completion_tokens": completion_tokens,
        "ttft_s": round(first_content - started, 3),
        "decode_tok_s": round(completion_tokens / decode_s, 2),
        "answer": "".join(content_parts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="8192,32768,131072")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gap-s", type=float, default=0.0,
                    help="think-time inserted between turns")
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metrics-url", default=None,
                    help="default: <base-url without /v1>/metrics")
    args = ap.parse_args()
    metrics_url = args.metrics_url or (
        args.base_url.rstrip("/").rsplit("/v1", 1)[0] + "/metrics")

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    out = open(args.output, "a", buffering=1)
    violations = 0
    summary = []

    for depth in depths:
        cold_ttfts, warm_ttfts, warm_ratios = [], [], []
        for s in range(args.sessions):
            salt = "%s-g%g-s%d-%d-%d" % (
                args.label, args.gap_s, s, depth, int(time.time()))
            print("\n=== depth %d session %d (gap %gs, %s) ==="
                  % (depth, s, args.gap_s, salt), flush=True)
            messages = [{"role": "user",
                         "content": build_context(depth, salt)}]
            for t in range(args.turns):
                if t > 0 and args.gap_s > 0:
                    time.sleep(args.gap_s)
                try:
                    q0, h0 = scrape_cache_counters(metrics_url)
                    r = run_turn(args.base_url, args.model, messages,
                                 args.max_tokens)
                    q1, h1 = scrape_cache_counters(metrics_url)
                except Exception as exc:                  # noqa: BLE001
                    print("  turn %d FAILED: %s" % (t + 1, exc), flush=True)
                    break
                r["metrics_queried_tokens"] = int(q1 - q0)
                r["metrics_hit_tokens"] = int(h1 - h0)
                r["metrics_hit_ratio"] = round(
                    (h1 - h0) / (q1 - q0), 4) if q1 > q0 else 0.0
                regime = "cold" if t == 0 else "warm"
                # The regime must be what we meant to measure (issue #29):
                # cold turns must miss, warm turns must hit. Either way the
                # record is written -- a broken cache is a finding, but a
                # flagged one, never a silent one. Engine counters are the
                # authority; usage-level cached_tokens is recorded but known
                # broken (always 0) in this build.
                hit_tokens = max(r["cached_tokens"], r["metrics_hit_tokens"])
                regime_ok = (hit_tokens == 0) if t == 0 else (hit_tokens > 0)
                if not regime_ok:
                    violations += 1
                answer = r.pop("answer")
                r.update(label=args.label, target_depth=depth, session=s,
                         turn=t + 1, gap_s=args.gap_s,
                         cache_regime=regime, regime_ok=regime_ok)
                out.write(json.dumps(r) + "\n")
                flag = "" if regime_ok else "  <-- REGIME VIOLATION (%s turn, hit=%d)" % (
                    regime, hit_tokens)
                print("  turn %d [%s]: ttft=%.3fs hit=%d/%d (%.1f%%) decode=%.1f tok/s%s"
                      % (t + 1, regime, r["ttft_s"], r["metrics_hit_tokens"],
                         r["prompt_tokens"], 100 * r["metrics_hit_ratio"],
                         r["decode_tok_s"], flag), flush=True)
                if t == 0:
                    cold_ttfts.append(r["ttft_s"])
                else:
                    warm_ttfts.append(r["ttft_s"])
                    warm_ratios.append(r["metrics_hit_ratio"])
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user",
                                 "content": _FOLLOWUPS[t % len(_FOLLOWUPS)]
                                 + " (turn %d)" % (t + 2)})
        if cold_ttfts and warm_ttfts:
            cold_med = statistics.median(cold_ttfts)
            warm_med = statistics.median(warm_ttfts)
            summary.append({
                "label": args.label, "target_depth": depth,
                "gap_s": args.gap_s,
                "sessions": len(cold_ttfts),
                "warm_turns": len(warm_ttfts),
                "cold_ttft_median_s": round(cold_med, 3),
                "warm_ttft_median_s": round(warm_med, 3),
                "warm_ttft_max_s": round(max(warm_ttfts), 3),
                "speedup_x": round(cold_med / warm_med, 1) if warm_med else None,
                "warm_hit_ratio_median": round(
                    statistics.median(warm_ratios), 4),
            })

    out.close()
    with open(args.output.replace(".jsonl", "-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== SUMMARY %s (gap %gs) ===" % (args.label, args.gap_s), flush=True)
    for s in summary:
        print("  %7d tok: cold %7.2fs -> warm %6.3fs (%sx, hit %.1f%%)"
              % (s["target_depth"], s["cold_ttft_median_s"],
                 s["warm_ttft_median_s"], s["speedup_x"],
                 100 * s["warm_hit_ratio_median"]), flush=True)
    if violations:
        print("\n%d REGIME VIOLATION(S) -- see records. Exiting nonzero."
              % violations, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
