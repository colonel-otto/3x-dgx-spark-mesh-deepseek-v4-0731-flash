#!/usr/bin/env python3
"""Long-context decode sweep: per-stream tok/s vs context depth.

Answers "does the third node still help when the prompt is large?" -- the
question the 2026-08-25 healthy-fabric re-run left open, because it used an
18-token prompt.

Three constraints this harness exists to satisfy, all learned the hard way:

1. PREFIX CACHE. Ascending depths where each prompt is a prefix of the next
   make every run after the first a cache hit, which measures nothing. Every
   prompt here gets a unique header, and cached_tokens is checked on every run.

2. JIT STALL TAIL. TileLang/CuTeDSL compile kernels DURING inference the first
   time they see a shape, costing ~5 s. Landing one inside a ~3 s run
   manufactures a 20% difference that is not real. So: warm every shape before
   measuring it, take >= 7 reps, and report the MEDIAN, never the mean.

3. DECODE, NOT END-TO-END. tok/s is measured from first content token to last,
   so prefill and queueing are excluded. A depth sweep that includes TTFT
   measures prefill, which we already know is at parity.

4. THE DECODE WINDOW MUST BE LONG ENOUGH TO AVERAGE MTP ACCEPTANCE, AND MUST
   BE VERIFIED. This harness originally ended its prompt with "In one sentence,
   state what this describes." The model obeyed: all 70 reps of the 2026-08-26
   sweep returned 25-26 tokens against a requested 256, giving 0.39-0.68 s
   windows. At MTP=5 that is ~5 speculative cycles, so one accepted-vs-rejected
   draft moves the rate double digits -- identical reps at 131K spanned
   37.0-64.3 tok/s, a 1.74x swing that was mistaken for JIT noise.
   Fix is threefold: do not ask for a short answer, pin the window with
   min_tokens + ignore_eos, and ASSERT completion_tokens == max_tokens so the
   run fails instead of publishing a collapsed window. See issue #26.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone

# Deterministic per-depth filler. Not random: the run must be reproducible.
_LOREM = (
    "The distributed inference engine partitions attention heads across nodes "
    "and exchanges activations over the fabric on every decode step. "
)


# Measured against this exact filler on DeepSeek-V4's tokenizer: the naive
# 4 chars/token rule undershot by ~38% (a "2048" prompt tokenized to 1,276),
# because repeated English prose tokenizes far denser than 4 chars/token.
# Depths must land on their nominal values or the 2-node and 3-node arms are
# not comparing the same shape.
_CHARS_PER_TOKEN = 6.42


def build_prompt(target_tokens: int, salt: str) -> str:
    """~target_tokens of filler with a unique header, so the prefix cache misses.

    The ratio is calibrated, not assumed -- see _CHARS_PER_TOKEN. The actual
    count is still reported from server-side usage and is what gets published.
    """
    header = "[session " + salt + "] Read the following log excerpt.\n\n"
    body_chars = max(int(target_tokens * _CHARS_PER_TOKEN) - len(header), 0)
    reps = body_chars // len(_LOREM) + 1
    body = (_LOREM * reps)[:body_chars]
    # NOTE: do NOT ask for a short answer here. An earlier version ended with
    # "In one sentence, state what this describes." and the model obeyed it,
    # returning 25-26 tokens against a requested 256 on every single rep. See
    # trap 4 below and issue #26.
    return header + body + "\n\nDescribe what this text is, in detail."


def run_once(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # min_tokens + ignore_eos pin the decode window to exactly max_tokens.
        # Without both, generation length is a property of the prompt rather
        # than of the harness, and the measured window silently collapses.
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
                if choice.get("delta", {}).get("content") and first_content is None:
                    first_content = time.perf_counter()
    finished = time.perf_counter()
    if first_content is None:
        raise RuntimeError("stream completed without content")
    if completion_tokens is None:
        raise RuntimeError("server did not return completion token usage")
    # Trap 4 (issue #26): the decode window must be the length we asked for.
    # Every rep of the 2026-08-26 sweep returned 25-26 tokens against a
    # requested 256 because the prompt asked for one sentence, and nothing
    # checked. A 0.4s window cannot average MTP acceptance: identical reps
    # spanned 37.0-64.3 tok/s. Fail loudly rather than publish that spread.
    if completion_tokens != max_tokens:
        raise RuntimeError(
            "decode window is %d tokens, expected %d -- the model stopped early. "
            "Check that min_tokens/ignore_eos are honoured by this server and "
            "that the prompt does not request a short answer (issue #26)."
            % (completion_tokens, max_tokens)
        )
    decode_s = max(finished - first_content, 1e-9)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": completion_tokens,
        "ttft_s": first_content - started,
        "decode_s": decode_s,
        "decode_tok_s": completion_tokens / decode_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="2048,8192,32768,131072,262144")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--label", required=True, help="e.g. tp3 or tp2")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    out = open(args.output, "a", buffering=1)
    summary = []

    for depth in depths:
        print("\n=== depth %d (%s) ===" % (depth, args.label), flush=True)
        # Warm EVERY shape before measuring it -- see constraint 2.
        for w in range(args.warmups):
            try:
                r = run_once(args.base_url, args.model,
                             build_prompt(depth, "%s-w%d-%d" % (args.label, w, depth)),
                             args.max_tokens)
                print("  warmup %d: %.1f tok/s (ptok=%s, cached=%s)"
                      % (w + 1, r["decode_tok_s"], r["prompt_tokens"],
                         r["cached_tokens"]), flush=True)
            except Exception as exc:                     # noqa: BLE001
                print("  warmup %d FAILED: %s" % (w + 1, exc), flush=True)

        samples = []
        records = []
        for i in range(args.reps):
            try:
                r = run_once(args.base_url, args.model,
                             build_prompt(depth, "%s-r%d-%d" % (args.label, i, depth)),
                             args.max_tokens)
            except Exception as exc:                     # noqa: BLE001
                print("  rep %d FAILED: %s" % (i + 1, exc), flush=True)
                continue
            r.update(label=args.label, target_depth=depth, rep=i)
            out.write(json.dumps(r) + "\n")
            records.append(r)
            samples.append(r["decode_tok_s"])
            flag = "  <-- CACHE HIT, INVALID" if r["cached_tokens"] else ""
            print("  rep %d: %.1f tok/s ttft=%.1fs ptok=%s%s"
                  % (i + 1, r["decode_tok_s"], r["ttft_s"],
                     r["prompt_tokens"], flag), flush=True)

        if not samples:
            print("  NO VALID SAMPLES", flush=True)
            continue
        med = statistics.median(samples)
        spread = (max(samples) - min(samples)) / med * 100
        cache_hits = sum(1 for r in records if r["cached_tokens"])
        ptok = statistics.median([r["prompt_tokens"] for r in records])
        summary.append({
            "label": args.label,
            "target_depth": depth,
            "actual_prompt_tokens": ptok,
            "n": len(samples),
            "median_decode_tok_s": round(med, 2),
            "min": round(min(samples), 2),
            "max": round(max(samples), 2),
            "spread_pct": round(spread, 1),
            "cache_hits": cache_hits,
            "median_ttft_s": round(statistics.median(
                [r["ttft_s"] for r in records]), 2),
        })
        print("  MEDIAN %.1f tok/s  (n=%d, spread %.1f%%, cache_hits=%d)"
              % (med, len(samples), spread, cache_hits), flush=True)

    out.close()
    with open(args.output.replace(".jsonl", "-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== SUMMARY %s ===" % args.label, flush=True)
    for s in summary:
        print("  %7s tok -> %7.2f tok/s  (spread %s%%, cache %d)"
              % (s["actual_prompt_tokens"], s["median_decode_tok_s"],
                 s["spread_pct"], s["cache_hits"]), flush=True)


if __name__ == "__main__":
    main()
