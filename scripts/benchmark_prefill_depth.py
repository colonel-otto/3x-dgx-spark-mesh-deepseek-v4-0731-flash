#!/usr/bin/env python3
"""Deep Prefill TTFT & Decode Benchmark Harness (Issue #33).

Measures TTFT and decode throughput across context depths (32K, 131K, 262K) with
a verified 256-token output window.

Asserts:
- completion_tokens == max_tokens (asserted per rep)
- cached_tokens == 0 (asserted cold path)
- full per-rep spread recorded
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone

_LOREM = (
    "The distributed inference engine partitions attention heads across nodes "
    "and exchanges activations over the fabric on every decode step. "
)
_CHARS_PER_TOKEN = 6.42


def build_prompt(target_tokens: int, salt: str) -> str:
    header = f"[session {salt}] Read the following log excerpt.\n\n"
    body_chars = max(int(target_tokens * _CHARS_PER_TOKEN) - len(header), 0)
    reps = body_chars // len(_LOREM) + 1
    body = (_LOREM * reps)[:body_chars]
    return header + body + "\n\nDescribe what this text is, in detail."


def run_single(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    with urllib.request.urlopen(request, timeout=3600) as response:
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
        raise RuntimeError("No content received")
    if completion_tokens != max_tokens:
        raise RuntimeError(f"Window collapse: requested {max_tokens}, got {completion_tokens}")
    if cached != 0:
        raise RuntimeError(f"Contaminated run: {cached} cached tokens on cold run")

    decode_s = max(finished - first_content, 1e-9)
    ttft_s = first_content - started
    total_s = finished - started

    return {
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tok_s": completion_tokens / decode_s,
    }


def run_sweep(url: str, model: str, bt: int, depths: list[int],
              reps: int, max_tokens: int) -> dict:
    depth_results = []
    print(f"\n=======================================================")
    print(f"Starting Prefill TTFT Sweep: bt={bt}, Depths={depths}, Reps={reps}")
    print(f"=======================================================\n")

    for depth in depths:
        print(f"--- Depth {depth} tokens (Reps={reps}) ---")
        # Warmup for this shape
        warmup_salt = f"warmup-bt{bt}-d{depth}-{int(time.time())}"
        print(f"  Running warmup for {depth} tokens...")
        run_single(url, model, build_prompt(depth, warmup_salt), max_tokens)
        print("  Warmup complete.")

        rep_samples = []
        for rep in range(reps):
            salt = f"bt{bt}-d{depth}-r{rep}-{int(time.time())}"
            sample = run_single(url, model, build_prompt(depth, salt), max_tokens)
            sample["rep"] = rep + 1
            rep_samples.append(sample)
            print(f"  Rep {rep+1}/{reps}: TTFT = {sample['ttft_s']:6.3f} s, "
                  f"Decode = {sample['decode_tok_s']:5.2f} tok/s (Prompt: {sample['prompt_tokens']} tokens)")

        ttfts = [s["ttft_s"] for s in rep_samples]
        decodes = [s["decode_tok_s"] for s in rep_samples]

        summary = {
            "depth_nominal": depth,
            "prompt_tokens_actual": rep_samples[0]["prompt_tokens"],
            "reps": reps,
            "ttft_median_s": statistics.median(ttfts),
            "ttft_spread_s": max(ttfts) - min(ttfts),
            "ttft_all_s": ttfts,
            "decode_tok_s_median": statistics.median(decodes),
            "decode_tok_s_spread": max(decodes) - min(decodes),
            "decode_tok_s_all": decodes,
            "samples": rep_samples,
        }
        depth_results.append(summary)
        print(f"  => Depth {depth} Summary: Median TTFT = {summary['ttft_median_s']:6.3f} s "
              f"(Spread: {summary['ttft_spread_s']:5.3f} s), "
              f"Median Decode = {summary['decode_tok_s_median']:5.2f} tok/s\n")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "batched_tokens": bt,
        "model": model,
        "depths": depth_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefill TTFT & Decode Depth Harness.")
    parser.add_argument("--url", default="http://192.168.10.223:8100/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--batched-tokens", type=int, required=True)
    parser.add_argument("--depths", default="32768,131072,262144")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    depths = [int(x.strip()) for x in args.depths.split(",")]
    res = run_sweep(args.url, args.model, args.batched_tokens, depths, args.reps, args.max_tokens)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
