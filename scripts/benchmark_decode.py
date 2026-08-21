#!/usr/bin/env python3
"""Small dependency-free OpenAI-compatible streaming decode benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_once(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
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
    text = []
    with urllib.request.urlopen(request, timeout=900) as response:
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
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_content is None:
                        first_content = time.perf_counter()
                    text.append(content)
    finished = time.perf_counter()
    if first_content is None:
        raise RuntimeError("stream completed without content")
    if completion_tokens is None:
        raise RuntimeError("server did not return completion token usage")
    decode_seconds = max(finished - first_content, 1e-9)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "completion_tokens": completion_tokens,
        "ttft_s": first_content - started,
        "decode_s": decode_seconds,
        "end_to_end_s": finished - started,
        "decode_tok_s": completion_tokens / decode_seconds,
        "output_chars": len("".join(text)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Example: http://node1:8100/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True, help="JSONL output path")
    args = parser.parse_args()
    with open(args.prompt_file, encoding="utf-8") as prompt_file:
        prompt = prompt_file.read()
    for _ in range(args.warmups):
        run_once(args.base_url, args.model, prompt, args.max_tokens)
    samples = []
    with open(args.output, "w", encoding="utf-8") as handle:
        for index in range(args.repetitions):
            sample = run_once(args.base_url, args.model, prompt, args.max_tokens)
            sample["run"] = index + 1
            samples.append(sample)
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
    speeds = [sample["decode_tok_s"] for sample in samples]
    ttfts = [sample["ttft_s"] for sample in samples]
    summary = {
        "repetitions": len(samples),
        "decode_tok_s": {
            "min": min(speeds), "median": statistics.median(speeds),
            "max": max(speeds), "p95": percentile(speeds, 0.95),
        },
        "ttft_s": {
            "min": min(ttfts), "median": statistics.median(ttfts),
            "max": max(ttfts), "p95": percentile(ttfts, 0.95),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
