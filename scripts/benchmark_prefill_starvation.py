#!/usr/bin/env python3
"""Measure long-prefill interference with a staggered streaming workload.

Long requests enter the scheduler first.  A short-prompt decoder then arrives
behind them.  Besides TTFT and throughput, the harness records content-event
gaps so zero-preemption decode starvation cannot hide behind a clean preemption
counter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


LOREM = (
    "The distributed inference engine partitions attention heads across nodes "
    "and exchanges activations over the fabric on every decode step. "
)
CHARS_PER_TOKEN = 6.42


def prompt(target_tokens: int, nonce: str) -> str:
    header = f"[starvation-probe {nonce}] Read this log.\n\n"
    body_chars = max(int(target_tokens * CHARS_PER_TOKEN) - len(header), 0)
    body = (LOREM * (body_chars // len(LOREM) + 1))[:body_chars]
    return header + body + "\n\nDescribe the system in detail."


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def stream(
    base_url: str,
    model: str,
    text: str,
    max_tokens: int,
    role: str,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    events: list[float] = []
    usage = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    events.append(time.perf_counter())
    finished = time.perf_counter()
    if not events:
        raise RuntimeError(f"{role}: stream completed without content")
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens != max_tokens:
        raise RuntimeError(
            f"{role}: completion_tokens={completion_tokens}, expected {max_tokens}"
        )
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    if cached:
        raise RuntimeError(f"{role}: cached_tokens={cached}, expected zero")
    gaps = [right - left for left, right in zip(events, events[1:])]
    decode_s = max(finished - events[0], 1e-9)
    return {
        "role": role,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": completion_tokens,
        "ttft_s": events[0] - started,
        "decode_s": decode_s,
        "decode_tok_s": completion_tokens / decode_s,
        "content_events": len(events),
        "max_event_gap_s": max(gaps, default=0.0),
        "p95_event_gap_s": percentile(gaps, 0.95),
        "elapsed_s": finished - started,
    }


def run_trial(args: argparse.Namespace, trial: int) -> dict:
    nonce = f"{args.label}-{trial}-{time.time_ns()}"
    long_prompts = [
        prompt(args.long_depth, f"{nonce}-long-{index}")
        for index in range(args.long_count)
    ]
    short_prompt = prompt(args.short_depth, f"{nonce}-decoder")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.long_count + 1
    ) as pool:
        long_futures = []
        for index, text in enumerate(long_prompts):
            long_futures.append(
                pool.submit(
                    stream,
                    args.base_url,
                    args.model,
                    text,
                    args.long_output_tokens,
                    f"long-{index}",
                )
            )
            if args.long_launch_gap:
                time.sleep(args.long_launch_gap)
        time.sleep(args.decoder_delay)
        decoder_future = pool.submit(
            stream,
            args.base_url,
            args.model,
            short_prompt,
            args.decoder_output_tokens,
            "decoder",
        )
        decoder = decoder_future.result()
        longs = [future.result() for future in long_futures]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "trial": trial,
        "long_depth_target": args.long_depth,
        "long_count": args.long_count,
        "decoder_delay_s": args.decoder_delay,
        "decoder": decoder,
        "long_requests": longs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--long-depth", type=int, default=131072)
    parser.add_argument("--long-count", type=int, default=2)
    parser.add_argument("--long-output-tokens", type=int, default=64)
    parser.add_argument("--short-depth", type=int, default=2048)
    parser.add_argument("--decoder-output-tokens", type=int, default=512)
    parser.add_argument("--long-launch-gap", type=float, default=0.2)
    parser.add_argument("--decoder-delay", type=float, default=0.5)
    args = parser.parse_args()

    output = Path(args.output)
    rows = []
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for trial in range(args.repeat):
            row = run_trial(args, trial)
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            decoder = row["decoder"]
            print(
                f"trial {trial}: decoder ttft={decoder['ttft_s']:.2f}s "
                f"decode={decoder['decode_tok_s']:.2f} tok/s "
                f"max_gap={decoder['max_event_gap_s']:.2f}s",
                flush=True,
            )

    summary = {
        "label": args.label,
        "trials": len(rows),
        "median_decoder_ttft_s": statistics.median(
            row["decoder"]["ttft_s"] for row in rows
        ),
        "median_decoder_decode_tok_s": statistics.median(
            row["decoder"]["decode_tok_s"] for row in rows
        ),
        "median_decoder_max_event_gap_s": statistics.median(
            row["decoder"]["max_event_gap_s"] for row in rows
        ),
        "max_decoder_event_gap_s": max(
            row["decoder"]["max_event_gap_s"] for row in rows
        ),
    }
    output.with_name(output.stem + "-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
