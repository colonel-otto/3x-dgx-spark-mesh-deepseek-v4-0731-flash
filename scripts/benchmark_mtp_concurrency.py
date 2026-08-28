#!/usr/bin/env python3
"""MTP Concurrency Sweep Benchmark Harness (Issue #32).

Measures decode throughput, TTFT, and speculative acceptance rates across
concurrency levels cc in {1, 4, 8, 16} at context depth 8192 with a pinned 256-token
decode window.

Asserts:
- completion_tokens == 256 (no window collapse)
- cached_tokens == 0 (cold path)
- full per-rep spread recorded
- Prometheus speculative metrics scraped and attributed per cell
"""

from __future__ import annotations

import argparse
import concurrent.futures
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


def scrape_metrics(metrics_url: str) -> dict[str, float]:
    accepted = draft = draft_iters = None
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                    accepted = float(line.rsplit(" ", 1)[1])
                elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                    draft = float(line.rsplit(" ", 1)[1])
                elif line.startswith("vllm:spec_decode_draft_iterations_total"):
                    draft_iters = float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return {
        "accepted": accepted or 0.0,
        "draft": draft or 0.0,
        "draft_iters": draft_iters or 0.0,
    }


def send_request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
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


def run_concurrency_batch(url: str, model: str, cc: int, depth: int,
                          max_tokens: int, batch_salt: str) -> dict:
    prompts = [build_prompt(depth, f"{batch_salt}-req{i}") for i in range(cc)]
    batch_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=cc) as executor:
        futures = [executor.submit(send_request, url, model, p, max_tokens) for p in prompts]
        results = [f.result() for f in futures]

    batch_finish = time.perf_counter()
    wall_time = batch_finish - batch_start

    total_tokens = sum(r["completion_tokens"] for r in results)
    agg_tok_s = total_tokens / wall_time
    per_stream_tok_s = [r["decode_tok_s"] for r in results]
    ttfts = [r["ttft_s"] for r in results]

    return {
        "concurrency": cc,
        "wall_time_s": wall_time,
        "aggregate_tok_s": agg_tok_s,
        "median_stream_tok_s": statistics.median(per_stream_tok_s),
        "mean_ttft_s": statistics.mean(ttfts),
        "median_ttft_s": statistics.median(ttfts),
        "all_stream_tok_s": per_stream_tok_s,
        "all_ttfts": ttfts,
    }


def run_sweep(url: str, metrics_url: str, model: str, ktok: int, depth: int,
              concurrencies: list[int], reps: int, max_tokens: int) -> dict:
    cells = []
    print(f"\n=======================================================")
    print(f"Starting MTP={ktok} Sweep: Context={depth}, Concurrencies={concurrencies}, Reps={reps}")
    print(f"=======================================================\n")

    # Warmup
    print("Running warmup request...")
    send_request(url, model, build_prompt(depth, f"warmup-k{ktok}"), max_tokens)
    print("Warmup complete.\n")

    for cc in concurrencies:
        print(f"--- Concurrency cc={cc} (Reps={reps}) ---")
        rep_results = []
        m_before = scrape_metrics(metrics_url)

        for rep in range(reps):
            batch_salt = f"k{ktok}-cc{cc}-r{rep}-{int(time.time())}"
            res = run_concurrency_batch(url, model, cc, depth, max_tokens, batch_salt)
            rep_results.append(res)
            print(f"  Rep {rep+1}/{reps}: Agg = {res['aggregate_tok_s']:6.2f} tok/s, "
                  f"Median Stream = {res['median_stream_tok_s']:5.2f} tok/s, "
                  f"Median TTFT = {res['median_ttft_s']:5.3f} s")

        m_after = scrape_metrics(metrics_url)
        d_acc = m_after["accepted"] - m_before["accepted"]
        d_draft = m_after["draft"] - m_before["draft"]
        d_iters = m_after["draft_iters"] - m_before["draft_iters"]

        acc_ratio = (d_acc / d_draft * 100.0) if d_draft > 0 else 0.0
        acc_len = (d_acc / d_iters) if d_iters > 0 else 0.0

        agg_rates = [r["aggregate_tok_s"] for r in rep_results]
        ttfts = [r["median_ttft_s"] for r in rep_results]

        cell = {
            "mtp_k": ktok,
            "concurrency": cc,
            "depth": depth,
            "reps": reps,
            "aggregate_tok_s_median": statistics.median(agg_rates),
            "aggregate_tok_s_spread": max(agg_rates) - min(agg_rates),
            "aggregate_tok_s_all": agg_rates,
            "ttft_s_median": statistics.median(ttfts),
            "ttft_s_all": ttfts,
            "spec_accepted_tokens": d_acc,
            "spec_draft_tokens": d_draft,
            "spec_acceptance_ratio_pct": acc_ratio,
            "spec_mean_accepted_length": acc_len,
            "rep_details": rep_results,
        }
        cells.append(cell)
        print(f"  => Cell cc={cc} Summary: Median Agg = {cell['aggregate_tok_s_median']:6.2f} tok/s "
              f"(Spread: {cell['aggregate_tok_s_spread']:5.2f}), "
              f"Acceptance = {acc_ratio:4.1f}%, Mean Accepted Len = {acc_len:4.2f}\n")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mtp_k": ktok,
        "depth": depth,
        "model": model,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MTP Concurrency Sweep Harness.")
    parser.add_argument("--url", default="http://192.168.10.223:8100/v1")
    parser.add_argument("--metrics-url", default="http://192.168.10.223:8100/metrics")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--mtp-k", type=int, required=True, help="MTP token depth (5, 3, 2)")
    parser.add_argument("--depth", type=int, default=8192)
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    ccs = [int(x.strip()) for x in args.concurrencies.split(",")]
    result = run_sweep(args.url, args.metrics_url, args.model, args.mtp_k,
                       args.depth, ccs, args.reps, args.max_tokens)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
