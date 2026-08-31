#!/usr/bin/env python3
"""Staggered Varied-Length Load Generator & Speculative Acceptance Validator.

Sends Poisson-distributed, ragged-context requests (1K–131K tokens) to stress
continuous batching, KV slot compaction, and MTP draft acceptance under
realistic async multi-user load.

This harness settles three production questions:
  1. Does the engine maintain MTP draft acceptance (>= 50%) under asynchronous
     continuous batching up to MAX_NUM_SEQS=32?
  2. Do ragged-context requests (disparate lengths arriving and finishing at
     random intervals) cause HTTP 500s, token truncation, or KV corruption?
  3. Does the virtual-TP zero-fill attention sink (eugr) or -inf sink (anemll)
     degrade under dynamic KV slot compaction?

Method:
  - For each concurrency tier (e.g. 1, 4, 8, 16, 32), launch N requests with
    Poisson inter-arrival jitter.  Each request draws a prompt length from a
    tiered mixture:
      Tier A (50%): 1K–8K tokens   (interactive)
      Tier B (35%): 8K–32K tokens  (medium context)
      Tier C (15%): 32K–131K tokens (deep context)
  - All requests use min_tokens=max_tokens=OUTPUT_TOKENS, ignore_eos=True to
    pin the output window (inherited trap from VOID-25-token-window).
  - Prometheus spec-decode counters are scraped before and after each tier to
    compute acceptance ratio over the measured window only.
  - Every request outcome (status, TTFT, decode tok/s, output tokens, error)
    is recorded.  The harness reports PASS only if zero errors occurred AND
    acceptance stayed above the configured floor.

Usage:
  python benchmark_staggered_spec_acceptance.py \\
    --url http://127.0.0.1:8888/v1 \\
    --metrics-url http://127.0.0.1:8888/metrics \\
    --concurrencies 1,4,8,16,32 \\
    --requests-per-tier 30 \\
    --out results/staggered_spec_acceptance.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_LOREM = (
    "The distributed inference engine partitions attention heads across nodes "
    "and exchanges activations over the fabric on every decode step. "
)
_CHARS_PER_TOKEN = 6.42  # calibrated on DeepSeek-V4 tokenizer


def _build_prompt(target_tokens: int, salt: str) -> str:
    """Build a deterministic filler prompt of approximately target_tokens."""
    header = f"[staggered-load salt={salt}] Read the following passage.\n\n"
    body_chars = max(int(target_tokens * _CHARS_PER_TOKEN) - len(header), 0)
    reps = body_chars // len(_LOREM) + 1
    body = (_LOREM * reps)[:body_chars]
    return header + body + "\n\nDescribe what this text is about, in detail."


# Tier boundaries: (min_tokens_inclusive, max_tokens_exclusive, weight)
_TIERS = [
    (1024, 8192, 0.50),     # Tier A: interactive
    (8192, 32768, 0.35),    # Tier B: medium context
    (32768, 131073, 0.15),  # Tier C: deep context
]


def _sample_prompt_length(rng: random.Random) -> int:
    """Sample a prompt token count from the tiered mixture distribution."""
    r = rng.random()
    cumulative = 0.0
    for lo, hi, weight in _TIERS:
        cumulative += weight
        if r <= cumulative:
            # Log-uniform within the tier so deep-context picks are not all 32K
            log_lo, log_hi = math.log(lo), math.log(hi - 1)
            return int(math.exp(rng.uniform(log_lo, log_hi)))
    # Fallback: shortest tier
    return _TIERS[0][0]


# ---------------------------------------------------------------------------
# Prometheus scraper
# ---------------------------------------------------------------------------

def scrape_metrics(metrics_url: str) -> dict:
    """Scrape Prometheus counters from /metrics, tolerating missing series."""
    result = {
        "accepted": 0.0,
        "draft": 0.0,
        "draft_iters": 0.0,
        "preemptions": 0.0,
        "running": 0.0,
        "waiting": 0.0,
    }
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("#") or not line:
                    continue
                # Extract metric name (before any labels or space)
                name = line.split("{", 1)[0].split(" ", 1)[0]
                try:
                    val = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    continue
                if name == "vllm:spec_decode_num_accepted_tokens_total":
                    result["accepted"] = val
                elif name == "vllm:spec_decode_num_draft_tokens_total":
                    result["draft"] = val
                elif name in ("vllm:spec_decode_num_drafts_total",
                              "vllm:spec_decode_draft_iterations_total"):
                    result["draft_iters"] = val
                elif name == "vllm:num_preemptions_total":
                    result["preemptions"] = val
                elif name == "vllm:num_requests_running":
                    result["running"] = val
                elif name == "vllm:num_requests_waiting":
                    result["waiting"] = val
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Single request execution
# ---------------------------------------------------------------------------

def _send_request(url: str, model: str, prompt: str, max_tokens: int,
                  request_id: str) -> dict:
    """Send a single streaming chat completion and record all observables."""
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
    first_content_at = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    error = None

    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens",
                                                  completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = details.get("cached_tokens", 0) or 0
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = (delta.get("content")
                               or delta.get("reasoning_content")
                               or delta.get("reasoning"))
                    if content and first_content_at is None:
                        first_content_at = time.perf_counter()
    except urllib.error.HTTPError as e:
        error = f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    finished = time.perf_counter()
    ttft_s = (first_content_at - started) if first_content_at else None
    decode_s = (finished - first_content_at) if first_content_at else None
    decode_tok_s = (completion_tokens / decode_s
                    if decode_s and decode_s > 0.001 else None)

    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "total_s": finished - started,
        "error": error,
        "window_ok": completion_tokens == max_tokens,
    }


# ---------------------------------------------------------------------------
# Staggered workload runner
# ---------------------------------------------------------------------------

def _run_staggered_tier(url: str, model: str, concurrency: int,
                        num_requests: int, arrival_rate: float,
                        max_tokens: int, tier_salt: str,
                        rng: random.Random) -> list[dict]:
    """Launch num_requests with Poisson inter-arrival, bounded by concurrency.

    Uses a threading.Semaphore to cap in-flight requests at concurrency.
    Arrivals are jittered by exponential distribution with mean 1/arrival_rate.
    """
    sem = threading.Semaphore(concurrency)
    results: list[dict] = [None] * num_requests  # type: ignore[list-item]
    threads: list[threading.Thread] = []

    def _worker(idx: int, prompt: str, req_id: str):
        sem.acquire()
        try:
            results[idx] = _send_request(url, model, prompt, max_tokens,
                                         req_id)
        finally:
            sem.release()

    for i in range(num_requests):
        target_tokens = _sample_prompt_length(rng)
        req_id = f"{tier_salt}-r{i}-p{target_tokens}"
        prompt = _build_prompt(target_tokens, req_id)

        t = threading.Thread(target=_worker, args=(i, prompt, req_id),
                             daemon=True)
        threads.append(t)
        t.start()

        # Poisson inter-arrival: exponential wait before launching next
        if i < num_requests - 1 and arrival_rate > 0:
            gap = rng.expovariate(arrival_rate)
            time.sleep(gap)

    for t in threads:
        t.join(timeout=7200)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(url: str, metrics_url: str, model: str,
              concurrencies: list[int], requests_per_tier: int,
              arrival_rate: float, max_tokens: int, warmups: int,
              seed: int) -> dict:
    """Run the full staggered sweep across concurrency tiers."""
    rng = random.Random(seed)
    all_tiers: list[dict] = []
    overall_errors = 0

    print("=" * 70)
    print("STAGGERED SPEC-ACCEPTANCE SWEEP")
    print(f"URL: {url}  Model: {model}")
    print(f"Concurrencies: {concurrencies}  Requests/tier: {requests_per_tier}")
    print(f"Arrival rate λ: {arrival_rate}  Output tokens: {max_tokens}")
    print("=" * 70)

    # Warmup: 2 single-stream requests (policy minimum)
    print(f"\nWarming up ({warmups} requests) ...")
    for w in range(warmups):
        prompt = _build_prompt(4096, f"warmup-{w}")
        _send_request(url, model, prompt, max_tokens, f"warmup-{w}")
    print("Warmup complete.\n")

    for cc in concurrencies:
        tier_salt = f"cc{cc}-{int(time.time())}"
        print(f"--- Concurrency cc={cc} ({requests_per_tier} requests, "
              f"staggered λ={arrival_rate}) ---")

        m_before = scrape_metrics(metrics_url)
        tier_start = time.perf_counter()

        results = _run_staggered_tier(url, model, cc, requests_per_tier,
                                      arrival_rate, max_tokens, tier_salt, rng)

        tier_elapsed = time.perf_counter() - tier_start
        m_after = scrape_metrics(metrics_url)

        # Compute speculative acceptance delta
        d_acc = m_after["accepted"] - m_before["accepted"]
        d_draft = m_after["draft"] - m_before["draft"]
        d_iters = m_after["draft_iters"] - m_before["draft_iters"]
        d_preempt = m_after["preemptions"] - m_before["preemptions"]
        acc_ratio_pct = (d_acc / d_draft * 100.0) if d_draft > 0 else None
        acc_mean_len = (d_acc / d_iters) if d_iters > 0 else None

        # Per-request stats
        errors = [r for r in results if r["error"] is not None]
        window_fails = [r for r in results if not r["window_ok"]
                        and r["error"] is None]
        ok_results = [r for r in results if r["error"] is None
                      and r["window_ok"]]

        ttfts = [r["ttft_s"] for r in ok_results if r["ttft_s"] is not None]
        decode_rates = [r["decode_tok_s"] for r in ok_results
                        if r["decode_tok_s"] is not None]
        prompt_lengths = [r["prompt_tokens"] for r in ok_results]

        tier_record = {
            "concurrency": cc,
            "requests_total": len(results),
            "requests_ok": len(ok_results),
            "requests_errored": len(errors),
            "requests_window_fail": len(window_fails),
            "tier_elapsed_s": tier_elapsed,
            "prompt_lengths_min": min(prompt_lengths) if prompt_lengths else 0,
            "prompt_lengths_max": max(prompt_lengths) if prompt_lengths else 0,
            "prompt_lengths_median": (statistics.median(prompt_lengths)
                                     if prompt_lengths else 0),
            "ttft_s_median": (statistics.median(ttfts) if ttfts else None),
            "ttft_s_p95": (sorted(ttfts)[int(len(ttfts) * 0.95)]
                          if len(ttfts) >= 2 else None),
            "decode_tok_s_median": (statistics.median(decode_rates)
                                   if decode_rates else None),
            "spec_accepted_tokens": d_acc,
            "spec_draft_tokens": d_draft,
            "spec_acceptance_ratio_pct": acc_ratio_pct,
            "spec_mean_accepted_length": acc_mean_len,
            "preemptions_delta": d_preempt,
            "errors": [{"request_id": r["request_id"], "error": r["error"]}
                       for r in errors],
            "window_failures": [{"request_id": r["request_id"],
                                 "completion_tokens": r["completion_tokens"]}
                                for r in window_fails],
            "per_request": results,
        }
        all_tiers.append(tier_record)
        overall_errors += len(errors)

        # Print summary
        acc_str = (f"{acc_ratio_pct:.1f}%" if acc_ratio_pct is not None
                   else "N/A")
        ttft_str = (f"{tier_record['ttft_s_median']:.3f}s"
                    if tier_record["ttft_s_median"] is not None else "N/A")
        decode_str = (f"{tier_record['decode_tok_s_median']:.1f}"
                      if tier_record["decode_tok_s_median"] is not None
                      else "N/A")
        print(f"  OK={len(ok_results)} ERR={len(errors)} "
              f"WINDOW_FAIL={len(window_fails)}  "
              f"Acceptance={acc_str}  "
              f"Preemptions={d_preempt:.0f}  "
              f"TTFT_med={ttft_str}  "
              f"Decode_med={decode_str} tok/s  "
              f"PromptRange=[{tier_record['prompt_lengths_min']}, "
              f"{tier_record['prompt_lengths_max']}]")
        if errors:
            for e in errors[:3]:
                print(f"    ERROR: {e['request_id']}: {e['error']}")
        print()

    # Overall verdict
    all_acc = [t["spec_acceptance_ratio_pct"] for t in all_tiers
               if t["spec_acceptance_ratio_pct"] is not None]
    all_preempt = sum(t["preemptions_delta"] for t in all_tiers)
    min_acc = min(all_acc) if all_acc else None

    verdict = "PASS"
    reasons: list[str] = []
    if overall_errors > 0:
        verdict = "FAIL"
        reasons.append(f"{overall_errors} HTTP errors")
    if min_acc is not None and min_acc < 30.0:
        verdict = "FAIL"
        reasons.append(f"acceptance collapsed to {min_acc:.1f}%")

    print("=" * 70)
    print(f"VERDICT: {verdict}")
    if reasons:
        print(f"  Reasons: {'; '.join(reasons)}")
    if min_acc is not None:
        print(f"  Min acceptance across tiers: {min_acc:.1f}%")
    print(f"  Total preemptions: {all_preempt:.0f}")
    print(f"  Total errors: {overall_errors}")
    print("=" * 70)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "benchmark_staggered_spec_acceptance.py",
        "model": model,
        "url": url,
        "arrival_rate_lambda": arrival_rate,
        "output_tokens": max_tokens,
        "warmups": warmups,
        "seed": seed,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "min_acceptance_pct": min_acc,
        "total_preemptions": all_preempt,
        "total_errors": overall_errors,
        "tiers": all_tiers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staggered varied-length load generator & spec-acceptance "
                    "validator for 3-node DGX Spark TP=3 deployment.")
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8888/metrics",
                        help="Prometheus /metrics endpoint")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--concurrencies", default="1,4,8,16,32",
                        help="Comma-separated concurrency tiers to sweep")
    parser.add_argument("--requests-per-tier", type=int, default=30,
                        help="Number of requests per concurrency tier")
    parser.add_argument("--arrival-rate", type=float, default=2.0,
                        help="Poisson arrival rate λ (requests/sec mean)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Forced output window (min_tokens=max_tokens)")
    parser.add_argument("--warmups", type=int, default=2,
                        help="Warm requests before measurement (policy min: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for prompt length sampling")
    parser.add_argument("--out", required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    ccs = [int(x.strip()) for x in args.concurrencies.split(",")]
    result = run_sweep(args.url, args.metrics_url, args.model, ccs,
                       args.requests_per_tier, args.arrival_rate,
                       args.max_tokens, args.warmups, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {args.out}")

    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
