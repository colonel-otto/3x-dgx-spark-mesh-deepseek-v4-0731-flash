#!/usr/bin/env python3
"""Probe DSpark proposer acceptance rate across long generation horizons (Issue #36).

Measures whether speculative draft acceptance degrades over long decode steps
(>50 steps) due to potential sliding-window cross-attention KV staleness.
"""
import argparse
import json
import time
import urllib.request
import urllib.parse
import re

METRICS_URL = "http://127.0.0.1:8100/metrics"
CHAT_URL = "http://127.0.0.1:8100/v1/chat/completions"

def get_spec_metrics(url: str = METRICS_URL) -> dict[str, float]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        return {}
    
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if m:
            name, val = m.group(1), float(m.group(2))
            if "spec_decode" in name:
                # Capture position labels if present
                if "position=" in line:
                    pos_m = re.search(r'position="(\d+)"', line)
                    if pos_m:
                        metrics[f"{name}_pos_{pos_m.group(1)}"] = val
                        continue
                metrics[name] = val
    return metrics

def run_generation_probe(prompt: str, max_tokens: int, url: str = CHAT_URL) -> dict:
    m_before = get_spec_metrics()
    
    payload = {
        "model": "deepseek-v4-flash-0731",
        "messages": [
            {"role": "system", "content": "You are a precise technical writer. Respond in detail."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    
    m_after = get_spec_metrics()
    
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    
    drafts_delta = m_after.get("vllm:spec_decode_num_drafts_total", 0) - m_before.get("vllm:spec_decode_num_drafts_total", 0)
    draft_tokens_delta = m_after.get("vllm:spec_decode_num_draft_tokens_total", 0) - m_before.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted_tokens_delta = m_after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m_before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    
    pos0_delta = m_after.get("vllm:spec_decode_num_accepted_tokens_per_pos_total_pos_0", 0) - m_before.get("vllm:spec_decode_num_accepted_tokens_per_pos_total_pos_0", 0)
    pos1_delta = m_after.get("vllm:spec_decode_num_accepted_tokens_per_pos_total_pos_1", 0) - m_before.get("vllm:spec_decode_num_accepted_tokens_per_pos_total_pos_1", 0)
    
    acceptance_rate = (accepted_tokens_delta / draft_tokens_delta) if draft_tokens_delta > 0 else 0.0
    mean_accepted_per_step = (accepted_tokens_delta / drafts_delta) if drafts_delta > 0 else 0.0
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0
    
    return {
        "max_tokens_target": max_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tok_per_sec": round(tok_per_sec, 2),
        "drafts_count": drafts_delta,
        "draft_tokens": draft_tokens_delta,
        "accepted_tokens": accepted_tokens_delta,
        "acceptance_rate": round(acceptance_rate, 4),
        "mean_accepted_per_step": round(mean_accepted_per_step, 3),
        "pos0_accepted": pos0_delta,
        "pos1_accepted": pos1_delta,
    }

def main():
    parser = argparse.ArgumentParser(description="Probe DSpark acceptance rate across generation horizons.")
    parser.add_argument("--lengths", type=str, default="256,512,1024,1536", help="Comma-separated max_tokens horizons.")
    parser.add_argument("--out-exclusivity", type=str, default=None, help="Path to write exclusivity.json")
    parser.add_argument("--allow-foreign", action="store_true", help="Warn instead of error on foreign traffic")
    args = parser.parse_args()

    # Issue #37: Exclusivity pre-flight gate
    try:
        from exclusivity import assert_idle, verify_exclusivity
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from exclusivity import assert_idle, verify_exclusivity

    start_success = assert_idle(allow_foreign=args.allow_foreign)
    
    horizons = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    prompt = (
        "Write an exhaustive, highly detailed architectural specification for a distributed key-value storage engine "
        "supporting Raft consensus, Multi-Raft partitioning, LSM-tree storage engines with write-ahead logging (WAL), "
        "compaction filters, block cache management with LRU/2Q eviction, bloom filters, and distributed transaction "
        "processing with two-phase commit (2PC) and snapshot isolation. Detail every data structure and protocol."
    )
    
    print(f"=== Probing DSpark Proposer Across Horizons: {horizons} ===")
    results = []
    for h in horizons:
        print(f"\n--- Testing Target Length: {h} tokens ---")
        res = run_generation_probe(prompt, max_tokens=h)
        print(f"Generated: {res['completion_tokens']} tokens in {res['elapsed_sec']}s ({res['tok_per_sec']} tok/s)")
        print(f"Acceptance Rate: {res['acceptance_rate']*100:.1f}% ({res['accepted_tokens']}/{res['draft_tokens']} draft tokens)")
        print(f"Mean Accepted / Step: {res['mean_accepted_per_step']} (Pos0: {res['pos0_accepted']}, Pos1: {res['pos1_accepted']})")
        results.append(res)
    
    print("\n=== Summary Table ===")
    print(f"{'Target Tokens':<15}{'Actual Tokens':<15}{'Time (s)':<12}{'Tok/s':<10}{'Acceptance %':<15}{'Accept/Step':<12}")
    print("-" * 79)
    for r in results:
        print(f"{r['max_tokens_target']:<15}{r['completion_tokens']:<15}{r['elapsed_sec']:<12}{r['tok_per_sec']:<10}{r['acceptance_rate']*100:<15.1f}{r['mean_accepted_per_step']:<12.3f}")

    # Issue #37: Exclusivity post-flight verification
    verify_exclusivity(
        start_success,
        expected_requests=len(horizons),
        output_file=args.out_exclusivity,
        allow_foreign=args.allow_foreign
    )

if __name__ == "__main__":
    main()
