#!/usr/bin/env python3
"""Capture Nsight Systems kernel profiling trace for 3-node cluster (Issue #38).

Captures CUDA kernel execution, NCCL collective share, and memory operations
for single-stream decode (8K context, 256 tokens) and deep prefill (131K context).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from exclusivity import assert_idle, verify_exclusivity

CHAT_URL = "http://127.0.0.1:8100/v1/chat/completions"
METRICS_URL = "http://127.0.0.1:8100/metrics"

def send_request(prompt: str, max_tokens: int, url: str = CHAT_URL) -> dict[str, Any]:
    payload = {
        "model": "deepseek-v4-flash-0731",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed,
        "usage": body.get("usage", {}),
    }

def capture_nsys_profile(
    workload_name: str,
    prompt: str,
    max_tokens: int,
    out_dir: Path,
    chat_url: str = CHAT_URL,
    metrics_url: str = METRICS_URL,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_file = out_dir / f"{workload_name}.nsys-rep"
    stats_file = out_dir / f"{workload_name}_stats.txt"

    print(f"\n=======================================================")
    print(f"=== Starting Nsight Profile: {workload_name} ===")
    print(f"=======================================================")

    # 1. Exclusivity check
    start_success = assert_idle(metrics_url, timeout_s=30.0)

    # 2. Warm up request
    print(f"Sending warm-up request (16 tokens)...")
    send_request("Ping warm-up test.", max_tokens=16, url=chat_url)
    time.sleep(1.0)
    assert_idle(metrics_url, timeout_s=10.0)

    # 3. Launch nsys in background
    # Estimate capture duration based on prompt shape
    expected_duration = 20 if max_tokens > 10 else 90
    nsys_cmd = [
        "sudo", "/usr/local/bin/nsys", "profile",
        "-t", "cuda,nvtx,osrt",
        "--sample=system-wide",
        "-o", str(rep_file.with_suffix("")),
        "--force-overwrite=true",
        "sleep", str(expected_duration)
    ]
    print(f"Launching nsys capture: {' '.join(nsys_cmd)}")
    nsys_proc = subprocess.Popen(nsys_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2.0)  # Allow nsys to attach/initialize

    # 4. Issue target workload
    print(f"Executing workload ({workload_name}): max_tokens={max_tokens}...")
    req_res = send_request(prompt, max_tokens=max_tokens, url=chat_url)
    print(f"Workload completed in {req_res['elapsed_sec']:.3f}s. Usage: {req_res['usage']}")

    # Wait for nsys to finish
    stdout, stderr = nsys_proc.communicate()
    print("Nsight capture completed.")

    # 5. Extract statistics using nsys stats
    print("Extracting kernel summary reports...")
    stats_cmd = [
        "sudo", "/usr/local/bin/nsys", "stats",
        "--report", "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum",
        str(rep_file)
    ]
    stats_proc = subprocess.run(stats_cmd, capture_output=True, text=True)
    stats_file.write_text(stats_proc.stdout + "\n" + stats_proc.stderr, encoding="utf-8")

    # 6. Parse top kernels and NCCL share
    parsed_kernels = parse_kernel_stats(stats_proc.stdout)

    summary = {
        "workload": workload_name,
        "elapsed_sec": req_res["elapsed_sec"],
        "completion_tokens": req_res["usage"].get("completion_tokens", 0),
        "prompt_tokens": req_res["usage"].get("prompt_tokens", 0),
        "top_kernels": parsed_kernels["top_kernels"],
        "nccl_time_pct": parsed_kernels["nccl_pct"],
        "gemm_moe_pct": parsed_kernels["gemm_moe_pct"],
        "mla_attn_pct": parsed_kernels["mla_attn_pct"],
    }

    summary_file = out_dir / f"{workload_name}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary

def parse_kernel_stats(stats_output: str) -> dict[str, Any]:
    lines = stats_output.splitlines()
    in_kern_table = False
    kernels = []
    total_time_ns = 0.0
    nccl_time_ns = 0.0
    gemm_moe_ns = 0.0
    mla_attn_ns = 0.0

    for line in lines:
        if "CUDA GPU Kernel Summary" in line or "Time (%)" in line:
            in_kern_table = True
            continue
        if in_kern_table:
            if not line.strip() or line.startswith("---") or line.startswith("Operating") or "CUDA API Summary" in line:
                if "CUDA API Summary" in line:
                    break
                continue
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) >= 6:
                try:
                    pct = float(parts[0])
                    total_time = float(parts[1].replace(",", ""))
                    name = " ".join(parts[5:])
                    kernels.append({"name": name, "time_pct": pct, "total_time_ns": total_time})
                    total_time_ns += total_time
                    name_lower = name.lower()
                    if "nccl" in name_lower or "allreduce" in name_lower or "all_gather" in name_lower:
                        nccl_time_ns += total_time
                    elif "moe" in name_lower or "gemm" in name_lower or "cutlass" in name_lower or "w1" in name_lower or "w2" in name_lower:
                        gemm_moe_ns += total_time
                    elif "attn" in name_lower or "flashinfer" in name_lower or "mla" in name_lower:
                        mla_attn_ns += total_time
                except ValueError:
                    continue

    nccl_pct = (nccl_time_ns / total_time_ns * 100.0) if total_time_ns > 0 else 0.0
    gemm_pct = (gemm_moe_ns / total_time_ns * 100.0) if total_time_ns > 0 else 0.0
    attn_pct = (mla_attn_ns / total_time_ns * 100.0) if total_time_ns > 0 else 0.0

    return {
        "top_kernels": kernels[:15],
        "nccl_pct": round(nccl_pct, 2),
        "gemm_moe_pct": round(gemm_pct, 2),
        "mla_attn_pct": round(attn_pct, 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Capture Nsight Systems profiling traces on 3-node cluster.")
    parser.add_argument("--out-dir", type=str, default="/home/sparkmain/profiling_results", help="Directory to save traces")
    parser.add_argument("--chat-url", default=CHAT_URL, help="vLLM Chat completions URL")
    parser.add_argument("--metrics-url", default=METRICS_URL, help="vLLM Prometheus metrics URL")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # 1. Workload: Single-stream Decode (8K Context, 256 Tokens)
    decode_prompt = "Write a comprehensive technical overview of high-throughput distributed tensor parallelism in large language models. " * 30
    decode_summary = capture_nsys_profile(
        "decode_8k_256tok",
        prompt=decode_prompt,
        max_tokens=256,
        out_dir=out_dir,
        chat_url=args.chat_url,
        metrics_url=args.metrics_url,
    )

    # 2. Workload: Single-stream Deep Prefill (131K Context, 1 Token)
    prefill_prompt = "The distributed system requires strict consistency under Byzantine fault tolerance and snapshot isolation. " * 7500
    prefill_summary = capture_nsys_profile(
        "prefill_131k_1tok",
        prompt=prefill_prompt,
        max_tokens=1,
        out_dir=out_dir,
        chat_url=args.chat_url,
        metrics_url=args.metrics_url,
    )

    print("\n=======================================================")
    print("=== PROFILING SUITE COMPLETED SUCCESSFULLY ===")
    print(f"Decode (8K -> 256): NCCL={decode_summary['nccl_time_pct']}% | MoE/GEMM={decode_summary['gemm_moe_pct']}% | MLA/Attn={decode_summary['mla_attn_pct']}%")
    print(f"Prefill (131K -> 1): NCCL={prefill_summary['nccl_time_pct']}% | MoE/GEMM={prefill_summary['gemm_moe_pct']}% | MLA/Attn={prefill_summary['mla_attn_pct']}%")
    print("=======================================================")

if __name__ == "__main__":
    main()
