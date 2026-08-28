#!/usr/bin/env python3
"""Run standardized GuideLLM benchmarks against the 3-node DeepSeek-V4 endpoint.

GuideLLM (https://github.com/vllm-project/guidellm) is the open industry standard for
LLM serving performance evaluation. It captures:
  - TTFT (Time to First Token) distributions (p50, p90, p95, p99)
  - ITL (Inter-Token Latency / streaming jitter)
  - TPOT (Time per Output Token)
  - Concurrency throughput vs latency curves ($cc \\in [1..32]$)
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

GUIDELLM_BIN = "/home/sparkmain/bench_env/bin/guidellm"
DEFAULT_TARGET = "http://127.0.0.1:8100/v1"
DEFAULT_MODEL = "deepseek-v4-flash-0731"
TOKENIZER_MODEL = "deepseek-ai/DeepSeek-V3"

def run_guidellm_benchmark(
    profile: str = "concurrent",
    streams: str = "1,4,8,16,32",
    prompt_tokens: int = 2048,
    output_tokens: int = 256,
    max_duration: int = 60,
    out_dir: str = "results/guidellm-report"
):
    p_out = pathlib.Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        GUIDELLM_BIN, "run",
        "--backend", f"kind=openai_http,target={DEFAULT_TARGET},model={DEFAULT_MODEL}",
        "--tokenizer", f"kind=hf_auto,model={TOKENIZER_MODEL}",
        "--profile", f"kind={profile}",
        "--override", "profile.streams", streams,
        "--data", f"kind=synthetic_text,prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
        "--constraint", f"kind=max_duration,seconds={max_duration}",
        "--output", "kind=console",
        "--output", f"kind=json,path={p_out / 'report.json'}",
        "--output", f"kind=html,path={p_out / 'report.html'}",
        "--disable-console-interactive"
    ]
    
    print("Executing GuideLLM command:")
    print(" ".join(cmd))
    
    res = subprocess.run(cmd, capture_output=False)
    return res.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GuideLLM industry standard serving benchmark")
    parser.add_argument("--profile", default="concurrent", choices=["concurrent", "synchronous", "throughput", "poisson"])
    parser.add_argument("--streams", default="1,4,8,16", help="Comma-separated concurrency streams")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--duration", type=int, default=45, help="Duration per sub-benchmark in seconds")
    parser.add_argument("--out-dir", default="results/guidellm-latest")
    
    args = parser.parse_args()
    code = run_guidellm_benchmark(
        profile=args.profile,
        streams=args.streams,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        max_duration=args.duration,
        out_dir=args.out_dir
    )
    sys.exit(code)
