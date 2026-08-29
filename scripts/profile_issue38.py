#!/usr/bin/env python3
"""
scripts/profile_issue38.py
Automated profiler capture script for Issue #38.
Captures both decode (8K/256tok) and deep prefill (131K/1tok) traces.
"""

import os
import sys
import time
import json
import subprocess
import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://192.168.10.10:8100")
CONTAINER_NAME = "dspark-vllm-gx10-vllm-dspark-1"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "20260829-issue38-kernel-profiling")

def run_ssh(host, cmd):
    res = subprocess.run(f"ssh {host} \"{cmd}\"", shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[{host}] Warning: {res.stderr.strip()}")
    return res

def wait_for_health():
    print("Checking cluster health...")
    for _ in range(30):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            if r.status_code == 200:
                print("Server is healthy!")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def clear_profiler_dir():
    print("Clearing /tmp/profiler_traces in container...")
    run_ssh("sparkmain", f"sudo docker exec {CONTAINER_NAME} rm -rf /tmp/profiler_traces && sudo docker exec {CONTAINER_NAME} mkdir -p /tmp/profiler_traces")

def start_profile():
    print("Calling POST /start_profile...")
    r = requests.post(f"{SERVER_URL}/start_profile", timeout=10)
    print(f"Start profile status: {r.status_code}, response: {r.text}")
    assert r.status_code == 200

def stop_profile():
    print("Calling POST /stop_profile...")
    try:
        r = requests.post(f"{SERVER_URL}/stop_profile", timeout=300)
        print(f"Stop profile status: {r.status_code}, response: {r.text[:200]}")
    except Exception as e:
        print(f"Stop profile call notice: {e}")
    time.sleep(3)

def send_request(prompt_tokens, max_tokens):
    print(f"Sending generation request: ~{prompt_tokens} input tokens -> {max_tokens} max output tokens...")
    # DeepSeek token ~ 4 chars
    prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)
    
    payload = {
        "model": "deepseek-v4-flash-0731",
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    
    t0 = time.perf_counter()
    r = requests.post(f"{SERVER_URL}/v1/completions", json=payload, timeout=600)
    t1 = time.perf_counter()
    print(f"Request finished in {t1 - t0:.2f}s, status: {r.status_code}")
    if r.status_code != 200:
        print(f"Error response: {r.text[:500]}")
    assert r.status_code == 200
    res_json = r.json()
    usage = res_json.get("usage", {})
    print(f"Usage: prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}")
    return res_json, t1 - t0

def fetch_and_save_traces(tag):
    tag_dir = os.path.join(OUTPUT_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    print(f"Extracting traces from sparkmain container into {tag_dir}...")
    
    # List files in container /tmp/profiler_traces
    res = run_ssh("sparkmain", f"sudo docker exec {CONTAINER_NAME} ls -la /tmp/profiler_traces")
    print(f"Container traces:\n{res.stdout}")
    
    # Copy files from container to host /tmp, then scp to local
    run_ssh("sparkmain", f"sudo rm -rf /tmp/host_traces && sudo docker cp {CONTAINER_NAME}:/tmp/profiler_traces /tmp/host_traces && sudo chmod -R 777 /tmp/host_traces")
    
    # scp to Windows host
    scp_cmd = f"scp -r sparkmain:/tmp/host_traces/* \"{tag_dir}\""
    print(f"Running scp: {scp_cmd}")
    subprocess.run(scp_cmd, shell=True, check=True)
    print(f"Saved traces to {tag_dir}: {os.listdir(tag_dir)}")

def run_suite():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not wait_for_health():
        print("Cluster not healthy, aborting.")
        return False
    
    # Warmup request to ensure CUDA graph / memory is hot
    print("\n--- Warmup Request (100 tokens) ---")
    send_request(100, 4)
    
    # 1. Profile Single-Stream Decode: 8K context, 16 tokens
    print("\n==========================================")
    print("1. PROFILING DECODE (8K context, 16 tokens)")
    print("==========================================")
    clear_profiler_dir()
    time.sleep(2)
    start_profile()
    time.sleep(1)
    decode_res, decode_time = send_request(8000, 16)
    time.sleep(1)
    stop_profile()
    time.sleep(2)
    fetch_and_save_traces("decode_8k_256tok")
    
    # 2. Profile Deep Prefill: 131K context, 1 token
    print("\n==========================================")
    print("2. PROFILING PREFILL (131K context, 1 token)")
    print("==========================================")
    clear_profiler_dir()
    time.sleep(2)
    start_profile()
    time.sleep(1)
    prefill_res, prefill_time = send_request(131072, 1)
    time.sleep(1)
    stop_profile()
    time.sleep(2)
    fetch_and_save_traces("prefill_131k_1tok")
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decode": {
            "prompt_tokens": decode_res.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": decode_res.get("usage", {}).get("completion_tokens"),
            "latency_sec": round(decode_time, 3),
            "tokens_per_sec": round(decode_res.get("usage", {}).get("completion_tokens", 0) / decode_time, 2)
        },
        "prefill": {
            "prompt_tokens": prefill_res.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": prefill_res.get("usage", {}).get("completion_tokens"),
            "latency_sec": round(prefill_time, 3),
            "prefill_tok_per_sec": round(prefill_res.get("usage", {}).get("prompt_tokens", 0) / prefill_time, 2)
        }
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCompleted profiling suite! Summary:\n{json.dumps(summary, indent=2)}")
    return True

if __name__ == "__main__":
    ok = run_suite()
    sys.exit(0 if ok else 1)
