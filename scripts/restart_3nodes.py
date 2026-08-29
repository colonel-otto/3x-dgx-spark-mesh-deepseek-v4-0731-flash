#!/usr/bin/env python3
"""
scripts/restart_3nodes.py
Reliable sequential launcher for 3-node DeepSeek-V4 TP=3 cluster.
"""

import subprocess
import time
import requests
import sys

def run_ssh(host, cmd, check=True):
    print(f"[{host}] Running: {cmd[:80]}...")
    res = subprocess.run(f"ssh {host} \"{cmd}\"", shell=True, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[{host}] ERROR ({res.returncode}): {res.stderr.strip() or res.stdout.strip()}")
        raise RuntimeError(f"Command failed on {host}")
    return res

def restart_cluster():
    print("=== Stopping running containers on all 3 nodes ===")
    for h in ["sparkmain", "spark1", "spark2"]:
        run_ssh(h, "sudo docker rm -f $(sudo docker ps -aq --filter label=com.docker.compose.service=vllm-dspark) 2>/dev/null || true", check=False)

    time.sleep(3)

    print("\n=== Starting Rank 1 (spark1) ===")
    run_ssh("spark1", "cd ~/localai/dspark-vllm-gx10 && ./scripts/start-node.sh config/tp3.env")
    print("Waiting 5 seconds...")
    time.sleep(5)

    print("\n=== Starting Rank 2 (spark2) ===")
    run_ssh("spark2", "cd ~/localai/dspark-vllm-gx10 && ./scripts/start-node.sh config/tp3.env")
    print("Waiting 12 seconds for workers to enter rendezvous...")
    time.sleep(12)

    print("\n=== Starting Rank 0 (sparkmain) ===")
    run_ssh("sparkmain", "cd ~/localai/dspark-vllm-gx10 && ./scripts/start-node.sh config/tp3.env")

    server_url = os.environ.get("SERVER_URL", "http://192.168.10.10:8100")
    print(f"\n=== Monitoring cluster readiness on {server_url}/health ===")
    for i in range(180):
        time.sleep(5)
        try:
            res = requests.get(f"{server_url}/health", timeout=3)
            if res.status_code == 200:
                print(f"\nSUCCESS: vLLM 3-node cluster is healthy and ready after {(i+1)*5}s!")
                return True
        except Exception:
            pass
        if (i + 1) % 4 == 0:
            print(f"Elapsed: {(i+1)*5}s... still initializing (weights / CUDA graphs)")
            # Check worker status
            for h in ["sparkmain", "spark1", "spark2"]:
                st = run_ssh(h, "sudo docker inspect -f '{{.State.Status}}' dspark-vllm-gx10-vllm-dspark-1 2>/dev/null || echo missing", check=False).stdout.strip()
                print(f"  - {h}: {st}")
    print("\nERROR: Timed out waiting for cluster readiness.")
    return False

if __name__ == "__main__":
    ok = restart_cluster()
    sys.exit(0 if ok else 1)
