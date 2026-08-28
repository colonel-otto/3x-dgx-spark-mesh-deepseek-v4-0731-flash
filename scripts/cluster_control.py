#!/usr/bin/env python3
"""Cluster control helper for 3spark-dsv4.

Manages syncing tp3.env across sparkmain, spark1, spark2, stopping and starting
the cluster, waiting for health readiness, and capturing live process configurations.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

NODES = ["sparkmain", "spark1", "spark2"]
ENDPOINT_HEALTH = "http://192.168.10.223:8100/health"
ENDPOINT_MODELS = "http://192.168.10.223:8100/v1/models"


def run_ssh(host: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    full_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, cmd]
    res = subprocess.run(full_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and res.returncode != 0:
        raise RuntimeError(f"SSH {host} failed ({res.returncode}):\n{res.stderr}\n{res.stdout}")
    return res


def update_remote_env(host: str, updates: dict[str, str | None]) -> None:
    """Updates ~/localai/dspark-vllm-gx10/config/tp3.env on a remote host."""
    cmd = "cat ~/localai/dspark-vllm-gx10/config/tp3.env"
    res = run_ssh(host, cmd)
    lines = res.stdout.splitlines()
    new_lines = []
    seen = set()

    for line in lines:
        matched = False
        for k, v in updates.items():
            pattern = rf"^{k}=.*$"
            if re.match(pattern, line):
                matched = True
                seen.add(k)
                if v is not None:
                    new_lines.append(f"{k}={v}")
                break
        if not matched:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in seen and v is not None:
            new_lines.append(f"{k}={v}")

    content = "\n".join(new_lines) + "\n"
    # Write back via ssh stdin
    write_proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", host, "cat > ~/localai/dspark-vllm-gx10/config/tp3.env"],
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    write_proc.communicate(input=content)
    if write_proc.returncode != 0:
        raise RuntimeError(f"Failed to write tp3.env to {host}")


def stop_cluster() -> None:
    print("[cluster_control] Stopping cluster via /home/sparkmain/bin/dsv4-service-stop ...")
    res = run_ssh("sparkmain", "/home/sparkmain/bin/dsv4-service-stop", check=False)
    print(f"Stop output:\n{res.stdout}")
    time.sleep(5)


def start_cluster() -> None:
    print("[cluster_control] Starting cluster via /home/sparkmain/bin/dsv4-service-start ...")
    res = run_ssh("sparkmain", "/home/sparkmain/bin/dsv4-service-start", check=False)
    print(f"Start output:\n{res.stdout}")
    if res.returncode != 0:
        print(f"Start error:\n{res.stderr}", file=sys.stderr)


def wait_for_ready(timeout_s: int = 600) -> bool:
    print(f"[cluster_control] Waiting up to {timeout_s}s for {ENDPOINT_HEALTH} ...")
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(ENDPOINT_MODELS)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.load(resp)
                    model_id = data["data"][0]["id"]
                    elapsed = int(time.time() - start)
                    print(f"[cluster_control] Cluster READY after {elapsed}s (serving {model_id})")
                    return True
        except Exception:
            pass
        time.sleep(5)
    print(f"[cluster_control] Timeout waiting for cluster readiness ({timeout_s}s)")
    return False


def get_live_config() -> dict:
    res = run_ssh("sparkmain", "ps -eo args | grep vllm | grep -v grep")
    cmd_str = res.stdout.strip()
    return {
        "sparkmain_ps": cmd_str,
    }


def reconfigure(updates: dict[str, str | None]) -> bool:
    print(f"[cluster_control] Applying updates across all 3 nodes: {updates}")
    for host in NODES:
        update_remote_env(host, updates)
        print(f"  [+] Updated tp3.env on {host}")

    stop_cluster()
    start_cluster()
    return wait_for_ready()


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster control orchestration.")
    subparsers = parser.add_subparsers(dest="action")

    subparsers.add_parser("stop")
    subparsers.add_parser("start")
    subparsers.add_parser("wait")
    subparsers.add_parser("status")

    reconfig_parser = subparsers.add_parser("reconfigure")
    reconfig_parser.add_argument("--set", action="append", help="KEY=VALUE to set")
    reconfig_parser.add_argument("--unset", action="append", help="KEY to remove")

    args = parser.parse_args()

    if args.action == "stop":
        stop_cluster()
    elif args.action == "start":
        start_cluster()
    elif args.action == "wait":
        if not wait_for_ready():
            return 1
    elif args.action == "status":
        cfg = get_live_config()
        print(json.dumps(cfg, indent=2))
    elif args.action == "reconfigure":
        updates = {}
        if args.set:
            for item in args.set:
                k, v = item.split("=", 1)
                updates[k] = v
        if args.unset:
            for item in args.unset:
                updates[item] = None
        if not reconfigure(updates):
            return 1
    else:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
