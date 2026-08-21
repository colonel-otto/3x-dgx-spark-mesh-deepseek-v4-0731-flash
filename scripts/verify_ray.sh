#!/usr/bin/env bash
set -euo pipefail
expected=${1:-3}
command -v ray >/dev/null || { echo 'ray CLI not found' >&2; exit 2; }
ray status
python3 - "$expected" <<'PY'
import json, subprocess, sys
expected=int(sys.argv[1])
try:
    raw=subprocess.check_output(["ray","list","nodes","--format=json"], text=True)
    nodes=json.loads(raw)
except Exception as exc:
    print(f"Unable to machine-check ray nodes: {exc}")
    sys.exit(2)
alive=[n for n in nodes if n.get("is_alive", n.get("state") == "ALIVE")]
print(f"Ray alive nodes: {len(alive)} / expected {expected}")
if len(alive) != expected:
    sys.exit(1)
PY
