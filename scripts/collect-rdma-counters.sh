#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

result = {"collected_utc": datetime.now(timezone.utc).isoformat(), "ports": {}}
for device in ("rocep1s0f0", "rocep1s0f1"):
    root = Path("/sys/class/infiniband") / device / "ports" / "1"
    entry = {}
    for name in ("state", "rate"):
        try:
            entry[name] = (root / name).read_text().strip()
        except OSError:
            entry[name] = None
    for name in ("port_xmit_data", "port_rcv_data", "port_xmit_discards",
                 "port_rcv_errors"):
        try:
            entry[name] = int((root / "counters" / name).read_text().strip())
        except OSError:
            entry[name] = None
    result["ports"][device] = entry
print(json.dumps(result, indent=2, sort_keys=True))
PY
