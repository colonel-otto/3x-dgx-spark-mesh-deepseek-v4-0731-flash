#!/usr/bin/env python3
import subprocess
from pathlib import Path

hosts = ["localhost", "spark1", "spark2"]

env_line = 'PROFILER_CONFIG=\'{"profiler":"torch","torch_profiler_dir":"/tmp/profiler_traces"}\''

for h in ["sparkmain", "spark1", "spark2"]:
    cmd = (
        f"ssh {h} \"python3 -c '\\n"
        f"from pathlib import Path\\n"
        f"p = Path.home() / \\\"localai/dspark-vllm-gx10/config/tp3.env\\\"\\n"
        f"lines = [l for l in p.read_text().splitlines() if not l.startswith(\\\"PROFILER_CONFIG\\\")]\\n"
        f"lines.append(\\\"{env_line}\\\")\\n"
        f"p.write_text(\\\"\\\\n\\\".join(lines) + \\\"\\\\n\\\")\\n"
        f"'\""
    )
    print(f"Configuring {h}...")
    subprocess.run(cmd, shell=True, check=True)
print("Done configuring all 3 nodes.")
