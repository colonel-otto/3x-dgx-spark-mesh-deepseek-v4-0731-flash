#!/usr/bin/env python3
import subprocess
from pathlib import Path

env_file = Path.home() / "localai/dspark-vllm-gx10/config/tp3.env"
lines = [l for l in env_file.read_text().splitlines() if not l.startswith("PROFILER_CONFIG")]
lines.append('PROFILER_CONFIG={"profiler":"torch","torch_profiler_dir":"/tmp/profiler_traces"}')
env_file.write_text("\n".join(lines) + "\n")
print(f"Updated {env_file}")
