#!/usr/bin/env bash
set -euo pipefail

# Read-only evidence collector. Review output against SECURITY.md before publishing.
printf 'collected_utc=%s\n' "$(date -u +%FT%TZ)"
printf 'hostname=REDACTED\n'
printf '\n[dgx-release]\n'
sed -E 's/(serial|uuid|hostname).*/\1=REDACTED/I' /etc/dgx-release 2>/dev/null || true
printf '\n[kernel]\n'
uname -srvmo
printf '\n[gpu-driver]\n'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf '\n[cuda]\n'
nvcc --version 2>/dev/null || true
printf '\n[rdma-map]\n'
ibdev2netdev 2>/dev/null || true
printf '\n[rdma-devices]\n'
ibv_devices 2>/dev/null || true
printf '\n[docker]\n'
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}' 2>/dev/null || true
printf '\n[python-packaged-nccl]\n'
python3 - <<'PY' 2>/dev/null || true
from importlib.metadata import version
for package in ("nvidia-nccl-cu13", "nvidia-nccl-cu12"):
    try:
        print(f"{package}={version(package)}")
    except Exception:
        pass
PY
