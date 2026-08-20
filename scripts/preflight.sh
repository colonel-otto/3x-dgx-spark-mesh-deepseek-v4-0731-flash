#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"
load_config "${1:?usage: preflight.sh CONFIG}"

expected=${#NODES[@]}
if [[ "${TOPOLOGY:-}" == ring && "$expected" -ne 3 ]]; then
  echo "FAIL: ring topology requires exactly 3 nodes; got $expected" >&2
  exit 1
fi
if [[ "${TOPOLOGY:-}" == direct && "$expected" -ne 2 ]]; then
  echo "FAIL: direct topology requires exactly 2 nodes; got $expected" >&2
  exit 1
fi

fail=0
for ip in "${NODES[@]}"; do
  echo "===== $ip ====="
  if ! ssh_node "$ip" 'true'; then
    echo "FAIL ssh: $ip"; fail=1; continue
  fi
  ssh_node "$ip" "
    set -e
    echo '[dgx-release]'
    cat /etc/dgx-release 2>/dev/null || true
    echo '[gpu]'
    nvidia-smi -L
    echo '[memory]'
    free -h
    echo '[cx7]'
    if command -v ibdev2netdev >/dev/null; then ibdev2netdev; else echo 'ibdev2netdev missing'; fi
    echo '[management]'
    ip -br addr show '${MGMT_IFNAME}' 2>/dev/null || true
    echo '[versions]'
    command -v python3 >/dev/null && python3 --version || true
    command -v vllm >/dev/null && vllm --version || true
    command -v ray >/dev/null && ray --version || true
  " || fail=1
done

[[ $fail -eq 0 ]] || { echo 'PREFLIGHT FAILED' >&2; exit 1; }
echo "PREFLIGHT PASSED: $expected nodes reachable"
