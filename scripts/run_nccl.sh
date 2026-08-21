#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
load_config "${1:?usage: run_nccl.sh CONFIG [OUTFILE]}"
OUT=${2:-/dev/stdout}

[[ ${#NODES[@]} -ge 2 ]] || { echo 'Need at least two nodes' >&2; exit 2; }
case "${TOPOLOGY:-}" in
  direct) [[ ${#NODES[@]} -eq 2 ]] || { echo 'direct requires 2 nodes' >&2; exit 2; } ;;
  ring) [[ ${#NODES[@]} -eq 3 ]] || { echo 'ring requires 3 nodes' >&2; exit 2; } ;;
  *) echo 'TOPOLOGY must be direct or ring' >&2; exit 2 ;;
esac

# Prefer an already-installed NVIDIA helper. Otherwise download the current
# official helper to a temporary file. This script does not modify network config.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
LAUNCH=${NVIDIA_NCCL_LAUNCH:-}
if [[ -z "$LAUNCH" ]]; then
  LAUNCH="$TMP/launch.sh"
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/${NVIDIA_PLAYBOOK_REF:-1fb66f059ee427c5a3678b3117ef73aab042b458}/nvidia/nccl/assets/launch.sh -o "$LAUNCH"
  chmod +x "$LAUNCH"
fi

export MGMT_IFNAME
export BEGIN=${NCCL_BEGIN:-256M}
export END=${NCCL_END:-16G}
export FACTOR=${NCCL_FACTOR:-2}

set +e
bash "$LAUNCH" --topology "$TOPOLOGY" "${NODES[@]}" 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
set -e
[[ $rc -eq 0 ]] || exit $rc

if [[ "$OUT" != /dev/stdout ]] && grep -Eq '#wrong[[:space:]]+0|#wrong = 0' "$OUT"; then
  echo 'NCCL correctness marker found: #wrong = 0'
else
  echo 'NCCL command completed. Inspect output for #wrong = 0 and bandwidth.'
fi
