#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
load_config "${1:?usage: bootstrap_nccl.sh CONFIG}"

REF=${NVIDIA_PLAYBOOK_REF:-1fb66f059ee427c5a3678b3117ef73aab042b458}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SETUP="$TMP/setup.sh"
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/${REF}/nvidia/nccl/assets/setup.sh" -o "$SETUP"
chmod +x "$SETUP"

# NVIDIA's setup helper runs locally on node 1 and remotely on the additional
# management IPs. Run this script from node 1.
workers=("${NODES[@]:1}")
echo "Pinned NVIDIA playbook ref: $REF"
echo "Bootstrapping NCCL/nccl-tests on ${#NODES[@]} nodes"
bash "$SETUP" "${workers[@]}"
