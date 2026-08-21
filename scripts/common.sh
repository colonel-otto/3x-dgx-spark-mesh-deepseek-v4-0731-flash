#!/usr/bin/env bash
set -euo pipefail

load_config() {
  local file="${1:?config file required}"
  [[ -f "$file" ]] || { echo "Config not found: $file" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$file"
  set +a
  : "${NODE_IPS:?NODE_IPS is required}"
  : "${SSH_USER:?SSH_USER is required}"
  : "${MGMT_IFNAME:=enP7s7}"
  : "${RUN_LABEL:=run}"
  read -r -a NODES <<< "$NODE_IPS"
}

stamp() { date -u +%Y%m%dT%H%M%SZ; }

# Per-node login override: SSH_USER_MAP="ip=user ip=user". Falls back to SSH_USER.
# Needed when the nodes do not share a username (common on DGX Spark, where each
# box is provisioned with its own account).
user_for_node() {
  local ip="$1" entry k v
  for entry in ${SSH_USER_MAP:-}; do
    k="${entry%%=*}"; v="${entry#*=}"
    if [[ "$k" == "$ip" ]]; then echo "$v"; return; fi
  done
  echo "${SSH_USER}"
}

ssh_node() {
  local ip="$1"; shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "$(user_for_node "$ip")@${ip}" "$@"
}
