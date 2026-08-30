#!/usr/bin/env bash
# discover_fabric_addrs.sh -- auto-discover the DGX Spark cluster's RDMA
# fabric addressing so FABRIC_ADDRS never has to be hand-edited again.
#
# WHY THIS EXISTS
#
# "NVIDIA Sync" periodically re-applies netplan CLUSTER-WIDE and RENUMBERS
# the fabric addresses (see feedback_nvidia_sync_kills_live_nccl and
# project_spark_fabric_sync_renumber). Every time that happens, a
# hand-written `FABRIC_ADDRS=` line in configs/3spark-live.env silently
# starts pointing at addresses that no longer exist on any interface. The
# fabric itself is usually perfectly healthy -- scripts/fabric_gate.sh just
# fails ~8 checks against stale IPs, and that misdiagnosis (fabric is "down")
# wastes real time chasing a problem that does not exist. This script
# replaces "remember to go re-read `ip addr` on 3 boxes and hand-edit the
# config" with a single discovery pass that is safe to run any time,
# including while an engine is up (it only ever runs `ip`, `ping`, `cat`,
# `grep` -- nothing here touches the GPUs or docker).
#
# TOPOLOGY
#
# Each node has 4 ConnectX RDMA NICs (interface names of the shape
# enp1s0f0np0 / enP2p1s0f1np1 -- "enp*s*f*np*" or "enP*p*s*f*np*") in a
# direct-attach mesh, no switch. Addresses currently live somewhere in
# 10.100.0.0/16, one /24 per physical cable segment -- but WHICH /24 lands on
# WHICH cable is exactly what Sync scrambles, so this script never hardcodes
# a /24. It only trusts two signals: the ConnectX interface-name shape, and
# (as a fallback / cross-check) the address falling inside 10.100.0.0/16.
# The management NIC (enP7s7, on the site's management /24 + DHCP) and any
# Wi-Fi interface (wl*) are explicitly excluded -- a node silently routing
# fabric traffic over WiFi or picking a management IP is a real failure this
# repo has hit before (see FABRIC_ADDRS comment in configs/3spark.env.example
# and scripts/fabric_gate.sh's mesh-reachability check).
#
# WHY "REACHABLE FROM BOTH PEERS" IS THE SELECTION CRITERION
#
# The 4 fabric NICs per node are 2 parallel point-to-point links to EACH of
# the other two nodes (dual-port mesh), so a node's 4 candidate addresses are
# not interchangeable: some routes only reach one peer, some reach both
# (this cluster's netplan installs host routes across the mesh -- see
# project_spark_mesh_gloo_routing). Gloo/NCCL need one address per node that
# every OTHER node can actually dial. Picking any address that merely exists
# locally is not good enough -- it has to be proven reachable by pinging it
# FROM each of the other two nodes over SSH. That is the whole point of this
# script: it doesn't guess, it verifies from both sides before it will ever
# print an address.
#
# USAGE
#   scripts/discover_fabric_addrs.sh                    # print FABRIC_ADDRS=... line
#   scripts/discover_fabric_addrs.sh --json              # full discovery detail as JSON
#   scripts/discover_fabric_addrs.sh --check [--env-file PATH]   # compare vs config, exit 1 on drift
#   scripts/discover_fabric_addrs.sh --write [--env-file PATH]   # update config in place (with .bak)
#   scripts/discover_fabric_addrs.sh --help
#
# EXIT CODES
#   0  success (or --check found no drift)
#   1  --check found drift
#   2  usage / config error
#   3  discovery failed (unreachable node, zero candidates, no dual-reachable address)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

# Node SSH aliases -- override with DISCOVER_NODES="a b c" if ever needed.
# Kept as a single variable so it is not hardcoded three separate times.
NODES=${DISCOVER_NODES:-"sparkmain spark1 spark2"}

DEFAULT_ENV_FILE="$ROOT/configs/3spark-live.env"
ENV_FILE="$DEFAULT_ENV_FILE"
MODE="print"   # print | json | check | write

SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
# -n redirects ssh's stdin from /dev/null: without it, ssh calls made from
# inside a `while read ... done <<< "$var"` loop silently drain the loop's
# stdin and break every read after the first iteration.

usage() {
  sed -n '/^# discover_fabric_addrs.sh/,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
}

die() { echo "discover_fabric_addrs.sh: ERROR: $*" >&2; exit "${2:-3}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) MODE="json" ;;
    --check) MODE="check" ;;
    --write) MODE="write" ;;
    --env-file) ENV_FILE="${2:?--env-file needs a path}"; shift ;;
    --env-file=*) ENV_FILE="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ---- ssh helper -------------------------------------------------------------
ssh_node() {
  local node="$1"; shift
  ssh "${SSH_OPTS[@]}" "$node" "$@"
}

# ---- 1. enumerate fabric interface candidates on each node -----------------
# For each node, print lines "IFACE ADDR" for every interface that:
#   - matches the ConnectX name shape enp<...>f<...>np<...> / enP<...>p<...>s<...>f<...>np<...>, AND
#   - is NOT the management NIC (enP7s7) or a Wi-Fi NIC (wl*)
# As a belt-and-braces cross-check we also require the address to fall
# inside 10.100.0.0/16 (catches any oddly-named NIC and rejects the site's
# management /24 and docker 172.17.0.0/16 addresses outright). We deliberately
# do NOT hardcode which /24 -- Sync changes that.
fetch_candidates() {
  local node="$1"
  # BusyBox-free, plain ip -o -4 addr parsing on the remote side. One line
  # per interface: "IFACE ADDR" for the ones that pass the filter.
  ssh_node "$node" '
    ip -o -4 addr show 2>/dev/null | while read -r _idx ifname _rest; do
      # extract "inet A.B.C.D/NN" from the line
      addr=$(echo "$_rest" | grep -oE "inet [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | awk "{print \$2}")
      [ -n "$addr" ] || continue
      case "$ifname" in
        enP7s7) continue ;;         # management NIC, always excluded
        wl*) continue ;;            # Wi-Fi, always excluded
        enp[0-9]*s[0-9]*f[0-9]*np[0-9]*|enP[0-9]*p[0-9]*s[0-9]*f[0-9]*np[0-9]*) : ;;
        *) continue ;;              # not a ConnectX-shaped fabric NIC
      esac
      case "$addr" in
        10.100.*) : ;;              # cross-check: must be in the fabric supernet
        *) continue ;;
      esac
      echo "$ifname $addr"
    done
  '
}

# ---- main discovery -----------------------------------------------------
declare -A CAND_LIST     # node -> newline-separated "iface addr"
declare -A CHOSEN        # node -> chosen address
declare -A REACH_DETAIL  # "node|addr" -> space-separated list of peers that reached it

read -r -a NODE_ARR <<< "$NODES"
[[ ${#NODE_ARR[@]} -ge 2 ]] || die "need at least 2 nodes, got: $NODES" 2

for n in "${NODE_ARR[@]}"; do
  # SSH liveness check first -- an unreachable node must fail loudly, not
  # silently produce a partial/garbage line.
  if ! ssh_node "$n" 'true' >/dev/null 2>&1; then
    die "node '$n' is not reachable over SSH -- cannot discover fabric addressing" 3
  fi

  out=$(fetch_candidates "$n") || die "failed to enumerate interfaces on '$n'"
  if [[ -z "$out" ]]; then
    die "node '$n' yielded zero fabric-NIC candidates (expected enp*/enP*...np* interfaces in 10.100.0.0/16) -- check interface naming or Sync state" 3
  fi
  CAND_LIST["$n"]="$out"
done

# For each node, test each of its candidate addresses against every OTHER
# node (ping FROM the peer TO the candidate, over SSH). An address only
# qualifies if EVERY other node can reach it.
for n in "${NODE_ARR[@]}"; do
  chosen_addr=""
  while read -r ifname addr; do
    [[ -n "$addr" ]] || continue
    reachers=()
    all_ok=1
    for peer in "${NODE_ARR[@]}"; do
      [[ "$peer" == "$n" ]] && continue
      if ssh_node "$peer" "ping -c1 -W2 $addr" >/dev/null 2>&1; then
        reachers+=("$peer")
      else
        all_ok=0
      fi
    done
    REACH_DETAIL["$n|$addr"]="${reachers[*]}"
    if [[ $all_ok -eq 1 && -z "$chosen_addr" ]]; then
      chosen_addr="$addr"
      # keep testing remaining candidates so JSON output has full detail,
      # but do not overwrite the already-chosen (first-match, deterministic) address
    fi
  done <<< "${CAND_LIST[$n]}"

  [[ -n "$chosen_addr" ]] || die "node '$n' has no fabric address reachable from BOTH other peers -- candidates were:
$(echo "${CAND_LIST[$n]}")
This usually means Sync mid-renumbered the mesh, or a cable/route is actually down. Not safe to emit a FABRIC_ADDRS line." 3
  CHOSEN["$n"]="$chosen_addr"
done

# ---- build the FABRIC_ADDRS line --------------------------------------------
build_line() {
  local parts=()
  for n in "${NODE_ARR[@]}"; do
    parts+=("$n=${CHOSEN[$n]}")
  done
  echo "FABRIC_ADDRS=\"${parts[*]}\""
}

FABRIC_LINE=$(build_line)

# ---- JSON emitter (hand-rolled, no jq dependency) ---------------------------
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

emit_json() {
  echo "{"
  echo "  \"discovered_at_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"nodes\": [$(printf '"%s",' "${NODE_ARR[@]}" | sed 's/,$//')],"
  echo "  \"fabric_addrs\": {"
  local i=0 n
  for n in "${NODE_ARR[@]}"; do
    i=$((i+1))
    sep=","; [[ $i -eq ${#NODE_ARR[@]} ]] && sep=""
    echo "    \"$n\": \"${CHOSEN[$n]}\"$sep"
  done
  echo "  },"
  echo "  \"fabric_addrs_line\": \"$(json_escape "$FABRIC_LINE")\","
  echo "  \"candidates\": {"
  i=0
  for n in "${NODE_ARR[@]}"; do
    i=$((i+1))
    sep=","; [[ $i -eq ${#NODE_ARR[@]} ]] && sep=""
    echo "    \"$n\": ["
    local lines total j
    mapfile -t lines <<< "${CAND_LIST[$n]}"
    total=${#lines[@]}
    j=0
    for line in "${lines[@]}"; do
      j=$((j+1))
      iface="${line%% *}"; addr="${line#* }"
      reach="${REACH_DETAIL["$n|$addr"]:-}"
      # peers-that-reached-it as a JSON array
      reach_json=""
      for p in $reach; do reach_json+="\"$p\","; done
      reach_json="${reach_json%,}"
      is_chosen="false"; [[ "$addr" == "${CHOSEN[$n]}" ]] && is_chosen="true"
      csep=","; [[ $j -eq $total ]] && csep=""
      echo "      {\"iface\": \"$iface\", \"addr\": \"$addr\", \"reachable_from\": [$reach_json], \"chosen\": $is_chosen}$csep"
    done
    echo "    ]$sep"
  done
  echo "  }"
  echo "}"
}

# ---- --check: compare against an env file's FABRIC_ADDRS= line -------------
do_check() {
  [[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE" 2
  local current
  current=$(grep -E '^FABRIC_ADDRS=' "$ENV_FILE" || true)
  if [[ -z "$current" ]]; then
    echo "No FABRIC_ADDRS= line found in $ENV_FILE" >&2
    echo "Discovered:" >&2
    echo "  $FABRIC_LINE" >&2
    exit 1
  fi
  if [[ "$current" == "$FABRIC_LINE" ]]; then
    echo "OK: FABRIC_ADDRS in $ENV_FILE matches live fabric addressing."
    echo "  $FABRIC_LINE"
    exit 0
  fi
  echo "DRIFT: FABRIC_ADDRS in $ENV_FILE does NOT match the live fabric." >&2
  echo "  file:      $current" >&2
  echo "  discovered: $FABRIC_LINE" >&2
  echo "This is expected after NVIDIA Sync renumbers the mesh -- re-run with --write to fix it." >&2
  exit 1
}

# ---- --write: update the FABRIC_ADDRS= line in place, keeping a backup -----
do_write() {
  [[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE" 2
  if ! grep -qE '^FABRIC_ADDRS=' "$ENV_FILE"; then
    die "no FABRIC_ADDRS= line found in $ENV_FILE -- refusing to guess where to insert one" 2
  fi
  local bak="${ENV_FILE}.bak-$(date -u +%Y%m%d-%H%M)"
  cp -p "$ENV_FILE" "$bak" || die "failed to create backup $bak"

  # Replace only the FABRIC_ADDRS= line, preserving every surrounding
  # comment/blank line untouched. Escape & and \ for sed's replacement side.
  local esc
  esc=$(printf '%s' "$FABRIC_LINE" | sed 's/[&\\]/\\&/g')
  sed -i.tmp -E "s|^FABRIC_ADDRS=.*|${esc}|" "$ENV_FILE" && rm -f "${ENV_FILE}.tmp"

  echo "Updated $ENV_FILE (backup: $bak)"
  echo "  $FABRIC_LINE"
}

case "$MODE" in
  print) echo "$FABRIC_LINE" ;;
  json)  emit_json ;;
  check) do_check ;;
  write) do_write ;;
esac
