#!/usr/bin/env bash
# Fabric gate -- run this BEFORE every benchmark, monitor, or tuning run.
#
# WHY THIS EXISTS
#
# On 2026-08-25 we discovered spark1 had been running at ~0.7 GB/s collective
# bandwidth against 4.6 GB/s for a healthy pair -- a 6.8x deficit, with ZERO
# error indicators. Port state ACTIVE, link speed 200,000 Mb/s, every error
# counter 0, identical firmware and PCIe width, and NCCL even selected the same
# merged NET/IB/2 transport. Nothing in ibstat, nvidia-smi, lspci, or the NCCL
# logs showed a problem.
#
# Months of benchmarks were taken on that fabric and had to be marked
# provisional (issues #14, #15). This gate exists so that never happens again:
# a degraded link now FAILS LOUDLY before a run instead of silently producing
# numbers that look plausible and mean nothing.
#
# TCP throughput is NOT a valid substitute. It showed the degraded link at
# 858 MB/s vs 1,019 for a healthy one -- a 1.19x difference hiding a 6.8x RDMA
# deficit -- because it never touches the RDMA verbs path. Only an NCCL
# collective exercises what vLLM actually uses.
#
# WHAT IT CHECKS, in ascending cost
#
#   1. SSH liveness      -- reads the banner. An open port 22 is not a live host;
#                           that lied to us twice during a wedge.
#   2. Mesh reachability -- all N*(N-1) directed pairs over the FABRIC addresses,
#                           not the management LAN. Gloo needs a full mesh, and a
#                           node silently routing over WiFi is a real failure we
#                           have hit.
#   3. Fabric latency    -- RTT per directed pair, with a ceiling. Catches a link
#                           that is up and routable but pathological.
#   4. NCCL bandwidth    -- the check that actually matters. Requires the GPUs to
#                           be free, so it is skipped by default when vLLM is up
#                           (see --nccl).
#
# EXIT CODES
#   0  all checks passed (or bandwidth skipped with --nccl=skip)
#   1  a check FAILED -- do not benchmark, investigate first
#   2  usage / config error
#
# USAGE
#   scripts/fabric_gate.sh configs/3spark.env                  # full gate
#   scripts/fabric_gate.sh configs/3spark.env --nccl=skip      # engine is up
#   scripts/fabric_gate.sh configs/3spark.env --nccl=pairs     # pairs only
#   scripts/fabric_gate.sh configs/3spark.env --json OUT.json
#
# From another script, gate a run in one line:
#   scripts/fabric_gate.sh "$CONFIG" || exit 1
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"
set +e   # common.sh sets -e; this script reports failures rather than aborting

CONFIG="${1:-}"
[[ -n "$CONFIG" && "$CONFIG" != --* ]] || {
  sed -n '/^# USAGE/,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d' >&2
  exit 2
}
shift
load_config "$CONFIG"

# ---- tunables (override in the config file or the environment) --------------
NCCL_MODE="${FABRIC_GATE_NCCL:-auto}"      # auto | pairs | all | full | skip
BUSBW_MIN="${FABRIC_GATE_BUSBW_MIN:-3.5}"  # GB/s @64MiB, pairwise. Healthy ~4.6
BUSBW_MIN_ALL="${FABRIC_GATE_BUSBW_MIN_ALL:-2.5}"  # N-rank. Healthy ~3.25 at N=3
RTT_MAX_MS="${FABRIC_GATE_RTT_MAX_MS:-2.0}"
JSON_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nccl=*) NCCL_MODE="${1#*=}" ;;
    --json)   JSON_OUT="${2:?--json needs a path}"; shift ;;
    --busbw-min=*) BUSBW_MIN="${1#*=}" ;;
    --rtt-max-ms=*) RTT_MAX_MS="${1#*=}" ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# FABRIC_ADDRS maps a node's SSH address to the address it advertises to peers
# over the RDMA fabric. These are NOT the management IPs -- checking the
# management LAN would have passed cleanly all through the degradation.
# Format: "ssh_addr=fabric_addr ssh_addr=fabric_addr"
: "${FABRIC_ADDRS:=}"
fabric_addr_for() {
  local n="$1" entry k v
  for entry in ${FABRIC_ADDRS}; do
    k="${entry%%=*}"; v="${entry#*=}"
    [[ "$k" == "$n" ]] && { echo "$v"; return; }
  done
  echo "$n"   # fall back to the SSH address
}

PASS=0; FAIL=0; SKIP=0
declare -a RESULTS
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); RESULTS+=("{\"check\":\"$2\",\"status\":\"pass\",\"detail\":\"$3\"}"); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); RESULTS+=("{\"check\":\"$2\",\"status\":\"fail\",\"detail\":\"$3\"}"); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIP=$((SKIP+1)); RESULTS+=("{\"check\":\"$2\",\"status\":\"skip\",\"detail\":\"$3\"}"); }

echo "== fabric gate: ${#NODES[@]} nodes from $(basename "$CONFIG") =="

# ---- 1. SSH liveness --------------------------------------------------------
# A host can answer on port 22 while being unable to run anything. Require a
# real command to return, not just a successful connect.
echo "-- ssh liveness"
for n in "${NODES[@]}"; do
  host=$(ssh_node "$n" 'hostname' 2>/dev/null)
  if [[ -n "$host" ]]; then ok "$n -> $host" "ssh:$n" "$host"
  else bad "$n unreachable (no command output; port 22 alone is not proof of life)" "ssh:$n" ""; fi
done
[[ $FAIL -eq 0 ]] || { echo; echo "GATE FAILED: a node is unreachable. Fix that before anything else." >&2; exit 1; }

# ---- 2 & 3. Mesh reachability and latency, over the FABRIC ------------------
# Every directed pair. Gloo's connectFullMesh needs all of them, and we have
# seen one direction fail while its reverse worked.
echo "-- fabric mesh (all directed pairs) + latency"
for s in "${NODES[@]}"; do
  for d in "${NODES[@]}"; do
    [[ "$s" == "$d" ]] && continue
    dst=$(fabric_addr_for "$d")
    rtt=$(ssh_node "$s" "ping -c3 -i0.2 -W2 -q $dst 2>/dev/null | awk -F'/' '/rtt|round-trip/{print \$5}'" 2>/dev/null)
    if [[ -z "$rtt" ]]; then
      bad "$s -> $d ($dst) UNREACHABLE over fabric" "mesh:$s>$d" ""
    elif awk -v r="$rtt" -v m="$RTT_MAX_MS" 'BEGIN{exit !(r>m)}'; then
      bad "$s -> $d ($dst) rtt ${rtt}ms exceeds ${RTT_MAX_MS}ms" "mesh:$s>$d" "$rtt"
    else
      ok "$s -> $d ($dst) rtt ${rtt}ms" "mesh:$s>$d" "$rtt"
    fi
  done
done

# ---- 4. NCCL collective bandwidth ------------------------------------------
# The only check that would have caught the 2026-08-25 degradation.
echo "-- nccl collective bandwidth"

engine_up() { ssh_node "$1" 'sudo docker ps --format "{{.Names}}" 2>/dev/null | grep -q vllm-dspark'; }

busy=0
for n in "${NODES[@]}"; do engine_up "$n" && busy=1; done

if [[ "$NCCL_MODE" == auto ]]; then
  if [[ $busy -eq 1 ]]; then
    NCCL_MODE=skip
    echo "  (engine is running; bandwidth needs the GPUs. Pass --nccl=pairs after stopping it.)"
  else
    NCCL_MODE=pairs
  fi
fi

run_nccl_group() {
  # $1 = space-separated node list, $2 = label, $3 = threshold
  local group=($1) label="$2" thresh="$3" world=${#1} rank=0 pids=() rc=0
  world=$(wc -w <<< "$1")
  local master; master=$(fabric_addr_for "${group[0]}")
  local tmp; tmp=$(mktemp -d)

  for n in "${group[@]}"; do
    local home; home=$(ssh_node "$n" 'echo $HOME')
    # These container flags MIRROR THE COMPOSE SERVICE deliberately. Dropping
    # --device /dev/infiniband silently measures socket fallback and yields a
    # plausible number that means nothing.
    ssh_node "$n" "sudo docker run --rm --network host --ipc host --shm-size 64gb \
        --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
        --device /dev/infiniband:/dev/infiniband \
        --entrypoint python3 \
        -v $home/results:/results \
        -e RANK=$rank -e WORLD_SIZE=$world \
        -e INIT_METHOD=tcp://$master:${FABRIC_GATE_PORT:-29555} \
        -e TAG=$label \
        -e NCCL_IB_HCA=${NCCL_IB_HCA_DEFAULT:-rocep1s0f0,rocep1s0f1} \
        -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB \
        -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
        ${VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1} /results/gate/agbench.py" \
      > "$tmp/$n.log" 2>&1 &
    pids+=($!); rank=$((rank+1))
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done

  local bw
  bw=$(awk '/^64MiB/{for(i=1;i<=NF;i++) if($i ~ /^busbw=/) {sub(/busbw=/,"",$i); print $i}}' "$tmp/${group[0]}.log" | tail -1)
  [[ -z "$bw" ]] && bw=$(awk '/64MiB/{print $(NF-1)}' "$tmp/${group[0]}.log" | tail -1)

  if [[ -z "$bw" ]]; then
    bad "$label: no bandwidth reported (rc=$rc) -- see $tmp/${group[0]}.log" "nccl:$label" ""
    cp "$tmp"/*.log "${TMPDIR:-/tmp}/" 2>/dev/null
  elif awk -v b="$bw" -v t="$thresh" 'BEGIN{exit !(b<t)}'; then
    bad "$label: ${bw} GB/s @64MiB is BELOW ${thresh} -- degraded node, reboot it before measuring" "nccl:$label" "$bw"
  else
    ok "$label: ${bw} GB/s @64MiB" "nccl:$label" "$bw"
  fi
  rm -rf "$tmp"
}

case "$NCCL_MODE" in
  skip)
    skip "bandwidth not measured (engine up, or --nccl=skip). THIS IS THE CHECK THAT MATTERS -- run it when the engine is down." "nccl" ""
    ;;
  pairs|all|full)
    # agbench.py must exist on every node; ship it from the repo so the gate is
    # self-contained rather than depending on hand-placed copies.
    AG_SRC="$HERE/../results/20260824-seqs32-nccl/agbench.py"
    if [[ ! -f "$AG_SRC" ]]; then
      bad "agbench.py not found at $AG_SRC" "nccl:deploy" ""
    else
      for n in "${NODES[@]}"; do
        home=$(ssh_node "$n" 'echo $HOME')
        ssh_node "$n" "mkdir -p $home/results/gate" </dev/null
        scp -q -o BatchMode=yes "$AG_SRC" "$(user_for_node "$n")@${n}:$home/results/gate/agbench.py"
      done
      if [[ "$NCCL_MODE" == pairs || "$NCCL_MODE" == full ]]; then
        for ((i=0; i<${#NODES[@]}; i++)); do
          for ((j=i+1; j<${#NODES[@]}; j++)); do
            run_nccl_group "${NODES[$i]} ${NODES[$j]}" "gate-p$i$j" "$BUSBW_MIN"
          done
        done
      fi
      if [[ "$NCCL_MODE" == all || "$NCCL_MODE" == full ]] && [[ ${#NODES[@]} -gt 2 ]]; then
        run_nccl_group "${NODES[*]}" "gate-all" "$BUSBW_MIN_ALL"
      fi
    fi
    ;;
  *) echo "unknown --nccl mode: $NCCL_MODE (auto|pairs|all|full|skip)" >&2; exit 2 ;;
esac

# ---- verdict ----------------------------------------------------------------
echo
echo "== $PASS passed, $FAIL failed, $SKIP skipped =="

if [[ -n "$JSON_OUT" ]]; then
  { printf '{"timestamp_utc":"%s","config":"%s","nccl_mode":"%s","pass":%d,"fail":%d,"skip":%d,"checks":[' \
      "$(stamp)" "$(basename "$CONFIG")" "$NCCL_MODE" "$PASS" "$FAIL" "$SKIP"
    (IFS=,; echo -n "${RESULTS[*]}")
    echo ']}'
  } > "$JSON_OUT"
  echo "wrote $JSON_OUT"
fi

if [[ $FAIL -gt 0 ]]; then
  echo "GATE FAILED -- do not trust any benchmark taken now." >&2
  exit 1
fi
if [[ $SKIP -gt 0 ]]; then
  echo "GATE PASSED (with skips). Bandwidth unverified -- see above." >&2
fi
exit 0
