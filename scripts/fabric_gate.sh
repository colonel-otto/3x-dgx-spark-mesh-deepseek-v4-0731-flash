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

# ---- 3b. Fabric addressing sanity -------------------------------------------
# We once found 192.168.100.2/24 assigned to BOTH of spark1's NICs, advertising
# sparkmain's subnet out the port physically cabled to spark2. rp_filter=2 hid
# it, traffic happened to take the right port anyway, and it was recorded as
# "latent, not active" and left in place. That is the kind of thing that stops
# being latent after a reboot, so check it every time rather than remembering.
#
# NCCL_IB_SUBNET_AWARE_ROUTING=1 picks the HCA whose SUBNET reaches the peer --
# so a duplicated or overlapping subnet directly mis-steers it.
echo "-- fabric addressing (duplicate/overlapping subnets)"
for n in "${NODES[@]}"; do
  dup=$(ssh_node "$n" \
    "ip -o -4 addr show 2>/dev/null | awk '\$2 ~ /^(enp1s0f|enP2p1s0f)/ {print \$4}' \
     | sort | uniq -d" 2>/dev/null)
  # Same address on two fabric NICs is the exact failure we hit.
  same=$(ssh_node "$n" \
    "ip -o -4 addr show 2>/dev/null | awk '\$2 ~ /^(enp1s0f|enP2p1s0f)/ {split(\$4,a,\"/\"); print a[1]}' \
     | sort | uniq -d" 2>/dev/null)
  if [[ -n "$same" ]]; then
    bad "$n: address on MULTIPLE fabric NICs: $(echo "$same" | tr '\n' ' ')" "subnet:$n" "$same"
  elif [[ -n "$dup" ]]; then
    bad "$n: duplicate fabric CIDR: $(echo "$dup" | tr '\n' ' ')" "subnet:$n" "$dup"
  else
    addrs=$(ssh_node "$n" "ip -o -4 addr show 2>/dev/null | awk '\$2 ~ /^(enp1s0f|enP2p1s0f)/ {print \$2\"=\"\$4}' | tr '\n' ' '" 2>/dev/null)
    ok "$n fabric addressing clean: ${addrs}" "subnet:$n" "$addrs"
  fi
done

# ---- 3c. Fabric ARP: every peer on its correct port -------------------------
# Recurring symptom. After the duplicate-address bug (192.168.100.2 on both of
# spark1's NICs) the kernel kept FAILED entries for fabric peers on the WRONG
# port on all three nodes, and those outlived the misconfiguration itself.
#
# A neighbour entry is a port-to-MAC mapping: there is exactly one correct
# answer, decided by which cable is plugged in. An entry naming a port the peer
# is not cabled to is not "stale", it is WRONG, and a wrong entry has no value
# worth preserving -- so this deletes it rather than merely reporting it. That
# is not a configuration change: the kernel immediately re-ARPs out the correct
# port. Every deletion is logged, never silent.
#
# STALE/DELAY on the RIGHT port is normal for an idle link and is left alone.
echo "-- fabric arp (peer on correct port)"
for n in "${NODES[@]}"; do
  # The port a peer is cabled to is the one whose subnet contains it -- ask the
  # routing table rather than hardcoding the topology.
  bad_entries=""
  for p in "${NODES[@]}"; do
    [[ "$n" == "$p" ]] && continue
    paddr=$(fabric_addr_for "$p")
    # Loopback-hosted addresses (sparkmain's /32) have no single cabled port.
    expect=$(ssh_node "$n" "ip route get $paddr 2>/dev/null | grep -oE 'dev [^ ]+' | head -1 | cut -d' ' -f2" 2>/dev/null)
    [[ -z "$expect" || "$expect" == lo ]] && continue
    while read -r dev state; do
      [[ -z "$dev" ]] && continue
      if [[ "$dev" != "$expect" ]]; then
        bad_entries+="$paddr@$dev(want $expect) "
        ssh_node "$n" "sudo ip -4 neigh del $paddr dev $dev" </dev/null >/dev/null 2>&1
      elif [[ "$state" == FAILED || "$state" == INCOMPLETE ]]; then
        bad_entries+="$paddr@$dev=$state "
        ssh_node "$n" "sudo ip -4 neigh del $paddr dev $dev" </dev/null >/dev/null 2>&1
      fi
    done < <(ssh_node "$n" "ip -4 neigh show | awk '\$1==\"$paddr\" {print \$3, \$NF}'" 2>/dev/null)
  done
  if [[ -n "$bad_entries" ]]; then
    bad "$n: wrong-port/failed ARP for a fabric peer, FLUSHED: $bad_entries" "arp:$n" "$bad_entries"
  else
    ok "$n: every fabric peer on its cabled port" "arp:$n" ""
  fi
done

# ---- 3d. Fabric config PERSISTENCE ------------------------------------------
# The live address is not the whole story: it must survive a reboot.
#
# On these boxes NetworkManager is only a RENDERER -- netplan owns the config.
# Every NM connection file lives under /run/NetworkManager/system-connections/,
# which is tmpfs and is wiped on reboot; /etc/NetworkManager/system-connections/
# is EMPTY. So an `nmcli con mod` that does not write through to /etc/netplan/
# looks perfectly applied and silently reverts on the next boot.
#
# This is not hypothetical. spark1's reboot lost runtime-only state and the
# cluster would not start: 192.168.200.1 existed only on sparkmain's loopback at
# runtime, and had sparkmain rebooted the cluster would have been unrecoverable
# without knowing that.
echo "-- fabric config persistence (survives reboot?)"
for n in "${NODES[@]}"; do
  missing=""
  # Every fabric address the node currently holds must appear in /etc/netplan.
  live=$(ssh_node "$n" "ip -o -4 addr show 2>/dev/null | awk '\$2 ~ /^(enp1s0f|enP2p1s0f)/ {print \$4}'" 2>/dev/null)
  # ...plus this node's own advertised fabric address, which may be on loopback.
  self=$(fabric_addr_for "$n")
  for a in $live "$self"; do
    bare="${a%%/*}"
    ssh_node "$n" "sudo grep -qF '$bare' /etc/netplan/*.yaml 2>/dev/null" </dev/null \
      || missing+="$bare "
  done
  if [[ -n "$missing" ]]; then
    bad "$n: NOT persisted in /etc/netplan (reverts on reboot): $missing" "persist:$n" "$missing"
  elif ! ssh_node "$n" 'sudo netplan generate' </dev/null >/dev/null 2>&1; then
    bad "$n: netplan generate FAILS -- config will not apply on boot" "persist:$n" "generate"
  else
    ok "$n: fabric addressing persisted and netplan generates" "persist:$n" ""
  fi
done

engine_up() { ssh_node "$1" 'sudo docker ps --format "{{.Names}}" 2>/dev/null | grep -q vllm-dspark'; }

# ---- 3e. RDMA completion errors in a RUNNING engine -------------------------
# NCCL init succeeding does NOT mean the fabric is healthy. On 2026-08-25 all
# three ranks completed init and every container stayed `running` while live
# RDMA completions failed with IBV_WC_RETRY_EXC_ERR -- the engine simply never
# finished loading. The container has no health check, so Docker could not flag
# it, and `docker ps` looked perfectly normal.
#
# So when an engine is up, read its log for completion errors rather than
# trusting its state.
echo "-- engine rdma health (if running)"
for n in "${NODES[@]}"; do
  if ! engine_up "$n"; then
    skip "$n: no engine running (nothing to check)" "rdma:$n" ""
    continue
  fi
  errs=$(ssh_node "$n" "sudo docker logs --tail 2000 \$(sudo docker ps --format '{{.Names}}' | grep vllm-dspark | head -1) 2>&1 \
      | grep -cE 'IBV_WC_RETRY_EXC_ERR|IBV_WC_[A-Z_]*ERR|GID table changed'" </dev/null 2>/dev/null)
  errs=${errs:-0}
  if [[ "$errs" -gt 0 ]] 2>/dev/null; then
    bad "$n: $errs RDMA completion errors in engine log -- fabric is degraded despite the container running" "rdma:$n" "$errs"
  else
    ok "$n: engine log clean of RDMA completion errors" "rdma:$n" "0"
  fi
done

# ---- 4. NCCL collective bandwidth ------------------------------------------
# The only check that would have caught the 2026-08-25 degradation.
echo "-- nccl collective bandwidth"


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
        -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
        -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
        ${VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1} /results/gate/agbench.py" \
      > "$tmp/$n.log" 2>&1 &
    pids+=($!); rank=$((rank+1))
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done

  # --- TRANSPORT VERIFICATION -----------------------------------------------
  # A bandwidth number is meaningless without knowing what carried it.
  # NCCL_NET=IB is a REQUEST, not a guarantee: if IB init fails NCCL falls back
  # to sockets and still reports a plausible-looking number. We measured exactly
  # that once -- NET/Socket at 0.44 GB/s -- and it looked like a real result.
  # So assert the transport from NCCL's own log rather than trusting the flag.
  local transport gdr
  transport=$(grep -ohE 'via NET/[A-Za-z]+/?[0-9]*' "$tmp/${group[0]}.log" | sort -u | tr '\n' ' ')
  if grep -q 'via NET/Socket' "$tmp/${group[0]}.log" 2>/dev/null; then
    bad "$label: NCCL fell back to NET/Socket -- this is NOT an RDMA measurement" \
        "transport:$label" "socket"
  elif [[ -n "$transport" ]]; then
    # Also surface the merged-device width; a merge that silently did not happen
    # halves the available bandwidth and is invisible in the throughput alone.
    local ndevs
    ndevs=$(grep -ohE 'Made virtual device \[[0-9]+\].*ndevs=[0-9]+' "$tmp/${group[0]}.log" \
            | grep -ohE 'ndevs=[0-9]+' | sort -u | tr '\n' ' ')
    ok "$label transport: ${transport}${ndevs:+(${ndevs% })}" "transport:$label" "$transport"

    # WHICH devices NCCL merged, not just how many. On a RING this is the whole
    # question: merging the two PCIe domains of the SAME port is what published
    # working configs do, while merging f0+f1 spans two DIFFERENT physical
    # ports -- which on a ring face DIFFERENT neighbours. NCCL then believes it
    # has a pipe to each neighbour that it does not have.
    # We previously captured only ndevs and threw the names away, so we could
    # not tell the two cases apart.
    local vdevs
    vdevs=$(grep -ohE 'Made virtual device \[[0-9]+\] name=[^ ]+' "$tmp/${group[0]}.log" \
            | sed -E 's/.*name=//' | sort -u | tr '\n' ' ')
    if [[ -n "$vdevs" ]]; then
      ok "$label merged devices: ${vdevs% }" "vdev:$label" "${vdevs% }"
    else
      skip "$label: no merged virtual device (NCCL using raw HCAs)" "vdev:$label" "none"
    fi
  else
    skip "$label: transport not reported (NCCL_DEBUG output missing)" "transport:$label" ""
  fi

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
