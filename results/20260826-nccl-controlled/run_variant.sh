#!/usr/bin/env bash
# Controlled NCCL all_gather reproduction runner (issue #18).
#
# Usage:
#   run_variant.sh <tag> <world> <fabric|mgmt> <sizeflag> [KEY=VAL ...]
#     sizeflag: "32M" or "16G"
#     KEY=VAL:  extra NCCL env vars, or "UNSET_IB_HCA" to drop NCCL_IB_HCA
#
# Runs one process per node (1 GPU each) under mpirun, inside the ntests:2.30.7
# container which carries NCCL 2.30.7 + OpenMPI 4.1.2 + an sshd on port 2299.
set -uo pipefail

TAG="$1"; WORLD="$2"; BOOT="$3"; SIZE="$4"; shift 4
EXTRA=("$@")

IMG=ntests:2.30.7
CN=ntrun
OUT=/tmp/ntresults
mkdir -p "$OUT"
LOG="$OUT/${TAG}_w${WORLD}_${SIZE}.log"

ALLNODES=(sparkmain spark1 spark2)
NODES=("${ALLNODES[@]:0:$WORLD}")

# bootstrap addresses
mgmt_of() { case "$1" in sparkmain) echo 192.168.1.223;; spark1) echo 192.168.1.50;; spark2) echo 192.168.1.27;; esac; }
fab_of()  { case "$1" in sparkmain) echo 192.168.200.1;; spark1) echo 192.168.100.2;; spark2) echo 192.168.101.2;; esac; }
# the fabric interface each node's advertised fabric address lives on
fabif_of(){ case "$1" in sparkmain) echo lo;; spark1) echo enp1s0f1np1;; spark2) echo enp1s0f0np0;; esac; }

cleanup() {
  for n in "${ALLNODES[@]}"; do
    ssh -n -o BatchMode=yes "$n" "docker rm -f $CN >/dev/null 2>&1" >/dev/null 2>&1
  done
}
# cleanup runs explicitly at the end, AFTER debug logs are collected
cleanup

echo "=== VARIANT $TAG | world=$WORLD | bootstrap=$BOOT | size=$SIZE | extra='${EXTRA[*]}' ===" | tee "$LOG"
echo "=== started $(date -Iseconds) ===" | tee -a "$LOG"

# --- start one container per participating node ------------------------------
for n in "${NODES[@]}"; do
  ssh -n -o BatchMode=yes "$n" \
    "docker run -d --name $CN --gpus all --network host \
       --device /dev/infiniband --ulimit memlock=-1 --shm-size 64gb --ipc=host \
       --cap-add IPC_LOCK --entrypoint bash $IMG -c '/usr/sbin/sshd -D -p 2299'" >/dev/null 2>&1
  if [ $? -ne 0 ]; then echo "FATAL: container start failed on $n" | tee -a "$LOG"; exit 1; fi
done
sleep 5

# verify sshd reachable container-to-container over the chosen bootstrap net
# NOTE: mpirun's OWN transport (oob/btl) always uses the management LAN.
# The fabric is a set of point-to-point /30 links, NOT a full mesh, so MPI's
# TCP BTL cannot route across it (spark2 has no route to 192.168.110.1). MPI
# here is only a process launcher; the variable under test is NCCL_SOCKET_IFNAME,
# which is what selects NCCL's own bootstrap channel.
HOSTS=""
for n in "${NODES[@]}"; do
  HOSTS="${HOSTS:+$HOSTS,}$(mgmt_of "$n"):1"
done
echo "mpirun hostlist (launcher only, always mgmt): $HOSTS" | tee -a "$LOG"

# --- NCCL environment --------------------------------------------------------
declare -A ENVMAP
ENVMAP[NCCL_DEBUG]=INFO
ENVMAP[NCCL_DEBUG_SUBSYS]="INIT,NET,GRAPH,ENV,TUNING"
ENVMAP[NCCL_IB_DISABLE]=0
ENVMAP[NCCL_IB_SUBNET_AWARE_ROUTING]=1
ENVMAP[NCCL_NET_PLUGIN]=none
ENVMAP[NCCL_IB_HCA]="=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"
ENVMAP[LD_LIBRARY_PATH]="/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/openmpi/lib"
# Send NCCL_DEBUG to its own file per rank so it does not interleave into and
# corrupt the results table on stdout. Collected back afterwards.
ENVMAP[NCCL_DEBUG_FILE]="/tmp/nccldbg.%h.%p.log"

DROP_HCA=0
for kv in "${EXTRA[@]}"; do
  if [ "$kv" = "UNSET_IB_HCA" ]; then DROP_HCA=1; continue; fi
  ENVMAP["${kv%%=*}"]="${kv#*=}"
done
[ "$DROP_HCA" = 1 ] && unset 'ENVMAP[NCCL_IB_HCA]'

# NCCL bootstrap interface: this is the ONE variable that A vs B changes.
if [ "$BOOT" = mgmt ]; then
  # B: one common interface on every node, as NVIDIA's launch.sh does.
  ENVMAP[NCCL_SOCKET_IFNAME]=wlP9s9
else
  # A: fabric bootstrap, mirroring the serving config which pins a QSFP-facing
  # iface per rank. sparkmain advertises its fabric address on lo.
  ENVMAP[NCCL_SOCKET_IFNAME]="enp1s0f0np0,enp1s0f1np1,enP2p1s0f0np0,enP2p1s0f1np1"
fi
# MPI launcher transport: always management (see note above).
BTL_IF="wlP9s9"

ENVARGS=()
for k in "${!ENVMAP[@]}"; do ENVARGS+=(-x "$k=${ENVMAP[$k]}"); done

{
  echo "--- env under test ---"
  for k in "${!ENVMAP[@]}"; do echo "  $k=${ENVMAP[$k]}"; done
  echo "----------------------"
} | tee -a "$LOG"

case "$SIZE" in
  32M) BFLAG="-b 32M -e 32M" ;;
  16G) BFLAG="-b 16G -e 16G" ;;
  *)   BFLAG="-b $SIZE -e $SIZE" ;;
esac

MPIRUN_CMD="mpirun --allow-run-as-root -np $WORLD -H $HOSTS \
  --mca plm_rsh_args '-p 2299 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
  --mca btl_tcp_if_include $BTL_IF \
  --mca oob_tcp_if_include $BTL_IF \
  --bind-to none --map-by node \
  ${ENVARGS[*]} \
  /src/build/all_gather_perf $BFLAG -f 2 -n 20 -w 5 -g 1 -c 1"

echo "--- mpirun command ---" | tee -a "$LOG"
echo "$MPIRUN_CMD" | tee -a "$LOG"
echo "----------------------" | tee -a "$LOG"

# launch from inside rank0's container
timeout 1800 ssh -n -o BatchMode=yes "${NODES[0]}" \
  "docker exec $CN bash -lc \"$MPIRUN_CMD\"" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== exit rc=$RC at $(date -Iseconds) ===" | tee -a "$LOG"

# --- collect per-rank NCCL_DEBUG logs ----------------------------------------
DBG="$OUT/${TAG}_w${WORLD}_${SIZE}.nccldebug.log"
: > "$DBG"
for n in "${NODES[@]}"; do
  echo "########## NCCL_DEBUG from $n ##########" >> "$DBG"
  ssh -n -o BatchMode=yes "$n" \
    "docker exec $CN bash -lc 'cat /tmp/nccldbg.*.log 2>/dev/null'" >> "$DBG" 2>&1
done
echo "debug log: $DBG ($(wc -l < "$DBG") lines)" | tee -a "$LOG"

# --- extract the headline facts ----------------------------------------------
{
  echo "---- EXTRACT $TAG w$WORLD $SIZE ----"
  echo "[nccl-version]"; grep -h -m3 'NCCL version' "$DBG" | sed 's/^/  /'
  echo "[vdev]";         grep -h 'Made virtual device' "$DBG" | sed 's/^.*NET\/IB : /  /' | sort -u
  echo "[transport]";    grep -hoE 'via NET/(IB/[0-9]+|Socket)' "$DBG" | sort | uniq -c | sed 's/^/  /'
  echo "[result]";       grep -hE '^ +[0-9]{6,}' "$LOG" | sed 's/^/  /'
  echo "[avgbusbw]";     grep -h 'Avg bus bandwidth' "$LOG" | sed 's/^/  /'
} | tee -a "$LOG"

cleanup
exit $RC
