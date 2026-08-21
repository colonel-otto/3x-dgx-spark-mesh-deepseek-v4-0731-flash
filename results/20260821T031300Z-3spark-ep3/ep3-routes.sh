#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Complete the 3-Spark fabric mesh.
#
# THE PROBLEM this solves:
#   The ConnectX-7 cabling is a TRIANGLE OF POINT-TO-POINT LINKS, not a shared
#   subnet. Each node has two ports, each on a different /30 or /24 with a
#   different peer:
#
#       sparkmain 192.168.100.1  <-- 100.0/24 -->  192.168.100.2  spark1
#       sparkmain 192.168.101.1  <-- 101.0/30 -->  192.168.101.2  spark-sep
#       spark1    192.168.102.1  <-- 102.0/30 -->  192.168.102.2  spark-sep
#
#   So no single address is directly reachable from all three nodes. spark-sep
#   cannot see 192.168.100.0/24 at all. NCCL/DP bootstrap requires every rank to
#   reach --data-parallel-address, so without these routes rank 2 hangs forever
#   at connect with no error.
#
#   This was invisible in the 2-node deployment because ranks 0 and 1 share the
#   192.168.100.0/24 link directly.
#
# THE FIX: ip_forward is already 1 on all three nodes, so the triangle only
#   needs routes. Subnet routes cover the forward path; the /32 host routes fix
#   the asymmetric return path (without them, replies to the far address of a
#   two-port peer have no route home and pings fail one direction only).
#
# Idempotent -- `ip route replace` is safe to re-run. Re-run after any reboot,
# or install as a systemd unit / netplan `routes:` stanza to make it persist.
#
# RUN THIS FROM A HOST THAT CAN SSH TO ALL THREE NODES BY NAME (e.g. the
# workstation whose ~/.ssh/config defines sparkmain / spark1 / spark-sep). The
# names are SSH aliases, not DNS, so running it ON a Spark fails with
# "Could not resolve hostname". Override with HEAD=/W1=/W2= if your aliases
# differ.
set -euo pipefail

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
HEAD=${HEAD:-sparkmain}
W1=${W1:-spark1}
W2=${W2:-spark-sep}

run() { ssh "${SSH_OPTS[@]}" "$1" "sudo $2"; }

echo "==> subnet routes (forward path across the third side of the triangle)"
run "$W2"   "ip route replace 192.168.100.0/24 via 192.168.101.1 dev enp1s0f0np0"
run "$W1"   "ip route replace 192.168.101.0/30 via 192.168.102.2 dev enp1s0f1np1"
run "$HEAD" "ip route replace 192.168.102.0/30 via 192.168.101.2 dev enp1s0f1np1"

echo "==> host routes (return path for the far address of each two-port peer)"
run "$HEAD" "ip route replace 192.168.102.1/32 via 192.168.100.2 dev enp1s0f0np0"
run "$W1"   "ip route replace 192.168.101.1/32 via 192.168.100.1 dev enp1s0f0np0"
run "$W2"   "ip route replace 192.168.100.2/32 via 192.168.102.1 dev enp1s0f1np1"

echo "==> verifying full 3x6 mesh"
fail=0
for h in "$HEAD" "$W1" "$W2"; do
  printf '%-12s ' "$h"
  out=$(ssh "${SSH_OPTS[@]}" "$h" '
    for p in 192.168.100.1 192.168.100.2 192.168.101.1 192.168.101.2 192.168.102.1 192.168.102.2; do
      if ping -c1 -W2 "$p" >/dev/null 2>&1; then printf "%s=OK " "$p"; else printf "%s=FAIL " "$p"; fi
    done')
  echo "$out"
  case "$out" in *FAIL*) fail=1 ;; esac
done

if [ "$fail" -ne 0 ]; then
  echo "MESH INCOMPLETE -- do not launch, DP bootstrap will hang." >&2
  exit 1
fi
echo "mesh complete: all 18 paths reachable"
