#!/usr/bin/env bash
# Full-reachability routing for the switchless 3-Spark mesh.
#
# Topology (NVIDIA cross-connected, re-cabled 2026-08-21):
#   head f0 192.168.100.1 <-> 192.168.100.2 f1 spark1
#   head f1 192.168.101.1 <-> 192.168.101.2 f0 spark-sep
#   spark1 f0 192.168.102.1 <-> 192.168.102.2 f1 spark-sep
#
# Each Spark has a DIFFERENT address per cable, so any peer address that is not
# on a node's own cable is unroutable by default. Gloo and NCCL choose their own
# addresses, so every fabric address must be reachable from every node -- each
# over that node's OWN direct cable. One hop, no transit, no switch.
#
# Without the peer-to-peer routes below, Gloo falls back to WiFi and stalls with
# local=[192.168.1.x] to a fabric remote.
set -euo pipefail
M=192.168.200.1

case "$(hostname)" in
  spark-5fa1)   # head: owns the master identity, directly on both cables
    sudo ip addr replace ${M}/32 dev lo
    ;;
  gx10-e146)    # spark1: head on f1 (.100), spark-sep on f0 (.102)
    sudo ip route replace ${M}/32          via 192.168.100.1 dev enp1s0f1np1
    sudo ip route replace 192.168.101.1/32 via 192.168.100.1 dev enp1s0f1np1
    sudo ip route replace 192.168.101.2/32 via 192.168.102.2 dev enp1s0f0np0
    ;;
  gx10-6b41)    # spark-sep: head on f0 (.101), spark1 on f1 (.102)
    sudo ip route replace ${M}/32          via 192.168.101.1 dev enp1s0f0np0
    sudo ip route replace 192.168.100.1/32 via 192.168.101.1 dev enp1s0f0np0
    sudo ip route replace 192.168.100.2/32 via 192.168.102.1 dev enp1s0f1np1
    ;;
  *) echo "unknown host $(hostname)" >&2; exit 1 ;;
esac
ping -c1 -W2 ${M} >/dev/null 2>&1 && echo "$(hostname): mesh routing OK" \
  || { echo "$(hostname): master ${M} UNREACHABLE" >&2; exit 1; }
