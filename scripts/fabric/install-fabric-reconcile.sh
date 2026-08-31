#!/usr/bin/env bash
# Install + enable the fabric reconcile unit on ONE node.
#
# Run on each node. The map is per node: pass the node's own rendezvous address
# and its peer routes as "<peer_addr>:<via>:<dev>" pairs.
#
# Example (sparkmain):
#   ./install-fabric-reconcile.sh 192.168.200.1 \
#       192.168.200.2:10.100.164.1:enp1s0f0np0 \
#       192.168.200.3:10.100.162.1:enp1s0f1np1
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <self_addr> <peer_addr:via:dev> [<peer_addr:via:dev> ...]" >&2
  exit 2
fi

self=$1; shift
here=$(cd "$(dirname "$0")" && pwd)

sudo install -m 0755 "$here/dsv4-fabric-reconcile" /usr/local/bin/
sudo install -m 0644 "$here/dsv4-fabric-reconcile.service" /etc/systemd/system/

printf 'SELF_ADDR=%s\nPEER_ROUTES="%s"\n' "$self" "$*" | sudo tee /etc/dsv4-fabric-map >/dev/null
sudo chmod 0644 /etc/dsv4-fabric-map

sudo systemctl daemon-reload
sudo systemctl enable --now dsv4-fabric-reconcile.service

echo "--- installed; unit output ---"
sudo systemctl status dsv4-fabric-reconcile.service --no-pager -n 12 || true
