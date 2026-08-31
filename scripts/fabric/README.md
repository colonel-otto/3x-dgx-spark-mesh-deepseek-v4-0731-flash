# Fabric rendezvous reconciliation

`dsv4-fabric-reconcile` restores, on every boot, the address each node
advertises to its peers for Gloo rendezvous — and that `scripts/fabric_gate.sh`
resolves peers by.

## Why

NVIDIA Sync renumbered the fabric to `10.100.16x` on 2026-08-30. The gate config
still named `192.168.100.2` / `192.168.101.2`, which no longer existed, so the
gate reported **7 failures on a healthy fabric**. That is a gate crying wolf,
which is worse than no gate.

sparkmain was unaffected because its rendezvous address lives on its **loopback**
(`192.168.200.1/32`) — a renumber of the fabric interfaces cannot touch it.
spark1 and spark2 had no equivalent. Every node now has one:

| node | rendezvous address |
|---|---|
| sparkmain | `192.168.200.1` |
| spark1 | `192.168.200.2` |
| spark2 | `192.168.200.3` |

Reachability between them is by host route over the correct point-to-point link
(the fabric is a triangle: each `/24` joins exactly one pair).

## Install (per node)

```bash
sudo install -m 0755 dsv4-fabric-reconcile /usr/local/bin/
sudo install -m 0644 dsv4-fabric-reconcile.service /etc/systemd/system/
sudo install -m 0644 map-<node>.env /etc/dsv4-fabric-map   # see below
sudo systemctl daemon-reload
sudo systemctl enable --now dsv4-fabric-reconcile.service
```

## The map file

`/etc/dsv4-fabric-map` is per node and gitignored in spirit — it holds LAN
addressing. `PEER_ROUTES` entries are `<peer_addr>:<via>:<dev>`:

```sh
SELF_ADDR=192.168.200.1
PEER_ROUTES="192.168.200.2:10.100.164.1:enp1s0f0np0 192.168.200.3:10.100.162.1:enp1s0f1np1"
```

## What it deliberately does NOT do

It never runs `netplan apply`. Applying netplan on a live fabric is what killed
a running cluster on 2026-08-30 (GID table change → `EngineDead` mid-benchmark).
Reboot persistence belongs to the netplan files and the NetworkManager loopback
address; this unit only repairs the **running** state, additively and
idempotently.

A peer being unreachable is reported but does not fail the unit — a powered-off
peer is not a local misconfiguration, and failing the boot for it would take the
whole node out over a neighbour's downtime.

## The three live maps (2026-08-31)

```sh
# sparkmain
./install-fabric-reconcile.sh 192.168.200.1 \
    192.168.200.2:10.100.164.1:enp1s0f0np0 \
    192.168.200.3:10.100.162.1:enp1s0f1np1

# spark1
./install-fabric-reconcile.sh 192.168.200.2 \
    192.168.200.1:10.100.164.2:enp1s0f1np1 \
    192.168.200.3:10.100.160.1:enp1s0f0np0

# spark2
./install-fabric-reconcile.sh 192.168.200.3 \
    192.168.200.1:10.100.162.2:enp1s0f0np0 \
    192.168.200.2:10.100.160.2:enp1s0f1np1
```

Reboot persistence of the loopback address is separately held by
`nmcli con mod lo +ipv4.addresses <addr>/32`, and of the routes by each node's
`/etc/netplan/96-dsv4-routes.yaml`. This unit is the belt to those braces: it
repairs the running state on every boot whatever those did.
