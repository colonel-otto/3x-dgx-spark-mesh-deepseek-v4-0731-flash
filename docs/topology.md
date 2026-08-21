# Three-Spark ring topology

## Physical wiring

Follow NVIDIA's canonical port rotation exactly:

```text
Node 1 p0 <----> Node 2 p1
Node 2 p0 <----> Node 3 p1
Node 3 p0 <----> Node 1 p1
```

On DGX Spark, port 0 is the CX-7 connector nearest the normal Ethernet connector and
port 1 is farther away. This is the convention in NVIDIA's
[three-Spark playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/README.md).

Do not infer a cable's peer from an old interface configuration. Verify the physical
neighbor with LLDP before assigning an address.

## Example single-rail address plan

| Node | Physical port | Linux interface | Address | Direct peer |
|---|---:|---|---|---|
| Node 1 | 0 | `enp1s0f0np0` | `192.168.100.1/24` | Node 2 p1 |
| Node 1 | 1 | `enp1s0f1np1` | `192.168.101.1/24` | Node 3 p0 |
| Node 2 | 1 | `enp1s0f1np1` | `192.168.100.2/24` | Node 1 p0 |
| Node 2 | 0 | `enp1s0f0np0` | `192.168.102.1/24` | Node 3 p1 |
| Node 3 | 0 | `enp1s0f0np0` | `192.168.101.2/24` | Node 1 p1 |
| Node 3 | 1 | `enp1s0f1np1` | `192.168.102.2/24` | Node 2 p0 |

Use subnets that do not overlap existing routes. MTU must match at both ends of every
cable; the measured setup used MTU 9000.

Each physical CX-7 port exposes two logical interfaces. NVIDIA's playbook recommends
addressing all four logical interfaces per node for symmetric full-bandwidth operation.
Our historical result used only `rocep1s0f0` and `rocep1s0f1`; the uppercase
`roceP2...` pair remained unaddressed. Treat configuring the second pair as a separate
multi-rail experiment, not a prerequisite for reproducing the historical result.

## Control-plane identity

A switchless ring gives a node a different address on each cable. The distributed
control plane still needs one master identity reachable by both workers. The measured
deployment used `192.168.200.1/32` on Node 1 loopback and explicit host routes from the
workers over their direct Node 1 links.

Example intent:

```text
Node 1: 192.168.200.1/32 on lo
Node 2: 192.168.200.1/32 via 192.168.100.1 on Node 2 port 1
Node 3: 192.168.200.1/32 via 192.168.101.1 on Node 3 port 0
```

Persist the routes with the host's network manager or a dedicated systemd unit. Do not
blindly install a route script: interface names and pre-existing subnets differ between
systems.

## Preflight checks

Run on every node:

```bash
ibdev2netdev
ip -br link show enp1s0f0np0 enp1s0f1np1
ip -br address show enp1s0f0np0 enp1s0f1np1
ip route get 192.168.200.1

for d in /sys/class/infiniband/rocep1s0f{0,1}/ports/1; do
  printf '%s state=%s rate=%s\n' "$d" "$(cat "$d/state")" "$(cat "$d/rate")"
done
```

Expected physical state is `ACTIVE` at `200 Gb/sec`. Verify jumbo frames along each
direct edge if MTU 9000 is configured:

```bash
ping -M do -s 8972 -c 3 PEER_FABRIC_IP
```

Ping proves IP reachability, not RDMA. Run NVIDIA's NCCL test procedure and confirm the
application log contains `NET/IB`.

## Cabling-change rule

A cable move is a coordinated network change. After any move:

1. confirm both new LLDP neighbors;
2. move the address to the correct interface;
3. update persistent NetworkManager/netplan configuration;
4. update master routes;
5. update rank-specific socket/HCA variables;
6. repeat jumbo ping, NCCL tests and the correctness suite.
