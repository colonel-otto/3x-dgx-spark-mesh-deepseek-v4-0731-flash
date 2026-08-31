# Troubleshooting

## eugr launcher bring-up (engine A/B, 2026-08-30)

Four issues hit in sequence bringing up `eugr/spark-vllm-b12x` at 3-node TP=3
via `run-recipe.py` (each fix verified by the next failure moving further):

1. **`Local IP (…) is not in the list of nodes`** — the hosts.json addresses
   for the Sparks are their **wifi** (`wlP9s9`) IPs. The launcher coordinates
   over the wired 10G management interface (`enP7s7`); pass THOSE addresses in
   `-n`, head first. Also map each worker's bare IP to its per-node username in
   the head's `~/.ssh/config` — the launcher SSHes workers by IP.
2. **`error while creating mount source path '/tmp/dsv4': mkdir … file exists`**
   — Docker refuses a symlink as a bind-mount source. Use a hardlink farm
   instead: `mkdir /tmp/dsv4 && cp -al $HOME/dsv4/hf-… /tmp/dsv4/` (instant,
   zero space, same filesystem). /tmp is volatile — recreate after reboots.
3. **`mkdir: cannot create directory '/home/sparkmain': Permission denied` on
   workers** — the launcher expands the head's `$HOME` cache paths literally in
   the worker docker args. `--no-cache-dirs` skips them (cost: AOT/FlashInfer
   caches not persisted between boots), and set `HF_HOME=/tmp/hfcache` so the
   HF cache mount is a uniform path too.
4. **`ibv_modify_qp failed with 110 … local GID ::ffff:10.100.164.2, remote GID
   ::ffff:10.100.162.1` → `NCCL error: unhandled system error`** — NCCL tried
   to reach a peer through the wrong point-to-point fabric link (each /24 in
   the triangle reaches exactly one peer). The launcher's mesh detection prints
   the right plan but its NCCL env did not reach the container. Fix: pass the
   proven values from `config/tp3.env` explicitly via `-e`:
   `NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1` (leading `=`
   = exact match), `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_NET_PLUGIN=none`,
   `NCCL_IB_MERGE_NICS=0`. Verified: the QP error disappeared on the next
   attempt and startup progressed to the memory check.

Then the known `Free memory … less than desired GPU memory utilization` check
fired at the recipe's 0.85 (see the dedicated section below) — the standing
0.82 applies to this engine too: `--gpu-memory-utilization 0.82`.

## NCCL reports `NET/Socket`

The run is using TCP, even if RDMA link state and ping are healthy.

Check that:

- `/dev/infiniband` is visible inside the container;
- `NCCL_IB_DISABLE=0` and `NCCL_NET=IB` reached the container;
- both peer-facing HCAs are listed in `NCCL_IB_HCA`;
- `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_NET_PLUGIN=none` reached the container;
- the loaded NCCL library supports subnet-aware routing.

`torch.cuda.nccl.version()` reports the NCCL version PyTorch was compiled against, not
necessarily the library mapped by the live process. Verify the live mapping through
`/proc/PID/maps` and query `ncclGetVersion` from that library.

## `ibv_modify_qp` fails during INIT to RTR

This commonly means the chosen HCA/GID cannot reach the remote rank. Confirm the GID is
RoCEv2/IPv4 and that subnet-aware routing can choose a peer-facing HCA. Do not begin by
moving cables when LLDP and the official ring already match.

## Ping succeeds but distributed startup times out

Ping does not validate new TCP flows or RDMA. Check the firewall on every rank, routes
to the master identity, container `/etc/hosts`, and a real NCCL test.

## Startup hangs after weight loading

Verify identical `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, MTP and model
arguments on every rank. Mismatched memory profiles can hang without a useful error.

## Correct-looking but wrong answers

Stop performance testing. Confirm that the TP=3 padding patch applied at every expected
site and every rank uses the same image. Run the correctness suite with enough output
budget to distinguish wrong answers from truncated reasoning.

## Node accepts TCP but SSH never sends a banner

This can be severe unified-memory exhaustion rather than a network problem. A single
unsharded copy of this checkpoint does not fit safely on one Spark. Calculate weights
per node before changing TP/PP settings; do not launch TP=1 for this checkpoint.

## Relaunch after a crash driver-OOMs at boot (external report, unverified)

A community TP=4 recipe ([tonyd2wild](https://github.com/tonyd2wild/Deepseek-V4-Flash-TP4-4x-DGX-Spark),
TROUBLESHOOTING §6) reports the GB10 driver can hold ~100 GiB of unified memory after a
crash, so an immediate relaunch driver-OOMs at boot despite `nvidia-smi` looking idle.
Their fix: `sync; echo 3 > /proc/sys/vm/drop_caches` and a fresh `docker run` (never
`docker restart`, never `--rm` — a crash then leaves no logs). **We have not reproduced
this**; it is recorded because it rhymes with our wedge patterns. If a relaunch OOMs
where the first launch fit, try `drop_caches` before concluding the config regressed.

## Avoid interpreting experimental dead ends as requirements

- A 200 GbE switch is not required for the proven three-node ring.
- A host-built NCCL is not automatically the runtime used by the Python process.
- The uppercase second CX-7 interface pair is optional multi-rail headroom, not a
  prerequisite for the historical result.

## Startup fails instantly with "Free memory on device ... is less than desired GPU memory utilization"

```
ValueError: Free memory on device cuda:0 (100.94/121.69 GiB) on startup is less
than desired GPU memory utilization (0.835, 101.61 GiB).
```

`GPU_MEMORY_UTILIZATION` is a fraction of **total** memory, but the check compares
it against **free** memory. On a GB10 node the OS, dockerd, `open-webui` (~1 GiB on
rank 0) and the exporters are already resident, so a node never frees more than
~100.9-101.5 GiB of its 121.69 GiB. Any value above ~0.828 therefore fails on a
warm box even though nothing is wrong with the cluster.

Fix by lowering the value on **all three ranks** (a mismatch hangs startup forever
with no error) or by freeing resident host RAM. See `DECISIONS.md` for the current
value and why it moved.

**`VLLM_SKIP_INIT_MEMORY_CHECK` does not work.** The engine logs it as
`Unknown vLLM environment variable detected` and `request_memory()` in
`vllm/v1/worker/utils.py` raises unconditionally — there is no skip branch in this
build. Do not rely on it to let a tight value through.

**Read the container log, not the journal.** `journalctl -u dsv4` shows only the
outer wrapper's traceback ending in `Engine core initialization failed. See root
cause above.` — the actual `ValueError` is upstream of it, in
`docker logs dspark-vllm-gx10-vllm-dspark-1`.

**Free is not available.** These are unified-memory nodes: `nvidia-smi` reports
`N/A` and the check reads *free*. A node showing 74 GiB free / 116 GiB available
has 42 GiB of reclaimable page cache — but vLLM tests the free number, so judge
headroom by that.

**Dropping caches is not the fix on its own.** `sysctl vm.drop_caches=3` makes the
*pre-start* free number look healthy, but the shortfall appears *during* startup:
measured on 2026-08-30, rank 0 had **110.59 GiB free** immediately before a launch
— 9 GiB more than the 101.61 GiB required, with no cache drop needed — and the
run still failed with vLLM reporting only **100.94 GiB**. The ~10 GiB goes to the
CUDA context and NCCL communicator buffers allocated before `request_memory()`
runs. Drop caches as an *extra* lever when a start is tight; do not rely on it to
carry a value the node cannot otherwise sustain.

**Leave a deliberate reserve.** The headroom left by `GPU_MEMORY_UTILIZATION` is
what keeps `sshd` and `open-webui` alive, and SSH is the only way back into a node
that goes bad. Per node (121.69 GiB total):

| util | engine | reserved for OS + sshd + open-webui |
|---|---|---|
| 0.835 | 101.61 GiB | 20.08 GiB |
| **0.82** | **99.79 GiB** | **21.90 GiB** |
| 0.80 | 97.35 GiB | 24.34 GiB |

Do not tune this purely for KV pool size. Recovery access is worth more than the
last GiB of cache.

**A clean cold start takes ~6 minutes.** Measured phase-by-phase from the
container log, 2026-08-30 15:37 boot (GPU_MEMORY_UTILIZATION=0.82, warm page
cache):

| Phase | Duration |
|---|---|
| systemd → engine init | ~30 s |
| Weight load (155 GiB main + MTP draft, "Model loading took 199 s") | ~3.3 min |
| KV profile + cache init (30.99 GiB → 4,390,838 tokens) | ~20 s |
| CUDA graph capture (15 PIECEWISE + 14 FULL, "finished in 13 secs") | **13 s** |
| `init engine (profile, create kv cache, warmup model)` | 86.7 s total |
| **`/v1/models` answering** | **~6 min** |

The earlier "~30 minutes, not 6-8" claim in this section is **superseded**: that
figure was measured across a bringup that included startup-memory-check failures
and retries (the 0.835→0.82 tuning session), not a clean start. `torch.compile`
is a red herring — the log warns the model does not support it. CUDA graph
capture, the other named suspect, is 13 seconds.

What remains true: **do not kill a start on a stopwatch.** A start that hits the
memory-check/retry path, cold page cache, or first-boot kernel autotune can
legitimately run far longer than 6 minutes, and `dsv4-service-start`'s 45-minute
budget exists because a 15-minute budget aborted a healthy bringup. Judge a start
by whether the container log is still advancing, never by elapsed time alone —
see "Startup hangs after weight loading" above for what a real hang looks like.

## `systemctl stop dsv4` leaves all three containers running

Fixed 2026-08-30 in the unit; this is what the symptom looked like and why.

`dsv4.service` is `Type=oneshot` with `RemainAfterExit=yes`, and a cold start
occupies `ExecStart` for ~30 minutes. **`ExecStop` only runs from an `active`
unit.** A `systemctl stop` issued while the unit is still `activating` instead
SIGTERMs the start script: the unit goes to `failed (result: signal)` and
`ExecStop` is never reached, so the head and both worker containers are orphaned
and keep holding ~100 GiB per node. `systemctl is-active` then reports `failed`
while `docker ps` shows all three still `Up` — and because the unit is now in
`failed` state rather than `active`, the *next* `systemctl stop` skips `ExecStop`
as well.

Two lines fix it:

```ini
# runs on EVERY exit path: clean stop, failed start, SIGTERM mid-startup
ExecStopPost=/home/sparkmain/bin/dsv4-service-stop
# stopping a deliberate startup is an operation, not a fault
SuccessExitStatus=SIGTERM
```

`dsv4-service-stop` is idempotent best-effort (it logs and skips a node that is
already down, and always `exit 0`), so having it in both `ExecStop` and
`ExecStopPost` is safe.

Verified by starting the cluster, issuing `systemctl stop` ~35 s in while it was
still `activating`, and confirming the teardown ran on all three nodes, the unit
reported `Deactivated successfully` / `inactive`, and no `dspark` container
survived on any node.

If you are on a build without these lines, tear down with
`~/bin/dsv4-service-stop` directly — `systemctl stop` alone is not sufficient.


## NVIDIA Sync kills a live cluster: netplan apply → GID change → EngineDead

**Root-caused 2026-08-30.** Opening the NVIDIA Sync desktop app while the
cluster is serving can kill the engine ~5 minutes later with no depth/load
correlation. Full proven chain, from journals on all three nodes:

1. The Sync app (running on another LAN machine; its mDNS lookup of a
   node's `.local` name may visibly fail there first) SSHes into each node and runs
   `sudo python3 ~/.config/NVIDIA/Sync/cache/cluster_node_probe.py
   --apply-netplan /tmp/netplan-<id>.yaml` — cluster-wide, same second.
2. netplan apply → NetworkManager device reconfigure ("user-requested" in the
   NM audit log) → avahi withdraws/re-adds addresses → **RoCE GID tables change
   on both fabric ports of every node**: `NCCL WARN NET/IB ... GID table changed`
   (the only warning NCCL gives).
3. In-flight collectives never complete. All GPUs show the spin signature:
   **~96 % util at ~20 W**. An in-flight request may return
   "stream completed without content".
4. Exactly one RPC timeout later (~5 min): `TimeoutError: RPC call to
   sample_tokens timed out` → `EngineDeadError`. The API then 500s and finally
   refuses connections — while `docker ps` shows the container Up and the
   oneshot unit shows `active (exited), Result=success`. Neither is health.

**Rules:**
- Never open/refresh NVIDIA Sync while the cluster serves or benchmarks.
- `GID table changed` in any worker log during a run **voids the run** —
  benchmark results after that line measure a broken fabric, not the engine.
- Recovery: full `systemctl stop dsv4` (teardown now reliable via ExecStopPost)
  then `start`. A plain `start` on the active-but-dead unit is a NO-OP.

**Same-day context (timeline 2026-08-30):** 11:09 and 11:19 engine start
failures were the separate `GPU_MEMORY_UTILIZATION=0.835` init memory check
(see above); 11:37 clean 6-min start; 12:55:13 the netplan event; 13:00:12
EngineDead. Three engine deaths in one day, three distinct causes — check the
docker log, not the unit state, before attributing.

## eugr engine: bimodal TTFT / aggregate (7s vs 2s) in the first minutes after boot

Two distinct effects, verified 2026-08-30 on the arm-1 run:

1. **`[b12x cute.compile] … status=disk-cache-miss reason=post-engine-start`** in the launcher log —
   B12X JIT-compiles each new CuTe kernel shape on first encounter; the batch that triggers one stalls.
   Booting with `--no-cache-dirs` leaves no persisted kernel cache, so this repeats every boot (20 misses in
   the first ~40 min of serving). Treat any throughput trial taken while the miss counter is still rising as
   contaminated; re-run when `grep -c "cute.compile.*disk-cache-miss" <launch log>` has stopped moving.
   Durable fix: mount uniform cache dirs (see ENGINE-AB-3NODE.md next steps).
2. **A steady-state cliff at the max_num_seqs cap** (c=16: TTFT 7.0s for 16×256 tokens; c=8: 1.9s) that
   persists after the miss counter freezes. Startup warns `max_num_scheduled_tokens is set to 8128 based on
   the speculative decoding settings` — nst=5 draft slots × 16 seqs. Lower `num_speculative_tokens` or
   raise the batched-token budget and re-measure; do not read the c=16 cell as an engine regression until
   that is separated.

## eugr launcher: the built-in cache mounts are `$HOME`-relative and break on workers

Verified 2026-08-30 while implementing the "persist kernel caches" step.

`launch-cluster.sh` mounts kernel caches **by default** (`MOUNT_CACHE_DIRS=true`);
`--no-cache-dirs` opts *out*. So the obvious fix — "just drop `--no-cache-dirs`" — is
wrong here, and this is why arm 1 passed that flag in the first place:

```bash
# launch-cluster.sh ~line 404
DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/vllm:/root/.cache/vllm"      # + flashinfer, .triton, .tilelang
CACHE_DIRS_TO_CREATE+=("$HOME/.cache/vllm")
```

`$HOME` expands on the **head** and the resulting absolute path is shipped verbatim to
the workers (`ssh "$worker" "mkdir -p ${CACHE_DIRS_TO_CREATE[*]}"`, then the same
`-v` string in the worker's `docker run`). Our homes are not uniform:

| node | user | `$HOME` |
|---|---|---|
| sparkmain | `sparkmain` | `/home/sparkmain` |
| spark1 | `spark1` | `/home/spark1` |
| spark2 | `spark2` | `/home/spark2` |

So the workers would get `/home/sparkmain/.cache/vllm` created and mounted on a box where
that path belongs to nobody — a root-owned junk dir, and no shared cache identity.
Same root cause as the recorded trap "the launcher expands head `$HOME` on workers".

**Fix that works:** keep `--no-cache-dirs` (to suppress the broken `$HOME` mounts) and
pass *uniform absolute* paths with `-v`, which `launch-cluster.sh` forwards unchanged
(`for mapping in "${VOLUME_MAPPINGS[@]}"; do DOCKER_ARGS="$DOCKER_ARGS -v $mapping"; done`):

```bash
-v /opt/eugrcache-vllm:/root/.cache/vllm \
-v /opt/eugrcache-flashinfer:/root/.cache/flashinfer \
-v /opt/eugrcache-triton:/root/.triton \
-v /opt/eugrcache-tilelang:/root/.tilelang
```

Use `/opt`, not `/tmp`: `/tmp` here is on the root NVMe (not tmpfs, so no RAM cost either
way) but systemd-tmpfiles wipes it on reboot, which would silently re-cold every kernel
cache after any node restart. `-v` passthrough does NOT mkdir on the workers, so
pre-create the four dirs on every node (`sudo mkdir -p … && chmod 777 …`);
`scripts/eugr-ab/eugr-boot.sh` does this as a precondition step.

## eugr recipes: override sweep parameters in the recipe, not via `--` passthrough

`run-recipe.py` **appends** post-`--` args after template substitution
(`command = command + " " + extra_args_str`). For a scalar that already exists in the
recipe this leaves the flag on the line **twice** and relies on argparse last-wins.
Confirmed with `--dry-run` for the K sweep:

```
--speculative-config '{"…","num_speculative_tokens":5,…}' --speculative-config '{"…","num_speculative_tokens":2,…}'
```

That is a silent-misconfiguration trap for a JSON blob, and `--served-model-name` is
worse (`nargs="+"`, so a second one may merge rather than replace). Since
`params = {**recipe["defaults"], **overrides}` substitutes `{num_speculative_tokens}` and
`{max_num_batched_tokens}` from `defaults:`, generate a per-sweep-point recipe instead —
one substitution site, exactly one flag on the command line, and the config that ran is
left on disk as `recipes/dsv4-tp3-nst<K>-mnbt<N>.yaml`. `eugr-boot.sh` generates it and
then **gates the launch on `--dry-run` assertions** (port, both served names, nst, mnbt,
and `grep -c -- '--speculative-config' == 1`) so a wrong config fails in seconds rather
than after an 8-minute boot.

## DSpark speculative depth has a HARD FLOOR at the checkpoint's block size

Verified 2026-08-30 by a failed boot. `num_speculative_tokens=2` is rejected at
config validation:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
  Value error, DSpark requires num_speculative_tokens >= dspark_block_size (5); got 2.
  Smaller values produce incorrect output. Use num_speculative_tokens=5 or larger (e.g. 7).
```

`dspark_block_size` is read from the **checkpoint's** `config.json`
(`dspark_block_size: 5`, alongside `dspark_markov_rank: 256` and
`dspark_target_layer_ids: [40,41,42]`) — it is not a launch flag and cannot be
lowered. DSpark is a semi-autoregressive **block** drafter: a speculative length
below the block size feeds the block/Markov-head machinery an unsupported layout
and produces garbled output rather than merely lower acceptance, which is why
vLLM refuses instead of serving it.

Two consequences:

1. Any plan to "match our anemll MTP K=2 to remove the speculator delta from the
   cross-engine A/B" is **impossible on this engine**. The comparison carries a
   permanent speculator delta (anemll MTP K=2 vs eugr DSpark K>=5); say so in
   every row's notes rather than implying a matched speculator.
2. The K sweep space is {5, 7}, not {2, 3, 5, 7}. Measured: **nst=5 wins every
   cell**; nst=7 never wins and costs 21% at c=8.

## `--no-cache-dirs` throughput numbers are a LOWER BOUND, not engine capability

Measured 2026-08-30, same engine / harness / node count / nst=5. The only change
is persistent kernel caches (`/opt/eugrcache-*`) replacing `--no-cache-dirs`:

| c | cold caches (arm 1) | warm caches | Δ |
|---|---:|---:|---:|
| 8 | 171.7 agg | 252.9 | **+47%** |
| 16 | 133.9 agg, TTFT 7000ms | 198.8 agg, TTFT 1755ms | **+48%, 4x TTFT** |

The arm-1 "c=16 scheduling cliff" was therefore mostly JIT compilation, not
scheduler budget. The tell is in arm-1's own c=1 log: decode **decayed** across
trials (83.8 → 80.3 → 64.6 → … → 57.8) as each new kernel shape hit the JIT.
With warm caches the same cell rises and holds (78.4 → 91.4 → 84.5 → 78.3).

So: a monotonic decay across repeated identical trials is a JIT signature, not
load or thermal drift. Check `grep -c 'cute.compile.*disk-cache-miss' <launcher
log>` and only record once it stops moving — `scripts/eugr-ab/eugr-sweep.sh`
automates exactly this and warms further when the counter is still climbing.
Boot time also halves (~8 min → ~4.3 min).

## `max_num_batched_tokens 16384` is a KV trap on the eugr engine too

The engine's own startup warning recommends raising it
(`max_num_scheduled_tokens is set to 8128 based on the speculative decoding
settings … Consider increasing max_num_batched_tokens`). Measured at nst=5:

| | mnbt 8192 | mnbt 16384 |
|---|---:|---:|
| c=4 agg | 152.8 | 165.0 (+8%) |
| c=8 agg | **252.9** | 241.8 (−4%) |
| c=16 agg | 198.8 | 214.3 (+8%) |
| KV cache | **2,357,009 tok** | 1,165,679 tok (**−50.5%**) |
| max concurrency @1M ctx | **2.25x** | 1.11x |

Raising it does silence the warning, but half the KV capacity for ~8% on two
cells is a bad trade on a 1M-context server. This reproduces the anemll finding
on a different engine AND a different KV dtype (fp8 vs nvfp4_ds_mla) — so treat
that startup warning as a known trap on both engines, and measure before acting
on it.

## `models-manifest-serve` has no `/v1/models`

It publishes named JSON documents (`opencode.models.json`,
`opencode.gateway.json`) and serves **any other path as a static file** from its
directory. So a wrong path returns an HTML *directory listing* with HTTP 200 —
which reads like "the service is up but broken" when it is perfectly healthy.
Query `http://<gateway>:8771/opencode.gateway.json`; that document is resolved
LIVE from the gateway's own `/v1/models` (3s cache TTL), so a model added to
LiteLLM appears there by itself with no manual edit.
