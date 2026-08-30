#!/usr/bin/env bash
# eugr-boot.sh — launch the eugr engine with persistent kernel caches, the local
# benchmark port/model aliases, and a settable speculative depth.
#
# Implements steps 1-3 of the post-arm-1 plan (docs/ENGINE-AB-3NODE.md) in one boot:
#   1. persistent uniform kernel-cache mounts (kills the JIT contamination)
#   2. num_speculative_tokens settable -> the K sweep
#   3. port 8100 + BOTH served names for local compatibility with existing clients
#
# Usage: eugr-boot.sh <nst> [max_num_batched_tokens] [logfile]
#
# WHY a generated recipe instead of `-- --speculative-config ...` passthrough:
# passthrough APPENDS, so the recipe's own --speculative-config stays on the
# command line and you get the flag twice, relying on argparse last-wins for a
# JSON blob. A generated recipe substitutes the value in ONE place, and leaves
# the exact config that ran on disk. Verified with --dry-run before use.
set -euo pipefail

NST="${1:?usage: eugr-boot.sh <nst> [mnbt] [logfile]}"
MNBT="${2:-8192}"
LOG="${3:-$HOME/eugr-boot-nst${NST}-mnbt${MNBT}.log}"
PIDFILE="${EUGR_PIDFILE:-$HOME/.eugr-launcher.pid}"

# The three WIRED (enP7s7, 10G) addresses -- head first. NOT the wifi addresses
# that hosts.json lists; the launcher SSHes workers by bare IP.
# Override for the real cluster, e.g. in ~/.eugr-nodes:
#   export NODE0=... NODE1=... NODE2=...
[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:-192.168.10.10}"
NODE1="${NODE1:-192.168.10.11}"
NODE2="${NODE2:-192.168.10.12}"
NODES="$NODE0,$NODE1,$NODE2"

# Reboot-durable and uniform across nodes.
#  - not $HOME/...: the launcher expands the HEAD's $HOME on the workers, whose
#    homes are /home/spark1 and /home/spark2 (this is why arm 1 used
#    --no-cache-dirs at all).
#  - not /tmp/...: systemd-tmpfiles wipes it on reboot, and /tmp here is on the
#    root NVMe, not tmpfs, so /opt costs nothing extra.
CACHE_ROOT=/opt/eugrcache

LAUNCHER="$HOME/eugr-launcher"
BASE_RECIPE="$LAUNCHER/recipes/dsv4-flash-0731-local-tp3.yaml"
GEN_RECIPE_NAME="dsv4-tp3-nst${NST}-mnbt${MNBT}"
GEN_RECIPE="$LAUNCHER/recipes/${GEN_RECIPE_NAME}.yaml"

echo "=== eugr-boot: nst=$NST mnbt=$MNBT ==="
echo "    recipe: $GEN_RECIPE"
echo "    log:    $LOG"

# ---- preconditions on EVERY node -------------------------------------------
for n in $NODE0 $NODE1 $NODE2; do
  ssh -o ConnectTimeout=10 "$n" "
    set -e
    mkdir -p /tmp/dsv4 /tmp/hfcache
    # Docker rejects a symlink bind source -> hardlink farm (instant, same fs).
    [ -d /tmp/dsv4/hf-DeepSeek-V4-Flash-0731 ] || cp -al \"\$HOME/dsv4/hf-DeepSeek-V4-Flash-0731\" /tmp/dsv4/
    sudo -n mkdir -p ${CACHE_ROOT}-vllm ${CACHE_ROOT}-flashinfer ${CACHE_ROOT}-triton ${CACHE_ROOT}-tilelang
    sudo -n chmod 777 ${CACHE_ROOT}-vllm ${CACHE_ROOT}-flashinfer ${CACHE_ROOT}-triton ${CACHE_ROOT}-tilelang
  " || { echo "PRECONDITION FAILED on $n"; exit 1; }
  echo "  preconditions OK: $n"
done

# ---- generate the sweep-point recipe ---------------------------------------
# Only two scalars change vs the base recipe; the served-name line gains the
# legacy model alias so local clients can continue using the existing name.
python3 - "$BASE_RECIPE" "$GEN_RECIPE" "$NST" "$MNBT" <<'PY'
import re, sys
src, dst, nst, mnbt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
t = open(src).read()

def set_default(text, key, val):
    new, n = re.subn(rf'^(\s*{key}:\s*).*$', rf'\g<1>{val}', text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"FATAL: expected exactly 1 '{key}:' in defaults, got {n}")
    return new

t = set_default(t, 'num_speculative_tokens', nst)
t = set_default(t, 'max_num_batched_tokens', mnbt)

# Serve BOTH names: the legacy model alias plus the A/B name used by arm-1 rows.
t, n = re.subn(r'--served-model-name deepseek-v4-flash-eugr-ab',
               '--served-model-name deepseek-v4-flash-dspark-abliterated deepseek-v4-flash-eugr-ab',
               t, count=1)
if n != 1:
    sys.exit(f"FATAL: expected exactly 1 --served-model-name, got {n}")

t = re.sub(r'^name:.*$', f'name: DSV4-Flash-0731-local-TP3-nst{nst}-mnbt{mnbt}', t, count=1, flags=re.M)
open(dst, 'w').write(t)
print(f"  generated {dst}: nst={nst} mnbt={mnbt}, both served names")
PY

# ---- verify the generated command BEFORE launching -------------------------
cd "$LAUNCHER"
DRY=$(HF_HOME=/tmp/hfcache python3 run-recipe.py "$GEN_RECIPE_NAME" \
        -t eugr/spark-vllm-b12x:latest -n "$NODES" \
        -v /tmp/dsv4:/models/dsv4host \
        -v ${CACHE_ROOT}-vllm:/root/.cache/vllm \
        -v ${CACHE_ROOT}-flashinfer:/root/.cache/flashinfer \
        -v ${CACHE_ROOT}-triton:/root/.triton \
        -v ${CACHE_ROOT}-tilelang:/root/.tilelang \
        --no-cache-dirs --gpu-memory-utilization 0.82 --port 8100 --dry-run 2>&1)

# Fail loudly rather than boot a subtly-wrong 8-minute config.
echo "$DRY" | grep -q -- "--port 8100"                        || { echo "DRYRUN FAIL: port"; echo "$DRY" | tail -40; exit 1; }
echo "$DRY" | grep -q "deepseek-v4-flash-dspark-abliterated"  || { echo "DRYRUN FAIL: legacy model alias"; exit 1; }
echo "$DRY" | grep -q "deepseek-v4-flash-eugr-ab"             || { echo "DRYRUN FAIL: ab name"; exit 1; }
echo "$DRY" | grep -q "\"num_speculative_tokens\":${NST}"     || { echo "DRYRUN FAIL: nst"; exit 1; }
echo "$DRY" | grep -q -- "--max-num-batched-tokens ${MNBT}"   || { echo "DRYRUN FAIL: mnbt"; exit 1; }
[ "$(echo "$DRY" | grep -c -- '--speculative-config')" -eq 1 ] || { echo "DRYRUN FAIL: duplicate --speculative-config"; exit 1; }
echo "  dry-run checks PASSED (port, both names, nst=$NST, mnbt=$MNBT, no dup flags)"

# ---- launch ----------------------------------------------------------------
# The engine streams to the LAUNCHER's stdout, not `docker logs` -> nohup+tee.
HF_HOME=/tmp/hfcache nohup python3 run-recipe.py "$GEN_RECIPE_NAME" \
  -t eugr/spark-vllm-b12x:latest \
  -n "$NODES" \
  -v /tmp/dsv4:/models/dsv4host \
  -v ${CACHE_ROOT}-vllm:/root/.cache/vllm \
  -v ${CACHE_ROOT}-flashinfer:/root/.cache/flashinfer \
  -v ${CACHE_ROOT}-triton:/root/.triton \
  -v ${CACHE_ROOT}-tilelang:/root/.tilelang \
  --no-cache-dirs \
  --gpu-memory-utilization 0.82 \
  --port 8100 \
  -e "NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_BUFFSIZE=16777216 \
  -e NCCL_TIMEOUT=3600 \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
chmod 600 "$PIDFILE"
echo "  launched pid $! (recorded in $PIDFILE) -> $LOG"
echo "  watch: tail -f $LOG   (expect ~8 min cold, faster once caches warm)"
