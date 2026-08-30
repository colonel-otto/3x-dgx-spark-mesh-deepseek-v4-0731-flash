#!/usr/bin/env bash
# Bring config/head.env (sparkmain) and config/worker.env (spark1) to the SAME
# engine profile the live TP=3 service runs, so the llama-benchy 2v3 arm changes
# node count and nothing else.
#
# WHY THIS EXISTS
#
# dsv4.service reads config/tp3.env. The TP=2 path reads head.env/worker.env.
# The matched run's Step-5 restore copied the *pre-matched* backups back over
# head.env/worker.env, so they currently sit at seqs=16 / MTP=5 / gpumem=0.80,
# are MISSING kv-cache-dtype, long-prefill-threshold and inflight-prefills
# entirely, and point at a DIFFERENT container image
# (ghcr.io/anemll/dspark-vllm-gx10:0.1.1) than the live engine (dsv4-3spark:0.1.1).
#
# Bringing TP=2 up from those files would change SEVEN variables at once and the
# comparison would be worthless -- silently, because vLLM starts fine either way.
#
# Rank-specific keys (NODE_RANK, VLLM_HOST_IP, MASTER_ADDR, NCCL_IB_HCA,
# NCCL_SOCKET_IFNAME) are deliberately NOT touched: the 2-node arm uses the
# direct sparkmain<->spark1 fabric (192.168.100.x) and that is a property of the
# topology under test, not a confound.
#
# Every file is backed up to .prebenchy.<STAMP> before modification.
#
# Usage: match_env_for_benchy.sh apply | verify | restore
set -euo pipefail

HEAD=sparkmain
WORKER=spark1
DIR='~/localai/dspark-vllm-gx10'

# The nine plan-mandated engine settings, plus the image and MoE backend that
# must match the live TP=3 engine build.
read -r -d '' KV <<'EOF' || true
GPU_MEMORY_UTILIZATION=0.835
MAX_NUM_SEQS=32
MTP_NUM_TOKENS=2
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=1048576
KV_CACHE_DTYPE=nvfp4_ds_mla
LONG_PREFILL_TOKEN_THRESHOLD=1024
DSPARK_MAX_INFLIGHT_PREFILLS=2
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
DSPARK_VLLM_IMAGE=dsv4-3spark:0.1.1
MOE_BACKEND=flashinfer_b12x
TP3_PATCH_DIR=/opt/dsv4-tp3-src
EOF

KEYS_RE='^(GPU_MEMORY_UTILIZATION|MAX_NUM_SEQS|MTP_NUM_TOKENS|MAX_NUM_BATCHED_TOKENS|MAX_MODEL_LEN|KV_CACHE_DTYPE|LONG_PREFILL_TOKEN_THRESHOLD|DSPARK_MAX_INFLIGHT_PREFILLS|VLLM_PREFIX_CACHE_RETENTION_INTERVAL|DSPARK_VLLM_IMAGE|MOE_BACKEND|TP3_PATCH_DIR)='

# Rewrite $2 on host $1 so every key in $KV holds the stated value: existing
# lines are replaced in place, absent keys are appended.
apply_one() {
  local host="$1" file="$2" stamp="$3"
  printf '%s\n' "$KV" | ssh -T "$host" "
    set -euo pipefail
    cd $DIR
    cp '$file' '$file.prebenchy.$stamp'
    want=\$(cat)
    tmp=\$(mktemp)
    # keep every line whose key we are not managing
    grep -vE '$KEYS_RE' '$file' > \"\$tmp\" || true
    printf '%s\n' \"\$want\" >> \"\$tmp\"
    mv \"\$tmp\" '$file'
    echo \"[$host] $file rewritten (backup: $file.prebenchy.$stamp)\"
  "
}

verify_one() {
  local host="$1" file="$2"
  echo "--- $host $file ---"
  ssh -n "$host" "cd $DIR && grep -E '$KEYS_RE' '$file' | sort"
}

case "${1:-}" in
  apply)
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    apply_one "$HEAD"   config/head.env   "$STAMP"
    apply_one "$WORKER" config/worker.env "$STAMP"
    echo
    echo "=== verify ==="
    verify_one "$HEAD"   config/head.env
    verify_one "$WORKER" config/worker.env
    echo
    # cluster_tp2.sh refuses to start if the ranks disagree; prove they agree now.
    h=$(ssh -n "$HEAD"   "cd $DIR && grep -E '$KEYS_RE' config/head.env   | sort")
    w=$(ssh -n "$WORKER" "cd $DIR && grep -E '$KEYS_RE' config/worker.env | sort")
    if [[ "$h" == "$w" ]]; then
      echo "MATCH OK: both ranks agree on all managed keys."
    else
      echo "MISMATCH -- startup would hang forever with no error:"
      diff <(echo "$h") <(echo "$w") || true
      exit 1
    fi
    ;;
  verify)
    verify_one "$HEAD"   config/head.env
    verify_one "$WORKER" config/worker.env
    ;;
  restore)
    for pair in "$HEAD:config/head.env" "$WORKER:config/worker.env"; do
      host="${pair%%:*}"; file="${pair##*:}"
      ssh -n "$host" "cd $DIR && b=\$(ls -t $file.prebenchy.* 2>/dev/null | head -1) && \
        [ -n \"\$b\" ] && cp \"\$b\" $file && echo \"[$host] restored $file from \$b\""
    done
    ;;
  *)
    echo "usage: $0 apply | verify | restore" >&2; exit 2 ;;
esac
