#!/usr/bin/env bash
# Bring the cluster up in its 2-node TP=2 shape (sparkmain + spark1).
#
# WHY THIS EXISTS
#
# dsv4.service is hardcoded to config/tp3.env and all three ranks -- deliberately,
# since TP=3 is the production profile. But a matched 2-node vs 3-node comparison
# needs the TP=2 arm too, and the old 2-node launcher (~/bin/dsv4, ~/dsv4/dsv4)
# is stale and must not be used on this cluster.
#
# So drive compose directly, the way HANDOFF section 5 prescribes: verify config
# across ranks, workers first, wait, then the head.
#
# The TP=2 arm reads config/head.env on sparkmain and config/worker.env on
# spark1. TP_SIZE/NNODES are absent from both, so docker-compose.yml's defaults
# (TP_SIZE=2, NNODES=2) apply. spark2 is not involved and is left alone.
#
# ORDERING IS LOAD-BEARING: rank 0 blocks on the NCCL rendezvous until every
# worker has joined, so the worker must be up first.
#
# A MISMATCH IN ANY PARALLELISM FLAG HANGS STARTUP FOREVER WITH NO ERROR. This
# script refuses to start rather than leave you staring at a silent hang.
#
# Usage:  cluster_tp2.sh up | down | status
set -euo pipefail

PROJECT="dspark-vllm-gx10"
REPO_MAIN="/home/sparkmain/localai/dspark-vllm-gx10"
URL="http://127.0.0.1:8100"
HEAD="sparkmain"
WORKER="spark1"
CNAME="dspark-vllm-gx10-vllm-dspark-1"

log() { echo "[$(date -Iseconds)] $*"; }

compose_on() {  # $1 host, $2 envfile, $3... args
  local host="$1" envfile="$2"; shift 2
  ssh -n -o BatchMode=yes "$host" \
    "cd ~/localai/dspark-vllm-gx10 && COMPOSE_DISABLE_ENV_FILE=1 \
     docker compose -p $PROJECT --env-file $envfile -f docker-compose.yml $*"
}

case "${1:-}" in
  up)
    # --- verify the two ranks agree on every engine-shaping flag -------------
    # Rank-specific keys (NODE_RANK, VLLM_HOST_IP, IFNAME, paths) MUST differ;
    # these must not.
    KEYS='MAX_MODEL_LEN|MAX_NUM_SEQS|MTP_NUM_TOKENS|GPU_MEMORY_UTILIZATION|MAX_NUM_BATCHED_TOKENS'
    h=$(ssh -n "$HEAD"   "grep -E '^($KEYS)=' ~/localai/dspark-vllm-gx10/config/head.env   | sort")
    w=$(ssh -n "$WORKER" "grep -E '^($KEYS)=' ~/localai/dspark-vllm-gx10/config/worker.env | sort")
    if [[ "$h" != "$w" ]]; then
      log "ERROR: head.env and worker.env disagree. Startup would hang forever with no error."
      echo "--- $HEAD ---";   echo "$h"
      echo "--- $WORKER ---"; echo "$w"
      exit 1
    fi
    log "Config verified identical across both ranks:"; echo "$h" | sed 's/^/    /'

    # Refuse to start on top of a 3-node cluster still holding the GPUs.
    if ssh -n spark2 'sudo docker ps --format "{{.Names}}" | grep -q vllm-dspark' 2>/dev/null; then
      log "ERROR: spark2 is still running a vLLM container (3-node cluster up?). Stop it first."
      exit 1
    fi

    log "Starting worker $WORKER (rank 1) ..."
    compose_on "$WORKER" config/worker.env up -d >/dev/null

    sleep 15

    log "Starting head $HEAD (rank 0) ..."
    compose_on "$HEAD" config/head.env up -d >/dev/null

    log "Waiting for $URL (cold start is ~7 min) ..."
    for i in $(seq 1 180); do
      if ssh -n "$HEAD" "curl -sf -m 5 -o /dev/null $URL/health" 2>/dev/null; then
        log "READY after ~$((i*5))s"
        ssh -n "$HEAD" "ps -eo args | grep -oE '\-\-tensor-parallel-size [0-9]+|--max-model-len [0-9]+|--max-num-seqs [0-9]+|--gpu-memory-utilization [0-9.]+' | sort -u"
        exit 0
      fi
      sleep 5
    done
    log "ERROR: timed out waiting for $URL"
    exit 1
    ;;

  down)
    log "Stopping head $HEAD ..."
    compose_on "$HEAD" config/head.env down >/dev/null 2>&1 || true
    log "Stopping worker $WORKER ..."
    compose_on "$WORKER" config/worker.env down >/dev/null 2>&1 || true
    log "Stopped."
    ;;

  status)
    for h in "$HEAD" "$WORKER" spark2; do
      printf '%-10s ' "$h"
      ssh -n "$h" 'sudo docker ps --format "{{.Names}} {{.Status}}" | grep vllm-dspark || echo "(no vllm container)"'
    done
    ;;

  *) sed -n '/^# Usage:/p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
