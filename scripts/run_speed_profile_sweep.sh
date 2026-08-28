#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

BATCHED_TOKENS="${1:-16384}"
MAX_MODEL_LEN="${2:-460800}"
LABEL="speed-bt${BATCHED_TOKENS}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DIR="$ROOT/results/${STAMP}-issue28-${LABEL}"
mkdir -p "$DIR"

LOG_MASTER="$DIR/orchestration.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_MASTER"; }

API_BASE="http://127.0.0.1:8100"
MODEL="deepseek-v4-flash-0731"
REPO_MAIN="/home/sparkmain/localai/dspark-vllm-gx10"

log "================================================================"
log "Issue #28: Speed Profile Sweep (MAX_NUM_BATCHED_TOKENS=${BATCHED_TOKENS}, MAX_MODEL_LEN=${MAX_MODEL_LEN})"
log "Target directory: $DIR"
log "================================================================"

log "Stopping 3-node cluster for pre-flight fabric gate..."
sudo systemctl stop dsv4.service | tee -a "$LOG_MASTER"

log "Fabric gate with engine stopped (3-node)..."
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=auto \
  --json "$DIR/fabric-gate-stopped.json" 2>&1 | tee -a "$LOG_MASTER"

log "Configuring speed profile across all 3 nodes..."
python3 "$HERE/configure_speed_profile.py" --repo "$REPO_MAIN" --batched-tokens "$BATCHED_TOKENS" --max-model-len "$MAX_MODEL_LEN" | tee -a "$LOG_MASTER"
scp "$HERE/configure_speed_profile.py" spark1:/tmp/ >/dev/null 2>&1
scp "$HERE/configure_speed_profile.py" spark2:/tmp/ >/dev/null 2>&1
ssh spark1 "python3 /tmp/configure_speed_profile.py --repo ~/localai/dspark-vllm-gx10 --batched-tokens $BATCHED_TOKENS --max-model-len $MAX_MODEL_LEN" | tee -a "$LOG_MASTER"
ssh spark2 "python3 /tmp/configure_speed_profile.py --repo ~/localai/dspark-vllm-gx10 --batched-tokens $BATCHED_TOKENS --max-model-len $MAX_MODEL_LEN" | tee -a "$LOG_MASTER"

log "Starting 3-node cluster under speed profile..."
sudo systemctl start dsv4.service | tee -a "$DIR/dsv4-start.log" "$LOG_MASTER"

log "Waiting for endpoint health (up to 15 min)..."
ok=0
for i in $(seq 1 90); do
  if curl -sf -m 5 "$API_BASE/health" >/dev/null 2>&1; then
    log "Cluster is HEALTHY at $API_BASE after ~$((i*10))s"
    ok=1
    break
  fi
  sleep 10
done

if [[ $ok -ne 1 ]]; then
  log "ERROR: Cluster failed to become healthy. Aborting."
  exit 1
fi

log "Capturing engine config and environment..."
curl -s "$API_BASE/v1/models" > "$DIR/models.json" || true
sudo docker inspect dspark-vllm-gx10-vllm-dspark-1 --format '{{json .Config.Env}}' > "$DIR/container-env.json" || true
sudo docker exec dspark-vllm-gx10-vllm-dspark-1 ps -eo args | grep vllm > "$DIR/engine-config.txt" || true

log "Running starvation probe (5 trials)..."
python3 "$HERE/benchmark_prefill_starvation.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --label "$LABEL" \
  --repeat 5 \
  --output "$DIR/starvation.jsonl" 2>&1 | tee "$DIR/starvation.log" "$LOG_MASTER"

log "Running depth sweep (2048, 8192, 32768, 131072, 262144)..."
python3 "$HERE/decode_depth_sweep.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --depths 2048,8192,32768,131072,262144 \
  --max-tokens 256 \
  --warmups 2 \
  --reps 7 \
  --label "$LABEL" \
  --output "$DIR/depth.jsonl" 2>&1 | tee "$DIR/depth.log" "$LOG_MASTER"

log "================================================================"
log "SPEED PROFILE SWEEP COMPLETE!"
log "================================================================"
