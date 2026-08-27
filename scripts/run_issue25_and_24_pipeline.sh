#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

LOG_MASTER="$ROOT/results/master-pipeline-$(date -u +%Y%m%dT%H%M%SZ).log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_MASTER"; }

API_BASE="http://127.0.0.1:8100"
MODEL="deepseek-v4-flash-0731"
REPO_MAIN="/home/sparkmain/localai/dspark-vllm-gx10"

log "================================================================"
log "Starting Issue #25 (Profile B) + Issue #24 (15-rep 131K)"
log "================================================================"

# Helper to configure all 3 nodes
config_profile() {
  local prof="$1"
  log "Configuring profile $prof on all 3 nodes..."
  python3 "$HERE/configure_issue25_profile.py" "$prof" --repo "$REPO_MAIN" | tee -a "$LOG_MASTER"
  scp "$HERE/configure_issue25_profile.py" spark1:/tmp/ >/dev/null 2>&1
  scp "$HERE/configure_issue25_profile.py" spark2:/tmp/ >/dev/null 2>&1
  ssh spark1 "python3 /tmp/configure_issue25_profile.py $prof --repo ~/localai/dspark-vllm-gx10" | tee -a "$LOG_MASTER"
  ssh spark2 "python3 /tmp/configure_issue25_profile.py $prof --repo ~/localai/dspark-vllm-gx10" | tee -a "$LOG_MASTER"
}

# -----------------------------------------------------------------------------
# PHASE 2: Profile B (TP=3 with published recipe deltas)
# -----------------------------------------------------------------------------
STAMP_B=$(date -u +%Y%m%dT%H%M%SZ)
DIR_B="$ROOT/results/${STAMP_B}-issue25-profile-b"
mkdir -p "$DIR_B"
log "=== PHASE 2: Starting Profile B run in $DIR_B ==="

log "Ensuring 3-node cluster is stopped before fabric gate..."
sudo systemctl stop dsv4.service | tee -a "$LOG_MASTER"

log "Fabric gate with engine stopped (3-node)..."
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=auto \
  --json "$DIR_B/fabric-gate-stopped.json" 2>&1 | tee -a "$LOG_MASTER"

config_profile b

log "Starting 3-node cluster under Profile B..."
sudo systemctl start dsv4.service | tee -a "$DIR_B/dsv4-start.log" "$LOG_MASTER"

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

log "Capturing Profile B engine config and environment..."
curl -s "$API_BASE/v1/models" > "$DIR_B/models.json" || true
sudo docker inspect dspark-vllm-gx10-vllm-dspark-1 --format '{{json .Config.Env}}' > "$DIR_B/container-env.json" || true
sudo docker exec dspark-vllm-gx10-vllm-dspark-1 ps -eo args | grep vllm > "$DIR_B/engine-config.txt" || true

log "Running Profile B starvation probe (5 trials)..."
python3 "$HERE/benchmark_prefill_starvation.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --label profile-b \
  --repeat 5 \
  --output "$DIR_B/starvation.jsonl" 2>&1 | tee "$DIR_B/starvation.log" "$LOG_MASTER"

log "Running Profile B depth sweep (2048, 8192, 32768, 131072, 262144)..."
python3 "$HERE/decode_depth_sweep.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --depths 2048,8192,32768,131072,262144 \
  --max-tokens 256 \
  --warmups 2 \
  --reps 7 \
  --label profile-b \
  --output "$DIR_B/depth.jsonl" 2>&1 | tee "$DIR_B/depth.log" "$LOG_MASTER"

# -----------------------------------------------------------------------------
# PHASE 3: Matched 15-rep 131K comparison (Issue #24) on TP=2 vs TP=3
# -----------------------------------------------------------------------------
STAMP_24=$(date -u +%Y%m%dT%H%M%SZ)
DIR_24="$ROOT/results/${STAMP_24}-node-count-131k-15rep"
mkdir -p "$DIR_24"
log "=== PHASE 3: Node-Count comparison with 15 reps at 131K in $DIR_24 ==="

log "Running 15 reps at 131K for TP=3 (Profile B)..."
python3 "$HERE/decode_depth_sweep.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --depths 131072 \
  --max-tokens 256 \
  --warmups 2 \
  --reps 15 \
  --label tp3-b \
  --output "$DIR_24/tp3-131k-15rep.jsonl" 2>&1 | tee "$DIR_24/tp3-131k-15rep.log" "$LOG_MASTER"

log "Stopping 3-node cluster to prepare for TP=2 arm..."
sudo systemctl stop dsv4.service | tee -a "$LOG_MASTER"

log "Fabric gate with engine stopped (2-node)..."
bash "$HERE/fabric_gate.sh" "$ROOT/configs/2spark-live.env" --nccl=auto \
  --json "$DIR_24/fabric-gate-tp2-stopped.json" 2>&1 | tee -a "$LOG_MASTER"

log "Bringing up 2-node TP=2 cluster..."
bash "$HERE/cluster_tp2.sh" up 2>&1 | tee "$DIR_24/cluster-tp2-up.log" "$LOG_MASTER"

log "Capturing TP=2 engine config..."
sudo docker exec dspark-vllm-gx10-vllm-dspark-1 ps -eo args | grep vllm > "$DIR_24/engine-config-tp2.txt" || true

log "Running TP=2 depth sweep (7 reps at 2K, 8K, 32K, 262K, and 15 reps at 131K)..."
python3 "$HERE/decode_depth_sweep.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --depths 2048,8192,32768,262144 \
  --max-tokens 256 \
  --warmups 2 \
  --reps 7 \
  --label tp2 \
  --output "$DIR_24/tp2-depth.jsonl" 2>&1 | tee "$DIR_24/tp2-depth.log" "$LOG_MASTER"

python3 "$HERE/decode_depth_sweep.py" \
  --base-url "$API_BASE/v1" \
  --model "$MODEL" \
  --depths 131072 \
  --max-tokens 256 \
  --warmups 2 \
  --reps 15 \
  --label tp2 \
  --output "$DIR_24/tp2-131k-15rep.jsonl" 2>&1 | tee "$DIR_24/tp2-131k-15rep.log" "$LOG_MASTER"

log "Bringing down 2-node TP=2 cluster..."
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$DIR_24/cluster-tp2-down.log" "$LOG_MASTER"

# -----------------------------------------------------------------------------
# PHASE 4: Restore Production TP=3 Profile
# -----------------------------------------------------------------------------
log "=== PHASE 4: Restoring Production Configuration ==="
config_profile b

log "Starting production dsv4.service..."
sudo systemctl start dsv4.service | tee -a "$LOG_MASTER"

log "Verifying 3-node cluster health..."
for i in $(seq 1 60); do
  if curl -sf -m 5 "$API_BASE/health" >/dev/null 2>&1; then
    log "Production 3-node cluster is UP and HEALTHY."
    break
  fi
  sleep 10
done

log "Fabric gate on restored cluster (engine running, --nccl=skip)..."
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=skip \
  --json "$DIR_24/fabric-gate-restored.json" 2>&1 | tee -a "$LOG_MASTER"

log "Testing live completion sanity check..."
curl -sf -m 30 -X POST "$API_BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is 17 times 23? Reply with only the number."}],"max_tokens":16,"temperature":0}' \
  | tee -a "$DIR_24/live-completion.json" "$LOG_MASTER"
echo | tee -a "$LOG_MASTER"

log "================================================================"
log "PIPELINE COMPLETE!"
log "================================================================"
