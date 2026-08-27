#!/usr/bin/env bash
# Orchestrates HANDOFF-2026-08-27 step 1: bring the cluster to 2 nodes,
# gate the fabric with the engine stopped, run the corrected (256-token,
# asserted) decode depth sweep, then restore the 3-node cluster.
#
# Designed to run ON sparkmain (where dsv4.service, cluster_tp2.sh, and the
# API endpoint 127.0.0.1:8100 live). Launch under nohup so it survives
# disconnects:
#   nohup bash ~/bench-repo/scripts/run_2node_corrected_sweep.sh \
#     > ~/bench-repo/results/orchestration-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${STAMP}-decode-2node-fixed"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/orchestration.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

API_BASE="http://127.0.0.1:8100"
MODEL="deepseek-v4-flash-0731"
DEPTHS="2048 8192 32768 131072 262144"

log "=== Step 1: stop 3-node cluster (dsv4.service) ==="
sudo systemctl stop dsv4.service 2>&1 | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  log "ERROR: failed to stop dsv4.service. Aborting before touching the cluster further."
  exit 1
fi

log "=== Step 2: fabric gate with engine stopped (2-node) ==="
bash "$HERE/fabric_gate.sh" "$ROOT/configs/2spark-live.env" --nccl=auto \
  --json "$RUN_DIR/fabric-gate-stopped.json" | tee -a "$LOG"
gate_rc=${PIPESTATUS[0]}
if [[ $gate_rc -ne 0 ]]; then
  log "FABRIC GATE FAILED with engine stopped. Restoring 3-node cluster and aborting."
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  exit 1
fi

log "=== Step 3: bring up 2-node TP=2 cluster ==="
bash "$HERE/cluster_tp2.sh" up 2>&1 | tee "$RUN_DIR/cluster-tp2-up.log" | tee -a "$LOG"
up_rc=${PIPESTATUS[0]}
if [[ $up_rc -ne 0 ]]; then
  log "ERROR: cluster_tp2.sh up failed. Attempting to restore 3-node cluster."
  bash "$HERE/cluster_tp2.sh" down 2>&1 | tee -a "$LOG"
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  exit 1
fi

log "=== Step 4: capture live engine config (2-node) ==="
PID=$(pgrep -f 'tensor-parallel-size 2' | head -1)
if [[ -n "${PID:-}" ]]; then
  tr '\0' ' ' < "/proc/$PID/cmdline" > "$RUN_DIR/engine-config.txt"
  echo >> "$RUN_DIR/engine-config.txt"
else
  echo "PID not found via pgrep pattern" > "$RUN_DIR/engine-config.txt"
  sudo docker exec dspark-vllm-gx10-vllm-dspark-1 ps -eo args 2>/dev/null | grep -m1 vllm >> "$RUN_DIR/engine-config.txt" || true
fi
cat "$RUN_DIR/engine-config.txt" | tee -a "$LOG"

log "=== Step 5: sanity check (correctness) ==="
curl -sf -m 30 -X POST "$API_BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | tee -a "$LOG"
echo | tee -a "$LOG"

log "=== Step 6: corrected decode depth sweep (256-tok window, asserted) ==="
for depth in $DEPTHS; do
  log "--- depth $depth ---"
  python3 "$HERE/decode_depth_sweep.py" \
    --base-url "$API_BASE/v1" \
    --model "$MODEL" \
    --depths "$depth" \
    --max-tokens 256 \
    --warmups 2 \
    --reps 7 \
    --label tp2-fixed \
    --output "$RUN_DIR/tp2-fixed-${depth}.jsonl" \
    2>&1 | tee "$RUN_DIR/tp2-fixed-${depth}.log" | tee -a "$LOG"
  sweep_rc=${PIPESTATUS[0]}
  if [[ $sweep_rc -ne 0 ]]; then
    log "WARNING: depth $depth sweep exited non-zero (rc=$sweep_rc) -- likely the window assertion. Continuing to next depth; inspect the .log."
  fi
done

log "=== Step 7: bring down 2-node cluster ==="
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$RUN_DIR/cluster-tp2-down.log" | tee -a "$LOG"

log "=== Step 8: restore 3-node cluster (dsv4.service) ==="
sudo systemctl start dsv4.service 2>&1 | tee "$RUN_DIR/dsv4-restart.log" | tee -a "$LOG"
restore_rc=${PIPESTATUS[0]}
if [[ $restore_rc -ne 0 ]]; then
  log "ERROR: dsv4.service failed to restart. MANUAL INTERVENTION NEEDED -- cluster may be down."
  exit 1
fi

log "=== Step 9: verify 3-node cluster is healthy ==="
ok=0
for i in $(seq 1 60); do
  if curl -sf -m 5 -o /dev/null "$API_BASE/health" 2>/dev/null; then
    ok=1
    break
  fi
  sleep 10
done
if [[ $ok -eq 1 ]]; then
  log "3-node cluster healthy after restart."
else
  log "WARNING: 3-node cluster did not report healthy within 10 minutes of restart. Check manually."
fi

log "=== Step 10: fabric gate on restored 3-node cluster (engine running, --nccl=skip) ==="
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=skip \
  --json "$RUN_DIR/fabric-gate-restored-3node.json" 2>&1 | tee -a "$LOG"

log "=== DONE. Run directory: $RUN_DIR ==="
