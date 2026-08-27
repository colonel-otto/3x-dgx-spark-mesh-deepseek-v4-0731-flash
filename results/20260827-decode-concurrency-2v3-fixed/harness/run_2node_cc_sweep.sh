#!/usr/bin/env bash
# Orchestrates the 2-node concurrency (cc=4/8/16) aggregate throughput arm,
# using the hardened bench_miaai_cc.py harness (min_tokens=max_tokens+ignore_eos,
# with a hard assertion). Mirrors run_2node_corrected_sweep.sh's structure.
set -uo pipefail
HERE="/home/sparkmain/bench-repo/scripts"
ROOT="/home/sparkmain/bench-repo"
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${STAMP}-cc-2node"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/orchestration.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

API_BASE="http://127.0.0.1:8100"
MODEL="deepseek-v4-flash-0731"
DEPTH=8192
MAXTOK=256

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

log "=== Step 4: sanity check (correctness) ==="
curl -sf -m 30 -X POST "$API_BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | tee -a "$LOG"
echo | tee -a "$LOG"

log "=== Step 5: concurrency sweep cc=4,8,16 at depth=$DEPTH, max_tokens=$MAXTOK (2-node) ==="
for cc in 4 8 16; do
  log "--- cc=$cc ---"
  python3 "$HERE/bench_miaai_cc.py" \
    --base-url "$API_BASE/v1" \
    --model "$MODEL" \
    --prompt "$DEPTH" --concurrency "$cc" --max-tokens "$MAXTOK" --repeat 3 \
    --output "$RUN_DIR/2node-cc${cc}.jsonl" \
    2>&1 | tee "$RUN_DIR/2node-cc${cc}.log" | tee -a "$LOG"
  cc_rc=${PIPESTATUS[0]}
  if [[ $cc_rc -ne 0 ]]; then
    log "WARNING: cc=$cc exited non-zero (rc=$cc_rc) -- likely the window assertion. Continuing to next cc; inspect the .log."
  fi
done

log "=== Step 6: bring down 2-node cluster ==="
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$RUN_DIR/cluster-tp2-down.log" | tee -a "$LOG"

log "=== Step 7: restore 3-node cluster (dsv4.service) ==="
sudo systemctl start dsv4.service 2>&1 | tee "$RUN_DIR/dsv4-restart.log" | tee -a "$LOG"
restore_rc=${PIPESTATUS[0]}
if [[ $restore_rc -ne 0 ]]; then
  log "ERROR: dsv4.service failed to restart. MANUAL INTERVENTION NEEDED -- cluster may be down."
  exit 1
fi

log "=== Step 8: verify 3-node cluster is healthy ==="
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

log "=== Step 9: fabric gate on restored 3-node cluster (engine running, --nccl=skip) ==="
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=skip \
  --json "$RUN_DIR/fabric-gate-restored-3node.json" 2>&1 | tee -a "$LOG"

log "=== Step 10: live completion sanity check on restored 3-node cluster ==="
curl -sf -m 30 -X POST "$API_BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | tee -a "$LOG"
echo | tee -a "$LOG"

log "=== DONE. Run directory: $RUN_DIR ==="
