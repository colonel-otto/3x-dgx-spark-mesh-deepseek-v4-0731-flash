#!/usr/bin/env bash
# Resume the matched 2v3 comparison: run the TP=2 arm ONLY.
#
# The TP=3 arm completed and its bundle is intact (5 depth cells, concurrency
# sweep, clock telemetry). The orchestrator then died mid-Step-4 because the
# script file was overwritten by scp WHILE bash was still reading it -- bash
# reads scripts incrementally by byte offset, so the file shifting underneath
# the interpreter produced a syntax error at line 260 and the run aborted.
#
# NEVER scp over a script that is currently executing. This resume writes to a
# NEW path so the same failure cannot recur.
#
# Preconditions when this starts (verified below, not assumed):
#   - dsv4.service stopped, no vllm containers on any node
#   - config/head.env + config/worker.env already carry the MATCHED config
#     (MAX_NUM_SEQS=32, MTP_NUM_TOKENS=2, GPU_MEMORY_UTILIZATION=0.835)
#     with .prematched.<STAMP> backups beside them
#   - tp2/fabric-gate.json already written and PASSED
#
# Run ON sparkmain under nohup:
#   nohup bash ~/bench-repo/scripts/run_matched_2v3_tp2only.sh \
#     > ~/bench-repo/results/tp2arm-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

# Reuse the EXISTING bundle so both arms live together.
RUN_DIR="${RUN_DIR:-$(ls -td "$ROOT"/results/*-matched-2v3 | head -1)}"
[[ -d "$RUN_DIR/tp3" ]] || { echo "FATAL: no tp3 arm at $RUN_DIR"; exit 1; }
mkdir -p "$RUN_DIR/tp2"
LOG="$RUN_DIR/orchestration-tp2.log"
STAMP=$(basename "$RUN_DIR" | sed 's/-matched-2v3//')

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

API="http://127.0.0.1:8100"
METRICS="$API/metrics"
MODEL="deepseek-v4-flash-0731"
DEPTHS="2048 8192 32768 131072 262144"
CCS="4,8,16"
REPS=7
CC_REPS=5
WARMUPS=2
M_SEQS=32; M_MTP=2; M_GPUMEM=0.835
COOL_TARGET_C=70; COOL_MAX_S=900

WEBUI_STOPPED=0
TELEM_PIDS=""
telem_stop() { [[ -n "$TELEM_PIDS" ]] && kill $TELEM_PIDS 2>/dev/null; TELEM_PIDS=""; }
restore_webui() {
  if [[ $WEBUI_STOPPED -eq 1 ]]; then
    log "Restoring open-webui ..."
    sudo docker start open-webui >/dev/null 2>&1 || log "WARN: could not restart open-webui"
    WEBUI_STOPPED=0
  fi
}
cleanup() { telem_stop; restore_webui; }
trap cleanup EXIT

telem_start() {
  local dir="$1" h
  for h in sparkmain spark1 spark2; do
    ssh -n "$h" "nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu \
      --format=csv,noheader -l 5" > "$dir/clocks-$h.csv" 2>/dev/null &
    TELEM_PIDS="$TELEM_PIDS $!"
  done
  log "telemetry started -> $dir/clocks-*.csv"
}

cooldown() {
  local target="$1" maxw="$2" waited=0 hot t h
  log "cooldown: waiting for all nodes <= ${target}C (max ${maxw}s)"
  while (( waited < maxw )); do
    hot=0
    for h in sparkmain spark1 spark2; do
      t=$(ssh -n "$h" "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader" 2>/dev/null | tr -dc '0-9')
      [[ -z "$t" ]] && continue
      (( t > target )) && hot=1
    done
    (( hot == 0 )) && { log "cooldown: all nodes <= ${target}C after ${waited}s"; return 0; }
    sleep 30; waited=$((waited+30))
  done
  log "cooldown: TIMEOUT after ${maxw}s -- proceeding; temps are in clocks-*.csv"
}

log "### TP=2 arm resume -> $RUN_DIR"

# --- Step 0: preconditions, verified not assumed --------------------------
log "=== Step 0: verify preconditions ==="
[[ -s "$RUN_DIR/tp2/fabric-gate.json" ]] || die "tp2 fabric gate artifact missing"
log "tp2 fabric gate artifact present."

for pair in "sparkmain:config/head.env" "spark1:config/worker.env"; do
  h="${pair%%:*}"; f="${pair##*:}"
  got=$(ssh -n "$h" "grep -E '^(MAX_NUM_SEQS|MTP_NUM_TOKENS|GPU_MEMORY_UTILIZATION)=' ~/localai/dspark-vllm-gx10/$f | sort | tr '\n' ' '")
  log "$h $f: $got"
  echo "$got" | grep -q "GPU_MEMORY_UTILIZATION=$M_GPUMEM" || die "$h $f not at gpumem=$M_GPUMEM"
  echo "$got" | grep -q "MAX_NUM_SEQS=$M_SEQS"            || die "$h $f not at seqs=$M_SEQS"
  echo "$got" | grep -q "MTP_NUM_TOKENS=$M_MTP"           || die "$h $f not at mtp=$M_MTP"
  ssh -n "$h" "ls ~/localai/dspark-vllm-gx10/${f}.prematched.* >/dev/null 2>&1" \
    || die "$h has no .prematched backup of $f -- refusing to proceed without a restore path"
done
log "Matched config verified on both ranks, backups present."

for h in sparkmain spark1 spark2; do
  ssh -n "$h" 'sudo docker ps --format "{{.Names}}" | grep -q vllm-dspark' 2>/dev/null \
    && die "$h still runs a vllm container -- cluster must be down before TP=2 bringup"
done
log "All nodes clear of vllm containers."

log "=== Step 0b: stop open-webui (Requirement 5) ==="
if sudo docker ps --format '{{.Names}}' | grep -q '^open-webui$'; then
  sudo docker stop open-webui >/dev/null 2>&1 && WEBUI_STOPPED=1 && log "open-webui stopped."
else
  log "open-webui not running."
fi

# --- Step 1: bring up TP=2 ------------------------------------------------
log "=== Step 1: bring up TP=2 ==="
bash "$HERE/cluster_tp2.sh" up 2>&1 | tee "$RUN_DIR/tp2/cluster-up.log" | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  log "cluster_tp2.sh up FAILED. Restoring env files and 3-node service."
  bash "$HERE/cluster_tp2.sh" down 2>&1 | tee -a "$LOG"
  for pair in "sparkmain:config/head.env" "spark1:config/worker.env"; do
    h="${pair%%:*}"; f="${pair##*:}"
    ssh -n "$h" "cd ~/localai/dspark-vllm-gx10 && cp \$(ls -t ${f}.prematched.* | head -1) $f" 2>&1 | tee -a "$LOG"
  done
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  die "TP=2 bringup failed"
fi

# --- Step 2: assert the arm really is matched -----------------------------
log "=== Step 2: verify live TP=2 engine matches the profile ==="
live=$(ps -eo args | grep -m1 '[v]llm.*tensor-parallel-size' || true)
echo "$live" > "$RUN_DIR/tp2/engine-config.txt"
log "live: $live"
echo "$live" | grep -q 'tensor-parallel-size 2'       || die "not TP=2"
echo "$live" | grep -q 'max-num-seqs 32'              || die "TP=2 not at max-num-seqs=32 -- MATCH FAILED"
echo "$live" | grep -q 'gpu-memory-utilization 0.835' || die "TP=2 not at gpu-mem=0.835 -- MATCH FAILED"
echo "$live" | grep -q '"num_speculative_tokens":2'   || die "TP=2 not at MTP=2 -- MATCH FAILED"
log "MATCH CONFIRMED: node count is the only variable."

sudo docker logs dspark-vllm-gx10-vllm-dspark-1 2>&1 \
  | grep -iE 'GPU KV cache size|maximum concurrency' | tail -5 > "$RUN_DIR/tp2/kv-pool-initlog.txt" || true
curl -s -m 20 "$METRICS" | grep -E 'cache_config_info' > "$RUN_DIR/tp2/kv-pool-metrics.txt" || true
cat "$RUN_DIR/tp2/kv-pool-initlog.txt" | tee -a "$LOG"

# --- Step 3: correctness gate ---------------------------------------------
log "=== Step 3: correctness ==="
out=$(curl -sf -m 90 -X POST "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}")
echo "$out" | tee -a "$LOG"
echo "$out" | grep -q 391 || die "correctness failed (expected 391)"

# --- Step 4: measure ------------------------------------------------------
cooldown "$COOL_TARGET_C" "$COOL_MAX_S"
telem_start "$RUN_DIR/tp2"

excl_out=$(python3 "$HERE/exclusivity.py" --url "$METRICS" --check-idle --timeout 60 2>&1)
echo "$excl_out" >> "$LOG"
start_total=$(echo "$excl_out" | sed -n 's/^IDLE_OK start_request_success_total=\([0-9.]*\).*/\1/p' | tail -1)
[[ -z "$start_total" ]] && die "cluster not idle: $excl_out"
log "exclusivity start_total=$start_total"

issued=0
log "=== [tp2] depth sweep ==="
for d in $DEPTHS; do
  log "--- [tp2] depth $d ---"
  python3 "$HERE/decode_depth_sweep.py" \
    --base-url "$API/v1" --model "$MODEL" --depths "$d" \
    --max-tokens 256 --warmups $WARMUPS --reps $REPS --label tp2 \
    --output "$RUN_DIR/tp2/decode-${d}.jsonl" \
    2>&1 | tee "$RUN_DIR/tp2/decode-${d}.log" | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && log "WARN: [tp2] depth $d exited non-zero"
  issued=$(( issued + REPS + WARMUPS ))
done

log "=== [tp2] concurrency sweep cc=$CCS @8K (H2) ==="
python3 "$HERE/benchmark_mtp_concurrency.py" \
  --url "$API/v1" --metrics-url "$METRICS" --model "$MODEL" \
  --mtp-k $M_MTP --depth 8192 --concurrencies "$CCS" \
  --reps $CC_REPS --max-tokens 256 --warmups $WARMUPS \
  --out "$RUN_DIR/tp2/concurrency.json" \
  2>&1 | tee "$RUN_DIR/tp2/concurrency.log" | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && log "WARN: [tp2] concurrency sweep exited non-zero"

# Ledger must include the concurrency harness's own traffic, or Requirement 5
# reports it as foreign (this is what produced the tp3 arm's false FAIL).
cc_issued=0
for c in ${CCS//,/ }; do cc_issued=$(( cc_issued + c * (CC_REPS + WARMUPS) )); done
issued=$(( issued + cc_issued + WARMUPS ))
log "[tp2] expected request ledger: $issued"

python3 "$HERE/exclusivity.py" --url "$METRICS" --verify \
  --start-total "$start_total" --expected "$issued" \
  --out "$RUN_DIR/tp2/exclusivity.json" 2>&1 | tee -a "$LOG"

telem_stop
for h in sparkmain spark1 spark2; do
  [[ -s "$RUN_DIR/tp2/clocks-$h.csv" ]] || continue
  awk -F', *' '{gsub(/ MHz/,"",$2); gsub(/ W/,"",$4);
                if($2+0>0){n++; s+=$2; if($2+0>mx||n==1)mx=$2+0; if($2+0<mn||n==1)mn=$2+0;
                           ts+=$3; if($3+0>tmx||n==1)tmx=$3+0}}
    END{if(n)printf "  %s: clock mean %.0f MHz (min %d, max %d), temp mean %.0f C (max %d), n=%d\n",
                     H, s/n, mn, mx, ts/n, tmx, n}' H="$h" "$RUN_DIR/tp2/clocks-$h.csv" | tee -a "$LOG"
done
log "=== [tp2] arm complete ==="

# --- Step 5: restore production ------------------------------------------
log "=== Step 5: tear down TP=2, restore env, restart 3-node ==="
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$RUN_DIR/tp2/cluster-down.log" | tee -a "$LOG"
for pair in "sparkmain:config/head.env" "spark1:config/worker.env"; do
  h="${pair%%:*}"; f="${pair##*:}"
  ssh -n "$h" "cd ~/localai/dspark-vllm-gx10 && cp \$(ls -t ${f}.prematched.* | head -1) $f && echo restored $f" 2>&1 | tee -a "$LOG"
done
sudo systemctl start dsv4.service 2>&1 | tee "$RUN_DIR/dsv4-restart.log" | tee -a "$LOG"

log "=== Step 6: wait for 3-node health ==="
ok=0
for i in $(seq 1 180); do
  curl -sf -m 5 -o /dev/null "$API/health" 2>/dev/null && { ok=1; break; }
  sleep 10
done
[[ $ok -eq 1 ]] && log "3-node cluster healthy." || log "WARNING: 3-node not healthy in 30 min -- CHECK MANUALLY"

restore_webui
log "### DONE. Both arms in: $RUN_DIR"
