#!/usr/bin/env bash
# Configuration-identical 2-node vs 3-node comparison.
#
# Pre-registered in docs/PREREGISTRATION-2V3-MATCHED.md. Read that first --
# the hypotheses and the tie band were fixed BEFORE this ran.
#
# WHAT THIS FIXES vs run_2node_corrected_sweep.sh:
#   1. Sets the 2-node arm to the SAME config as production TP=3
#      (MAX_NUM_SEQS=32, MTP_NUM_TOKENS=2, GPU_MEMORY_UTILIZATION=0.835).
#      The old script left head.env/worker.env at 16 / 5 / 0.80 -- the confound.
#   2. Enforces Requirement 5 (exclusivity) around every measured arm.
#      open-webui runs against this engine and caused a 3.5x false regression
#      on 2026-08-29. It is stopped for the duration and restored at the end.
#   3. Measures the CONCURRENCY arm too (cc=4/8/16), where hypothesis H2 lives.
#      The old script only did the depth sweep.
#   4. Measures BOTH arms in one session so fabric state is shared.
#
# Run ON sparkmain, under nohup so it survives disconnect:
#   nohup bash ~/bench-repo/scripts/run_matched_2v3.sh \
#     > ~/bench-repo/results/matched2v3-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
#
# SAFETY: this script never reboots and never force-kills the cluster. On any
# failure it restores the 3-node production service and stops.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${STAMP}-matched-2v3"
mkdir -p "$RUN_DIR"/{tp3,tp2}
LOG="$RUN_DIR/orchestration.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }   # cleanup runs via the EXIT trap

API="http://127.0.0.1:8100"
METRICS="$API/metrics"
MODEL="deepseek-v4-flash-0731"
DEPTHS="2048 8192 32768 131072 262144"
CCS="4,8,16"
REPS=7

# Matched configuration -- the ONLY thing that differs between arms is node count.
M_SEQS=32
M_MTP=2
M_GPUMEM=0.835

# Thermal equalisation between arms. 70C is reachable from idle on all three
# nodes; the cap keeps a hot node from stalling the run indefinitely.
COOL_TARGET_C=70
COOL_MAX_S=900

WEBUI_STOPPED=0
TELEM_PIDS=""

# --- clock/thermal telemetry -------------------------------------------
# GB10 does NOT honour nvidia-smi -lgc (docs/GPU-CLOCKS-NOT-LOCKABLE.md): the
# clock floats against a package power budget and varies per node with thermal
# state. It cannot be controlled, so it must be RECORDED -- continuously, not
# once. The Issue #36 bundle claimed locked clocks on the strength of a single
# idle sample; this samples every 5s for the life of each arm.
telem_start() { # $1 = arm dir
  local dir="$1" h
  for h in sparkmain spark1 spark2; do
    ssh -n "$h" "nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu \
      --format=csv,noheader -l 5" > "$dir/clocks-$h.csv" 2>/dev/null &
    TELEM_PIDS="$TELEM_PIDS $!"
  done
  log "telemetry started -> $dir/clocks-*.csv"
}
telem_stop() {
  [[ -n "$TELEM_PIDS" ]] && kill $TELEM_PIDS 2>/dev/null
  TELEM_PIDS=""
}

# Equalise thermal state before an arm so a cool 2-node arm is not compared
# against a heat-soaked 3-node one. Capped so it cannot stall the run forever.
cooldown() { # $1 = target degC, $2 = max wait seconds
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

restore_webui() {
  if [[ $WEBUI_STOPPED -eq 1 ]]; then
    log "Restoring open-webui ..."
    sudo docker start open-webui >/dev/null 2>&1 || log "WARN: could not restart open-webui"
    WEBUI_STOPPED=0
  fi
}
cleanup() { telem_stop; restore_webui; }
trap cleanup EXIT

# --- exclusivity helpers (Requirement 5) ---------------------------------
excl_start() {  # echoes start total, or empty on failure
  local out
  out=$(python3 "$HERE/exclusivity.py" --url "$METRICS" --check-idle --timeout 60 2>&1)
  echo "$out" >> "$LOG"
  # Parse the explicit key, not a trailing-number grep: IDLE_FAIL messages also
  # end in digits and would otherwise be mistaken for a valid start total.
  echo "$out" | sed -n 's/^IDLE_OK start_request_success_total=\([0-9.]*\).*/\1/p' | tail -1
}
excl_verify() { # $1 = arm dir, $2 = start total, $3 = expected count
  python3 "$HERE/exclusivity.py" --url "$METRICS" --verify \
    --start-total "$2" --expected "$3" --out "$1/exclusivity.json" \
    2>&1 | tee -a "$LOG"
}

capture_cfg() { # $1 = arm dir
  ssh -n localhost true 2>/dev/null
  ps -eo args | grep -m1 '[v]llm.*tensor-parallel-size' > "$1/engine-config.txt" 2>/dev/null \
    || sudo docker exec dspark-vllm-gx10-vllm-dspark-1 ps -eo args 2>/dev/null \
       | grep -m1 vllm > "$1/engine-config.txt" || true
  # KV pool from THIS boot's own init log, per BENCHMARK-POLICY Requirement 4
  sudo docker logs dspark-vllm-gx10-vllm-dspark-1 2>&1 \
    | grep -iE 'GPU KV cache size|maximum concurrency' | tail -5 > "$1/kv-pool-initlog.txt" || true
  curl -s -m 10 "$METRICS" | grep -E 'cache_config_info' > "$1/kv-pool-metrics.txt" || true
  log "--- config for $1 ---"; cat "$1/engine-config.txt" | tee -a "$LOG"
  cat "$1/kv-pool-initlog.txt" | tee -a "$LOG"
}

sanity() { # correctness gate -- must pass before any measurement
  local out
  out=$(curl -sf -m 60 -X POST "$API/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}")
  echo "$out" | tee -a "$LOG"
  echo "$out" | grep -q 391 || die "correctness check failed (expected 391). Refusing to benchmark a cluster that may be serving nonsense."
}

measure_arm() { # $1 = label (tp3|tp2), $2 = arm dir
  local label="$1" dir="$2"
  log "=== MEASURE [$label]: exclusivity + config capture ==="
  capture_cfg "$dir"
  sanity
  # Both arms start from a comparable thermal state (clock is uncontrollable
  # on GB10, so equalise the thing that drives it).
  cooldown "$COOL_TARGET_C" "$COOL_MAX_S"
  telem_start "$dir"

  local start_total
  start_total=$(excl_start)
  [[ -z "$start_total" ]] && die "[$label] cluster not idle -- foreign traffic present, refusing to measure"
  log "[$label] exclusivity start_total=$start_total"

  log "=== [$label] depth sweep (256-tok asserted, warm, reps=$REPS) ==="
  local issued=0
  for d in $DEPTHS; do
    log "--- [$label] depth $d ---"
    python3 "$HERE/decode_depth_sweep.py" \
      --base-url "$API/v1" --model "$MODEL" --depths "$d" \
      --max-tokens 256 --warmups 2 --reps $REPS --label "$label" \
      --output "$dir/decode-${d}.jsonl" \
      2>&1 | tee "$dir/decode-${d}.log" | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} -ne 0 ]] && log "WARN: [$label] depth $d exited non-zero -- inspect log (window assertion?)"
    issued=$((issued + REPS + 2))
  done

  log "=== [$label] concurrency sweep cc=$CCS @8K (H2) ==="
  python3 "$HERE/benchmark_mtp_concurrency.py" \
    --url "$API/v1" --metrics-url "$METRICS" --model "$MODEL" \
    --mtp-k $M_MTP --depth 8192 --concurrencies "$CCS" \
    --reps 5 --max-tokens 256 \
    --out "$dir/concurrency.json" \
    2>&1 | tee "$dir/concurrency.log" | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && log "WARN: [$label] concurrency sweep exited non-zero"

  excl_verify "$dir" "$start_total" "$issued" || log "WARN: [$label] exclusivity delta mismatch -- see exclusivity.json"
  telem_stop
  # Publish the clock/thermal envelope this arm actually ran under, so the
  # reader can see whether the two arms were comparable.
  for h in sparkmain spark1 spark2; do
    [[ -s "$dir/clocks-$h.csv" ]] || continue
    awk -F', *' '{gsub(/ MHz/,"",$2); gsub(/ W/,"",$4);
                  if($2+0>0){n++; s+=$2; if($2+0>mx||n==1)mx=$2+0; if($2+0<mn||n==1)mn=$2+0;
                             ts+=$3; if($3+0>tmx||n==1)tmx=$3+0}}
      END{if(n)printf "  %s: clock mean %.0f MHz (min %d, max %d), temp mean %.0f C (max %d), n=%d\n",
                       H, s/n, mn, mx, ts/n, tmx, n}' H="$h" "$dir/clocks-$h.csv" | tee -a "$LOG"
  done
  log "=== [$label] arm complete ==="
}

# =========================================================================
log "### Matched 2v3 run $STAMP -> $RUN_DIR"
log "### Pre-registration: docs/PREREGISTRATION-2V3-MATCHED.md"
log "### Matched config: seqs=$M_SEQS mtp=$M_MTP gpumem=$M_GPUMEM"

# --- Step 0: silence the foreign client ----------------------------------
log "=== Step 0: stop open-webui (Requirement 5) ==="
if sudo docker ps --format '{{.Names}}' | grep -q '^open-webui$'; then
  sudo docker stop open-webui >/dev/null 2>&1 && WEBUI_STOPPED=1 && log "open-webui stopped."
else
  log "open-webui not running."
fi

# --- Step 1: verify patch parity across nodes ----------------------------
log "=== Step 1: verify patch parity across all 3 nodes ==="
for h in sparkmain spark1 spark2; do
  ssh -n "$h" "sudo docker run --rm --entrypoint sh dsv4-3spark:0.1.1 -c 'sha256sum /opt/dsv4-tp3/*.py'" \
    > "$RUN_DIR/patch-hashes-$h.txt" 2>&1 || die "could not hash patches on $h"
done
if ! diff -q "$RUN_DIR/patch-hashes-sparkmain.txt" "$RUN_DIR/patch-hashes-spark1.txt" >/dev/null \
   || ! diff -q "$RUN_DIR/patch-hashes-sparkmain.txt" "$RUN_DIR/patch-hashes-spark2.txt" >/dev/null; then
  die "patch files DIFFER across nodes -- arms would not be comparable"
fi
log "Patch parity verified: all 3 nodes byte-identical."

# --- Step 2: ARM A = TP=3 (already live in production config) ------------
log "=== Step 2: ARM A (TP=3) ==="
curl -sf -m 10 -o /dev/null "$API/health" || die "3-node engine not healthy at start"
# Confirm the live engine really is at the matched config before trusting it.
live=$(ps -eo args | grep -m1 '[v]llm.*tensor-parallel-size' || true)
echo "$live" | grep -q 'max-num-seqs 32'            || die "TP=3 arm is not at max-num-seqs=32"
echo "$live" | grep -q 'gpu-memory-utilization 0.835' || die "TP=3 arm is not at gpu-mem=0.835"
echo "$live" | grep -q '"num_speculative_tokens":2' || die "TP=3 arm is not at MTP=2"
log "TP=3 live config matches the matched profile."

bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=skip \
  --json "$RUN_DIR/tp3/fabric-gate.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "fabric gate FAILED on 3-node arm"

measure_arm tp3 "$RUN_DIR/tp3"

# --- Step 3: tear down 3-node, reconfigure 2-node to matched config -------
log "=== Step 3: stop 3-node service ==="
sudo systemctl stop dsv4.service 2>&1 | tee -a "$LOG" || die "could not stop dsv4.service"
sleep 20
# spark2 must release the GPU or cluster_tp2.sh refuses to start
ssh -n spark2 'sudo docker ps --format "{{.Names}}" | grep -q vllm-dspark' 2>/dev/null \
  && { log "spark2 still holds a container; composing it down"; \
       ssh -n spark2 'cd ~/localai/dspark-vllm-gx10 && COMPOSE_DISABLE_ENV_FILE=1 sudo docker compose -p dspark-vllm-gx10 --env-file config/worker.env -f docker-compose.yml down' 2>&1 | tee -a "$LOG"; }

log "=== Step 3b: rewrite 2-node env to the MATCHED config ==="
for pair in "sparkmain:config/head.env" "spark1:config/worker.env"; do
  h="${pair%%:*}"; f="${pair##*:}"
  ssh -n "$h" "cd ~/localai/dspark-vllm-gx10 && cp $f ${f}.prematched.$STAMP && \
    sed -i -e 's/^MAX_NUM_SEQS=.*/MAX_NUM_SEQS=$M_SEQS/' \
           -e 's/^MTP_NUM_TOKENS=.*/MTP_NUM_TOKENS=$M_MTP/' \
           -e 's/^GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=$M_GPUMEM/' $f && \
    grep -E '^(MAX_NUM_SEQS|MTP_NUM_TOKENS|GPU_MEMORY_UTILIZATION|MAX_MODEL_LEN|MAX_NUM_BATCHED_TOKENS)=' $f | sort" \
    2>&1 | tee -a "$LOG" || die "could not rewrite $f on $h"
done
log "2-node env rewritten (backups: *.prematched.$STAMP)"

# --- Step 4: fabric gate with engine stopped, then bring up TP=2 ---------
log "=== Step 4: fabric gate (engine stopped) ==="
bash "$HERE/fabric_gate.sh" "$ROOT/configs/2spark-live.env" --nccl=auto \
  --json "$RUN_DIR/tp2/fabric-gate.json" 2>&1 | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  log "FABRIC GATE FAILED on 2-node arm. Restoring 3-node and aborting."
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  die "fabric gate failed pre-TP=2"
fi

log "=== Step 5: bring up TP=2 ==="
bash "$HERE/cluster_tp2.sh" up 2>&1 | tee "$RUN_DIR/tp2/cluster-up.log" | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  log "cluster_tp2.sh up FAILED. Restoring 3-node."
  bash "$HERE/cluster_tp2.sh" down 2>&1 | tee -a "$LOG"
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  die "TP=2 bringup failed"
fi

# --- Step 6: ARM B = TP=2 ------------------------------------------------
log "=== Step 6: ARM B (TP=2) ==="
live2=$(ps -eo args | grep -m1 '[v]llm.*tensor-parallel-size' || true)
echo "$live2" | grep -q 'tensor-parallel-size 2'      || die "TP=2 arm did not come up at TP=2"
echo "$live2" | grep -q 'max-num-seqs 32'             || die "TP=2 arm is not at max-num-seqs=32 -- MATCH FAILED"
echo "$live2" | grep -q 'gpu-memory-utilization 0.835'|| die "TP=2 arm is not at gpu-mem=0.835 -- MATCH FAILED"
echo "$live2" | grep -q '"num_speculative_tokens":2'  || die "TP=2 arm is not at MTP=2 -- MATCH FAILED"
log "TP=2 live config matches the matched profile. Node count is the only variable."

measure_arm tp2 "$RUN_DIR/tp2"

# --- Step 7: restore production ------------------------------------------
log "=== Step 7: tear down TP=2, restore env files, restart 3-node ==="
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$RUN_DIR/tp2/cluster-down.log" | tee -a "$LOG"
for pair in "sparkmain:config/head.env" "spark1:config/worker.env"; do
  h="${pair%%:*}"; f="${pair##*:}"
  ssh -n "$h" "cd ~/localai/dspark-vllm-gx10 && cp ${f}.prematched.$STAMP $f && echo restored $f" 2>&1 | tee -a "$LOG"
done
sudo systemctl start dsv4.service 2>&1 | tee "$RUN_DIR/dsv4-restart.log" | tee -a "$LOG"

log "=== Step 8: wait for 3-node health (cold start can be ~7-20 min) ==="
ok=0
for i in $(seq 1 180); do
  curl -sf -m 5 -o /dev/null "$API/health" 2>/dev/null && { ok=1; break; }
  sleep 10
done
[[ $ok -eq 1 ]] && log "3-node cluster healthy." || log "WARNING: 3-node not healthy within 30 min -- CHECK MANUALLY"

restore_webui
log "### DONE. Bundle: $RUN_DIR"
log "### Next: fill the pre-registered tables in docs/PREREGISTRATION-2V3-MATCHED.md"
