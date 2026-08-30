#!/usr/bin/env bash
# llama-benchy 2-node vs 3-node arm -- an INDEPENDENT, third-party check on the
# result our own harness produced in RESULT-2V3-MATCHED-2026-08-30.md.
#
# Implements docs/PLAN-LLAMA-BENCHY-2V3.md. Read that first; it carries the
# pre-registered expectations (L1/L2/L3) and the load-bearing caveat that
# cross-harness ABSOLUTE numbers must never be compared -- only the 2v3 ratio
# computed within each harness.
#
# ORDER DIFFERS FROM THE PLAN, DELIBERATELY.
# The plan runs TP=2 first. The cluster is already up, healthy and verified at
# the matched TP=3 profile, so this runs TP=3 FIRST and spends one cold start
# instead of two (~30 min saved). Nothing about the comparison depends on arm
# order; both arms are cooled to the same temperature gate before measuring.
#
# THREE DEFECTS IN THE PLAN'S COMMANDS ARE CORRECTED HERE:
#   1. llama-benchy has no --output flag. It is --save-result. The plan's
#      commands as written die with "unrecognized arguments".
#   2. --tokenizer is necessary but NOT sufficient: llama-benchy falls back to
#      gpt2 SILENTLY on a load failure, which is the 11.6% token-count error the
#      plan documents. This script proves the real tokenizer loaded, per arm,
#      and refuses to measure if it did not.
#   3. --no-cache is required for the concurrency sweep. Without it llama-benchy
#      reuses one prompt across clients, so vLLM's prefix cache (enabled on this
#      engine) serves most of the prefill and the sweep measures cache hits.
#
# NEVER scp over a script that is currently executing -- bash reads scripts
# incrementally by byte offset and the file shifting underneath the interpreter
# produces a syntax error mid-run. That is how the previous orchestrator died.
#
# Run ON sparkmain under nohup:
#   nohup bash ~/bench-repo/scripts/run_llama_benchy_2v3.sh \
#     > ~/bench-repo/results/benchy-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${STAMP}-llama-benchy-2v3"
mkdir -p "$RUN_DIR/tp3" "$RUN_DIR/tp2"
LOG="$RUN_DIR/orchestration.log"

BENCHY_DIR="$HOME/llama-benchy"
BENCHY="$BENCHY_DIR/.venv/bin/llama-benchy"
TOKENIZER="$HOME/dsv4-tokenizer"
API="http://127.0.0.1:8100"
METRICS="$API/metrics"
MODEL="deepseek-v4-flash-0731"

RUNS=10
WARMUPS=3
DEPTHS="0 8192 32768 131072"
CONCS="1 4 8 16"
COOL_TARGET_C=70
COOL_MAX_S=900

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

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

clock_summary() {
  local dir="$1" h
  for h in sparkmain spark1 spark2; do
    [[ -s "$dir/clocks-$h.csv" ]] || continue
    awk -F', *' '{gsub(/ MHz/,"",$2); gsub(/ W/,"",$4);
                  if($2+0>0){n++; s+=$2; if($2+0>mx||n==1)mx=$2+0; if($2+0<mn||n==1)mn=$2+0;
                             ts+=$3; if($3+0>tmx||n==1)tmx=$3+0}}
      END{if(n)printf "  %s: clock mean %.0f MHz (min %d, max %d), temp mean %.0f C (max %d), n=%d\n",
                       H, s/n, mn, mx, ts/n, tmx, n}' H="$h" "$dir/clocks-$h.csv" | tee -a "$LOG"
  done
}

# Wait for every node to fall to COOL_TARGET_C. GB10 clocks cannot be locked, so
# equalising thermal state is the only way to stop arm order from biasing the
# result.
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

# Assert the live engine really is the profile this arm claims. A silent
# mismatch here is exactly what makes a 2v3 number worthless.
assert_engine() {
  # NOTE: assignments in a single `local` are evaluated left-to-right, and a
  # later initialiser referencing an earlier name in the SAME statement sees it
  # unset -- which under `set -u` aborts the run. Bind them separately.
  local arm="$1"
  local tp="$2"
  local dir="$RUN_DIR/$arm"
  local live
  live=$(ps -eo args | grep -m1 '[v]llm.*tensor-parallel-size' || true)
  [[ -n "$live" ]] || die "[$arm] no vllm process found"
  echo "$live" > "$dir/engine-config.txt"
  log "[$arm] live engine args captured -> $dir/engine-config.txt"

  echo "$live" | grep -q -- "--tensor-parallel-size $tp"       || die "[$arm] not TP=$tp"
  echo "$live" | grep -q -- '--max-num-seqs 32'                || die "[$arm] max-num-seqs != 32"
  echo "$live" | grep -q -- '--gpu-memory-utilization 0.835'   || die "[$arm] gpu-mem != 0.835"
  echo "$live" | grep -q -- '--max-num-batched-tokens 8192'    || die "[$arm] batched-tokens != 8192"
  echo "$live" | grep -q -- '--max-model-len 1048576'          || die "[$arm] model-len != 1048576"
  echo "$live" | grep -q -- '--kv-cache-dtype nvfp4_ds_mla'    || die "[$arm] kv-cache-dtype wrong"
  echo "$live" | grep -q -- '--long-prefill-token-threshold 1024' || die "[$arm] long-prefill != 1024"
  echo "$live" | grep -q '"num_speculative_tokens":2'          || die "[$arm] MTP != 2"
  echo "$live" | grep -q 'flashinfer_b12x'                     || die "[$arm] moe-backend != flashinfer_b12x"
  log "[$arm] ENGINE MATCH CONFIRMED (TP=$tp; all 9 settings + MoE backend)"

  sudo docker logs dspark-vllm-gx10-vllm-dspark-1 2>&1 \
    | grep -iE 'GPU KV cache size|maximum concurrency' | tail -5 > "$dir/kv-pool-initlog.txt" || true
  curl -s -m 20 "$METRICS" | grep -E 'cache_config_info' > "$dir/kv-pool-metrics.txt" || true
  cat "$dir/kv-pool-initlog.txt" | tee -a "$LOG"
}

# 17 x 23 = 391. The TP=3 padding trap serves FLUENT NONSENSE rather than
# erroring, so a correctness gate is not optional on this cluster.
assert_correct() {
  local arm="$1" out
  out=$(curl -sf -m 120 -X POST "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}")
  echo "$out" | tee -a "$LOG"
  echo "$out" | grep -q 391 || die "[$arm] correctness gate FAILED (expected 391)"
  log "[$arm] correctness 391 OK"
}

# llama-benchy falls back to gpt2 SILENTLY when the tokenizer will not load.
# That fallback measured an 11.6% token-count error against this corpus, which
# would propagate into every t/s figure. Prove the real one loads before we
# trust a single number from this tool.
assert_tokenizer() {
  local dir="$1"
  "$BENCHY_DIR/.venv/bin/python" - "$TOKENIZER" > "$dir/tokenizer-check.txt" 2>&1 <<'PYEOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
n = len(tok)
probe = tok.encode("The quick brown fox jumps over the lazy dog.")
print(f"class={tok.__class__.__name__}")
print(f"vocab_size={n}")
print(f"probe_len={len(probe)}")
# gpt2 is 50257. The DeepSeek tokenizer for this checkpoint is ~128k.
assert n > 100000, f"REFUSING: vocab {n} looks like the gpt2 fallback, not DeepSeek"
print("TOKENIZER_OK")
PYEOF
  cat "$dir/tokenizer-check.txt" | tee -a "$LOG"
  grep -q TOKENIZER_OK "$dir/tokenizer-check.txt" \
    || die "tokenizer verification FAILED -- refusing to measure with a gpt2 fallback"
}

# --- the two llama-benchy invocations -------------------------------------
# NOTE --save-result (NOT --output; that flag does not exist) and --no-cache on
# the concurrency sweep so prefix caching cannot serve the prefill.
benchy_depth() {
  local arm="$1"
  local dir="$RUN_DIR/$arm"
  log "=== [$arm] llama-benchy depth sweep: depths=$DEPTHS runs=$RUNS ==="
  ( cd "$BENCHY_DIR" && "$BENCHY" \
      --base-url "$API/v1" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --pp 2048 --tg 256 --exact-tg \
      --depth $DEPTHS \
      --runs $RUNS --warmup-runs $WARMUPS \
      --latency-mode generation \
      --format json --save-result "$dir/${arm}-depth.json" \
  ) 2>&1 | tee "$dir/${arm}-depth.log" | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && log "WARN: [$arm] depth sweep exited non-zero"
  return 0
}

benchy_conc() {
  local arm="$1"
  local dir="$RUN_DIR/$arm"
  log "=== [$arm] llama-benchy concurrency sweep: cc=$CONCS runs=$RUNS ==="
  ( cd "$BENCHY_DIR" && "$BENCHY" \
      --base-url "$API/v1" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --pp 8192 --tg 256 --exact-tg \
      --depth 0 --concurrency $CONCS \
      --runs $RUNS --warmup-runs $WARMUPS \
      --no-cache \
      --latency-mode generation \
      --format json --save-result "$dir/${arm}-concurrency.json" \
  ) 2>&1 | tee "$dir/${arm}-concurrency.log" | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && log "WARN: [$arm] concurrency sweep exited non-zero"
  return 0
}

measure_arm() {
  local arm="$1"
  local dir="$RUN_DIR/$arm"
  cooldown "$COOL_TARGET_C" "$COOL_MAX_S"
  telem_start "$dir"
  benchy_depth "$arm"
  benchy_conc  "$arm"
  telem_stop
  clock_summary "$dir"
  log "=== [$arm] arm complete ==="
}

# ==========================================================================
log "### llama-benchy 2v3 -> $RUN_DIR"
log "plan: docs/PLAN-LLAMA-BENCHY-2V3.md ; arm order TP=3 then TP=2 (see header)"

# --- Step 0: preconditions -------------------------------------------------
log "=== Step 0: preconditions ==="
[[ -x "$BENCHY" ]] || die "llama-benchy not installed at $BENCHY"
[[ -d "$TOKENIZER" ]] || die "tokenizer missing at $TOKENIZER"
assert_tokenizer "$RUN_DIR"

# Record the software stack. Driver is a first-class variable on GB10 -- a ~3.5x
# regression is documented between two 580.x releases -- so a number published
# without it is not externally comparable.
{
  echo "stamp=$STAMP"
  echo "benchy_commit=$(cd "$BENCHY_DIR" && git rev-parse --short HEAD 2>/dev/null)"
  echo "benchy_version=$("$BENCHY" --version 2>&1 | head -1)"
  for h in sparkmain spark1 spark2; do
    echo "--- $h ---"
    ssh -n "$h" "nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; uname -r" 2>/dev/null
  done
  echo "--- image ---"
  sudo docker inspect dspark-vllm-gx10-vllm-dspark-1 --format '{{.Config.Image}}' 2>/dev/null
} > "$RUN_DIR/software-stack.txt" 2>&1
cat "$RUN_DIR/software-stack.txt" | tee -a "$LOG"

log "=== Step 0b: stop open-webui (Requirement 5) ==="
if sudo docker ps --format '{{.Names}}' | grep -q '^open-webui$'; then
  sudo docker stop open-webui >/dev/null 2>&1 && WEBUI_STOPPED=1 && log "open-webui stopped."
else
  log "open-webui not running."
fi

# --- Step 1: TP=3 arm (cluster is already up) ------------------------------
log "=== Step 1: TP=3 arm ==="
curl -sf -m 10 -o /dev/null "$API/health" || die "TP=3 endpoint not healthy at start"
# --nccl=skip is mandatory while the engine holds the GPUs: the NCCL bandwidth
# check needs them free. SSH liveness, full-mesh reachability and per-pair
# latency still run, which is what catches a node silently routing off-fabric.
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=skip \
  --json "$RUN_DIR/tp3/fabric-gate.json" 2>&1 | tee "$RUN_DIR/tp3/fabric-gate.log" | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "TP=3 fabric gate FAILED -- refusing to benchmark a degraded fabric"
assert_engine tp3 3
assert_correct tp3
measure_arm tp3

# --- Step 2: switch to TP=2 ------------------------------------------------
log "=== Step 2: bring the cluster down and switch to TP=2 ==="
sudo systemctl stop dsv4.service 2>&1 | tee -a "$LOG"
for i in $(seq 1 60); do
  curl -sf -m 3 -o /dev/null "$API/health" 2>/dev/null || break
  sleep 5
done
for h in sparkmain spark1 spark2; do
  ssh -n "$h" 'sudo docker ps --format "{{.Names}}" | grep -q vllm-dspark' 2>/dev/null \
    && log "WARN: $h still shows a vllm container after stop"
done
log "cluster down."

log "--- rewriting head.env/worker.env to the matched profile ---"
bash "$HERE/match_env_for_benchy.sh" apply 2>&1 | tee "$RUN_DIR/tp2/env-match.log" | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "env match failed -- refusing to run an unmatched TP=2 arm"

log "--- TP=2 bringup (cold start ~30 min) ---"
bash "$HERE/cluster_tp2.sh" up 2>&1 | tee "$RUN_DIR/tp2/cluster-up.log" | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  log "cluster_tp2.sh up FAILED -- restoring 3-node production."
  bash "$HERE/cluster_tp2.sh" down 2>&1 | tee -a "$LOG"
  bash "$HERE/match_env_for_benchy.sh" restore 2>&1 | tee -a "$LOG"
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  die "TP=2 bringup failed"
fi

bash "$HERE/fabric_gate.sh" "$ROOT/configs/2spark-live.env" --nccl=skip \
  --json "$RUN_DIR/tp2/fabric-gate.json" 2>&1 | tee "$RUN_DIR/tp2/fabric-gate.log" | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "TP=2 fabric gate FAILED -- refusing to benchmark a degraded fabric"
assert_engine tp2 2
assert_correct tp2
measure_arm tp2

# --- Step 3: restore production -------------------------------------------
log "=== Step 3: tear down TP=2, restore env, restart 3-node ==="
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$RUN_DIR/tp2/cluster-down.log" | tee -a "$LOG"
bash "$HERE/match_env_for_benchy.sh" restore 2>&1 | tee -a "$LOG"
sudo systemctl start dsv4.service 2>&1 | tee "$RUN_DIR/dsv4-restart.log" | tee -a "$LOG"

log "=== Step 4: wait for 3-node health ==="
ok=0
for i in $(seq 1 270); do
  curl -sf -m 5 -o /dev/null "$API/health" 2>/dev/null && { ok=1; break; }
  sleep 10
done
if [[ $ok -eq 1 ]]; then
  log "3-node cluster healthy; production restored."
  assert_correct restored || true
else
  log "ERROR: 3-node NOT healthy after 45 min -- cluster needs attention."
fi

restore_webui
log "### RUN COMPLETE -> $RUN_DIR"
ls -la "$RUN_DIR/tp3" "$RUN_DIR/tp2" | tee -a "$LOG"
