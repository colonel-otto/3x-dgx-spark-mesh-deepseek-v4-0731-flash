#!/usr/bin/env bash
# Re-measure the TWO cells that did not resolve in the 2026-08-30 llama-benchy
# 2v3 run, at higher n.
#
# WHICH CELLS, AND WHY THEY NEED DIFFERENT COMMANDS
#
#   8K decode   came from the DEPTH sweep:       --pp 2048 --depth 8192
#   cc=1 decode came from the CONCURRENCY sweep: --pp 8192 --depth 0 --no-cache
#
# These are DIFFERENT measurements that both happen to run at concurrency 1,
# which is exactly why they disagreed (+4.0% vs +6.3%) and why one command
# cannot cover both.
#
# WHAT THIS RUN CAN AND CANNOT SETTLE -- read before interpreting the output.
#
# The original n=10 was NOT underpowered for the effect under test. At the
# measured pooled CVs (10.8% and 9.4%), n=8 and n=6 respectively would resolve
# a +15.4% effect, and we ran n=10. The cells did not resolve because the effect
# at these two shapes is genuinely SMALLER, not because the measurement was too
# noisy to see a large one.
#
# The 95% CIs from that run make this concrete:
#   8K decode:   +4.0%  CI [-6.5%, +14.5%]  -- excludes +15.4%, includes 0
#   cc=1 decode: +6.3%  CI [-2.6%, +15.2%]  -- excludes +15.4%, includes 0
#
# So both cells already RULE OUT the +15.4% depth-sweep effect, while failing to
# rule out no effect at all. That is a finding, not a gap.
#
# Resolving the OBSERVED differences would need n=115 (8K) and n=36 (cc=1) per
# arm. n=115 is out of reach for a confirmatory re-run. This script uses n=30,
# which resolves an effect of roughly 8-9% at these CVs. Expected outcomes:
#
#   - If the true effect is ~0-5%: cells stay inconclusive, but the CI tightens
#     enough to state "smaller than 9%" rather than "smaller than 15.4%".
#   - If the true effect is ~10%+: cells resolve and we get a real number.
#
# EITHER OUTCOME IS PUBLISHABLE. Do not re-run again hoping for significance --
# that is p-hacking. n=30 is the pre-committed stopping point.
#
# Node count is the only variable, asserted against the live engine per arm, as
# in run_llama_benchy_2v3.sh. Same nine settings, same image, same tokenizer.
#
# Run ON sparkmain under nohup:
#   nohup bash ~/bench-repo/scripts/rerun_inconclusive_cells.sh \
#     > ~/bench-repo/results/rerun-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${STAMP}-rerun-inconclusive"
mkdir -p "$RUN_DIR/tp3" "$RUN_DIR/tp2"
LOG="$RUN_DIR/orchestration.log"

BENCHY_DIR="$HOME/llama-benchy"
BENCHY="$BENCHY_DIR/.venv/bin/llama-benchy"
TOKENIZER="$HOME/dsv4-tokenizer"
API="http://127.0.0.1:8100"
MODEL="deepseek-v4-flash-0731"

RUNS=30          # pre-committed. Do NOT raise this after seeing the result.
WARMUPS=3
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
  local dir="$1"
  local h
  for h in sparkmain spark1 spark2; do
    ssh -n "$h" "nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu \
      --format=csv,noheader -l 5" > "$dir/clocks-$h.csv" 2>/dev/null &
    TELEM_PIDS="$TELEM_PIDS $!"
  done
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

assert_engine() {
  local arm="$1"
  local tp="$2"
  local dir="$RUN_DIR/$arm"
  local live
  live=$(ps -eo args | grep -m1 '[v]llm.*tensor-parallel-size' || true)
  [[ -n "$live" ]] || die "[$arm] no vllm process found"
  echo "$live" > "$dir/engine-config.txt"
  echo "$live" | grep -q -- "--tensor-parallel-size $tp"          || die "[$arm] not TP=$tp"
  echo "$live" | grep -q -- '--max-num-seqs 32'                   || die "[$arm] max-num-seqs != 32"
  echo "$live" | grep -q -- '--gpu-memory-utilization 0.835'      || die "[$arm] gpu-mem != 0.835"
  echo "$live" | grep -q -- '--max-num-batched-tokens 8192'       || die "[$arm] batched-tokens != 8192"
  echo "$live" | grep -q -- '--max-model-len 1048576'             || die "[$arm] model-len != 1048576"
  echo "$live" | grep -q -- '--kv-cache-dtype nvfp4_ds_mla'       || die "[$arm] kv-cache-dtype wrong"
  echo "$live" | grep -q -- '--long-prefill-token-threshold 1024' || die "[$arm] long-prefill != 1024"
  echo "$live" | grep -q '"num_speculative_tokens":2'             || die "[$arm] MTP != 2"
  echo "$live" | grep -q 'flashinfer_b12x'                        || die "[$arm] moe-backend wrong"
  log "[$arm] ENGINE MATCH CONFIRMED (TP=$tp)"
}

assert_correct() {
  local arm="$1" out
  out=$(curl -sf -m 120 -X POST "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Reply with only the number.\"}],\"max_tokens\":16,\"temperature\":0}")
  echo "$out" | grep -q 391 || die "[$arm] correctness gate FAILED (expected 391)"
  log "[$arm] correctness 391 OK"
}

assert_tokenizer() {
  local dir="$1"
  "$BENCHY_DIR/.venv/bin/python" - "$TOKENIZER" > "$dir/tokenizer-check.txt" 2>&1 <<'PYEOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
n = len(tok)
print(f"vocab_size={n}")
assert n > 100000, f"REFUSING: vocab {n} looks like the gpt2 fallback"
print("TOKENIZER_OK")
PYEOF
  grep -q TOKENIZER_OK "$dir/tokenizer-check.txt" || die "tokenizer verification FAILED"
  log "tokenizer verified (real DeepSeek, not gpt2 fallback)"
}

# Cell A: 8K decode, reproducing the DEPTH sweep's shape exactly.
cell_depth8k() {
  local arm="$1"
  local dir="$RUN_DIR/$arm"
  log "=== [$arm] cell A: 8K decode (depth-sweep shape) n=$RUNS ==="
  ( cd "$BENCHY_DIR" && "$BENCHY" \
      --base-url "$API/v1" --model "$MODEL" --tokenizer "$TOKENIZER" \
      --pp 2048 --tg 256 --exact-tg \
      --depth 8192 \
      --runs $RUNS --warmup-runs $WARMUPS \
      --latency-mode generation \
      --format json --save-result "$dir/${arm}-depth8k.json" \
  ) 2>&1 | tee "$dir/${arm}-depth8k.log" | tee -a "$LOG"
  return 0
}

# Cell B: cc=1 decode, reproducing the CONCURRENCY sweep's shape exactly
# (pp=8192, depth=0, --no-cache). NOT the same measurement as cell A.
cell_cc1() {
  local arm="$1"
  local dir="$RUN_DIR/$arm"
  log "=== [$arm] cell B: cc=1 decode (concurrency-sweep shape) n=$RUNS ==="
  ( cd "$BENCHY_DIR" && "$BENCHY" \
      --base-url "$API/v1" --model "$MODEL" --tokenizer "$TOKENIZER" \
      --pp 8192 --tg 256 --exact-tg \
      --depth 0 --concurrency 1 \
      --no-cache \
      --runs $RUNS --warmup-runs $WARMUPS \
      --latency-mode generation \
      --format json --save-result "$dir/${arm}-cc1.json" \
  ) 2>&1 | tee "$dir/${arm}-cc1.log" | tee -a "$LOG"
  return 0
}

measure_arm() {
  local arm="$1"
  cooldown "$COOL_TARGET_C" "$COOL_MAX_S"
  telem_start "$RUN_DIR/$arm"
  cell_depth8k "$arm"
  cell_cc1 "$arm"
  telem_stop
  log "=== [$arm] arm complete ==="
}

# ==========================================================================
log "### rerun of inconclusive cells -> $RUN_DIR (n=$RUNS, pre-committed)"

[[ -x "$BENCHY" ]] || die "llama-benchy not installed at $BENCHY"
assert_tokenizer "$RUN_DIR"

if sudo docker ps --format '{{.Names}}' | grep -q '^open-webui$'; then
  sudo docker stop open-webui >/dev/null 2>&1 && WEBUI_STOPPED=1 && log "open-webui stopped."
fi

# --- TP=3 arm (cluster is already up) -------------------------------------
log "=== TP=3 arm ==="
curl -sf -m 10 -o /dev/null "$API/health" || die "TP=3 endpoint not healthy"
bash "$HERE/fabric_gate.sh" "$ROOT/configs/3spark-live.env" --nccl=skip \
  --json "$RUN_DIR/tp3/fabric-gate.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "TP=3 fabric gate FAILED"
assert_engine tp3 3
assert_correct tp3
measure_arm tp3

# --- switch to TP=2 -------------------------------------------------------
log "=== switch to TP=2 ==="
sudo systemctl stop dsv4.service 2>&1 | tee -a "$LOG"
for i in $(seq 1 60); do
  curl -sf -m 3 -o /dev/null "$API/health" 2>/dev/null || break
  sleep 5
done
log "cluster down."

bash "$HERE/match_env_for_benchy.sh" apply 2>&1 | tee "$RUN_DIR/tp2/env-match.log" | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "env match failed -- refusing an unmatched TP=2 arm"

bash "$HERE/cluster_tp2.sh" up 2>&1 | tee "$RUN_DIR/tp2/cluster-up.log" | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  log "TP=2 bringup FAILED -- restoring production."
  bash "$HERE/cluster_tp2.sh" down 2>&1 | tee -a "$LOG"
  bash "$HERE/match_env_for_benchy.sh" restore 2>&1 | tee -a "$LOG"
  sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"
  die "TP=2 bringup failed"
fi

bash "$HERE/fabric_gate.sh" "$ROOT/configs/2spark-live.env" --nccl=skip \
  --json "$RUN_DIR/tp2/fabric-gate.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} -ne 0 ]] && die "TP=2 fabric gate FAILED"
assert_engine tp2 2
assert_correct tp2
measure_arm tp2

# --- restore production ---------------------------------------------------
log "=== restore production ==="
bash "$HERE/cluster_tp2.sh" down 2>&1 | tee "$RUN_DIR/tp2/cluster-down.log" | tee -a "$LOG"
bash "$HERE/match_env_for_benchy.sh" restore 2>&1 | tee -a "$LOG"
sudo systemctl start dsv4.service 2>&1 | tee -a "$LOG"

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
log "### RERUN COMPLETE -> $RUN_DIR"
