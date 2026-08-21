#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
CONFIG=${1:?usage: run_experiment.sh CONFIG}
load_config "$CONFIG"
: "${API_BASE:?API_BASE required}"
: "${MODEL:=auto}"

ROOT=$(cd "$HERE/.." && pwd)
RUN_DIR="$ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)-${RUN_LABEL}"
mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"
bash "$HERE/collect_environment.sh" "$CONFIG" "$RUN_DIR"

# Snapshot Ray when available on the local launcher; this is informational.
(ray status || true) > "$RUN_DIR/ray-status.txt" 2>&1

python3 "$HERE/benchmark.py" \
  --api-base "$API_BASE" \
  --model "$MODEL" \
  --label "$RUN_LABEL" \
  --contexts "${CONTEXTS:-2048,8192,32768}" \
  --concurrencies "${CONCURRENCIES:-1,3,6}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --repetitions "${REPETITIONS:-3}" \
  --output "$RUN_DIR/benchmark.jsonl"

echo "$RUN_DIR"
