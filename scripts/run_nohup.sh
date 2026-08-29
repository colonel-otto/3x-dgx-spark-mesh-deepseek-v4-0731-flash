#!/usr/bin/env bash
# =============================================================================
# scripts/run_nohup.sh -- Detached Background Execution Runner for 3spark-dsv4
#
# Runs long-running benchmarks, fabric gates, or verification scripts inside
# nohup with output redirection, PID tracking, start/end timestamps, and exit
# code recording so unstable SSH connections will never abort execution.
#
# Usage:
#   bash scripts/run_nohup.sh <command> [args...]
#
# Example:
#   bash scripts/run_nohup.sh python3 scripts/benchmark_mtp_concurrency.py --mtp-k 2
# =============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
CMD_NAME=$(basename "$1" | tr -cs 'a-zA-Z0-9._-' '_')
LOG_DIR="$ROOT/results/nohup-runs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/${TIMESTAMP}_${CMD_NAME}.log"
PID_FILE="$LOG_DIR/${TIMESTAMP}_${CMD_NAME}.pid"
EXIT_FILE="$LOG_DIR/${TIMESTAMP}_${CMD_NAME}.exit"

echo "=== Launching detached background execution ==="
echo "Command   : $*"
echo "Log File  : $LOG_FILE"
echo "PID File  : $PID_FILE"
echo "Exit File : $EXIT_FILE"
echo "==============================================="

nohup bash -c '
  CMD="$*"
  LOG_FILE="'"$LOG_FILE"'"
  EXIT_FILE="'"$EXIT_FILE"'"
  
  echo "=== RUN STARTED: $(date -u --iso-8601=seconds) ===" > "$LOG_FILE"
  echo "Command: $CMD" >> "$LOG_FILE"
  echo "PID: $$" >> "$LOG_FILE"
  echo "================================================" >> "$LOG_FILE"
  
  set +e
  eval "$CMD" >> "$LOG_FILE" 2>&1
  EXIT_CODE=$?
  set -e
  
  echo "" >> "$LOG_FILE"
  echo "================================================" >> "$LOG_FILE"
  echo "=== RUN FINISHED: $(date -u --iso-8601=seconds) ===" >> "$LOG_FILE"
  echo "Exit Code: $EXIT_CODE" >> "$LOG_FILE"
  echo "================================================" >> "$LOG_FILE"
  
  echo "$EXIT_CODE" > "$EXIT_FILE"
' _ "$@" > /dev/null 2>&1 &

RUN_PID=$!
echo "$RUN_PID" > "$PID_FILE"
echo "Started with PID: $RUN_PID"
echo "To monitor live logs:"
echo "  tail -f $LOG_FILE"
echo "To check if running:"
echo "  kill -0 $RUN_PID 2>/dev/null && echo 'Running' || echo 'Finished (Exit: '\$(cat $EXIT_FILE 2>/dev/null || echo 'N/A')')'"