#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
CONFIG=${1:?usage: capture_fabric.sh CONFIG}
load_config "$CONFIG"
ROOT=$(cd "$HERE/.." && pwd)
RUN_DIR="$ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)-${RUN_LABEL}-fabric"
mkdir -p "$RUN_DIR"

bash "$HERE/preflight.sh" "$CONFIG" | tee "$RUN_DIR/preflight.txt"
bash "$HERE/collect_environment.sh" "$CONFIG" "$RUN_DIR"
bash "$HERE/run_nccl.sh" "$CONFIG" "$RUN_DIR/nccl.txt"

echo "Fabric run captured: $RUN_DIR"
