#!/usr/bin/env bash
# Install the eugr boot, sweep, and systemd artifacts on sparkmain.
# This deliberately installs but does not start the service; starting it takes
# over all three GPUs and must happen only after the current nohup engine is
# explicitly torn down.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST="${EUGR_INSTALL_DIR:-$HOME}"

if [ "$(id -un)" != "sparkmain" ] && [ "${ALLOW_NON_SPARKMAIN_INSTALL:-0}" != "1" ]; then
  echo "FATAL: run this on sparkmain as the sparkmain user (or set ALLOW_NON_SPARKMAIN_INSTALL=1)." >&2
  exit 1
fi

mkdir -p "$DEST"
install -m 0755 "$HERE/eugr-boot.sh" "$DEST/eugr-boot.sh"
install -m 0755 "$HERE/eugr-sweep.sh" "$DEST/eugr-sweep.sh"
install -m 0755 "$HERE/bench-miaai.py" "$DEST/bench-miaai.py"
install -m 0755 "$HERE/eugr-service-start" "$DEST/eugr-service-start"
install -m 0755 "$HERE/eugr-service-stop" "$DEST/eugr-service-stop"
install -m 0644 "$(cd "$HERE/.." && pwd)/exclusivity.py" "$DEST/exclusivity.py"

sudo -n install -m 0644 "$HERE/eugr.service" /etc/systemd/system/eugr.service
sudo -n systemctl daemon-reload
sudo -n systemctl enable eugr.service

echo "Installed eugr artifacts under $DEST and enabled eugr.service."
echo "The service was NOT started. Stop the current nohup engine first, then run:"
echo "  sudo systemctl start eugr.service"
