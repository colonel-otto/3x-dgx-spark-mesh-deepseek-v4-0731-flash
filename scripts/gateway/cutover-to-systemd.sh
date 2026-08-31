#!/usr/bin/env bash
# One-time cutover: replace the bare nohup LiteLLM process with litellm.service.
#
# The unit is already installed and ENABLED, so a reboot alone would also do
# this. Run it to cut over WITHOUT rebooting. Idempotent and safe to re-run.
#
#   bash $HOME/litellm/cutover-to-systemd.sh
set -euo pipefail

say() { printf "%s\n" "$*"; }

say "== before =="
systemctl is-enabled litellm.service || true
pid="$(ss -ltnp 2>/dev/null | grep -oP ":4000\s.*pid=\K[0-9]+" | head -1 || true)"

if [ -n "${pid:-}" ] && ! systemctl is-active --quiet litellm.service; then
  say "stopping bare LiteLLM process (pid $pid)"
  kill "$pid"
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then say "did not exit, SIGKILL"; kill -9 "$pid"; sleep 2; fi
else
  say "no bare process to stop"
fi

say "starting litellm.service"
sudo systemctl start litellm.service

say "== after =="
systemctl is-active litellm.service
curl -fsS -m 10 -o /dev/null -w "  /health/liveliness -> %{http_code}\n" \
  http://127.0.0.1:4000/health/liveliness
curl -fsS -m 10 http://127.0.0.1:4000/v1/models \
  | python3 -c "import json,sys;[print(\"  model:\",m[\"id\"]) for m in json.load(sys.stdin).get(\"data\",[])]"
say "done. Apply future config edits with: sudo systemctl restart litellm"
