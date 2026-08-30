#!/usr/bin/env bash
# verify-gateway.sh — step 3's verify half: prove the LAN gateway route is live
# end-to-end, not just that :8100 answers locally.
#
# Chain: client -> bigdog LiteLLM :4000 -> sparkmain :8100 -> engine
# Also checks the manifest service :8771, which clients use for discovery.
set -uo pipefail

# The LAN gateway host (bigdog) running LiteLLM :4000 and the manifest :8771.
# Placeholder default -- set the real address in ~/.eugr-nodes (gitignored):
#   export GW_HOST=...
[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
GW="${GW_HOST:-192.168.10.20}"
LOCAL_EP="${LOCAL_EP:-http://127.0.0.1:8100}"
LEGACY_NAME=deepseek-v4-flash-dspark-abliterated
AB_NAME=deepseek-v4-flash-eugr-ab
fail=0
ok()   { echo "  PASS  $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }

echo "== 1. local engine on :8100 =="
models=$(curl -fsS -m 10 "$LOCAL_EP/v1/models" 2>/dev/null) || { bad "no answer from $LOCAL_EP"; models=""; }
echo "$models" | grep -q "$LEGACY_NAME" && ok "serves $LEGACY_NAME (the name the gateway routes to)" \
                                        || bad "missing $LEGACY_NAME -- gateway route will 404"
echo "$models" | grep -q "$AB_NAME"     && ok "serves $AB_NAME (the A/B row identity)" \
                                        || bad "missing $AB_NAME"

echo "== 2. through the gateway on $GW:4000 =="
gw=$(curl -fsS -m 10 "http://$GW:4000/v1/models" 2>/dev/null) || bad "gateway :4000 unreachable"
echo "$gw" | grep -q "$LEGACY_NAME" && ok "gateway lists $LEGACY_NAME" || bad "gateway does not list $LEGACY_NAME"

echo "== 3. end-to-end completion through the gateway =="
# The real test: a token round-trip, not just a model listing. A stale LiteLLM
# config lists a model it can no longer reach (config edits need a restart).
body='{"model":"'"$LEGACY_NAME"'","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
reply=$(curl -fsS -m 120 -H 'Content-Type: application/json' -d "$body" \
        "http://$GW:4000/v1/chat/completions" 2>/dev/null \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null)
if [ -n "$reply" ]; then ok "gateway completion returned: ${reply:0:40}"
else bad "no completion through the gateway (route lists but does not serve)"; fi

echo "== 4. manifest service on $GW:8771 =="
# models-manifest-serve does NOT expose /v1/models -- it publishes named JSON
# documents and serves any other path as a static file (so a wrong path returns
# an HTML directory listing and looks like a broken service when it is fine).
# The gateway-backed document is opencode.gateway.json, which the exporter
# resolves LIVE from the gateway's own /v1/models -- so once section 2 passes,
# this should follow within the 3s cache TTL with no manual edit.
man=$(curl -fsS -m 15 "http://$GW:8771/opencode.gateway.json" 2>/dev/null) || bad "manifest :8771 unreachable"
echo "$man" | grep -q "$LEGACY_NAME" && ok "manifest advertises $LEGACY_NAME (auto-discovered)"                                      || bad "manifest does not advertise $LEGACY_NAME"

echo
[ "$fail" -eq 0 ] && echo "GATEWAY ROUTE VERIFIED" || echo "GATEWAY ROUTE INCOMPLETE (see FAILs)"
exit $fail
