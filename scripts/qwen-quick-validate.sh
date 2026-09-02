#!/usr/bin/env bash
# qwen-quick-validate.sh — validation suite for Qwen 3.8 on Spark cluster
set -uo pipefail

URL=${1:-http://127.0.0.1:8100/v1/chat/completions}
BASE=${URL%/v1/chat/completions}
DETECTED_MODEL=$(curl -sS --max-time 10 "$BASE/v1/models" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || echo "qwen3.8-27b-nvfp4")
MODEL=${2:-$DETECTED_MODEL}

pass=0; fail=0
say() { printf '%s\n' "$*"; }

ask() { # ask "prompt" max_tokens
  curl -sS --max-time 300 "$URL" -H 'Content-Type: application/json' -d "$(
    python3 - "$1" "$2" "$MODEL" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[3],
                  "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0}))
PY
  )" | python3 -c 'import json,sys; data=json.load(sys.stdin); msg=data["choices"][0]["message"]; print((msg.get("content") or "") + " " + (msg.get("reasoning") or ""))' 2>/dev/null
}

check() { # check name needle haystack
  if grep -qi "$2" <<<"$3"; then say "PASS  $1"; pass=$((pass+1));
  else say "FAIL  $1 -- got: $(head -c 200 <<<"$3")"; fail=$((fail+1)); fi
}

say "=== Qwen Validation Battery ==="
say "== 1. Models Endpoint =="
models=$(curl -sS --max-time 10 "$BASE/v1/models" || true)
check "models endpoint serves $MODEL" "$MODEL" "$models"

say "== 2. Core Reasoning & Correctness =="
check "capital lookup"  "Paris"  "$(ask 'What is the capital of France? Answer with just the city name.' 50)"
check "17 x 23"         "391"    "$(ask 'What is 17 multiplied by 23? Show the final number.' 400)"
check "red/blue"        "7"      "$(ask 'Red is 7 and blue is 3. A ball is red. What number is the ball? Reason briefly, then answer.' 400)"

say "== 3. Needle In Haystack Retrieval =="
needle=$(python3 -c "print('The sky report follows. ' + 'Filler sentence about weather patterns. '*260 + 'The secret code is FALCON42. ' + 'More filler about clouds. '*40 + 'What is the secret code mentioned above? Answer with just the code.')")
check "needle ~1.5k tok" "FALCON42" "$(ask "$needle" 60)"

say "== 4. Text Quality & Degeneration Check =="
deg=$(ask 'Write one short paragraph about the ocean.' 150)
uw=$(python3 -c "import sys; w=sys.argv[1].split(); print('OK' if not w or len(set(w))/len(w) > 0.4 else 'DEGENERATE')" "$deg")
check "no degeneration (unique-word ratio)" "OK" "$uw"

say ""
say "=== Result: $pass PASS / $fail FAIL ==="
exit "$fail"
