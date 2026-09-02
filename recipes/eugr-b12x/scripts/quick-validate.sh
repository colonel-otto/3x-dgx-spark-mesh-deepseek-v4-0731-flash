#!/usr/bin/env bash
# Quick post-boot validation of the eugr engine A/B arm (docs/ENGINE-AB-3NODE.md).
#
# Run on sparkmain AFTER the endpoint answers. Fast subset first (~2 min),
# then the full battery (garble sweep + RULER-lite + tool batteries) whose
# expected results are the anemll-engine baseline in
# results/20260827-quality-suite-3node/ (7/7, 8/8, ALL CLEAN, 12/12).
# Suite scripts: ~/xrepo/2spark-suite — verify checksums against
# results/20260827-quality-suite-3node/vendored-SHA256SUMS.txt before trusting
# a run (byte-identical harness or the comparison is void).
set -uo pipefail

# Port 8100, not 8000: eugr.service serves DSv4 on :8100 and :8000 is DEAD for
# this model. The old default made every content check fail with an empty body.
URL=${1:-http://127.0.0.1:8100/v1/chat/completions}
MODEL=${2:-deepseek-v4-flash-eugr-ab}
SUITE=${SUITE:-$HOME/xrepo/2spark-suite}
BASE=${URL%/v1/chat/completions}

pass=0; fail=0
say() { printf '%s\n' "$*"; }
ask() { # ask "prompt" max_tokens -> prints content
  curl -sS --max-time 300 "$URL" -H 'Content-Type: application/json' -d "$(
    python3 - "$1" "$2" "$MODEL" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[3],
                  "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0}))
PY
  )" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null
}
check() { # check name needle haystack
  if grep -qi "$2" <<<"$3"; then say "PASS  $1"; pass=$((pass+1));
  else say "FAIL  $1 -- got: $(head -c 200 <<<"$3")"; fail=$((fail+1)); fi
}

say "== endpoint =="
models=$(curl -sS --max-time 10 "$BASE/v1/models" || true)
check "models endpoint serves $MODEL" "$MODEL" "$models"

say "== engine identity (virtual TP must be active for TP=3) =="
# The virtual_tp warning is printed by the APIServer, which streams to the
# LAUNCHER's stdout (our ~/eugr-ab-launch*.log), not the container log.
vtp=$(grep -h -m1 "virtual TP padding" $(ls -t ~/eugr-ab-launch*.log 2>/dev/null | head -1) 2>/dev/null || docker logs vllm_node 2>&1 | grep -m1 "virtual TP padding" || true)
check "virtual-TP plan activated (heads 64->72)" "output groups 8 -> 9" "$vtp"

say "== acceptance items (docs/patch.md; 400-token budget on reasoning) =="
check "capital lookup"  "Paris"  "$(ask 'What is the capital of France? Answer with just the city name.' 50)"
check "17 x 23"         "391"    "$(ask 'What is 17 multiplied by 23? Show the final number.' 400)"
check "red/blue"        "7"      "$(ask 'Red is 7 and blue is 3. A ball is red. What number is the ball? Reason briefly, then answer.' 400)"
needle=$(python3 -c "print('The sky report follows. ' + 'Filler sentence about weather patterns. '*260 + 'The secret code is FALCON42. ' + 'More filler about clouds. '*40 + 'What is the secret code mentioned above? Answer with just the code.')")
check "needle ~1.5k tok" "FALCON42" "$(ask "$needle" 60)"
deg=$(ask 'Write one short paragraph about the ocean.' 150)
# An EMPTY body is a failure, not a pass: the previous 'not w or ...' spelling
# reported OK when the endpoint returned nothing, so a dead port scored a PASS
# on this line while every other check failed.
uw=$(python3 -c "import sys; w=sys.argv[1].split(); print('EMPTY' if not w else ('OK' if len(set(w))/len(w) > 0.4 else 'DEGENERATE'))" "$deg")
check "no degeneration (unique-word ratio)" "OK" "$uw"

say ""
say "quick subset: $pass pass / $fail fail"
say ""
say "Full battery (run each, compare to results/20260827-quality-suite-3node/):"
say "  python3 $SUITE/tool-battery.py $URL $MODEL          # expect 7/7"
say "  python3 $SUITE/deepctx-tool-battery.py $URL $MODEL  # expect 8/8"
say "  python3 $SUITE/context-garble-sweep.py $URL $MODEL  # expect ALL CLEAN"
say "  python3 $SUITE/ruler-lite.py $URL $MODEL            # expect 12/12"
exit $fail
