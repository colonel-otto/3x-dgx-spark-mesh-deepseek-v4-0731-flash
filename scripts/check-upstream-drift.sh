#!/usr/bin/env bash
# Report upstream drift against the pins in upstream.lock.
#
# Answers three questions without touching any deployment:
#   1. Are there vLLM releases newer than our pinned tag?
#   2. Have the upstream vLLM files our patches/overlay replace changed
#      since the pinned commit? (These are the files an upgrade must re-port.)
#   3. Has Anemll's overlay/ moved past our pinned overlay commit?
#
# Requires: gh (authenticated). Read-only; safe to run anywhere.
#
# Usage: scripts/check-upstream-drift.sh [target-ref]   (default: main)
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source "$here/upstream.lock"
target=${1:-main}

# Stock vLLM files we carry edits for. Two groups:
#  - patched: rewritten in place by patches/apply_tp3_patch.py (anchors must
#    still match after any upgrade -- run it with --check against the new tree)
#  - overlaid: replaced wholesale by Anemll's overlay (new-file overlay entries
#    like models/deepseek_v4/* have no upstream counterpart and are not listed)
watched=(
  "vllm/config/model.py"                                    # patched: head-divisibility gate
  "vllm/model_executor/parameter.py"                        # patched: pad-aware loaders
  "vllm/model_executor/layers/fused_moe/config.py"          # patched: MoE intermediate pad
  "vllm/model_executor/layers/vocab_parallel_embedding.py"  # patched: vocab pad
  "vllm/v1/worker/gpu_worker.py"                            # patched: profiler activities
  "vllm/config/cache.py"                                    # overlaid
  "vllm/config/vllm.py"                                     # overlaid
  "vllm/envs.py"                                            # overlaid
  "vllm/utils/torch_utils.py"                               # overlaid
  "vllm/v1/kv_cache_interface.py"                           # overlaid
)

# True when $2 already contains commit $1 (i.e. no drift past our pin).
contains() {
  local status
  status=$(gh api "repos/$3/compare/$1...$2" --jq '.status' 2>/dev/null) || return 2
  [[ "$status" == "identical" || "$status" == "behind" ]]
}

echo "== vLLM releases (pinned: ${VLLM_TAG}) =="
gh api "repos/vllm-project/vllm/releases?per_page=5" \
  --jq '.[] | "\(.tag_name)  \(.published_at)"' \
  | sed "s|^${VLLM_TAG} |${VLLM_TAG} (PINNED) |"

echo
echo "== watched vLLM files, ${VLLM_COMMIT:0:12} vs ${target} =="
drift=0
for path in "${watched[@]}"; do
  latest=$(gh api "repos/vllm-project/vllm/commits?sha=${target}&path=${path}&per_page=1" \
    --jq '.[0] | "\(.sha) \(.commit.committer.date)"') || { echo "ERR    ${path}"; continue; }
  sha=${latest%% *}
  when=${latest#* }
  if contains "$VLLM_COMMIT" "$sha" "vllm-project/vllm"; then
    echo "ok     ${path}"
  else
    echo "DRIFT  ${path}  (last touched ${sha:0:12} ${when})"
    drift=$((drift + 1))
  fi
done

echo
echo "== Anemll overlay, pinned ${ANEMLL_OVERLAY_COMMIT:0:12} vs their main =="
latest=$(gh api "repos/Anemll/dspark-vllm-gx10/commits?sha=main&path=overlay&per_page=1" \
  --jq '.[0] | "\(.sha) \(.commit.committer.date)"')
sha=${latest%% *}
if contains "$ANEMLL_OVERLAY_COMMIT" "$sha" "Anemll/dspark-vllm-gx10"; then
  echo "ok     overlay/ unchanged since our pin"
else
  echo "DRIFT  overlay/ last touched ${sha:0:12} ${latest#* }"
  drift=$((drift + 1))
fi

echo
if [[ $drift -eq 0 ]]; then
  echo "No drift on any watched path."
else
  echo "${drift} watched path(s) drifted. An upgrade means: bump upstream.lock,"
  echo "re-run apply_tp3_patch.py --check against the new tree, re-port any"
  echo "MISSed anchors and changed overlay files, then re-run the validation suite."
fi
exit $((drift > 0))
