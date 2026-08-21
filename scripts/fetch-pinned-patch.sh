#!/usr/bin/env bash
set -euo pipefail

revision=496c6a146a383f1b7c3f5991f4f1930091420720
repository=https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark.git
destination=${1:-vendor/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark}

if [[ -e "$destination" ]]; then
  printf 'Refusing to overwrite existing path: %s\n' "$destination" >&2
  exit 1
fi

git clone --filter=blob:none --no-checkout "$repository" "$destination"
git -C "$destination" checkout --detach "$revision"
actual=$(git -C "$destination" rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
  printf 'Revision mismatch: expected %s, got %s\n' "$revision" "$actual" >&2
  exit 1
fi

sha256sum "$destination/patches/tp3/apply_tp3_patch.py"
