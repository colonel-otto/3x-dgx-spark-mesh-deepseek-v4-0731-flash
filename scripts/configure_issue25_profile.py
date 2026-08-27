#!/usr/bin/env python3
"""Apply or restore the temporary issue #25 runtime profile.

This helper is intentionally strict: it edits only known anchors, creates one
immutable backup per file, and always restores that backup before applying a
profile.  It is used on each Spark checkout so all ranks receive identical
Compose and environment inputs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MARK = "# issue25-profile-b"


def backup(path: Path) -> Path:
    saved = path.with_name(path.name + ".issue25-original")
    if not saved.exists():
        shutil.copy2(path, saved)
    return saved


def restore(path: Path) -> None:
    saved = backup(path)
    shutil.copy2(saved, path)


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(anchor, replacement, 1)


def apply_profile_b(compose_path: Path, env_path: Path) -> None:
    compose = compose_path.read_text(encoding="utf-8")
    env = env_path.read_text(encoding="utf-8")
    if MARK in compose or MARK in env:
        raise RuntimeError("profile marker already present after restore")

    compose = replace_once(
        compose,
        '      MTP_NUM_TOKENS: "${MTP_NUM_TOKENS:-3}"\n',
        '      MTP_NUM_TOKENS: "${MTP_NUM_TOKENS:-3}"\n'
        f"      {MARK}\n"
        '      VLLM_PREFIX_CACHE_RETENTION_INTERVAL: '
        '"${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-}"\n'
        '      DSPARK_MAX_INFLIGHT_PREFILLS: '
        '"${DSPARK_MAX_INFLIGHT_PREFILLS:-}"\n',
        "compose environment",
    )
    compose = replace_once(
        compose,
        "        if [ -f /opt/dsv4-tp3/apply_tp3_patch.py ]; then "
        "python3 /opt/dsv4-tp3/apply_tp3_patch.py || exit 1; fi;\n",
        "        if [ -f /opt/dsv4-tp3/apply_tp3_patch.py ]; then "
        "python3 /opt/dsv4-tp3/apply_tp3_patch.py || exit 1; fi;\n"
        "        if [ -f /opt/dsv4-tp3/hotfix-dsv4-issue26-hybrid-swa-min.py ]; "
        "then python3 /opt/dsv4-tp3/hotfix-dsv4-issue26-hybrid-swa-min.py || exit 1; fi;\n"
        "        if [ -f /opt/dsv4-tp3/hotfix-dsv4-issue27-partial-prefill-concurrency.py ]; "
        "then python3 /opt/dsv4-tp3/hotfix-dsv4-issue27-partial-prefill-concurrency.py || exit 1; fi;\n",
        "compose patch entrypoint",
    )
    compose = replace_once(
        compose,
        "        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192}\n",
        "        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192}\n"
        "        --long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD:-0}\n",
        "long-prefill CLI",
    )

    env = replace_once(
        env,
        "GPU_MEMORY_UTILIZATION=0.80\n",
        "GPU_MEMORY_UTILIZATION=0.835\n",
        "GPU memory utilization",
    )
    if not env.endswith("\n"):
        env += "\n"
    env += (
        f"\n{MARK}\n"
        "LONG_PREFILL_TOKEN_THRESHOLD=1024\n"
        "DSPARK_MAX_INFLIGHT_PREFILLS=2\n"
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096\n"
    )

    compose_path.write_text(compose, encoding="utf-8", newline="\n")
    env_path.write_text(env, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("a", "b"))
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    compose = args.repo / "docker-compose.yml"
    env = args.repo / "config" / "tp3.env"
    restore(compose)
    restore(env)
    if args.profile == "b":
        apply_profile_b(compose, env)
    print(f"issue25 profile {args.profile} configured in {args.repo}")


if __name__ == "__main__":
    main()
