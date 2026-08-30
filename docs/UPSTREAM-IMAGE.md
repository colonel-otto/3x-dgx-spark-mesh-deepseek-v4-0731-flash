# Building the runtime from upstream vLLM

`docker/Dockerfile.upstream` builds the 3-node runtime **directly from stock
`vllm/vllm-openai:v0.25.1`** instead of the prebuilt
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` image that `docker/Dockerfile.runtime`
extends. The result is functionally interchangeable — same overlay, same
patches — but the whole lineage is now pinned and rebuildable by us.

## Why

Inspecting Anemll's repo showed their image is not a vLLM fork; it is a thin,
fully public recipe:

```text
vllm/vllm-openai:v0.25.1            (stock upstream image)
  + pip: cuda-python / cutlass-dsl / tvm-ffi   (pinned versions)
  + pip: lukealonso/b12x @ commit              (public repo)
  + pip: flashinfer @ commit                   (public repo)
  + COPY overlay/vllm/                         (13 files, MIT)
```

Owning that recipe means we can audit every layer, apply fixes without waiting
on a prebuilt tag, and see upstream changes coming instead of discovering them
inside someone else's image bump.

## Pins

Everything is pinned in [`upstream.lock`](../upstream.lock) (mirroring
Anemll's own `upstream.lock` at our pinned overlay commit, plus
`ANEMLL_OVERLAY_COMMIT` for the overlay itself). The Dockerfile `ARG` defaults
duplicate the lock so a bare `docker build` is reproducible; change them
together.

## Build

On a Spark (the base image is linux/arm64; this does not build on an x86 PC):

```bash
docker build -f docker/Dockerfile.upstream -t dsv4-3spark-upstream:v0.25.1 .
```

The build fails hard if any TP=3 patch anchor MISSes
(`apply_tp3_patch.py --check` runs as the last layer), so a base bump that
invalidates an anchor cannot produce a silently unpatched image.

## Monitoring upstream

```bash
scripts/check-upstream-drift.sh          # vs vLLM main
scripts/check-upstream-drift.sh v0.26.0  # vs a candidate upgrade tag
```

Read-only (GitHub API via `gh`). Reports newer vLLM releases, drift on every
stock file our patches or the overlay replace, and movement in Anemll's
`overlay/` past our pin. Exit code 1 on any drift, so it can run in CI or a
cron.

## Upgrading the vLLM base

A version bump is a project, not a knob (the overlay files are whole-file
replacements against v0.25.1 internals, and the patch anchors target the same
tree):

1. `scripts/check-upstream-drift.sh <new-tag>` — see exactly which watched
   files moved.
2. Bump `upstream.lock` + Dockerfile ARGs; build. `apply_tp3_patch.py --check`
   will list any MISSed anchors.
3. Re-port MISSed anchors and any drifted overlay files (or take a newer
   Anemll overlay commit if they have already ported).
4. Re-run the TP=3 validation suite (14/14 in
   [`TP3-TUNING.md`](TP3-TUNING.md) lineage) before any node runs it.
