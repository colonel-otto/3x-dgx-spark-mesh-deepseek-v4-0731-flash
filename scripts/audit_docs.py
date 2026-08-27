#!/usr/bin/env python3
"""Report documentation size, discoverability, stale markers, and broken links."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
WORD_RE = re.compile(r"\b[\w-]+\b")
STALE_RE = re.compile(
    r"\b(SUPERSEDED|RETRACTED|VOID|IN[- ]PROGRESS|TODO|TBD|provisional)\b",
    re.IGNORECASE,
)


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts
    )


def local_target(source: Path, raw: str) -> Path | None:
    target = raw.split("#", 1)[0].strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    # GitHub repository routes are intentionally relative so forks keep working.
    if re.match(r"^(?:\.\./)+(?:issues|pull)/\d+$", target):
        return None
    return (source.parent / unquote(target)).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-links", action="store_true", help="exit non-zero for broken local links"
    )
    args = parser.parse_args()

    all_markdown = markdown_files()
    docs = sorted(DOCS.glob("*.md"))
    inbound: Counter[Path] = Counter()
    broken: list[tuple[Path, str]] = []

    for source in all_markdown:
        text = source.read_text(encoding="utf-8", errors="replace")
        for match in LINK_RE.finditer(text):
            target = local_target(source, match.group(1))
            if target is None:
                continue
            if target.exists():
                inbound[target] += 1
            else:
                broken.append((source.relative_to(ROOT), match.group(1)))

    repeated: defaultdict[str, set[str]] = defaultdict(set)
    rows = []
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            normalized = re.sub(r"[`*_>#|\[\]()]", "", line).strip().lower()
            normalized = re.sub(r"\s+", " ", normalized)
            if len(normalized) >= 100:
                repeated[normalized].add(path.name)
        rows.append(
            (
                path.name,
                len(WORD_RE.findall(text)),
                len(text.splitlines()),
                inbound[path.resolve()],
                len(STALE_RE.findall(text)),
            )
        )

    print("# Documentation audit\n")
    print("| Page | Words | Lines | Inbound links | Stale markers |")
    print("|---|---:|---:|---:|---:|")
    for name, words, lines, links, stale in sorted(rows, key=lambda row: -row[1]):
        print(f"| `{name}` | {words} | {lines} | {links} | {stale} |")

    duplicates = [
        (line, names) for line, names in repeated.items() if len(names) > 1
    ]
    print(f"\nPages: {len(docs)}; words: {sum(row[1] for row in rows):,}")
    print(f"Repeated long lines across pages: {len(duplicates)}")
    print(f"Broken local links: {len(broken)}")
    for source, target in broken:
        print(f"- `{source}` -> `{target}`")

    return 1 if args.fail_links and broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
