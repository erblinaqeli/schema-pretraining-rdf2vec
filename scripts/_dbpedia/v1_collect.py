"""Collect http://dbpedia.org/resource/... IRIs from all text splits under v1/dbpedia."""

from __future__ import annotations

import re
from pathlib import Path

# DBpedia resource path segment: letters, digits, and common URI-safe punctuation in titles.
_RESOURCE_INNER = r"[^\s<>`\"#]+"
RESOURCE_URI_RE = re.compile(rf"http://dbpedia\.org/resource/{_RESOURCE_INNER}")


def iter_txt_under(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def collect_dbpedia_resource_uris(root: Path) -> set[str]:
    """
    Scan every ``*.txt`` under ``root`` (e.g. ``v1/dbpedia``) and return unique
    ``http://dbpedia.org/resource/...`` strings found anywhere on a line.
    """
    out: set[str] = set()
    for path in iter_txt_under(root):
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                out.update(RESOURCE_URI_RE.findall(line))
    return out


def write_sorted_uris(path: Path, uris: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for u in sorted(uris):
            out.write(u + "\n")
