"""Load entity URI sets from positives/negatives/train lists (one URI per line)."""

from __future__ import annotations

from pathlib import Path


def load_entity_uris(*paths: Path) -> set[str]:
    """
    Each file: one resource URI per line (optional tab + label after first column).
    Blank lines and lines starting with # are skipped.
    """
    out: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                uri = line.split("\t", 1)[0].strip()
                if uri:
                    out.add(uri)
    return out
