"""Stream N-Triples (optionally bzip2-compressed) and filter by subject/object URI sets."""

from __future__ import annotations

import bz2
import gzip
import re
from collections.abc import Iterator
from pathlib import Path

TRIPLE_LINE = re.compile(r"^\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*(?:#.*)?$")


def open_text_maybe_compressed(path: Path):
    s = str(path)
    if s.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if s.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def iter_nt_iri_triples(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (s, p, o) for lines that match strict IRI–IRI–IRI N-Triples."""
    with open_text_maybe_compressed(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                continue
            yield m.group(1), m.group(2), m.group(3)


def filter_triples_by_endpoints(
    input_nt: Path,
    output_nt: Path,
    *,
    uris: set[str],
    include_if_subject: bool,
    include_if_object: bool,
) -> int:
    """Write triples matching subject/object inclusion flags. Returns count written."""
    output_nt.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_nt.open("w", encoding="utf-8") as out:
        for s, p, o in iter_nt_iri_triples(input_nt):
            if include_if_subject and s in uris:
                out.write(f"<{s}> <{p}> <{o}> .\n")
                n += 1
            elif include_if_object and o in uris:
                out.write(f"<{s}> <{p}> <{o}> .\n")
                n += 1
    return n


def _endpoints(triples: set[tuple[str, str, str]]) -> set[str]:
    nodes: set[str] = set()
    for s, _p, o in triples:
        nodes.add(s)
        nodes.add(o)
    return nodes


def extract_subgraph_k_hops(
    input_nt: Path,
    output_nt: Path,
    *,
    seed_uris: set[str],
    hops: int,
) -> int:
    """
    Undirected k-hop closure from ``seed_uris`` over IRI–IRI–IRI triples.

    Each hop adds every triple that touches any node already in the active set,
    then extends the active set with all endpoints of those triples.
    ``hops=1`` keeps triples with at least one endpoint in ``seed_uris``.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    active: set[str] = set(seed_uris)
    kept: set[tuple[str, str, str]] = set()
    for _ in range(hops):
        added_this_round = False
        with open_text_maybe_compressed(input_nt) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                m = TRIPLE_LINE.match(line)
                if not m:
                    continue
                s, p, o = m.group(1), m.group(2), m.group(3)
                if s in active or o in active:
                    t = (s, p, o)
                    if t not in kept:
                        kept.add(t)
                        added_this_round = True
        active = _endpoints(kept)
        if not added_this_round:
            break
    output_nt.parent.mkdir(parents=True, exist_ok=True)
    with output_nt.open("w", encoding="utf-8") as out:
        for s, p, o in sorted(kept):
            out.write(f"<{s}> <{p}> <{o}> .\n")
    return len(kept)
