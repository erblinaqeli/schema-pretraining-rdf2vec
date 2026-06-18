"""Stream RDF (Turtle / N-Triples, optionally bzip2) into strict IRI–IRI–IRI N-Triples."""

from __future__ import annotations

import bz2
import gzip
import shutil
from pathlib import Path

from pyoxigraph import NamedNode, RdfFormat, parse


def _open_binary_maybe_compressed(path: Path):
    s = str(path)
    if s.endswith(".bz2"):
        return bz2.open(path, "rb")
    if s.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _guess_format(path: Path) -> RdfFormat:
    s = str(path).lower()
    if s.endswith(".ttl") or s.endswith(".ttl.bz2") or s.endswith(".ttl.gz"):
        return RdfFormat.TURTLE
    if s.endswith(".nt") or s.endswith(".nt.bz2") or s.endswith(".nt.gz"):
        return RdfFormat.N_TRIPLES
    raise ValueError(f"Cannot guess RDF format from path: {path}")


def _triple_line_iri_only(t) -> str | None:
    if not isinstance(t.subject, NamedNode):
        return None
    if not isinstance(t.predicate, NamedNode):
        return None
    if not isinstance(t.object, NamedNode):
        return None
    s, p, o = t.subject.value, t.predicate.value, t.object.value
    return f"<{s}> <{p}> <{o}> .\n"


def stream_rdf_sources_to_iri_nt(sources: list[Path], output_nt: Path, *, append: bool = False) -> int:
    """
    Parse each ``sources`` path (Turtle or NT, gzip/bzip2 ok) with pyoxigraph and append
    only triples whose subject, predicate, and object are IRIs (``NamedNode``).
    Returns total triples written.
    """
    output_nt.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    total = 0
    with output_nt.open(mode, encoding="utf-8") as out:
        for src in sources:
            fmt = _guess_format(src)
            with _open_binary_maybe_compressed(src) as bio:
                # DBpedia dumps may contain IRI code points outside strict RDF 1.1.
                for t in parse(input=bio, format=fmt, lenient=True):
                    line = _triple_line_iri_only(t)
                    if line is None:
                        continue
                    out.write(line)
                    total += 1
                    if total % 10_000_000 == 0:
                        print(f"  ... wrote {total} IRI triples (from {src.name})", flush=True)
    return total


def copy_ontology_nt(src: Path, dst: Path) -> None:
    """Copy an existing ontology ``.nt`` file to ``dst`` (preserves literals and comments)."""
    if not src.is_file():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
