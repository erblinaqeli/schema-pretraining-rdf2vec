"""Build / validate instance graphs for v1/dbpedia seeds."""

from __future__ import annotations

from pathlib import Path

from _dbpedia.nt_stream import iter_nt_iri_triples


def seeds_with_subject_triple(nt: Path, seeds: set[str]) -> set[str]:
    """Seeds that appear as **subject** of at least one IRI–IRI–IRI triple in ``nt``."""
    out: set[str] = set()
    for s, _p, _o in iter_nt_iri_triples(nt):
        if s in seeds:
            out.add(s)
    return out


def seeds_touching_graph(nt: Path, seeds: set[str]) -> set[str]:
    """Seeds that appear as subject or object in at least one triple."""
    out: set[str] = set()
    for s, _p, o in iter_nt_iri_triples(nt):
        if s in seeds:
            out.add(s)
        if o in seeds:
            out.add(o)
    return out


def report_seed_coverage(nt: Path, seeds: set[str]) -> tuple[int, int, int]:
    """
    Returns (n_seeds, n_seeds_in_graph, n_seeds_as_subject).
    ``n_seeds_in_graph`` counts seeds that appear as subject or object in some triple.
    """
    as_subj = seeds_with_subject_triple(nt, seeds)
    touching = seeds_touching_graph(nt, seeds)
    return len(seeds), len(touching & seeds), len(as_subj & seeds)
