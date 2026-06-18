#!/usr/bin/env python3
"""
Build MASCHInE protographs P1 and P2 from RDF/S schema triples (Hubert et al., arXiv:2306.03659).

P1: For each relation r with axioms domain(r, Ci) and range(r, Cj), emit (Ci, r, Cj).
    Protograph entities are the class IRIs themselves (one node per class).

P2: Start from P1; for each triple (Ci, r, Cj), for each direct subclass Ci' of Ci add
    (Ci', r, Cj), and for each direct subclass Cj' of Cj add (Ci, r, Cj'). Do not add
    (Ci', r, Cj') when both ends are replaced (direct subclasses only, not transitive).

Also writes entity-to-class mappings from instance rdf:type triples in the schema file:
  entity2classes.json — entity -> list of most-specific declared classes
  entity2classes_hier.json — entity -> class hierarchy (most-specific first, then superclasses)

Example:
  uv run protograph --schema schema.nt --out-dir ./out
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Same pattern as convert.py / random_walks.py
TRIPLE_LINE = re.compile(r"^\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*(?:#.*)?$")

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
RDFS_DOMAIN = f"{RDFS}domain"
RDFS_RANGE = f"{RDFS}range"
RDFS_SUBCLASS = f"{RDFS}subClassOf"


def iter_nt_iris(path: Path) -> Iterable[tuple[str, str, str]]:
    """Yield (subject, predicate, object) for IRI object triples; skip literals and bad lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                continue
            yield m.group(1), m.group(2), m.group(3)


def iter_rdf_iris(path: Path) -> Iterable[tuple[str, str, str]]:
    """
    Yield (s, p, o) for triples where all terms are IRIs (no blank nodes, no literals).

    Supported inputs:
      - `.nt`  (fast regex-based scan; same behavior as before)
      - `.ttl` / `.turtle` (parsed with RDFLib)
    """
    suffix = path.suffix.lower()
    if suffix == ".nt":
        yield from iter_nt_iris(path)
        return

    if suffix in {".ttl", ".turtle"}:
        try:
            import rdflib  # type: ignore
            from rdflib.term import BNode, Literal, URIRef  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Reading Turtle (.ttl) requires rdflib. "
                "Install dependencies (e.g. `uv sync`) or add rdflib to your environment."
            ) from e

        g = rdflib.Graph()
        g.parse(path.as_posix(), format="turtle")
        for s, p, o in g:
            if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
                yield str(s), str(p), str(o)
            else:
                # Skip bnodes and literals (protograph generation assumes IRI-IRI-IRI).
                continue
        return

    raise ValueError(f"Unsupported schema/KG format for {path.name!r}. Expected .nt or .ttl.")


def load_ontology(
    path: Path,
    *,
    relation_filter: set[str] | None = None,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """
    From N-Triples, collect schema axioms and instance/class typing in one pass.

    Returns:
        domains: property -> set of domain class IRIs
        ranges: property -> set of range class IRIs
        children: superclass IRI -> set of direct subclass IRIs
        instance_types: instance IRI -> set of declared class IRIs (rdf:type)
        class_parents: class IRI -> set of direct superclass IRIs (rdfs:subClassOf)
    """
    domains: dict[str, set[str]] = defaultdict(set)
    ranges: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    instance_types: dict[str, set[str]] = defaultdict(set)
    class_parents: dict[str, set[str]] = defaultdict(set)

    for s, p, o in iter_rdf_iris(path):
        if p == RDFS_SUBCLASS and s.startswith("C_") and o.startswith("C_"):
            # s subclassOf o  => o is superclass, s is direct subclass
            children[o].add(s)
            class_parents[s].add(o)
            continue
        if p == RDF_TYPE and s.startswith("I_") and o.startswith("C_"):
            instance_types[s].add(o)
            continue
        if p == RDFS_DOMAIN:
            if relation_filter is not None and s not in relation_filter:
                continue
            domains[s].add(o)
            continue
        if p == RDFS_RANGE:
            if relation_filter is not None and s not in relation_filter:
                continue
            ranges[s].add(o)

    return (
        dict(domains),
        dict(ranges),
        dict(children),
        dict(instance_types),
        dict(class_parents),
    )


def load_schema(
    path: Path,
    *,
    relation_filter: set[str] | None = None,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """
    From N-Triples, collect rdfs:domain / rdfs:range per property and direct subclasses.

    Returns:
        domains: property -> set of domain class IRIs
        ranges: property -> set of range class IRIs
        children: superclass IRI -> set of direct subclass IRIs
    """
    domains, ranges, children, _instance_types, _class_parents = load_ontology(
        path, relation_filter=relation_filter
    )
    return domains, ranges, children


def build_ancestor_index(class_parents: dict[str, set[str]]) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {}

    def get_ancestors(c: str) -> set[str]:
        if c in ancestors:
            return ancestors[c]
        parents = class_parents.get(c, set())
        out = set(parents)
        for parent in parents:
            out |= get_ancestors(parent)
        ancestors[c] = out
        return out

    for c in class_parents:
        get_ancestors(c)
    return ancestors


def get_most_specific_declared(classes: set[str], ancestors: dict[str, set[str]]) -> list[str]:
    """Keep only declared classes that are not ancestors of another declared class."""
    if len(classes) <= 1:
        return sorted(classes)

    cls_list = sorted(classes)
    remaining: list[str] = []
    for c in cls_list:
        is_general = any(c != d and c in ancestors.get(d, set()) for d in cls_list)
        if not is_general:
            remaining.append(c)

    return remaining if remaining else cls_list


def expand_upward(start_classes: list[str], class_parents: dict[str, set[str]]) -> list[str]:
    """Layered BFS from specific to general classes (deduplicated, deterministic)."""
    visited: set[str] = set()
    result: list[str] = []
    frontier = sorted(set(start_classes))

    while frontier:
        next_frontier: set[str] = set()
        for c in frontier:
            if c in visited:
                continue
            visited.add(c)
            result.append(c)
        for c in frontier:
            for parent in class_parents.get(c, set()):
                if parent not in visited:
                    next_frontier.add(parent)
        frontier = sorted(next_frontier)

    return result


def build_entity2classes(
    instance_types: dict[str, set[str]],
    class_parents: dict[str, set[str]],
) -> dict[str, list[str]]:
    ancestors = build_ancestor_index(class_parents)
    mapping: dict[str, list[str]] = {}
    for inst, classes in sorted(instance_types.items()):
        mapping[inst] = get_most_specific_declared(classes, ancestors)
    return mapping


def build_entity2classes_hier(
    instance_types: dict[str, set[str]],
    class_parents: dict[str, set[str]],
) -> dict[str, list[str]]:
    ancestors = build_ancestor_index(class_parents)
    mapping: dict[str, list[str]] = {}

    for inst, classes in sorted(instance_types.items()):
        specific = get_most_specific_declared(classes, ancestors)
        mapping[inst] = expand_upward(specific, class_parents)

    return mapping


def to_nt_token(iri: str) -> str:
    return iri if iri.startswith("<") else f"<{iri}>"


def mapping_to_json(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        to_nt_token(entity): [to_nt_token(cls) for cls in classes]
        for entity, classes in mapping.items()
    }


def check_entity_mappings(
    entity2classes: dict[str, list[str]],
    entity2classes_hier: dict[str, list[str]],
) -> None:
    if set(entity2classes) != set(entity2classes_hier):
        raise ValueError(
            "entity2classes and entity2classes_hier have different entity sets: "
            f"{len(entity2classes)} vs {len(entity2classes_hier)}."
        )

    for inst in entity2classes:
        specific = entity2classes[inst]
        hier = entity2classes_hier[inst]
        if set(specific) != set(hier[: len(specific)]):
            raise ValueError(
                f"Most-specific classes mismatch for {inst!r}: "
                f"entity2classes={specific!r}, hier prefix={hier[: len(specific)]!r}."
            )


def build_p1(
    domains: dict[str, set[str]],
    ranges: dict[str, set[str]],
) -> set[tuple[str, str, str]]:
    """Cartesian product of domain × range per relation (multiple axioms -> multiple triples)."""
    triples: set[tuple[str, str, str]] = set()
    relations = set(domains) & set(ranges)
    for r in relations:
        for ci in domains[r]:
            for cj in ranges[r]:
                triples.add((ci, r, cj))
    return triples


def build_p1_notebook(
    domains: dict[str, set[str]],
    ranges: dict[str, set[str]],
) -> set[tuple[str, str, str]]:
    """P1 as in simple_correct/build_protographs.py: one (domain, rel, range) per relation."""
    triples: set[tuple[str, str, str]] = set()
    for rel in sorted(set(domains) & set(ranges)):
        dom = sorted(domains[rel])[0]
        ran = sorted(ranges[rel])[0]
        triples.add((dom, rel, ran))
    return triples


def relations_with_domain_and_range(
    domains: dict[str, set[str]],
    ranges: dict[str, set[str]],
) -> set[str]:
    """Property IRIs that have at least one rdfs:domain and one rdfs:range axiom."""
    return set(domains) & set(ranges)


def expected_p1_triple_count(
    domains: dict[str, set[str]],
    ranges: dict[str, set[str]],
) -> int:
    """
    Number of P1 triples implied by the schema: one (Ci, r, Cj) per domain×range pair per relation.

    This matches the count of relations with both domain and range only when each such relation
    has exactly one domain class and one range class.
    """
    rels = relations_with_domain_and_range(domains, ranges)
    return sum(len(domains[r]) * len(ranges[r]) for r in rels)


def check_p1_notebook_size(
    p1: set[tuple[str, str, str]],
    domains: dict[str, set[str]],
    ranges: dict[str, set[str]],
) -> None:
    """Sanity-check notebook-style P1: one triple per relation with both domain and range."""
    expected = len(relations_with_domain_and_range(domains, ranges))
    got = len(p1)
    if got != expected:
        raise ValueError(
            f"P1 (notebook) size mismatch: got {got} triples, expected {expected} "
            f"(one triple per relation with both rdfs:domain and rdfs:range)."
        )


def check_p1_size(
    p1: set[tuple[str, str, str]],
    domains: dict[str, set[str]],
    ranges: dict[str, set[str]],
) -> None:
    """
    Sanity-check P1: |P1| must equal the number of (Ci, r, Cj) tuples from domain×range per
    relation with both domain and range (equivalently: as many P1 triples as that Cartesian
    product produces; when each relation has a single domain and range, that equals the number
    of such relations).
    """
    expected = expected_p1_triple_count(domains, ranges)
    got = len(p1)
    if got != expected:
        rels = relations_with_domain_and_range(domains, ranges)
        raise ValueError(
            f"P1 size mismatch: got {got} triples, expected {expected} "
            f"(from {len(rels)} relations with both rdfs:domain and rdfs:range; "
            "sum of |domain(r)|×|range(r) per relation)."
        )


def build_p2(
    p1: set[tuple[str, str, str]],
    children: dict[str, set[str]],
) -> set[tuple[str, str, str]]:
    """P2 expansion: direct subclasses on one side at a time (Section 3.1)."""
    out = set(p1)
    for ci, r, cj in p1:
        for ci_prime in children.get(ci, ()):
            out.add((ci_prime, r, cj))
        for cj_prime in children.get(cj, ()):
            out.add((ci, r, cj_prime))
    return out


def check_p2_subclass_expansion(
    p1: set[tuple[str, str, str]],
    p2: set[tuple[str, str, str]],
    children: dict[str, set[str]],
) -> None:
    """
    P2 sanity check (Hubert et al., Sec. 3.1): for each prototriple in P1, each
    direct subclass of the domain or range class must appear in an additional
    triple in P2 — (Ci', r, Cj) for Ci' subclass of Ci, (Ci, r, Cj') for Cj'
    subclass of Cj; both ends are not replaced at once (direct subclasses only).
    """
    for ci, r, cj in p1:
        for ci_prime in children.get(ci, ()):
            t = (ci_prime, r, cj)
            if t not in p2:
                raise ValueError(
                    f"P2 missing subclass expansion for domain: expected {t!r} "
                    f"from P1 triple {(ci, r, cj)!r} and rdfs:subClassOf({ci_prime!r}, {ci!r})."
                )
        for cj_prime in children.get(cj, ()):
            t = (ci, r, cj_prime)
            if t not in p2:
                raise ValueError(
                    f"P2 missing subclass expansion for range: expected {t!r} "
                    f"from P1 triple {(ci, r, cj)!r} and rdfs:subClassOf({cj_prime!r}, {cj!r})."
                )


def triples_to_nt_lines(triples: set[tuple[str, str, str]]) -> list[str]:
    lines = [f"<{s}> <{p}> <{o}> .\n" for s, p, o in sorted(triples)]
    return lines


def relations_in_kg(kg_path: Path) -> set[str]:
    """Collect predicate IRIs appearing in a KG (.nt or .ttl) to filter schema relations."""
    preds: set[str] = set()
    for _s, p, _o in iter_rdf_iris(kg_path):
        preds.add(p)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MASCHInE protographs P1 and P2 (arXiv:2306.03659).")
    ap.add_argument(
        "--schema",
        type=Path,
        required=True,
        help="Schema file (.nt or .ttl) with rdfs:domain, rdfs:range, rdfs:subClassOf (IRI triples).",
    )
    ap.add_argument(
        "--kg",
        type=Path,
        default=None,
        help="Optional KG (.nt or .ttl): only relations whose IRIs appear as predicates here are used.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for protograph_p1.nt, protograph_p2.nt, and entity-class JSON files",
    )
    ap.add_argument(
        "--p1-name",
        default="protograph_p1.nt",
        help="Output filename for P1 (N-Triples).",
    )
    ap.add_argument(
        "--p2-name",
        default="protograph_p2.nt",
        help="Output filename for P2 (N-Triples).",
    )
    ap.add_argument(
        "--ontology-only",
        action="store_true",
        help="Use ontology.nt only (no KG relation filter) and notebook-style P1 "
        "(one domain/range triple per relation, matching simple_correct/build_protographs.py).",
    )
    args = ap.parse_args()

    rel_filter = None if args.ontology_only else (
        relations_in_kg(args.kg) if args.kg is not None else None
    )
    domains, ranges, children, instance_types, class_parents = load_ontology(
        args.schema, relation_filter=rel_filter
    )

    if args.ontology_only:
        p1 = build_p1_notebook(domains, ranges)
        check_p1_notebook_size(p1, domains, ranges)
        print("P1 sanity check passed (notebook style: one triple per relation).")
    else:
        p1 = build_p1(domains, ranges)
        check_p1_size(p1, domains, ranges)
        print("P1 sanity check passed (|P1| matches domain×range count per relation).")
    p2 = build_p2(p1, children)
    check_p2_subclass_expansion(p1, p2, children)
    print(
        "P2 sanity check passed (each direct subclass of a P1 domain/range class "
        "appears in an additional triple, per Sec. 3.1)."
    )

    entity2classes = build_entity2classes(instance_types, class_parents)
    entity2classes_hier = build_entity2classes_hier(instance_types, class_parents)
    check_entity_mappings(entity2classes, entity2classes_hier)
    print(f"Entity-class mapping sanity check passed ({len(entity2classes)} instances).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1_path = args.out_dir / args.p1_name
    p2_path = args.out_dir / args.p2_name
    entity2classes_path = args.out_dir / "entity2classes.json"
    entity2classes_hier_path = args.out_dir / "entity2classes_hier.json"

    p1_path.write_text("".join(triples_to_nt_lines(p1)), encoding="utf-8")
    p2_path.write_text("".join(triples_to_nt_lines(p2)), encoding="utf-8")
    entity2classes_path.write_text(
        json.dumps(mapping_to_json(entity2classes), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    entity2classes_hier_path.write_text(
        json.dumps(mapping_to_json(entity2classes_hier), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(p1)} triples -> {p1_path}")
    print(f"Wrote {len(p2)} triples -> {p2_path}")
    print(f"Wrote {len(entity2classes)} instances -> {entity2classes_path}")
    print(f"Wrote {len(entity2classes_hier)} instances -> {entity2classes_hier_path}")


if __name__ == "__main__":
    main()
