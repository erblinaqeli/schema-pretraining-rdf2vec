from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
RDFS_SUBCLASS = "<http://www.w3.org/2000/01/rdf-schema#subClassOf>"


def parse_tc_arg(tc_arg: str) -> list[int]:
    s = str(tc_arg).strip().lower()
    if s == "all":
        return list(range(1, 13))
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def parse_ontology_nt(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Parse ontology.nt and collect:
      - instance_types: {<I_...>: set(<C_...>)}
      - class_parents:  {<C_child>: set(<C_parent>)}
    """
    instance_types: dict[str, set[str]] = {}
    class_parents: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw == ".":
                continue
            parts = raw.split(" ", 3)
            if len(parts) < 3:
                continue

            s, p, o = parts[0], parts[1], parts[2]
            if o.endswith("."):
                o = o[:-1].rstrip()

            if not (s.startswith("<") and s.endswith(">")):
                continue
            if not (p.startswith("<") and p.endswith(">")):
                continue
            if not (o.startswith("<") and o.endswith(">")):
                continue

            if p == RDF_TYPE and s.startswith("<I_") and o.startswith("<C_"):
                instance_types.setdefault(s, set()).add(o)

            if p == RDFS_SUBCLASS and s.startswith("<C_") and o.startswith("<C_"):
                class_parents.setdefault(s, set()).add(o)

    return instance_types, class_parents


def build_ancestor_index(class_parents: dict[str, set[str]]) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {}

    def get_ancestors(c: str) -> set[str]:
        if c in ancestors:
            return ancestors[c]
        parents = class_parents.get(c, set())
        out = set(parents)
        for p in parents:
            out |= get_ancestors(p)
        ancestors[c] = out
        return out

    for c in class_parents:
        get_ancestors(c)
    return ancestors


def get_most_specific_declared(classes: set[str], ancestors: dict[str, set[str]]) -> list[str]:
    """
    Keep only declared classes that are not ancestors of another declared class.
    """
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
    """
    Layered BFS from specific to general classes (deduplicated, deterministic).
    """
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


def build_entity2classes_hier(
    instance_types: dict[str, set[str]],
    class_parents: dict[str, set[str]],
) -> dict[str, list[str]]:
    ancestors = build_ancestor_index(class_parents)
    mapping: dict[str, list[str]] = {}

    for inst, classes in sorted(instance_types.items()):
        specific = get_most_specific_declared(classes, ancestors)
        expanded = expand_upward(specific, class_parents)
        mapping[inst] = expanded

    return mapping


def process_tc(tc: int) -> None:
    tc_str = f"tc{tc:02d}"
    ontology_path = (
        ROOT
        / "data"
        / "DLCC"
        / "synthetic_ontology"
        / tc_str
        / "synthetic_ontology"
        / "ontology.nt"
    )
    if not ontology_path.exists():
        raise FileNotFoundError(f"ontology.nt not found: {ontology_path}")

    instance_types, class_parents = parse_ontology_nt(ontology_path)
    mapping = build_entity2classes_hier(instance_types, class_parents)

    out_path = ROOT / "training_output" / "synthetic_ontology" / tc_str / "entity2classes_hier.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[OK] {tc_str}: wrote {len(mapping)} instances to {out_path} "
        f"(subclass edges={sum(len(v) for v in class_parents.values())})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build entity2classes_hier.json: most-specific rdf:type classes + all transitive superclasses."
        )
    )
    ap.add_argument("--tc", "-t", required=True, help='TC(s): 1, 1-12, 1,3,5, or "all".')
    args = ap.parse_args()

    for tc in parse_tc_arg(args.tc):
        process_tc(tc)


if __name__ == "__main__":
    main()