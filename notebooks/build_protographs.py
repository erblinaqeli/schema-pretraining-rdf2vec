from pathlib import Path
import sys

ROOT = Path(r"C:\Users\Erblina\thesis").resolve()

RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


def parse_ontology_nt(path: Path):
    """Simple parser: assumes clean N-Triples with <s> <p> <o> . lines"""
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw == ".":
                continue

            if raw.endswith("."):
                raw = raw[:-1].strip()

            parts = raw.split()
            if len(parts) < 3:
                continue

            s, p, o = parts[0], parts[1], parts[2]

            def strip_b(x):
                return x[1:-1] if x.startswith("<") and x.endswith(">") else x

            triples.append((strip_b(s), strip_b(p), strip_b(o)))

    return triples


def build_indices(triples):
    rel2domain = {}
    rel2range = {}
    class2subclasses = {} # it stores children grouped by parent

    for s, p, o in triples:
        if p == RDFS_DOMAIN:
            rel2domain[s] = o
        elif p == RDFS_RANGE:
            rel2range[s] = o
        elif p == RDFS_SUBCLASS:
            # s subClassOf o
            parent = o
            child = s
            class2subclasses.setdefault(parent, set()).add(child)

    return rel2domain, rel2range, class2subclasses


def write_nt(triples, out_path: Path):
    """Write triples in N-Triples format, wrapping with <...>."""
    def wrap(x):
        return x if (x.startswith("<") and x.endswith(">")) else f"<{x}>"

    with out_path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{wrap(s)} {wrap(p)} {wrap(o)} .\n")


def build_prototype_v1(rel2domain, rel2range, out_path: Path):
    """P1: DomainClass --rel--> RangeClass."""
    out_triples = []
    for rel, dom in rel2domain.items():
        if rel in rel2range:
            ran = rel2range[rel]
            out_triples.append((dom, rel, ran))

    write_nt(out_triples, out_path)
    print(f"[P1] wrote {out_path} ({len(out_triples)} triples)")


def build_prototype_v2(rel2domain, rel2range, class2subclasses, out_path: Path):
    """P2: P1 + direct subclasses expansion."""
    # Implements MASCHInE P2 protograph exactly:
    # (Ci, r, Cj), (C'i, r, Cj), (Ci, r, C'j)
    # No (C'i, r, C'j), direct subclasses only

    out_triples = []

    for rel, dom in rel2domain.items():
        if rel not in rel2range:
            continue
        ran = rel2range[rel]

        out_triples.append((dom, rel, ran)) # (pCi, r, pCj)

        # domain subclasses
        for dsub in class2subclasses.get(dom, []):
            out_triples.append((dsub, rel, ran)) # (pC′i, r, pCj)

        # range subclasses
        for rsub in class2subclasses.get(ran, []):
            out_triples.append((dom, rel, rsub)) # (pCi, r, pC′j)

    # dedupe
    seen = set()
    deduped = []
    for tri in out_triples:
        if tri not in seen:
            seen.add(tri)
            deduped.append(tri)

    write_nt(deduped, out_path)
    print(f"[P2] wrote {out_path} ({len(deduped)} triples)")


def process_tc(tc_num: int):
    tc_str = f"tc{tc_num:02d}"
    ont_path = (
        ROOT / "data" / "DLCC" / "synthetic_ontology" /
        tc_str / "synthetic_ontology" / "ontology.nt"
    )

    if not ont_path.exists():
        print(f"[WARN] ontology.nt not found at {ont_path}")
        return

    print(f"Parsing ontology for {tc_str}: {ont_path}")
    triples = parse_ontology_nt(ont_path)
    rel2domain, rel2range, class2subclasses = build_indices(triples)

    out_dir = (
        ROOT / "training_output" / "synthetic_ontology" /
        tc_str / "protographs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    build_prototype_v1(rel2domain, rel2range, out_dir / "protograph_p1.nt")
    build_prototype_v2(rel2domain, rel2range, class2subclasses, out_dir / "protograph_p2.nt")


def main():
    if len(sys.argv) > 1:
        tcs = [int(x) for x in sys.argv[1:]]
    else:
        tcs = list(range(1, 13))

    for tc in tcs:
        print(f"=== building protographs for TC {tc:02d} ===")
        process_tc(tc)


if __name__ == "__main__":
    main()