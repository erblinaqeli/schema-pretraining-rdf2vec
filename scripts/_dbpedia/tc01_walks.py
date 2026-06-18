#!/usr/bin/env python3
"""
Generate RDF2Vec jRDF2Vec-style duplicate-free random walks for all entity lists under
v1/dbpedia/tc01/**/{positives,negatives}.txt, scanning the (large) DBpedia graph only once.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _dbpedia.entities import load_entity_uris
from _walks import (
    build_forward_adjacency,
    collect_jrdf2vec_entities,
    write_jrdf2vec_duplicate_free_walks,
)


def _norm_uri(u: str) -> str:
    u = u.strip()
    if u.startswith("<") and u.endswith(">"):
        return u[1:-1]
    return u


def main() -> None:
    p = argparse.ArgumentParser(description="Generate tc01 walks (single graph scan).")
    p.add_argument(
        "--tc01-root",
        type=Path,
        default=Path("v1/dbpedia/tc01"),
        help="Root directory containing domain/size/{positives,negatives}.txt (default: v1/dbpedia/tc01)",
    )
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("dbpedia_graph/graph.nt"),
        help="DBpedia instance graph in N-Triples (default: dbpedia_graph/graph.nt)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Default: output/dbpedia_tc01_walks/<datetime>/",
    )
    p.add_argument("--walks-per-entity", type=int, default=100)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--threads", type=int, default=None, help="Default: same as random_walks.py")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--token-format", choices=("angled", "bare"), default="angled")
    args = p.parse_args()

    if args.out_root is None:
        run_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_root = Path("output") / "dbpedia_tc01_walks" / run_dt

    if not args.tc01_root.exists():
        raise SystemExit(f"Missing tc01 root: {args.tc01_root}")
    if not args.graph.exists():
        raise SystemExit(f"Missing graph: {args.graph}")

    threads = args.threads
    if threads is None:
        # Keep default behavior from random_walks.py if user didn't specify.
        import os

        threads = max(1, (os.cpu_count() or 2) // 2)

    pos_files = sorted(args.tc01_root.rglob("positives.txt"))
    if not pos_files:
        raise SystemExit(f"No positives.txt found under {args.tc01_root}")

    print(f"Scanning graph once: {args.graph}", flush=True)
    adj = build_forward_adjacency(args.graph)
    if not adj:
        raise SystemExit("No object triples found; nothing to walk on.")
    present = set(collect_jrdf2vec_entities(adj))

    for pos in pos_files:
        neg = pos.with_name("negatives.txt")
        if not neg.exists():
            print(f"Skip (missing negatives.txt): {pos.parent}", flush=True)
            continue

        # Expect: tc01/<domain>/<size>/positives.txt
        size_dir = pos.parent
        domain_dir = size_dir.parent
        domain = domain_dir.name
        size = size_dir.name
        out_dir = args.out_root / domain / size
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, path in (("positives", pos), ("negatives", neg)):
            requested = sorted({_norm_uri(u) for u in load_entity_uris(path)})
            if not requested:
                print(f"Skip empty {name}: {path}", flush=True)
                continue

            entities = [u for u in requested if u in present]
            missing = len(requested) - len(entities)
            if missing:
                print(f"{domain}/{size} {name}: skipping {missing}/{len(requested)} missing entities", flush=True)
            if not entities:
                print(f"{domain}/{size} {name}: no entities present in graph; skip", flush=True)
                continue

            out_path = out_dir / f"walks_{name}.txt"
            print(f"{domain}/{size} {name}: {len(entities)} entities -> {out_path}", flush=True)
            write_jrdf2vec_duplicate_free_walks(
                adj,
                entities,
                out_path,
                walks_per_entity=args.walks_per_entity,
                depth=args.depth,
                threads=threads,
                seed=args.seed,
                token_format=args.token_format,
            )

    print(f"Done. Walks under: {args.out_root}", flush=True)


if __name__ == "__main__":
    main()

