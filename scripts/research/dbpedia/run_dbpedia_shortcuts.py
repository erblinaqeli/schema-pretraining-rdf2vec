#!/usr/bin/env python3
"""Compute the DLCC shortcut/correlation analysis on v1/dbpedia and cache it.

  .venv/bin/python scripts/run_dbpedia_shortcuts.py

Writes:
  notebooks/dbpedia_shortcuts/adjacency.pkl   (eval-entity out/in predicate sets)
  notebooks/dbpedia_shortcuts/results.json    (per-split shortcuts + bag probe)

The notebook (notebooks/dbpedia_shortcuts.ipynb) only reads results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPTS = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _dbpedia_compare import load_eval_splits  # noqa: E402
from _dbpedia_investigate import split_meta  # noqa: E402
from _dbpedia_shortcuts import (  # noqa: E402
    build_adjacency,
    collect_eval_entities,
    existence_bag_probe,
    load_population,
    single_feature_shortcuts,
)

ROOT = _SCRIPTS.parent
GRAPH = ROOT / "dbpedia_graph" / "graph.nt"
DBPEDIA = ROOT / "v1" / "dbpedia"
OUT = ROOT / "notebooks" / "dbpedia_shortcuts"
K = 5000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("loading eval splits …", flush=True)
    splits = load_eval_splits(DBPEDIA, k=K, include_hard=True)
    print(f"  {len(splits)} splits", flush=True)

    entities = collect_eval_entities(splits)
    print(f"  {len(entities):,} unique eval entities", flush=True)

    print("building adjacency (one pass over graph.nt) …", flush=True)
    adj = build_adjacency(GRAPH, entities, cache_path=OUT / "adjacency.pkl")
    print("  adjacency:", json.dumps(adj.stats), flush=True)

    results: dict[str, dict] = {}
    for i, sp in enumerate(splits, 1):
        tc, domain, hard = split_meta(sp.name)
        dom = f"{domain}_hard" if hard else domain
        pos, neg = load_population(DBPEDIA, tc, dom, k=K)
        shortcuts = single_feature_shortcuts(pos, neg, adj, top_k=15)
        bag_acc = existence_bag_probe(sp, adj)
        best = shortcuts[0] if shortcuts else None
        results[sp.name] = dict(
            tc=tc,
            domain=domain,
            hard=hard,
            n_pos=len(pos),
            n_neg=len(neg),
            best_feature=best["feature"] if best else None,
            best_p_pos=best["p_pos"] if best else None,
            best_p_neg=best["p_neg"] if best else None,
            best_single_acc=best["acc"] if best else None,
            bag_acc=bag_acc,
            top=shortcuts,
        )
        single = results[sp.name]["best_single_acc"]
        print(f"  [{i:2d}/{len(splits)}] {sp.name:24s} "
              f"single={single:.3f} " if single is not None
              else f"  [{i:2d}/{len(splits)}] {sp.name:24s} single=  n/a ",
              end="", flush=True)
        print(f"bag={bag_acc:.3f}  best={results[sp.name]['best_feature']}", flush=True)

    payload = {"k": K, "adjacency_stats": adj.stats, "splits": results}
    (OUT / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / 'results.json'}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
