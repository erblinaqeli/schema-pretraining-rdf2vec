#!/usr/bin/env python3
"""
Train a damped bound-offset ("mixed") DBpedia variant: protograph pretrain ->
``(1-λ)·class + λ·bound`` init -> finetune. Same corpus / recipe as
run_dbpedia_compare.py so the resulting KeyedVectors share its 1.28M vocab and
are directly comparable (GEval clustering, DLCC accuracy).

Example (background):

  nohup .venv/bin/python scripts/run_dbpedia_mixed.py --kind p2 --lam 0.5 \
      --out output/dbpedia/p2_bound_mixed_lam05 \
      > output/dbpedia/p2_bound_mixed_lam05/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _dbpedia_compare import (  # noqa: E402
    corpus_vocab,
    ensure_walks,
    load_eval_splits,
    load_instance_types,
    load_schema,
    run_protograph_mixed,
    write_protographs,
)

ROOT = _SCRIPTS_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="p2", choices=["p1", "p2", "p3"])
    ap.add_argument("--lam", type=float, default=0.5, help="mix coefficient (0=class, 1=bound)")
    ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "dbpedia" / "p2_bound_mixed_lam05")
    ap.add_argument("--protographs", type=Path, default=ROOT / "notebooks" / "dbpedia_compare" / "protographs")
    ap.add_argument("--ontology", type=Path, default=ROOT / "dbpedia_graph" / "ontology.nt")
    ap.add_argument("--graph", type=Path, default=ROOT / "dbpedia_graph" / "graph.nt")
    ap.add_argument("--dbpedia-root", type=Path, default=ROOT / "v1" / "dbpedia")
    ap.add_argument("--eval-k", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--pretrain-epochs", type=int, default=5)
    ap.add_argument("--proto-walks-per-entity", type=int, default=200)
    ap.add_argument("--proto-depth", type=int, default=3)
    ap.add_argument("--finetune-alpha", type=float, default=0.0025)
    ap.add_argument("--min-alpha", type=float, default=0.0001)
    ap.add_argument("--target-norm", type=float, default=8.0)
    ap.add_argument("--direction-tag", default="rolled")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.kind}_bound_mixed_{args.lam:g}"
    kv_path = out / f"{name}.kv"

    print(f"[{datetime.now():%F %T}] mixed run: {name}", flush=True)
    print(f"  walks: {args.walks}\n  out:   {out}\n  lambda: {args.lam}", flush=True)

    t0 = time.time()
    schema = load_schema(args.ontology)
    print(f"  schema: {schema.stats}", flush=True)

    # Reuse the protographs/walks already written by run_dbpedia_compare if present.
    proto_paths = write_protographs(schema, args.protographs)
    proto_walk = ensure_walks(
        proto_paths[args.kind],
        args.protographs / f"walks_{args.kind}.txt",
        walks_per_entity=args.proto_walks_per_entity,
        depth=args.proto_depth,
        seed=args.seed,
    )
    print(f"  protograph walks ready ({time.time()-t0:.0f}s)", flush=True)

    print("  loading instance types from graph.nt ...", flush=True)
    itypes = load_instance_types(args.graph, schema)
    print(f"  instance types: {itypes.stats} ({time.time()-t0:.0f}s)", flush=True)

    print("  building/loading corpus vocab cache ...", flush=True)
    vocab_cache = args.walks.with_name(args.walks.stem + "_vocab.pkl")
    freq, n_lines, n_tokens = corpus_vocab(args.walks, vocab_cache)
    print(f"  corpus: {n_lines} walks, {n_tokens} tokens, {len(freq)} distinct "
          f"({time.time()-t0:.0f}s)", flush=True)

    splits = load_eval_splits(args.dbpedia_root, k=args.eval_k)
    print(f"  eval splits: {len(splits)} (k={args.eval_k})", flush=True)

    print(f"[{datetime.now():%F %T}] {name}: start", flush=True)
    res = run_protograph_mixed(
        name, proto_walk, args.walks, args.graph,
        freq, n_lines, n_tokens, splits, itypes, schema,
        mix_lambda=args.lam,
        dim=args.dim, epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
        finetune_alpha=args.finetune_alpha, min_alpha=args.min_alpha,
        target_norm=args.target_norm, direction_tag=args.direction_tag,
        seed=args.seed, workers=args.workers, save_kv=kv_path,
    )
    res["direction_tag"] = args.direction_tag
    (out / "results.json").write_text(json.dumps({name: res}, indent=2))
    print(f"[{datetime.now():%F %T}] {name}: done in {res['seconds']:.0f}s "
          f"(init {res['init_mean_acc']:.4f} -> final {res['final_mean_acc']:.4f})", flush=True)
    print(f"  saved: {kv_path}", flush=True)


if __name__ == "__main__":
    main()
