#!/usr/bin/env python3
"""
Run the DBpedia vanilla / classic / concept-bound comparison end to end.

Trains up to seven variants on the same instance-walk corpus and evaluates
LogReg test accuracy on every DLCC DBpedia split (v1/dbpedia, k=5000) at init
and after every finetune epoch. Results are cached per variant in
``<out>/results.json``; finished variants are skipped on re-run.

Example (full run, ~1.5h per variant):

  nohup .venv/bin/python scripts/run_dbpedia_compare.py \
      --walks walks/all_walks.txt --out notebooks/dbpedia_compare \
      > notebooks/dbpedia_compare/run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _dbpedia_compare import (  # noqa: E402
    VARIANTS,
    corpus_vocab,
    ensure_walks,
    load_eval_splits,
    load_instance_types,
    load_results,
    load_schema,
    run_protograph_variant,
    run_vanilla,
    save_results,
    write_protographs,
)

ROOT = _SCRIPTS_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "dbpedia_compare")
    ap.add_argument("--ontology", type=Path, default=ROOT / "dbpedia_graph" / "ontology.nt")
    ap.add_argument("--graph", type=Path, default=ROOT / "dbpedia_graph" / "graph.nt")
    ap.add_argument("--dbpedia-root", type=Path, default=ROOT / "v1" / "dbpedia")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--eval-k", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--pretrain-epochs", type=int, default=5)
    ap.add_argument("--proto-walks-per-entity", type=int, default=200)
    ap.add_argument("--proto-depth", type=int, default=3)
    ap.add_argument("--vanilla-alpha", type=float, default=0.025)
    ap.add_argument("--finetune-alpha", type=float, default=0.0025)
    ap.add_argument("--min-alpha", type=float, default=0.0001)
    ap.add_argument("--target-norm", type=float, default=8.0)
    ap.add_argument(
        "--classic-strategy",
        choices=["all_init", "most_specific"],
        default="all_init",
        help="Instance init for *_classic variants: all_init (climb subClassOf to the "
        "nearest coded ancestor, default) or most_specific (leaf class direct code, no "
        "fallback -> random init when the leaf is OOV in pretrain).",
    )
    ap.add_argument(
        "--direction-tag",
        default="rolled",
        help="Incoming-edge tag for bound variants: rolled|shared|negated|keyed "
        "(see notebooks/direction_aware_cardinality.ipynb)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--no-save-kv", action="store_true", help="Do not save final KeyedVectors")
    args = ap.parse_args()

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; expected from {VARIANTS}")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    results_json = out / "results.json"

    print(f"[{datetime.now():%F %T}] DBpedia compare run", flush=True)
    print(f"  walks:   {args.walks}", flush=True)
    print(f"  out:     {out}", flush=True)
    print(f"  variants: {args.variants}", flush=True)

    t0 = time.time()
    schema = load_schema(args.ontology)
    print(f"  schema: {schema.stats}", flush=True)
    proto_paths = write_protographs(schema, out / "protographs")

    proto_walks = {
        kind: ensure_walks(
            path,
            out / "protographs" / f"walks_{kind}.txt",
            walks_per_entity=args.proto_walks_per_entity,
            depth=args.proto_depth,
            seed=args.seed,
        )
        for kind, path in proto_paths.items()
    }
    print(f"  protograph walks ready ({time.time()-t0:.0f}s)", flush=True)

    print("  loading instance types from graph.nt ...", flush=True)
    itypes = load_instance_types(args.graph, schema)
    print(f"  instance types: {itypes.stats} ({time.time()-t0:.0f}s)", flush=True)

    print("  building/loading corpus vocab cache ...", flush=True)
    vocab_cache = args.walks.with_name(args.walks.stem + "_vocab.pkl")
    freq, n_lines, n_tokens = corpus_vocab(args.walks, vocab_cache)
    print(
        f"  corpus: {n_lines} walks, {n_tokens} tokens, {len(freq)} distinct "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    splits = load_eval_splits(args.dbpedia_root, k=args.eval_k)
    print(f"  eval splits: {len(splits)} (k={args.eval_k})", flush=True)

    results = load_results(results_json)
    shared = dict(
        dim=args.dim,
        epochs=args.epochs,
        min_alpha=args.min_alpha,
        seed=args.seed,
        workers=args.workers,
    )

    for variant in args.variants:
        if variant in results:
            print(f"[{datetime.now():%F %T}] {variant}: cached, skipping", flush=True)
            continue
        print(f"[{datetime.now():%F %T}] {variant}: start", flush=True)
        kv_path = None if args.no_save_kv else out / f"{variant}.kv"
        if variant == "vanilla":
            res = run_vanilla(
                args.walks, freq, n_lines, n_tokens, splits,
                alpha=args.vanilla_alpha, save_kv=kv_path, **shared,
            )
        else:
            kind, mode = variant.split("_", 1)
            res = run_protograph_variant(
                variant, proto_walks[kind], args.walks, args.graph,
                freq, n_lines, n_tokens, splits, itypes, schema,
                bound=(mode == "bound"),
                pretrain_epochs=args.pretrain_epochs,
                finetune_alpha=args.finetune_alpha,
                target_norm=args.target_norm,
                classic_strategy=args.classic_strategy,
                direction_tag=args.direction_tag,
                save_kv=kv_path,
                **shared,
            )
            if mode == "bound":
                res["direction_tag"] = args.direction_tag
        results[variant] = res
        save_results(results, results_json)
        print(
            f"[{datetime.now():%F %T}] {variant}: done in {res['seconds']:.0f}s "
            f"(init {res['init_mean_acc']:.4f} -> final {res['final_mean_acc']:.4f})",
            flush=True,
        )

    print(f"[{datetime.now():%F %T}] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
