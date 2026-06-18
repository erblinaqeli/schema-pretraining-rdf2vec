#!/usr/bin/env python3
"""
DBpedia vanilla stability check — 5 repeated runs on a 10% walk subset.

Trains the ``vanilla`` (from-scratch) variant 5 times with different gensim
seeds on the cached 10% instance-walk corpus (``walks/all_walks_010.txt``),
evaluating LogReg test accuracy on every DLCC DBpedia split at init and after
every epoch. The 10% corpus is FIXED across the 5 runs (reused, not resampled),
so only the training/init randomness varies — the same methodology as
``notebooks/5_times.ipynb`` uses for the synthetic suite.

vanilla needs neither protographs nor instance types, so this skips all of that
setup and is ~10x faster than ``run_dbpedia_compare.py`` on the full corpus
(~6-8 min/run instead of ~67).

Results are cached per seed in ``<out>/results.json``; finished seeds are
skipped on re-run.

Example:

  .venv/bin/python scripts/run_dbpedia_vanilla_5x.py \
      --walks walks/all_walks_010.txt \
      --out notebooks/dbpedia_vanilla_5x_10pct
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
    load_eval_splits,
    run_vanilla,
)

ROOT = _SCRIPTS_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks_010.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "dbpedia_vanilla_5x_10pct")
    ap.add_argument("--dbpedia-root", type=Path, default=ROOT / "v1" / "dbpedia")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--eval-k", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--min-alpha", type=float, default=0.0001)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    results_json = out / "results.json"

    print(f"[{datetime.now():%F %T}] DBpedia vanilla 5x (10% corpus)", flush=True)
    print(f"  walks: {args.walks}", flush=True)
    print(f"  out:   {out}", flush=True)
    print(f"  seeds: {args.seeds}", flush=True)

    t0 = time.time()
    vocab_cache = args.walks.with_name(args.walks.stem + "_vocab.pkl")
    freq, n_lines, n_tokens = corpus_vocab(args.walks, vocab_cache)
    print(
        f"  corpus: {n_lines} walks, {n_tokens} tokens, {len(freq)} distinct "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    splits = load_eval_splits(args.dbpedia_root, k=args.eval_k)
    print(f"  eval splits: {len(splits)} (k={args.eval_k})", flush=True)

    results: dict[str, dict] = {}
    if results_json.is_file():
        results = json.loads(results_json.read_text(encoding="utf-8"))

    for seed in args.seeds:
        key = str(seed)
        if key in results:
            print(f"[{datetime.now():%F %T}] seed={seed}: cached, skipping", flush=True)
            continue
        print(f"[{datetime.now():%F %T}] seed={seed}: start", flush=True)
        res = run_vanilla(
            args.walks, freq, n_lines, n_tokens, splits,
            dim=args.dim, epochs=args.epochs, alpha=args.alpha,
            min_alpha=args.min_alpha, seed=seed, workers=args.workers, save_kv=None,
        )
        res["seed"] = seed
        results[key] = res
        results_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{datetime.now():%F %T}] seed={seed}: done in {res['seconds']:.0f}s "
            f"(init {res['init_mean_acc']:.4f} -> final {res['final_mean_acc']:.4f})",
            flush=True,
        )

    print(f"[{datetime.now():%F %T}] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
