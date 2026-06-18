#!/usr/bin/env python3
"""Ablation: does unit-normalizing class codes BEFORE averaging help the *classic*
(pure class-mean transfer, no concept-bound) init?

The concept-bound class mix averages UNIT class codes (``_class_codes`` -> ``_unit``),
while ``classic_init`` averages RAW resolved stage-1 class codes (``_resolved_class_codes``)
and only renormalizes the FINAL mean to ``target_norm`` -- so classes with larger
pretrain norms dominate a multi-typed entity's mean. This script runs the classic
init both ways on the 10% corpus (identical everything except the flag) and reports
per-epoch DLCC accuracy.

  classic_raw  -- normalize_class_codes=False (current default)
  classic_unit -- normalize_class_codes=True  (unit each class code first)

Same testbed as exp3/exp6 (10% corpus, standardize lens), so numbers are comparable
to the references vanilla 0.805 / p2_classic 0.832.

Usage:
  .venv/bin/python scripts/_invest_classic_norm.py            # 10% corpus, p2
  .venv/bin/python scripts/_invest_classic_norm.py --walks walks/all_walks.txt
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

from _dbpedia_compare import classic_init, load_eval_splits  # noqa: E402
from _dbpedia_investigate import (  # noqa: E402
    Model,
    evaluate_models,
    load_emb,
    summarize,
)

ART = ROOT / "notebooks" / "dbpedia_investigate" / "artifacts"
OUT = ROOT / "notebooks" / "dbpedia_investigate" / "classic_norm"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks_010.txt")
ap.add_argument("--kind", default="p2")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--workers", type=int, default=20)
ap.add_argument("--n-jobs", type=int, default=12)
ap.add_argument("--alpha", type=float, default=0.0025)
ap.add_argument("--min-alpha", type=float, default=0.0001)
args = ap.parse_args()

DIM = 200
LENSES = ("standardize", "raw")
t_global = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t_global:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Corpus vocab + cached schema/itypes/stage-1 artifacts
# --------------------------------------------------------------------------- #
vocab_pkl = args.walks.with_name(args.walks.stem + "_vocab.pkl")
with vocab_pkl.open("rb") as fh:
    cv = pickle.load(fh)
freq, n_lines, n_tokens = cv["freq"], cv["n_lines"], cv["n_tokens"]
log(f"corpus {args.walks.name}: {n_lines:,} lines, {n_tokens:,} tokens, {len(freq):,} vocab")

splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
schema = pickle.loads((ART / "schema.pkl").read_bytes())
itypes = pickle.loads((ART / "itypes.pkl").read_bytes())
stage1 = load_emb(ART / f"stage1_{args.kind}")
used_norm = json.loads((ART / f"used_norm_{args.kind}.json").read_text())["used_norm"]
log(f"artifacts: {len(stage1)} stage1 codes, used_norm={used_norm:.3f}")

res_path = OUT / "results.json"
results = json.loads(res_path.read_text()) if res_path.is_file() else {}


def alpha_boundaries(start: float, end: float, epochs: int) -> list[float]:
    return [start + (end - start) * i / epochs for i in range(epochs + 1)]


def evaluate(model) -> dict:
    kv = model.wv
    m = Model("ft", kv=kv, dim=kv.vectors.shape[1])
    rec = {}
    for lens in LENSES:
        accs = evaluate_models(splits, [m], scaling=lens, n_jobs=args.n_jobs)
        rec[lens] = {"accs": accs, "summary": summarize(accs)}
    return rec


def run(name: str, normalize_class_codes: bool) -> None:
    if name in results:
        log(f"[skip] {name} cached")
        return
    t0 = time.time()
    bounds = alpha_boundaries(args.alpha, args.min_alpha, args.epochs)
    model = Word2Vec(
        vector_size=DIM, window=5, sg=1, hs=0, negative=5, min_count=1, sample=0.0,
        alpha=bounds[0], min_alpha=bounds[-1], workers=args.workers, seed=42,
        compute_loss=True,
    )
    model.build_vocab_from_freq(freq)
    model.corpus_count = n_lines
    model.corpus_total_words = n_tokens

    # pure classic transfer (no bound vectors); only the flag differs between variants
    stats = classic_init(
        model, stage1, itypes, schema,
        target_norm=used_norm, bound_vectors=None,
        normalize_class_codes=normalize_class_codes,
    )
    log(f"{name}: init_stats {stats}")

    rows = [evaluate(model)]            # epoch 0 (init quality, pre-training)
    losses = [0.0]
    s0 = rows[0]["standardize"]["summary"]
    log(f"{name} ep0(init): std all {s0['all']:.3f} (normal {s0['normal']:.3f} hard {s0['hard']:.3f})")

    corpus = LineSentence(str(args.walks))
    for ep in range(args.epochs):
        a0, a1 = bounds[ep], bounds[ep + 1]
        model.train(corpus, total_examples=n_lines, epochs=1,
                    start_alpha=a0, end_alpha=a1, compute_loss=True)
        loss = float(model.get_latest_training_loss())
        rec = evaluate(model)
        rows.append(rec)
        losses.append(loss)
        s = rec["standardize"]["summary"]
        log(f"{name} ep{ep + 1} lr[{a0:.4f}->{a1:.4f}] loss {loss:,.0f}: "
            f"std all {s['all']:.3f} (normal {s['normal']:.3f} hard {s['hard']:.3f})")

    model.wv.save(str(OUT / f"{name}.kv"))
    results[name] = {
        "name": name, "normalize_class_codes": normalize_class_codes,
        "kind": args.kind, "epochs": args.epochs, "corpus": args.walks.name,
        "alpha_bounds": bounds, "seconds": round(time.time() - t0, 1),
        "per_epoch": [
            {lens: rows[i][lens]["summary"] for lens in LENSES} | {"loss": losses[i]}
            for i in range(len(rows))
        ],
        "final_accs": {lens: rows[-1][lens]["accs"] for lens in LENSES},
        "init_accs": {lens: rows[0][lens]["accs"] for lens in LENSES},
        "losses": losses,
    }
    res_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fin = results[name]["per_epoch"][-1]["standardize"]
    log(f"[done] {name} in {time.time() - t0:.0f}s -> std all {fin['all']:.3f} "
        f"(normal {fin['normal']:.3f} hard {fin['hard']:.3f}); saved {name}.kv")
    del model


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        log(f"[ERROR] {fn.__name__}{a}:\n{traceback.format_exc()}")
        return None


safe(run, "classic_raw", False)
safe(run, "classic_unit", True)

# --------------------------------------------------------------------------- #
# Side-by-side summary
# --------------------------------------------------------------------------- #
if "classic_raw" in results and "classic_unit" in results:
    log("=" * 64)
    for lens in LENSES:
        a = results["classic_raw"]["per_epoch"][-1][lens]
        b = results["classic_unit"]["per_epoch"][-1][lens]
        log(f"[{lens:11s}] raw  all {a['all']:.3f} normal {a['normal']:.3f} hard {a['hard']:.3f}")
        log(f"[{lens:11s}] unit all {b['all']:.3f} normal {b['normal']:.3f} hard {b['hard']:.3f}")
        log(f"[{lens:11s}] dlt  all {b['all']-a['all']:+.3f} "
            f"normal {b['normal']-a['normal']:+.3f} hard {b['hard']-a['hard']:+.3f}")
log("ALL DONE")
