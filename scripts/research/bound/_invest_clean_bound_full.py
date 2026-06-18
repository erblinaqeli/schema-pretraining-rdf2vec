#!/usr/bin/env python3
"""Train the two *cleaner* concept-bound P2 init variants from
notebooks/clean_bound.ipynb on the FULL 1.16 B-token DBpedia corpus, each with
and without the per-row norm cap @ 16, and save every run under output/dbpedia/.

The cleaner variants reweight how the asserted class hierarchy enters the
concept-bound init (the ``class_weight_mode`` knob on
``_dbpedia_compare.concept_bound_vectors``):

  * decay         — every materialized ancestor class, weighted ``0.5 ** hops``
                    from the leaf (leaf=1.0, parent=0.5, grandparent=0.25 ...);
  * most_specific — only the most-specific asserted class(es), weight 1.0.

2x2 grid (init mode x norm policy), all LR 0.0025 -> 0.0001 (protected finetune):

  1. p2_bound_decay_cap16      decay init      -> per-row norm cap @ 16
  2. p2_bound_specific_cap16   most_specific   -> per-row norm cap @ 16
  3. p2_bound_decay            decay init      -> global rescale only (no cap)
  4. p2_bound_specific         most_specific   -> global rescale only (no cap)

cap16 and no-cap share one cached init artifact per mode (built with the single
global mean->8 rescale; the cap is applied only at load time), so only TWO init
builds run. ``itypes`` is REBUILT from graph.nt here (not loaded from the cached
pickle, which predates ``parent_ids`` and would crash the decay weighting).

Each config folder output/dbpedia/<name>/ gets:
  model.kv (+ .vectors.npy)     trained KeyedVectors
  losses.json                   per-epoch finetuning training loss
  accuracies_per_epoch.json     per-epoch LogReg test acc summaries (standardize + raw),
                                including epoch 0 (the init), with the matching loss
  final_accs.json / init_accs.json  per-split accuracies at the last / 0th epoch
  params.json                   full training hyper-parameters + corpus/init metadata
  results.json                  everything above in one record (exp6-compatible schema)

Usage:
  .venv/bin/python scripts/_invest_clean_bound_full.py
  .venv/bin/python scripts/_invest_clean_bound_full.py --walks walks/all_walks_010.txt   # smoke test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

from _dbpedia_compare import (  # noqa: E402
    classic_init,
    concept_bound_vectors,
    load_eval_splits,
    load_instance_types,
)
from _dbpedia_investigate import (  # noqa: E402
    Model,
    cap_norms,
    evaluate_models,
    load_emb,
    save_emb,
    summarize,
)

ART = ROOT / "notebooks" / "dbpedia_investigate" / "artifacts"

ap = argparse.ArgumentParser()
ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks.txt")
ap.add_argument("--vocab", type=Path, default=None,
                help="corpus vocab pkl (default: <walks stem>_vocab.pkl)")
ap.add_argument("--graph", type=Path, default=ROOT / "dbpedia_graph" / "graph.nt")
ap.add_argument("--kind", default="p2")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--workers", type=int, default=20)
ap.add_argument("--cap", type=float, default=16.0)
ap.add_argument("--ancestor-decay", type=float, default=0.5)
ap.add_argument("--n-jobs", type=int, default=12)
ap.add_argument("--dim", type=int, default=200)
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--only", default=None,
                help="comma-separated config names to run (default: all four)")
ap.add_argument("--out-root", type=Path, default=ROOT / "output" / "dbpedia",
                help="where to write <config>/ folders (override for smoke tests)")
args = ap.parse_args()

DIM = args.dim
LENSES = ("standardize", "raw")
t_global = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t_global:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Corpus vocab + cached schema / stage-1 artifacts; itypes rebuilt from graph
# --------------------------------------------------------------------------- #
vocab_pkl = args.vocab or args.walks.with_name(args.walks.stem + "_vocab.pkl")
with vocab_pkl.open("rb") as fh:
    cv = pickle.load(fh)
freq, n_lines, n_tokens = cv["freq"], cv["n_lines"], cv["n_tokens"]
vocab_set = set(freq.keys())
log(f"corpus {args.walks.name}: {n_lines:,} lines, {n_tokens:,} tokens, {len(vocab_set):,} vocab")

splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
log(f"eval splits: {len(splits)}")

schema = pickle.loads((ART / "schema.pkl").read_bytes())
stage1 = load_emb(ART / f"stage1_{args.kind}")
used_norm = json.loads((ART / f"used_norm_{args.kind}.json").read_text())["used_norm"]
log(f"artifacts: {len(stage1)} stage1 codes, used_norm={used_norm:.3f}")

# Rebuild instance types from graph.nt so parent_ids is populated (the cached
# itypes.pkl predates parent_ids and would crash class_weight_mode='decay').
log("rebuilding instance types from graph.nt (populates parent_ids) ...")
itypes = load_instance_types(args.graph, schema)
log(f"itypes: {itypes.stats}")


# --------------------------------------------------------------------------- #
# Init builders
# --------------------------------------------------------------------------- #
def ensure_init(tag: str, **kw) -> Path:
    """Build & cache a bound init artifact built against THIS corpus' full vocab.

    Cached under artifacts/boundinit_<tag>_full_<kind>. cap16 and no-cap configs
    of the same mode reuse this single artifact (cap is applied at load time)."""
    path = ART / f"boundinit_{tag}_full_{args.kind}"
    keys_file = path.with_suffix(".keys.json")
    if keys_file.is_file():
        n = len(json.loads(keys_file.read_text()))
        log(f"init '{tag}' (full vocab): cached, {n:,} vecs")
        return path
    log(f"init '{tag}' (full vocab): streaming graph.nt ({kw}) ...")
    bound = concept_bound_vectors(
        args.graph, itypes, stage1, vocab_set,
        direction_tag="rolled", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
        target_norm=used_norm, **kw,
    )
    save_emb(path, bound)
    norms = np.array([float(np.linalg.norm(v)) for v in bound.values()])
    log(f"init '{tag}' (full vocab): {len(bound):,} vecs, norm median {np.median(norms):.2f} "
        f"p99 {np.percentile(norms, 99):.2f} max {norms.max():.1f}")
    del bound
    return path


def alpha_boundaries(start: float, end: float, epochs: int) -> list[float]:
    """epochs+1 boundary LRs for a linear start->end decay across the whole run."""
    return [start + (end - start) * i / epochs for i in range(epochs + 1)]


def evaluate(model) -> dict:
    kv = model.wv
    m = Model("ft", kv=kv, dim=kv.vectors.shape[1])
    rec = {}
    for lens in LENSES:
        accs = evaluate_models(splits, [m], scaling=lens, n_jobs=args.n_jobs, seed=args.seed)
        rec[lens] = {"accs": accs, "summary": summarize(accs)}
    return rec


# --------------------------------------------------------------------------- #
# One full finetune run -> writes output/dbpedia/<name>/{...}
# --------------------------------------------------------------------------- #
def run(name: str, init_path: Path, norm_policy: str, bounds: list[float],
        init_desc: str, class_weight_mode: str) -> None:
    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "results.json").is_file():
        log(f"[skip] {name}: results.json already present in {out_dir}")
        return

    t0 = time.time()
    model = Word2Vec(
        vector_size=DIM, window=args.window, sg=1, hs=0, negative=5, min_count=1,
        sample=0.0, alpha=bounds[0], min_alpha=bounds[-1], workers=args.workers,
        seed=args.seed, compute_loss=True,
    )
    model.build_vocab_from_freq(freq)
    model.corpus_count = n_lines
    model.corpus_total_words = n_tokens

    bound = load_emb(init_path)
    if norm_policy.startswith("cap:"):
        bound = cap_norms(bound, float(norm_policy.split(":")[1]))
    # "global": init already mean-rescaled at build time -> use as-is
    init_stats = classic_init(model, stage1, itypes, schema,
                              target_norm=used_norm, bound_vectors=bound)
    del bound
    log(f"{name}: init_stats {init_stats}")

    rows = [evaluate(model)]            # epoch 0 (init quality, before any training)
    losses = [0.0]
    s0 = rows[0]["standardize"]["summary"]
    log(f"{name} ep0(init): std all {s0['all']:.3f} (normal {s0['normal']:.3f} hard {s0['hard']:.3f})")

    corpus = LineSentence(str(args.walks))
    for ep in range(args.epochs):
        a0, a1 = bounds[ep], bounds[ep + 1]
        te = time.time()
        model.train(corpus, total_examples=n_lines, epochs=1,
                    start_alpha=a0, end_alpha=a1, compute_loss=True)
        loss = float(model.get_latest_training_loss())
        rec = evaluate(model)
        rows.append(rec)
        losses.append(loss)
        s = rec["standardize"]["summary"]
        log(f"{name} ep{ep + 1} lr[{a0:.4f}->{a1:.4f}] loss {loss:,.0f} ({time.time()-te:.0f}s): "
            f"std all {s['all']:.3f} (normal {s['normal']:.3f} hard {s['hard']:.3f})")

    # ----- save artifacts -----------------------------------------------------
    model.wv.save(str(out_dir / "model.kv"))

    per_epoch = [
        {"epoch": i, "loss": losses[i],
         **{lens: rows[i][lens]["summary"] for lens in LENSES}}
        for i in range(len(rows))
    ]
    params = {
        "name": name,
        "init": "bound",
        "init_desc": init_desc,
        "init_path": str(init_path.relative_to(ROOT)),
        "kind": args.kind,
        "class_weight_mode": class_weight_mode,
        "ancestor_decay": args.ancestor_decay if class_weight_mode == "decay" else None,
        "norm_policy": norm_policy,
        "cap": args.cap if norm_policy.startswith("cap:") else None,
        "alpha_start": bounds[0],
        "alpha_end": bounds[-1],
        "alpha_bounds_per_epoch": bounds,
        "epochs": args.epochs,
        "dim": DIM,
        "window": args.window,
        "sg": 1, "hs": 0, "negative": 5, "min_count": 1, "sample": 0.0,
        "seed": args.seed,
        "workers": args.workers,
        "target_norm": used_norm,
        "corpus": args.walks.name,
        "corpus_vocab_pkl": vocab_pkl.name,
        "corpus_lines": n_lines,
        "corpus_tokens": n_tokens,
        "corpus_vocab": len(vocab_set),
        "init_stats": init_stats,
        "eval": {"benchmark": "v1/dbpedia DLCC", "k": 5000, "n_splits": len(splits),
                 "lenses": list(LENSES), "logreg_C": 1.0, "logreg_max_iter": 2000},
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    (out_dir / "losses.json").write_text(
        json.dumps({"per_epoch_loss": losses,
                    "note": "loss[0]=0.0 placeholder for the pre-training init; "
                            "loss[i] = gensim training loss for epoch i"},
                   indent=2) + "\n", encoding="utf-8")
    (out_dir / "accuracies_per_epoch.json").write_text(
        json.dumps(per_epoch, indent=2) + "\n", encoding="utf-8")
    (out_dir / "final_accs.json").write_text(
        json.dumps({lens: rows[-1][lens]["accs"] for lens in LENSES}, indent=2) + "\n",
        encoding="utf-8")
    (out_dir / "init_accs.json").write_text(
        json.dumps({lens: rows[0][lens]["accs"] for lens in LENSES}, indent=2) + "\n",
        encoding="utf-8")
    results = {
        **params,
        "per_epoch": per_epoch,
        "losses": losses,
        "final_accs": {lens: rows[-1][lens]["accs"] for lens in LENSES},
        "init_accs": {lens: rows[0][lens]["accs"] for lens in LENSES},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    fin = per_epoch[-1]["standardize"]
    log(f"[done] {name} in {time.time() - t0:.0f}s -> std all {fin['all']:.3f} "
        f"(normal {fin['normal']:.3f} hard {fin['hard']:.3f}); saved -> {out_dir}")
    del model


def safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        log(f"[ERROR] {fn.__name__}{a}:\n{traceback.format_exc()}")
        return None


# --------------------------------------------------------------------------- #
# Configs (2x2: init mode x norm policy), all protected LR 0.0025 -> 0.0001
# --------------------------------------------------------------------------- #
E = args.epochs
CAP = f"cap:{args.cap:g}"
DECAY_KW = dict(class_weight_mode="decay", ancestor_decay=args.ancestor_decay)
SPEC_KW = dict(class_weight_mode="most_specific")
PROTECTED = alpha_boundaries(0.0025, 0.0001, E)

CONFIGS = {
    "p2_bound_decay_cap16": dict(
        init=lambda: ensure_init("decay", **DECAY_KW), norm_policy=CAP,
        bounds=PROTECTED, mode="decay",
        init_desc=f"decay-weighted concept-bound init (full vocab, ancestor_decay="
                  f"{args.ancestor_decay:g}), per-row norm cap @ {args.cap:g}, "
                  "protected LR 0.0025->0.0001"),
    "p2_bound_specific_cap16": dict(
        init=lambda: ensure_init("specific", **SPEC_KW), norm_policy=CAP,
        bounds=PROTECTED, mode="most_specific",
        init_desc="most-specific concept-bound init (full vocab, leaf class only), "
                  f"per-row norm cap @ {args.cap:g}, protected LR 0.0025->0.0001"),
    "p2_bound_decay": dict(
        init=lambda: ensure_init("decay", **DECAY_KW), norm_policy="global",
        bounds=PROTECTED, mode="decay",
        init_desc=f"decay-weighted concept-bound init (full vocab, ancestor_decay="
                  f"{args.ancestor_decay:g}), global mean->{used_norm:g} rescale "
                  "(no cap), protected LR 0.0025->0.0001"),
    "p2_bound_specific": dict(
        init=lambda: ensure_init("specific", **SPEC_KW), norm_policy="global",
        bounds=PROTECTED, mode="most_specific",
        init_desc="most-specific concept-bound init (full vocab, leaf class only), "
                  f"global mean->{used_norm:g} rescale (no cap), "
                  "protected LR 0.0025->0.0001"),
}

selected = args.only.split(",") if args.only else list(CONFIGS)
log(f"running configs: {selected}")
for cname in selected:
    cfg = CONFIGS[cname]
    log(f"\n========== {cname} ==========")
    init_path = cfg["init"]()
    if init_path is None:
        log(f"[ERROR] {cname}: init build failed, skipping")
        continue
    safe(run, cname, init_path, cfg["norm_policy"], cfg["bounds"], cfg["init_desc"], cfg["mode"])

log("ALL DONE")
