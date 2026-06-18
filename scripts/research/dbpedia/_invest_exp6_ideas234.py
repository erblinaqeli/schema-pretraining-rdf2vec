#!/usr/bin/env python3
"""Experiment 6: push the capped concept-bound init higher on DBpedia via three
INDEPENDENT levers (no fusion). Each run is a one-variable change from the
winning control ``p2_bound_cap16`` + protected LR 0.0025:

  Idea 2  count compression at construction  -- replace the post-hoc per-row
          norm cap with a monotone (sqrt / log1p) compression of the accumulated
          row norm, baked into the init, so the hub-norm tail is tamed at the
          source and no hard clip is needed (finetune with policy "global").
  Idea 3  re-open the learning rate          -- the cap removed the tail that
          forced LR 0.0025; sweep higher constant-decay LRs and a
          warmup-then-decay schedule on the SAME cached cap16 init.
  Idea 4  specificity-weighted class init    -- IDF-weight ancestor classes in
          the alpha class-mean term (rarer/more-specific types weigh more),
          keeping cap16 + protected LR.

Runs on the 10% corpus (same testbed as exp3) so numbers are comparable to the
exp3 references: vanilla 0.805 / p2_classic 0.832 / p2_bound_cap16 0.847
(all-splits, standardize lens). Every run saves the trained .kv model plus
per-epoch accuracies (standardize + raw, incl. epoch 0 init) and per-epoch
training loss.

Usage:
  .venv/bin/python scripts/_invest_exp6_ideas234.py            # 10% corpus
  .venv/bin/python scripts/_invest_exp6_ideas234.py --walks walks/all_walks.txt
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import traceback
from collections import Counter
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
OUT = ROOT / "notebooks" / "dbpedia_investigate" / "exp6_ideas234"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks_010.txt")
ap.add_argument("--graph", type=Path, default=ROOT / "dbpedia_graph" / "graph.nt")
ap.add_argument("--kind", default="p2")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--workers", type=int, default=20)
ap.add_argument("--cap", type=float, default=16.0)
ap.add_argument("--n-jobs", type=int, default=12)
args = ap.parse_args()

DIM = 200
LENSES = ("standardize", "raw")
t_global = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t_global:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Corpus + cached schema/itypes/stage-1 artifacts (built by _invest_build_artifacts)
# --------------------------------------------------------------------------- #
vocab_pkl = args.walks.with_name(args.walks.stem + "_vocab.pkl")
with vocab_pkl.open("rb") as fh:
    cv = pickle.load(fh)
freq, n_lines, n_tokens = cv["freq"], cv["n_lines"], cv["n_tokens"]
vocab_set = set(freq.keys())
log(f"corpus {args.walks.name}: {n_lines:,} lines, {n_tokens:,} tokens, {len(vocab_set):,} vocab")

splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
res_path = OUT / "results.json"
results = json.loads(res_path.read_text()) if res_path.is_file() else {}

schema = pickle.loads((ART / "schema.pkl").read_bytes())
itypes = pickle.loads((ART / "itypes.pkl").read_bytes())
stage1 = load_emb(ART / f"stage1_{args.kind}")
used_norm = json.loads((ART / f"used_norm_{args.kind}.json").read_text())["used_norm"]
log(f"artifacts: {len(stage1)} stage1 codes, used_norm={used_norm:.3f}")


# --------------------------------------------------------------------------- #
# Init builders (graph streams; cached as artifacts so a rerun is free)
# --------------------------------------------------------------------------- #
def build_idf_weights() -> dict[int, float]:
    """IDF specificity weight per class id, document-frequency over vocab entities'
    materialized ancestor sets: idf(c) = log(N / df(c)); rarer/specific -> higher."""
    mat_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
    df: Counter[int] = Counter()
    n = 0
    for ent, cids in itypes.ent_types.items():
        if ent not in vocab_set:
            continue
        m = mat_cache.get(cids)
        if m is None:
            m = itypes.materialized(cids)
            mat_cache[cids] = m
        n += 1
        for c in m:
            df[c] += 1
    idf = {c: math.log(n / d) for c, d in df.items() if d > 0}
    vals = list(idf.values())
    log(f"idf: {n:,} vocab entities, {len(idf)} classes, weight min {min(vals):.2f} "
        f"max {max(vals):.2f} (generic types ~0, specific ~max)")
    return idf


def ensure_init(tag: str, **kw) -> Path:
    """Build & cache a bound init artifact (mean-rescaled to used_norm)."""
    path = ART / f"boundinit_{tag}_{args.kind}"
    if path.with_suffix(".keys.json").is_file():
        log(f"init '{tag}': cached")
        return path
    log(f"init '{tag}': streaming graph.nt ({kw}) ...")
    bound = concept_bound_vectors(
        args.graph, itypes, stage1, vocab_set,
        direction_tag="rolled", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
        target_norm=used_norm, **kw,
    )
    save_emb(path, bound)
    norms = np.array([float(np.linalg.norm(v)) for v in bound.values()])
    log(f"init '{tag}': {len(bound):,} vecs, norm median {np.median(norms):.2f} "
        f"p99 {np.percentile(norms, 99):.2f} max {norms.max():.1f}")
    del bound
    return path


# --------------------------------------------------------------------------- #
# Finetune (explicit per-epoch loop -> per-epoch acc + training loss)
# --------------------------------------------------------------------------- #
def alpha_boundaries(start: float, end: float, epochs: int) -> list[float]:
    """epochs+1 boundary LRs for a linear start->end decay across the whole run."""
    return [start + (end - start) * i / epochs for i in range(epochs + 1)]


def evaluate(model) -> dict:
    kv = model.wv
    m = Model("ft", kv=kv, dim=kv.vectors.shape[1])
    rec = {}
    for lens in LENSES:
        accs = evaluate_models(splits, [m], scaling=lens, n_jobs=args.n_jobs)
        rec[lens] = {"accs": accs, "summary": summarize(accs)}
    return rec


def run(name: str, init_path: Path, norm_policy: str, bounds: list[float]) -> None:
    if name in results:
        log(f"[skip] {name} cached")
        return
    t0 = time.time()
    model = Word2Vec(
        vector_size=DIM, window=5, sg=1, hs=0, negative=5, min_count=1, sample=0.0,
        alpha=bounds[0], min_alpha=bounds[-1], workers=args.workers, seed=42,
        compute_loss=True,
    )
    model.build_vocab_from_freq(freq)
    model.corpus_count = n_lines
    model.corpus_total_words = n_tokens

    bound = load_emb(init_path)
    if norm_policy.startswith("cap:"):
        bound = cap_norms(bound, float(norm_policy.split(":")[1]))
    # "global": init already mean-rescaled at build time -> use as-is
    stats = classic_init(model, stage1, itypes, schema, target_norm=used_norm, bound_vectors=bound)
    del bound
    log(f"{name}: init_stats {stats}")

    rows = [evaluate(model)]            # epoch 0 (init quality, before any training)
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
        "name": name, "init_path": str(init_path.relative_to(ROOT)),
        "norm_policy": norm_policy, "alpha_bounds": bounds, "epochs": args.epochs,
        "corpus": args.walks.name, "seconds": round(time.time() - t0, 1),
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


# --------------------------------------------------------------------------- #
# Run the sweep. Order matters for fail-fast: the control + idea-3 runs reuse the
# already-cached plain init (no graph stream), so any bug in run() surfaces in a
# couple of minutes BEFORE the 3 graph-stream init builds. Builds are lazy/cached.
# --------------------------------------------------------------------------- #
plain_init = ART / f"boundinit_{args.kind}"          # cached plain init (build_artifacts)
CAP = f"cap:{args.cap:g}"
E = args.epochs

# Control: cap16 + protected LR 0.0025 (reproduce the exp3/exp4 winner, with loss curve)
safe(run, "B0_cap16_lr025", plain_init, CAP, alpha_boundaries(0.0025, 0.0001, E))

# Idea 3: re-open the LR on the SAME cap16 init (no rebuild)
safe(run, "E3a_cap16_lr005", plain_init, CAP, alpha_boundaries(0.005, 0.0005, E))
safe(run, "E3b_cap16_lr01", plain_init, CAP, alpha_boundaries(0.01, 0.0005, E))
safe(run, "E3c_cap16_warmup", plain_init, CAP, [0.0005, 0.005, 0.005, 0.0025, 0.001, 0.0001][:E + 1])

# Idea 2: compression baked into init, no hard cap, protected LR (graph rebuilds).
# NB: the compression must be passed as the concept_bound_vectors kwarg `compress=`;
# the tag is only the artifact filename.
try:
    log1p_init = ensure_init("log1p", compress="log1p")
    safe(run, "E2a_log1p_lr025", log1p_init, "global", alpha_boundaries(0.0025, 0.0001, E))
except Exception:
    log(f"[ERROR] log1p build/run:\n{traceback.format_exc()}")
try:
    sqrt_init = ensure_init("sqrt", compress="sqrt")
    safe(run, "E2b_sqrt_lr025", sqrt_init, "global", alpha_boundaries(0.0025, 0.0001, E))
except Exception:
    log(f"[ERROR] sqrt build/run:\n{traceback.format_exc()}")

# Idea 4: IDF-weighted alpha class-mean term, cap16 + protected LR (graph rebuild)
try:
    idf_init = ensure_init("idf", alpha_weights=build_idf_weights())
    safe(run, "E4a_idf_cap16_lr025", idf_init, CAP, alpha_boundaries(0.0025, 0.0001, E))
except Exception:
    log(f"[ERROR] idf build/run:\n{traceback.format_exc()}")

log("ALL DONE")
