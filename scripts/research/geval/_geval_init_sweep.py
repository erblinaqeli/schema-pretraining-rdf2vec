#!/usr/bin/env python3
"""Cheap init-space lambda sweep for the damped bound-offset init.

Builds the class-mean init and the bound init ONCE, then for a range of lambda
applies mixed_init in place and clusters the (frozen) GEval entities. Because
class and bound codes share one space at init, mixing here is geometrically
valid (the finetuned-space blend is not). This maps GEval ARI vs lambda in ~5
min so we only spend a full finetune run on a promising lambda.

  uv run python scripts/_geval_init_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_S = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
if str(_S) not in sys.path:
    sys.path.insert(0, str(_S))

import _geval  # noqa: E402
from _dbpedia_compare import (  # noqa: E402
    build_model_with_vocab,
    concept_bound_vectors,
    corpus_vocab,
    ensure_walks,
    load_instance_types,
    load_schema,
    mixed_init,
    normalized_stage1_vectors,
    pretrain_protograph,
    write_protographs,
)

ROOT = _S.parent
LAMS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0]
DATASETS = ["CC", "CCB", "CAMAP"]


def main() -> None:
    schema = load_schema(ROOT / "dbpedia_graph" / "ontology.nt")
    protos = write_protographs(schema, ROOT / "notebooks" / "dbpedia_compare" / "protographs")
    pw = ensure_walks(
        protos["p2"],
        ROOT / "notebooks" / "dbpedia_compare" / "protographs" / "walks_p2.txt",
        walks_per_entity=200, depth=3, seed=42,
    )
    itypes = load_instance_types(ROOT / "dbpedia_graph" / "graph.nt", schema)
    freq, n_lines, n_tokens = corpus_vocab(
        ROOT / "walks" / "all_walks.txt", ROOT / "walks" / "all_walks_vocab.pkl"
    )
    pre = pretrain_protograph(pw, dim=200, epochs=5, seed=42)
    stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=8.0)
    del pre
    model = build_model_with_vocab(
        freq, n_lines, n_tokens, dim=200, alpha=0.0025, min_alpha=0.0001, seed=42, workers=20
    )
    bound = concept_bound_vectors(
        ROOT / "dbpedia_graph" / "graph.nt", itypes, stage1, model.wv.key_to_index,
        direction_tag="rolled", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, target_norm=used_norm,
    )
    print("  setup done; sweeping lambda (init-only clustering)\n", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler, normalize

    # CC only, with metrics that (unlike k-means ARI) respond to lambda at init:
    #  margin = mean within-class cosine - mean between-class cosine (class geometry)
    #  logreg = supervised balanced accuracy (is class linearly readable?)
    #  ari    = unsupervised k-means (the actual GEval metric; degenerate at init)
    uris, labels = _geval.load_gold("CC")
    keep = [(u, l) for u, l in zip(uris, labels) if u in model.wv.key_to_index]
    y = np.asarray([l for _, l in keep])
    rng = np.random.default_rng(0)
    ci = np.where(y == 1)[0]; co = np.where(y == 0)[0]
    sc = rng.choice(ci, min(800, len(ci)), replace=False)
    so = rng.choice(co, min(800, len(co)), replace=False)
    wv = model.wv
    table = {}
    for lam in LAMS:
        mixed_init(model, stage1, itypes, schema,
                   target_norm=used_norm, bound_vectors=bound, mix_lambda=lam)
        X = np.vstack([wv[u] for u, _ in keep])
        Xn = normalize(X)
        within = (float((Xn[sc] @ Xn[sc].T).mean()) + float((Xn[so] @ Xn[so].T).mean())) / 2
        between = float((Xn[sc] @ Xn[so].T).mean())
        logreg = cross_val_score(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            StandardScaler().fit_transform(X), y, cv=3, scoring="balanced_accuracy").mean()
        ari = _geval.cluster_and_score(X, y, 2)["ARI"]
        table[lam] = {"margin": round(within - between, 3),
                      "logreg_balacc": round(float(logreg), 3),
                      "kmeans_ARI": round(ari, 3)}
        print(f"lambda={lam:.2f}  margin={table[lam]['margin']:+.3f}  "
              f"logreg={table[lam]['logreg_balacc']:.3f}  ARI={table[lam]['kmeans_ARI']:+.3f}", flush=True)

    df = pd.DataFrame(table).T
    df.index.name = "lambda"
    out = ROOT / "output" / "dbpedia" / "geval_init_lambda_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out)
    print("\n=== CC init geometry vs lambda (0=class, 1=bound) ===")
    print(df.to_string())
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
