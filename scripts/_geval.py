"""GEval clustering evaluation for DBpedia RDF2Vec variants.

Reusable helpers behind ``notebooks/geval_cc.ipynb``. Mirrors the entity
clustering protocol of MASCHInE (Hubert et al. 2023, Sec. 4.2) on the GEval
gold standards: CC / CCB (Cities & Countries) and CAMAP (5 classes).

Gold standards are the ``*_cluster.tsv`` files shipped with GEval
(mariaangelapellegrino/Evaluation-Framework), downloaded to ``v1/geval/``.
Columns: ``DBpedia_URI``, ``cluster``, ``DBpedia_URI_Base32``.

All embedding variants in ``notebooks/dbpedia_compare/`` share one 1.28M-key
vocab of full DBpedia URIs, so coverage is identical across variants and we
only need to filter the gold standard against the vocab once.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import normalize as l2_normalize
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent

# GEval clustering gold standards (relative to repo root) and their k.
GEVAL_DATASETS = {
    "CC": "v1/geval/citiesAndCountries_cluster.tsv",
    "CCB": "v1/geval/cities2000AndCountries_cluster.tsv",
    "CAMAP": "v1/geval/citiesMoviesAlbumsCompaniesUni_cluster.tsv",
}
GEVAL_K = {"CC": 2, "CCB": 2, "CAMAP": 5}

# DBpedia RDF2Vec embeddings to compare (relative to repo root). All share vocab.
VARIANTS = {
    "vanilla": "notebooks/dbpedia_compare/vanilla.kv",
    "p1_classic": "notebooks/dbpedia_compare/p1_classic.kv",
    "p1_bound": "notebooks/dbpedia_compare/p1_bound.kv",
    "p2_classic": "notebooks/dbpedia_compare/p2_classic.kv",
    "p3_classic": "notebooks/dbpedia_investigate/p3_classic_FULL.kv",
    "p2_bound": "notebooks/dbpedia_compare/p2_bound.kv",
    "p3_bound": "notebooks/dbpedia_compare/p3_bound.kv",
    # cap16 = per-row norm cap of 16 on the bound init (unfreezes hubs so the
    # protected finetune LR can move them); winning bound variant on DLCC.
    # Only p2/p3 were built as _FULL (shared 1.28M vocab); no p1_bound_cap16 exists.
    "p2_bound_cap16": "notebooks/dbpedia_investigate/p2_bound_cap16_lr025_FULL.kv",
    "p3_bound_cap16": "notebooks/dbpedia_investigate/p3_bound_cap16_lr025_FULL.kv",
    # damped bound-offset init: (1-λ)·class + λ·bound at INIT, then finetuned (λ=0.5)
    "p2_bound_mixed_0.5": "output/dbpedia/p2_bound_mixed_lam05/p2_bound_mixed_0.5.kv",
    "p2_bound_mixed_0.1": "output/dbpedia/p2_bound_mixed_lam01/p2_bound_mixed_0.1.kv",
}

# Metric name -> sklearn function. Matches MASCHInE Table 3 (+ NMI from Fig. 3).
METRICS = {
    "ARI": adjusted_rand_score,
    "AMI": adjusted_mutual_info_score,
    "NMI": normalized_mutual_info_score,
    "VM": v_measure_score,
    "FM": fowlkes_mallows_score,
    "H": homogeneity_score,
    "C": completeness_score,
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_gold(name: str) -> tuple[list[str], np.ndarray]:
    """Return (uris, integer labels) for a GEval dataset name (CC/CCB/CAMAP)."""
    path = _resolve(GEVAL_DATASETS[name])
    uris, labels = [], []
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for row in reader:
            uris.append(row[0])
            labels.append(int(row[1]))
    return uris, np.asarray(labels)


def load_vocab(variant: str = "vanilla") -> set[str]:
    """Load just the key set of a variant (all variants share one vocab)."""
    kv = KeyedVectors.load(str(_resolve(VARIANTS[variant])), mmap="r")
    return set(kv.key_to_index.keys())


def coverage_report(name: str, vocab: set[str]) -> pd.DataFrame:
    """Per-cluster and overall coverage of a GEval dataset against ``vocab``."""
    uris, labels = load_gold(name)
    rows = []
    for c in sorted(set(labels)):
        idx = labels == c
        total = int(idx.sum())
        present = sum(1 for u, l in zip(uris, labels) if l == c and u in vocab)
        rows.append({"cluster": c, "covered": present, "total": total,
                     "coverage": present / total})
    present_all = sum(1 for u in uris if u in vocab)
    rows.append({"cluster": "ALL", "covered": present_all, "total": len(uris),
                 "coverage": present_all / len(uris)})
    return pd.DataFrame(rows)


def embed_subset(kv: KeyedVectors, uris: list[str], labels: np.ndarray):
    """Filter (uris, labels) to those in the vocab; return (X, y, kept_uris)."""
    keep = [(u, l) for u, l in zip(uris, labels) if u in kv.key_to_index]
    kept_uris = [u for u, _ in keep]
    y = np.asarray([l for _, l in keep])
    X = np.vstack([kv[u] for u in kept_uris])
    return X, y, kept_uris


def _fit_labels(X: np.ndarray, k: int, algo: str, random_state: int) -> np.ndarray:
    if algo == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
    if algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)
    raise ValueError(f"unknown algo {algo!r}")


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {m: float(fn(y_true, y_pred)) for m, fn in METRICS.items()}


def cluster_and_score(X, y_true, k, algo="kmeans", normalize=False,
                      random_state=42) -> dict[str, float]:
    """Cluster X into k groups and score against y_true. ``normalize`` L2s rows."""
    Xc = l2_normalize(X) if normalize else X
    y_pred = _fit_labels(Xc, k, algo, random_state)
    return score(y_true, y_pred)


def mix_subset(uris, class_kv, bound_kv, lam: float):
    """Damped bound-offset blend of two (finetuned) spaces, per entity.

    For each entity e present in both: scale v_bound to ||v_class|| (so norm
    doesn't dominate the mix), then v_mixed = (1-lam)*v_class + lam*v_bound_scaled.
    Returns (X, kept_uris). NOTE: this blends two *finetuned* embedding spaces
    post-hoc — it is not the init-then-finetune method.
    """
    kept = [u for u in uris
            if u in class_kv.key_to_index and u in bound_kv.key_to_index]
    rows = []
    for u in kept:
        c = class_kv[u]
        b = bound_kv[u]
        cn = float(np.linalg.norm(c))
        bn = float(np.linalg.norm(b))
        b_scaled = b / bn * cn if bn > 0 else b
        rows.append((1.0 - lam) * c + lam * b_scaled)
    return np.vstack(rows), kept


def evaluate_mix(name: str, class_variant: str = "p2_classic",
                 bound_variant: str = "p2_bound", lam: float = 0.5,
                 algo: str = "kmeans", normalize: bool = False,
                 random_state: int = 42) -> dict[str, float]:
    """Cluster the damped bound-offset blend (class⊕bound at lam) on a dataset."""
    uris, labels = load_gold(name)
    ckv = KeyedVectors.load(str(_resolve(VARIANTS[class_variant])), mmap="r")
    bkv = KeyedVectors.load(str(_resolve(VARIANTS[bound_variant])), mmap="r")
    keep_idx = [i for i, u in enumerate(uris)
                if u in ckv.key_to_index and u in bkv.key_to_index]
    y = labels[keep_idx]
    X, _ = mix_subset([uris[i] for i in keep_idx], ckv, bkv, lam)
    metrics = cluster_and_score(X, y, GEVAL_K[name], algo=algo,
                                normalize=normalize, random_state=random_state)
    metrics["n"] = len(y)
    return metrics


def evaluate(name: str, variants: list[str] | None = None, algo: str = "kmeans",
             normalize: bool = False, random_state: int = 42) -> pd.DataFrame:
    """Run clustering for each variant on a GEval dataset; one row per variant.

    Only entities present in the (shared) vocab are clustered. Returns a
    DataFrame indexed by variant with the GEval metrics plus ``n`` (entities
    clustered) and ``coverage``.
    """
    variants = variants or list(VARIANTS)
    uris, labels = load_gold(name)
    k = GEVAL_K[name]
    rows = {}
    for v in variants:
        kv = KeyedVectors.load(str(_resolve(VARIANTS[v])), mmap="r")
        X, y, kept = embed_subset(kv, uris, labels)
        metrics = cluster_and_score(X, y, k, algo=algo, normalize=normalize,
                                    random_state=random_state)
        metrics["n"] = len(y)
        metrics["coverage"] = len(y) / len(uris)
        rows[v] = metrics
    cols = list(METRICS) + ["n", "coverage"]
    return pd.DataFrame(rows).T[cols]


# ---------------------------------------------------------------------------
# Classification (GEval task 1) — supervised, entity embedding -> categorical
# label, scored by accuracy under k-fold CV. Mirrors the GEval protocol
# (Pellegrino et al.): a small panel of classifiers (Naive Bayes, k-NN k=3,
# C4.5 decision tree, linear SVM), 10-fold stratified CV, accuracy per model.
# Gold files: evaluation_framework/Classification/data/*.tsv -> v1/geval/classification/.
# Each TSV carries a `DBpedia_URI` column (the entity) and a `label` column
# (the categorical target, e.g. high/medium/low). Only entities in the shared
# vocab are usable; the rest are GEval's "ignored data".
# ---------------------------------------------------------------------------

CLASSIFICATION_DATASETS = {
    "AAUP": "v1/geval/classification/AAUP.tsv",
    "Cities": "v1/geval/classification/Cities.tsv",
    "Forbes": "v1/geval/classification/Forbes.tsv",
    "MetacriticAlbums": "v1/geval/classification/MetacriticAlbums.tsv",
    "MetacriticMovies": "v1/geval/classification/MetacriticMovies.tsv",
}

# classifier name -> zero-arg factory. Matches the GEval model panel; SVM uses a
# fixed linear C=1.0 (GEval grids C — we keep one setting for a fair, cheap
# cross-variant comparison) and C4.5 is sklearn's CART decision tree.
CLASSIFIERS = {
    "NB": lambda: GaussianNB(),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=3),
    "C4.5": lambda: DecisionTreeClassifier(random_state=42),
    "SVM": lambda: SVC(kernel="linear", C=1.0),
}


def load_classification_gold(name: str) -> tuple[list[str], np.ndarray]:
    """Return (uris, string labels) for a GEval classification dataset."""
    path = _resolve(CLASSIFICATION_DATASETS[name])
    # utf-8-sig strips the BOM some of these files carry on the first header.
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str)
    df = df[["DBpedia_URI", "label"]].dropna()
    return df["DBpedia_URI"].tolist(), df["label"].to_numpy()


def classification_coverage(name: str, vocab: set[str]) -> pd.DataFrame:
    """Per-label and overall coverage of a classification dataset vs ``vocab``."""
    uris, labels = load_classification_gold(name)
    rows = []
    for c in sorted(set(labels)):
        idx = labels == c
        total = int(idx.sum())
        present = sum(1 for u, l in zip(uris, labels) if l == c and u in vocab)
        rows.append({"label": c, "covered": present, "total": total,
                     "coverage": present / total})
    present_all = sum(1 for u in uris if u in vocab)
    rows.append({"label": "ALL", "covered": present_all, "total": len(uris),
                 "coverage": present_all / len(uris)})
    return pd.DataFrame(rows)


def _classify_one(X, y, n_splits: int, normalize: bool, random_state: int) -> dict[str, float]:
    """Mean CV accuracy per classifier for one (X, y). Drops classes too small
    to stratify, and caps folds at the smallest surviving class count."""
    if normalize:
        X = l2_normalize(X)
    # stratified CV needs >=2 members per class and folds <= smallest class.
    uniq, counts = np.unique(y, return_counts=True)
    keep_classes = set(uniq[counts >= 2])
    mask = np.array([c in keep_classes for c in y])
    X, y = X[mask], y[mask]
    if len(y) == 0:
        return {c: float("nan") for c in CLASSIFIERS} | {"n": 0}
    min_class = int(np.unique(y, return_counts=True)[1].min())
    k = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    out = {}
    for cname, make in CLASSIFIERS.items():
        scores = cross_val_score(make(), X, y, cv=cv, scoring="accuracy")
        out[cname] = float(scores.mean())
    out["n"] = int(len(y))
    return out


def evaluate_classification(name: str, variants: list[str] | None = None,
                            n_splits: int = 10, normalize: bool = False,
                            random_state: int = 42) -> pd.DataFrame:
    """Per-variant classification accuracy on a GEval dataset; one row per variant.

    Columns: one mean-CV-accuracy column per classifier in ``CLASSIFIERS``,
    plus ``mean`` (average across classifiers), ``n`` (entities used after
    coverage + stratifiability filtering) and ``coverage`` (n / |gold|).
    """
    variants = variants or list(VARIANTS)
    uris, labels = load_classification_gold(name)
    rows = {}
    for v in variants:
        kv = KeyedVectors.load(str(_resolve(VARIANTS[v])), mmap="r")
        keep = [(u, l) for u, l in zip(uris, labels) if u in kv.key_to_index]
        if keep:
            kept_uris = [u for u, _ in keep]
            y = np.asarray([l for _, l in keep])
            X = np.vstack([kv[u] for u in kept_uris])
            acc = _classify_one(X, y, n_splits, normalize, random_state)
        else:
            acc = {c: float("nan") for c in CLASSIFIERS} | {"n": 0}
        clf_cols = [c for c in CLASSIFIERS]
        acc["mean"] = float(np.nanmean([acc[c] for c in clf_cols]))
        acc["coverage"] = acc["n"] / len(uris)
        rows[v] = acc
    cols = list(CLASSIFIERS) + ["mean", "n", "coverage"]
    return pd.DataFrame(rows).T[cols]


# ---------------------------------------------------------------------------
# Regression (GEval task 2) — same 5 datasets as classification, but predict
# the numeric `rating` column instead of the categorical `label`. Scored by
# RMSE (lower is better) under 10-fold CV with a small regressor panel.
# The gold files are shared with classification (the TSVs carry both columns).
# ---------------------------------------------------------------------------

REGRESSION_DATASETS = CLASSIFICATION_DATASETS  # same TSVs, different target col

# regressor name -> factory. Mirrors the classifier panel: linear model,
# k-NN (k=3), CART tree (sklearn's stand-in for GEval's M5), linear SVR.
REGRESSORS = {
    "LR": lambda: LinearRegression(),
    "KNN": lambda: KNeighborsRegressor(n_neighbors=3),
    "C4.5": lambda: DecisionTreeRegressor(random_state=42),
    "SVM": lambda: SVR(kernel="linear", C=1.0),
}


def load_regression_gold(name: str) -> tuple[list[str], np.ndarray]:
    """Return (uris, float ratings) for a GEval regression dataset."""
    path = _resolve(REGRESSION_DATASETS[name])
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str)
    df = df[["DBpedia_URI", "rating"]].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["DBpedia_URI", "rating"])
    return df["DBpedia_URI"].tolist(), df["rating"].to_numpy(dtype=float)


def evaluate_regression(name: str, variants: list[str] | None = None,
                        n_splits: int = 10, normalize: bool = False,
                        random_state: int = 42) -> pd.DataFrame:
    """Per-variant regression RMSE on a GEval dataset; one row per variant.

    Columns: one RMSE column per regressor in ``REGRESSORS`` (lower is better),
    plus ``mean`` (average across regressors), ``n`` and ``coverage``.
    """
    variants = variants or list(VARIANTS)
    uris, ratings = load_regression_gold(name)
    rows = {}
    for v in variants:
        kv = KeyedVectors.load(str(_resolve(VARIANTS[v])), mmap="r")
        keep = [(u, r) for u, r in zip(uris, ratings) if u in kv.key_to_index]
        out = {}
        if keep:
            kept_uris = [u for u, _ in keep]
            y = np.asarray([r for _, r in keep], dtype=float)
            X = np.vstack([kv[u] for u in kept_uris])
            if normalize:
                X = l2_normalize(X)
            cv = KFold(n_splits=min(n_splits, len(y)), shuffle=True,
                       random_state=random_state)
            for rname, make in REGRESSORS.items():
                scores = cross_val_score(make(), X, y, cv=cv,
                                         scoring="neg_root_mean_squared_error")
                out[rname] = float(-scores.mean())
            out["n"] = len(y)
        else:
            out = {r: float("nan") for r in REGRESSORS} | {"n": 0}
        out["mean"] = float(np.nanmean([out[r] for r in REGRESSORS]))
        out["coverage"] = out["n"] / len(uris)
        rows[v] = out
    cols = list(REGRESSORS) + ["mean", "n", "coverage"]
    return pd.DataFrame(rows).T[cols]


# ---------------------------------------------------------------------------
# Semantic analogies (GEval task) — pure vector arithmetic, no ML. Each line is
# a quadruplet (a, b, c, d) of DBpedia URIs; predict d via 3CosAdd: the nearest
# neighbour (cosine) to (b - a + c) over the FULL vocab, excluding a/b/c. Scored
# by top-k accuracy. Mirrors GEval's semanticAnalogies_model.py.
# Gold: evaluation_framework/SemanticAnalogies/data/*.txt -> v1/geval/analogies/.
# ---------------------------------------------------------------------------

ANALOGY_DATASETS = {
    "capital_country": "v1/geval/analogies/capital_country_entities.txt",
    "all_capital_country": "v1/geval/analogies/all_capital_country_entities.txt",
    "currency": "v1/geval/analogies/currency_entities.txt",
    "city_state": "v1/geval/analogies/city_state_entities.txt",
}


def load_analogies(name: str) -> list[tuple[str, str, str, str]]:
    """Return the (a, b, c, d) URI quadruplets of a GEval analogy dataset."""
    path = _resolve(ANALOGY_DATASETS[name])
    quads = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 4:
                quads.append(tuple(parts))
    return quads


def analogy_coverage(name: str, vocab: set[str]) -> dict:
    """How many quadruplets are fully answerable (all 4 URIs in ``vocab``)."""
    quads = load_analogies(name)
    answerable = sum(1 for q in quads if all(e in vocab for e in q))
    return {"dataset": name, "total": len(quads), "answerable": answerable,
            "coverage": answerable / len(quads) if quads else float("nan")}


def _normed_matrix(kv: KeyedVectors) -> np.ndarray:
    """Full L2-normalized embedding matrix (rows aligned to kv.key_to_index)."""
    M = np.asarray(kv.vectors, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def evaluate_analogies(name: str, variants: list[str] | None = None,
                       top_k: tuple[int, ...] = (1, 2),
                       batch_size: int = 256) -> pd.DataFrame:
    """Per-variant 3CosAdd top-k accuracy on a GEval analogy dataset.

    Only quadruplets whose four URIs are all in the vocab are scored. The
    candidate set is the FULL vocab (a/b/c excluded per query), as in GEval.
    Columns: one ``top{k}`` accuracy column per k, plus ``n`` (quads scored)
    and ``coverage`` (n / |quads|).
    """
    variants = variants or list(VARIANTS)
    quads = load_analogies(name)
    maxk = max(top_k)
    rows = {}
    for v in variants:
        kv = KeyedVectors.load(str(_resolve(VARIANTS[v])), mmap="r")
        idx = kv.key_to_index
        usable = [q for q in quads if all(e in idx for e in q)]
        hits = {k: 0 for k in top_k}
        if usable:
            M = _normed_matrix(kv)
            for start in range(0, len(usable), batch_size):
                batch = usable[start:start + batch_size]
                A = M[[idx[a] for a, b, c, d in batch]]
                B = M[[idx[b] for a, b, c, d in batch]]
                C = M[[idx[c] for a, b, c, d in batch]]
                Q = B - A + C
                Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-12)
                sims = Q @ M.T  # (batch, |vocab|)
                for i, (a, b, c, d) in enumerate(batch):
                    sims[i, idx[a]] = -np.inf
                    sims[i, idx[b]] = -np.inf
                    sims[i, idx[c]] = -np.inf
                part = np.argpartition(-sims, maxk, axis=1)[:, :maxk]
                for i, (a, b, c, d) in enumerate(batch):
                    order = part[i][np.argsort(-sims[i, part[i]])]
                    di = idx[d]
                    for k in top_k:
                        if di in order[:k]:
                            hits[k] += 1
        n = len(usable)
        row = {f"top{k}": (hits[k] / n if n else float("nan")) for k in top_k}
        row["n"] = n
        row["coverage"] = n / len(quads) if quads else float("nan")
        rows[v] = row
    cols = [f"top{k}" for k in top_k] + ["n", "coverage"]
    return pd.DataFrame(rows).T[cols]
