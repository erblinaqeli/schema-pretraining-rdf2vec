"""DLCC "correlations and shortcuts" analysis on the *real* DBpedia gold standard.

Reproduces the argument of Portisch & Paulheim (DLCC, Sec. 6.3 / Fig. 6) on our
own ``v1/dbpedia`` splits: a complex class constructor (e.g. tc07 ``∃dbo:team.T``)
is, on DBpedia, statistically *redundant* with much simpler features. The paper's
illustration: 95.6 % of the tc07 positive examples also satisfy the tc01 problem
``∃dbo:position``, but only 13.6 % of the trivial negatives — so a learner that can
only solve tc01 already scores 0.91 on the balanced tc07 set, *overestimating* its
ability to learn tc07.

Here every relation-existence is a simple candidate feature:

    out:r   ==  ∃r.⊤   (a tc01 constructor)
    in:r    ==  ∃r⁻.⊤  (a tc02 constructor)

``rdf:type`` is not treated as a relation (consistent with the rest of the
project). For each ``(tc, domain[/hard])`` split we report:

  * the single best shortcut feature (max |p_pos − p_neg|) with the exact
    DLCC-style numbers p_pos / p_neg / balanced single-feature accuracy;
  * the full ranking of the top correlated features;
  * an **existence-bag probe** — one LogReg on the binary {out:r, in:r} vector,
    *no embeddings* — i.e. the upper bound of any model that can learn tc01/tc02
    classes. This is the relation-existence analogue of dbpedia_compare's
    degree probe.

Heavy lifting: one streaming pass over ``dbpedia_graph/graph.nt`` (29 M triples)
collecting, for the eval entities only, their outgoing and incoming predicate
sets (cached as a pickle).
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _dbpedia_compare import (  # noqa: E402
    RDF_TYPE,
    EvalSplit,
    load_eval_splits,
)
from _evaluate import load_labeled_txt  # noqa: E402
from _protograph_gen import iter_nt_iris  # noqa: E402

DBO = "http://dbpedia.org/ontology/"


def short(iri: str) -> str:
    """Human label for a predicate IRI (``…/ontology/team`` -> ``dbo:team``)."""
    if iri.startswith(DBO):
        return "dbo:" + iri[len(DBO):]
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


# ---------------------------------------------------------------------------
# Adjacency: eval entity -> outgoing / incoming predicate id sets
# ---------------------------------------------------------------------------

@dataclass
class Adjacency:
    pred_list: list[str]               # id -> predicate IRI
    pred_index: dict[str, int]         # predicate IRI -> id
    out_preds: dict[str, tuple[int, ...]]  # entity -> outgoing predicate ids
    in_preds: dict[str, tuple[int, ...]]   # entity -> incoming predicate ids
    stats: dict


def collect_eval_entities(splits: list[EvalSplit]) -> set[str]:
    ents: set[str] = set()
    for sp in splits:
        ents.update(sp.train_tokens)
        ents.update(sp.test_tokens)
    return ents


def build_adjacency(
    graph_nt: Path,
    entities: set[str],
    cache_path: Path | None = None,
) -> Adjacency:
    """One pass over graph.nt; keep out/in predicate sets for ``entities`` only.

    rdf:type edges are skipped (not a relation). Result is cached as a pickle.
    """
    if cache_path is not None and cache_path.is_file():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    pred_index: dict[str, int] = {}
    out_sets: dict[str, set[int]] = defaultdict(set)
    in_sets: dict[str, set[int]] = defaultdict(set)
    n_triples = 0
    n_type = 0

    def pid(p: str) -> int:
        i = pred_index.get(p)
        if i is None:
            i = len(pred_index)
            pred_index[p] = i
        return i

    for s, p, o in iter_nt_iris(graph_nt):
        n_triples += 1
        if p == RDF_TYPE:
            n_type += 1
            continue
        s_eval = s in entities
        o_eval = o in entities
        if not (s_eval or o_eval):
            continue
        i = pid(p)
        if s_eval:
            out_sets[s].add(i)
        if o_eval:
            in_sets[o].add(i)

    pred_list = [""] * len(pred_index)
    for p, i in pred_index.items():
        pred_list[i] = p

    out_preds = {e: tuple(sorted(v)) for e, v in out_sets.items()}
    in_preds = {e: tuple(sorted(v)) for e, v in in_sets.items()}
    n_no_out = sum(1 for e in entities if e not in out_preds)
    adj = Adjacency(
        pred_list=pred_list,
        pred_index=pred_index,
        out_preds=out_preds,
        in_preds=in_preds,
        stats={
            "graph_triples": n_triples,
            "rdf_type_rows": n_type,
            "predicates": len(pred_list),
            "eval_entities": len(entities),
            "entities_with_out_edge": len(out_preds),
            "entities_with_in_edge": len(in_preds),
            "entities_without_out_edge": n_no_out,
        },
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return adj


# ---------------------------------------------------------------------------
# Feature presence per entity  ->  "f:<dir>:<pred-id>" tokens
# ---------------------------------------------------------------------------

def entity_features(ent: str, adj: Adjacency) -> set[tuple[str, int]]:
    """{('out', pid), ('in', pid), …} — relation-existence features of ``ent``."""
    feats: set[tuple[str, int]] = set()
    for i in adj.out_preds.get(ent, ()):  # ∃r.⊤
        feats.add(("out", i))
    for i in adj.in_preds.get(ent, ()):   # ∃r⁻.⊤
        feats.add(("in", i))
    return feats


def feature_label(direction: str, pid: int, adj: Adjacency) -> str:
    r = short(adj.pred_list[pid])
    return f"{r}⁻" if direction == "in" else r


# ---------------------------------------------------------------------------
# Single-feature shortcuts (DLCC Fig. 6 style)
# ---------------------------------------------------------------------------

def single_feature_shortcuts(
    pos_entities: list[str],
    neg_entities: list[str],
    adj: Adjacency,
    *,
    top_k: int = 15,
    min_support: int = 5,
) -> list[dict]:
    """For each relation-existence feature: p_pos, p_neg, balanced single-feature
    accuracy. Returns the ``top_k`` most discriminative, accuracy-sorted.

    Balanced accuracy of the threshold rule "predict positive iff feature present"
    (or its complement, whichever is better) on a balanced set is
    ``0.5 + 0.5·|p_pos − p_neg|`` — exactly DLCC's 0.5·(p_pos + 1 − p_neg).
    """
    npos, nneg = len(pos_entities), len(neg_entities)
    pos_count: dict[tuple[str, int], int] = defaultdict(int)
    neg_count: dict[tuple[str, int], int] = defaultdict(int)
    for e in pos_entities:
        for f in entity_features(e, adj):
            pos_count[f] += 1
    for e in neg_entities:
        for f in entity_features(e, adj):
            neg_count[f] += 1

    rows = []
    for f in set(pos_count) | set(neg_count):
        cp, cn = pos_count.get(f, 0), neg_count.get(f, 0)
        if cp + cn < min_support:
            continue
        p_pos, p_neg = cp / npos, cn / nneg
        direction, pid = f
        rows.append(dict(
            feature=feature_label(direction, pid, adj),
            direction=direction,
            p_pos=p_pos,
            p_neg=p_neg,
            acc=0.5 + 0.5 * abs(p_pos - p_neg),
            sign="+" if p_pos >= p_neg else "-",
        ))
    rows.sort(key=lambda r: r["acc"], reverse=True)
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Existence-bag probe: LogReg on the binary {out:r, in:r} vector, no embeddings
# ---------------------------------------------------------------------------

def existence_bag_probe(
    split: EvalSplit,
    adj: Adjacency,
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> float:
    """Test accuracy of one LogReg on the relation-existence bag (tc01/tc02
    learner upper bound). Features = all {out:r, in:r} seen in the TRAIN set."""
    feat_id: dict[tuple[str, int], int] = {}
    train_feats = [entity_features(e, adj) for e in split.train_tokens]
    for fs in train_feats:
        for f in fs:
            if f not in feat_id:
                feat_id[f] = len(feat_id)
    d = len(feat_id)
    if d == 0:
        return 0.5

    def matrix(token_feats: list[set]) -> np.ndarray:
        x = np.zeros((len(token_feats), d), dtype=np.float32)
        for i, fs in enumerate(token_feats):
            for f in fs:
                j = feat_id.get(f)
                if j is not None:
                    x[i, j] = 1.0
        return x

    x_tr = matrix(train_feats)
    x_te = matrix([entity_features(e, adj) for e in split.test_tokens])
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(x_tr, split.y_train)
    return float(accuracy_score(split.y_test, clf.predict(x_te)))


# ---------------------------------------------------------------------------
# Full-population positives / negatives (DLCC uses "all entities fulfilling …")
# ---------------------------------------------------------------------------

def load_population(dbpedia_root: Path, tc: str, domain: str, *, k: int = 5000):
    """Return (positive_entities, negative_entities) from the *.txt populations.

    ``domain`` may carry a ``_hard`` suffix; the hard population is the union of
    the hard train+test split (no separate negatives.txt exists for hard)."""
    base = dbpedia_root / tc / domain.removesuffix("_hard") / str(int(k))
    if domain.endswith("_hard"):
        toks, ys = [], []
        for part in ("train", "test"):
            t, y = load_labeled_txt(base / "train_test_hard" / f"{part}.txt")
            toks += t
            ys.append(y)
        ys = np.concatenate(ys)
        pos = [t for t, y in zip(toks, ys) if y == 1]
        neg = [t for t, y in zip(toks, ys) if y == 0]
        return pos, neg
    pos = (base / "positives.txt").read_text(encoding="utf-8").split()
    neg = (base / "negatives.txt").read_text(encoding="utf-8").split()
    return pos, neg
