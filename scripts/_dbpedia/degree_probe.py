#!/usr/bin/env python3
"""
Degree-shortcut probe for the DLCC DBpedia splits.

Trains LogReg on FOUR trivial graph statistics per entity — log1p(out-degree),
log1p(in-degree), #distinct outgoing predicates, #distinct incoming predicates
(rdf:type edges excluded) — and reports test accuracy per split. No embeddings
involved: this measures how much of each split is solvable by popularity /
degree alone, i.e. the shortcut signal a corpus-trained skip-gram can absorb
for free on a real graph. Cached in notebooks/dbpedia_compare/degree_probe.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _dbpedia_compare import RDF_TYPE, load_eval_splits  # noqa: E402
from _protograph_gen import iter_nt_iris  # noqa: E402

ROOT = _SCRIPTS_DIR.parent
GRAPH = ROOT / "dbpedia_graph" / "graph.nt"
OUT = ROOT / "notebooks" / "dbpedia_compare" / "degree_probe.json"


def main() -> None:
    splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
    ents = {t for sp in splits for t in (*sp.train_tokens, *sp.test_tokens)}
    print(f"{len(splits)} splits, {len(ents)} distinct eval entities", flush=True)

    out_deg = {e: 0 for e in ents}
    in_deg = {e: 0 for e in ents}
    out_preds: dict[str, set[str]] = {e: set() for e in ents}
    in_preds: dict[str, set[str]] = {e: set() for e in ents}

    t0 = time.time()
    for i, (s, r, o) in enumerate(iter_nt_iris(GRAPH), 1):
        if i % 5_000_000 == 0:
            print(f"  {i/1e6:.0f}M triples in {time.time()-t0:.0f}s", flush=True)
        if r == RDF_TYPE:
            continue
        if s in out_deg:
            out_deg[s] += 1
            out_preds[s].add(r)
        if o in in_deg:
            in_deg[o] += 1
            in_preds[o].add(r)

    feats = {
        e: [
            float(np.log1p(out_deg[e])),
            float(np.log1p(in_deg[e])),
            float(len(out_preds[e])),
            float(len(in_preds[e])),
        ]
        for e in ents
    }

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    accs: dict[str, float] = {}
    for sp in splits:
        x_tr = np.array([feats[t] for t in sp.train_tokens])
        x_te = np.array([feats[t] for t in sp.test_tokens])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(x_tr, sp.y_train)
        accs[sp.name] = float(accuracy_score(sp.y_test, clf.predict(x_te)))

    OUT.write_text(json.dumps({"features": ["log1p_out_deg", "log1p_in_deg",
                                            "n_out_preds", "n_in_preds"],
                               "split_accs": accs}, indent=1), encoding="utf-8")
    mean = float(np.mean(list(accs.values())))
    print(f"done in {time.time()-t0:.0f}s, mean acc {mean:.4f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
