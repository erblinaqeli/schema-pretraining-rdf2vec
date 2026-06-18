#!/usr/bin/env python3
"""Experiment 2: init-side analysis (no training).

  * reproduce the epoch-0 quality of the true concept-bound init (P2/P3);
  * per-row norm-cap sweep on the init — does capping the pathological norm
    tail hurt init quality (esp. the cardinality family)?  This is the fix that
    should unfreeze hub rows during finetune without erasing counts;
  * fusion with the TRUE init (vs the cached frozen kv) + an adversarial
    random-block control proving the fusion gain is real signal, not dimensions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

from _dbpedia_investigate import (  # noqa: E402
    Model, evaluate_models, summarize, load_emb, cap_norms,
)
from _dbpedia_compare import load_eval_splits  # noqa: E402

ART = ROOT / "notebooks" / "dbpedia_investigate" / "artifacts"
COMPARE = ROOT / "notebooks" / "dbpedia_compare"
OUT = ROOT / "notebooks" / "dbpedia_investigate"

t0 = time.time()
splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
all_eval_tokens = set()
for sp in splits:
    all_eval_tokens.update(sp.train_tokens)
    all_eval_tokens.update(sp.test_tokens)
print(f"[{time.time()-t0:.0f}s] {len(splits)} splits, {len(all_eval_tokens)} distinct eval tokens", flush=True)

results = {}

def run(label, models, scaling, store=True):
    accs = evaluate_models(splits, models, scaling=scaling, n_jobs=12)
    s = summarize(accs)
    if store:
        results[f"{label}__{scaling}"] = {"label": label, "scaling": scaling, "summary": s, "accs": accs}
    card = f"card N {s['cardinality (tc09-12) N']:.3f} H {s['cardinality (tc09-12) H']:.3f}"
    print(f"[{time.time()-t0:.0f}s] {label:34s}[{scaling:11s}] "
          f"normal {s['normal']:.3f} hard {s['hard']:.3f} all {s['all']:.3f}  {card}", flush=True)
    return s

for kind in ("p2", "p3"):
    bound = load_emb(ART / f"boundinit_{kind}")
    cov = sum(1 for t in all_eval_tokens if t in bound) / len(all_eval_tokens)
    print(f"\n[{time.time()-t0:.0f}s] === {kind} bound init: {len(bound)} vecs, "
          f"eval-token coverage {cov:.1%} ===", flush=True)
    # epoch-0 quality, two lenses
    run(f"{kind}_bound_init", [Model(kind, emb=bound)], "raw")
    run(f"{kind}_bound_init", [Model(kind, emb=bound)], "standardize")
    # norm-cap sweep (standardize lens)
    for cap in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        capped = cap_norms(bound, cap)
        run(f"{kind}_bound_cap{cap:g}", [Model(kind, emb=capped)], "standardize")
    del bound

# fusion with TRUE p2 init + adversarial random control
vanilla = KeyedVectors.load(str(COMPARE / "vanilla.kv"), mmap="r")
bound2 = load_emb(ART / "boundinit_p2")
run("vanilla_alone", [Model("vanilla", kv=vanilla)], "standardize")
run("vanilla+trueP2bound", [Model("vanilla", kv=vanilla), Model("p2b", emb=bound2)], "standardize")

# random 200-d block control (same dimensionality bump, zero signal)
rng = np.random.default_rng(0)
rand = {t: rng.standard_normal(200).astype(np.float32) for t in all_eval_tokens}
run("vanilla+random200", [Model("vanilla", kv=vanilla), Model("rand", emb=rand)], "standardize")

(OUT / "exp2_init.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"\n[{time.time()-t0:.0f}s] wrote {OUT/'exp2_init.json'}", flush=True)
