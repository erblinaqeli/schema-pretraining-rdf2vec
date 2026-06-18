#!/usr/bin/env python3
"""Experiment 1: scaling lenses + feature fusion on the CACHED dbpedia_compare kv.

No training. Tests whether (a) the bound vectors' pathological norm tail makes
raw LogReg under-report their quality, and (b) the bound channel is complementary
to vanilla (fusion [vanilla|bound] beating either alone), especially on hard
splits / the constructor families it was designed for.
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

from _dbpedia_investigate import Model, evaluate_models, summarize  # noqa: E402
from _dbpedia_compare import load_eval_splits  # noqa: E402

COMPARE = ROOT / "notebooks" / "dbpedia_compare"
OUT = ROOT / "notebooks" / "dbpedia_investigate"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
print(f"[{time.time()-t0:.0f}s] {len(splits)} eval splits", flush=True)

kvs = {}
for v in ("vanilla", "p2_classic", "p2_bound", "p3_bound"):
    p = COMPARE / f"{v}.kv"
    if p.is_file():
        kvs[v] = KeyedVectors.load(str(p), mmap="r")
# p3_classic was not part of dbpedia_compare; trained here (exp5).
p3c = OUT / "p3_classic_FULL.kv"
if p3c.is_file():
    kvs["p3_classic"] = KeyedVectors.load(str(p3c), mmap="r")
print(f"[{time.time()-t0:.0f}s] loaded kv: {list(kvs)}", flush=True)

results = {}

def run(label, models, scaling):
    accs = evaluate_models(splits, models, scaling=scaling, n_jobs=12)
    s = summarize(accs)
    results[f"{label}__{scaling}"] = {"label": label, "scaling": scaling, "summary": s, "accs": accs}
    print(f"[{time.time()-t0:.0f}s] {label:38s} [{scaling:18s}] "
          f"normal {s['normal']:.3f}  hard {s['hard']:.3f}  all {s['all']:.3f}", flush=True)

# --- single models under three scaling lenses ---
for v, kv in kvs.items():
    for scaling in ("raw", "standardize", "l2row"):
        run(f"{v}", [Model(v, kv=kv)], scaling)

# --- fusion: vanilla + bound / classic (standardize each block first) ---
if "vanilla" in kvs and "p2_bound" in kvs:
    run("vanilla+p2_bound", [Model("vanilla", kv=kvs["vanilla"]),
                              Model("p2_bound", kv=kvs["p2_bound"])], "standardize")
if "vanilla" in kvs and "p3_bound" in kvs:
    run("vanilla+p3_bound", [Model("vanilla", kv=kvs["vanilla"]),
                              Model("p3_bound", kv=kvs["p3_bound"])], "standardize")
if "vanilla" in kvs and "p2_classic" in kvs:
    run("vanilla+p2_classic", [Model("vanilla", kv=kvs["vanilla"]),
                                Model("p2_classic", kv=kvs["p2_classic"])], "standardize")
if {"vanilla", "p2_classic", "p2_bound"} <= set(kvs):
    run("vanilla+p2_classic+p2_bound",
        [Model("vanilla", kv=kvs["vanilla"]),
         Model("p2_classic", kv=kvs["p2_classic"]),
         Model("p2_bound", kv=kvs["p2_bound"])], "standardize")
# --- P3 fusions (once p3_classic is trained) ---
if {"vanilla", "p3_classic"} <= set(kvs):
    run("vanilla+p3_classic", [Model("vanilla", kv=kvs["vanilla"]),
                                Model("p3_classic", kv=kvs["p3_classic"])], "standardize")
if {"vanilla", "p3_classic", "p3_bound"} <= set(kvs):
    run("vanilla+p3_classic+p3_bound",
        [Model("vanilla", kv=kvs["vanilla"]),
         Model("p3_classic", kv=kvs["p3_classic"]),
         Model("p3_bound", kv=kvs["p3_bound"])], "standardize")

(OUT / "exp1_fusion.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"[{time.time()-t0:.0f}s] wrote {OUT/'exp1_fusion.json'}", flush=True)
