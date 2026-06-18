#!/usr/bin/env python3
"""Experiment 4: confirm the winning finetune policy (norm-cap 16 + protected LR)
on the FULL 1.16 B-token corpus, for both P2 and P3.

vanilla / p2_classic on the full corpus are taken from exp1 (cached dbpedia_compare
kv, same eval harness + standardize lens), so only the two bound-cap configs run
here. Each is ~70 min.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

from _dbpedia_investigate import finetune_run  # noqa: E402
from _dbpedia_compare import load_eval_splits  # noqa: E402

ART = ROOT / "notebooks" / "dbpedia_investigate" / "artifacts"
OUT = ROOT / "notebooks" / "dbpedia_investigate"
WALKS = ROOT / "walks" / "all_walks.txt"
VOCAB = ROOT / "walks" / "all_walks_vocab.pkl"

splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
res_path = OUT / "exp4_full.json"
results = json.loads(res_path.read_text()) if res_path.is_file() else {}

CONFIGS = [
    ("p2_bound_cap16_lr025_FULL", "bound", "p2", "cap:16", 0.0025),
    ("p3_bound_cap16_lr025_FULL", "bound", "p3", "cap:16", 0.0025),
]

common = dict(artifacts_dir=ART, walks=WALKS, corpus_vocab_pkl=VOCAB,
              splits=splits, epochs=5, lenses=("standardize", "raw"))

for name, init, kind, policy, lr in CONFIGS:
    if name in results:
        print(f"[skip] {name} cached", flush=True)
        continue
    print(f"\n========== {name} (policy={policy} lr={lr}) ==========", flush=True)
    t = time.time()
    out = finetune_run(name=name, init=init, kind=kind, norm_policy=policy,
                       finetune_alpha=lr, save_kv=OUT / f"{name}.kv", **common)
    results[name] = out
    res_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fin = out["per_epoch"][-1]["standardize"]; ini = out["per_epoch"][0]["standardize"]
    print(f"[done] {name} in {time.time()-t:.0f}s  init {ini['all']:.3f} -> final {fin['all']:.3f} "
          f"(normal {fin['normal']:.3f} hard {fin['hard']:.3f})", flush=True)

print("\nALL DONE", flush=True)
