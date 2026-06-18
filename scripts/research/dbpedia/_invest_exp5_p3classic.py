#!/usr/bin/env python3
"""Experiment 5: train the missing p3_classic on the FULL corpus.

p3_bound (frozen + cap16) already exist; p3_classic (MASCHInE own-class-mean init
on P3 codes, protected LR 0.0025) was never trained. This is the P3 analog of the
cached p2_classic, so the P3 family is complete for the analysis.
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
res_path = OUT / "exp5_p3classic.json"
results = json.loads(res_path.read_text()) if res_path.is_file() else {}

name = "p3_classic_FULL"
if name in results:
    print(f"[skip] {name} cached")
else:
    print(f"========== {name} (init=classic kind=p3 lr=0.0025) ==========", flush=True)
    t = time.time()
    out = finetune_run(name=name, init="classic", kind="p3", norm_policy="global",
                       finetune_alpha=0.0025, artifacts_dir=ART, walks=WALKS,
                       corpus_vocab_pkl=VOCAB, splits=splits, epochs=5,
                       lenses=("standardize", "raw"), save_kv=OUT / f"{name}.kv")
    results[name] = out
    res_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fin = out["per_epoch"][-1]["standardize"]; ini = out["per_epoch"][0]["standardize"]
    print(f"[done] {name} in {time.time()-t:.0f}s init {ini['all']:.3f} -> final {fin['all']:.3f} "
          f"(normal {fin['normal']:.3f} hard {fin['hard']:.3f})", flush=True)
print("ALL DONE", flush=True)
