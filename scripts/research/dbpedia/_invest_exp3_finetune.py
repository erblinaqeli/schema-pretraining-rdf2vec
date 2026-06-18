#!/usr/bin/env python3
"""Experiment 3: finetune-policy sweep on a sub-sampled corpus.

Tests the two fixes prescribed by dbpedia_compare but never run on DBpedia:
relax the protected LR, and cap the init norm tail so hub rows are no longer
frozen. All bound runs start from the SAME cached epoch-0 concept-bound init;
only the norm policy and finetune LR change. vanilla + classic on the same
corpus are the references (same-corpus comparison is the fair one for a
finetune-policy study).

Usage:
  .venv/bin/python scripts/_invest_exp3_finetune.py [--walks PATH] [--tag NAME]
"""
from __future__ import annotations

import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--walks", type=Path, default=ROOT / "walks" / "all_walks_010.txt")
ap.add_argument("--tag", default="ft010")
ap.add_argument("--epochs", type=int, default=5)
args = ap.parse_args()

vocab_pkl = args.walks.with_name(args.walks.stem + "_vocab.pkl")
splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
res_path = OUT / f"exp3_{args.tag}.json"
results = json.loads(res_path.read_text()) if res_path.is_file() else {}

# (name, init, kind, norm_policy, finetune_alpha)
CONFIGS = [
    ("vanilla",              "vanilla", "p2", "global",  0.025),
    ("p2_classic",           "classic", "p2", "global",  0.0025),
    ("p2_bound_global_lr25", "bound",   "p2", "global",  0.025),   # relax protection
    ("p2_bound_global_lr025","bound",   "p2", "global",  0.0025),  # reproduce freeze
    ("p2_bound_cap16_lr025", "bound",   "p2", "cap:16",  0.0025),  # cap + protected
    ("p2_bound_cap16_lr25",  "bound",   "p2", "cap:16",  0.025),   # cap + relaxed
    ("p2_bound_cap8_lr01",   "bound",   "p2", "cap:8",   0.01),    # intermediate
]

def main():
    common = dict(artifacts_dir=ART, walks=args.walks, corpus_vocab_pkl=vocab_pkl,
                  splits=splits, epochs=args.epochs, lenses=("standardize", "raw"))
    for name, init, kind, policy, lr in CONFIGS:
        if name in results:
            print(f"[skip] {name} cached", flush=True)
            continue
        print(f"\n========== {name} (init={init} policy={policy} lr={lr}) ==========", flush=True)
        t = time.time()
        out = finetune_run(name=name, init=init, kind=kind, norm_policy=policy,
                           finetune_alpha=lr, **common)
        results[name] = out
        res_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        fin = out["per_epoch"][-1]["standardize"]
        ini = out["per_epoch"][0]["standardize"]
        print(f"[done] {name} in {time.time()-t:.0f}s  init all {ini['all']:.3f} -> "
              f"final all {fin['all']:.3f} (normal {fin['normal']:.3f} hard {fin['hard']:.3f})",
              flush=True)
    print("\nALL DONE", flush=True)

if __name__ == "__main__":
    main()
