"""
One-off experiment: does anchoring on the LAST fine-tune epoch too help?

The production ``AnchoringCallback`` (scripts/_anchoring.py) deliberately SKIPS
the final fine-tune epoch -- ``self._epoch >= self.total_epochs`` -- so the saved
model keeps the freely-learned vectors of the last epoch. This script asks: at
``lambda = 0.25``, what if we ALSO apply the proximal pull-back after the last
epoch, on P1/P2/P3?

Efficiency / rigor:
  The "all-but-last" and "all-epochs" variants share an *identical* SGNS
  trajectory -- same pretrain, same init, same per-epoch anchoring for epochs
  1..n-1. They differ by exactly ONE extra proximal step after epoch n. So we
  fine-tune ONCE with the standard (all-but-last) AnchoringCallback at
  lambda=0.25 -- which reproduces the notebook's lambda=0.25 column -- then derive
  the all-epochs result by applying the single closed-form step

      v_e <- (1 - lambda) * v_e + lambda * v^0_e

  to the anchored tokens and re-evaluating. Perfectly paired (one trajectory),
  one fine-tune per (tc, proto). lambda=0 (no anchoring) and vanilla are read
  from the cached results.json for reference.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# This driver lives under scripts/research/anchoring/; put the repo's scripts/ on
# sys.path so the bare `_*` library imports below resolve.
sys.path.insert(
    0,
    str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"),
)

import numpy as np
from gensim.models.word2vec import LineSentence

from _anchoring import (
    ALL_TCS, CFG, OUT_ROOT, PROTO_STRATEGY, PROTOGRAPHS,
    AnchoringCallback, _ensure_corpora, load_results,
)
from _init_strategies import tc_paths
from _synthetic_compare import (
    make_eval_fn, new_skipgram_model, normalized_stage1_vectors,
    pretrain_protograph, protograph_init, train_with_eval,
)

LAM = 0.25
OUT_JSON = OUT_ROOT / "results_lastepoch_lam025.json"


def run_one(tc: str, proto: str, inst_walks: Path, proto_walks: dict, eval_fn):
    """Fine-tune once (all-but-last, lambda=0.25); derive the all-epochs final."""
    strategy = PROTO_STRATEGY[proto]
    pre = pretrain_protograph(
        proto_walks[proto], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
        seed=CFG["seed"], workers=CFG["workers"],
    )
    stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])

    model = new_skipgram_model(
        dim=CFG["dim"], alpha=CFG["finetune_alpha"], min_alpha=CFG["min_alpha"],
        seed=CFG["seed"], workers=CFG["workers"],
    )
    model.build_vocab(LineSentence(str(inst_walks)))
    class_tokens: list[str] = []
    protograph_init(
        model, stage1, tc_paths(tc)["ontology"],
        strategy=strategy, ancestor_decay=CFG["ancestor_decay"],
        target_norm=used_norm, renormalize=CFG["normalize"], noise=0.0, mode="both",
        record_class_init=class_tokens,
    )
    anchor_vecs = {
        tok: np.asarray(model.wv.vectors[model.wv.key_to_index[tok]], dtype=np.float32).copy()
        for tok in class_tokens
    }

    # Single fine-tune with the standard all-but-last callback (reproduces the
    # notebook's lambda=0.25). accs = [init, e1, ..., e5]; final == all-but-last.
    accs = train_with_eval(
        model, inst_walks, epochs=CFG["epochs"], alpha=CFG["finetune_alpha"],
        min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
        extra_callbacks=[AnchoringCallback(anchor_vecs, LAM, total_epochs=CFG["epochs"])],
    )
    all_but_last_final = accs[-1]

    # Derive the all-epochs final: one extra proximal step on the anchored tokens.
    b, a = 1.0 - LAM, LAM
    wv = model.wv
    for tok, v_init in anchor_vecs.items():
        idx = wv.key_to_index[tok]
        wv.vectors[idx] = b * wv.vectors[idx] + a * v_init
    all_epochs_final = float(eval_fn(model))

    return {
        "tc": tc, "proto": proto, "lam": LAM,
        "accs_all_but_last": accs,
        "final_all_but_last": float(all_but_last_final),
        "final_all_epochs": float(all_epochs_final),
        "n_anchored": len(anchor_vecs),
    }


def main():
    cached = load_results()
    rows = []
    t0 = time.time()
    for tc in ALL_TCS:
        p = tc_paths(tc)
        inst_walks, proto_walks = _ensure_corpora(p)
        eval_fn = make_eval_fn(p["train"], p["test"])
        for proto in PROTOGRAPHS:
            r = run_one(tc, proto, inst_walks, proto_walks, eval_fn)
            # Reference finals from the cached sweep (same seed/budget).
            by_var = {x.get("variant"): x for x in cached.get(tc, [])}
            r["ref_vanilla"] = by_var.get("vanilla", {}).get("final_acc")
            r["ref_lambda0"] = by_var.get(f"{proto}_lambda0", {}).get("final_acc")
            r["ref_lambda025_cached"] = by_var.get(f"{proto}_lambda0.25", {}).get("final_acc")
            rows.append(r)
            d = r["final_all_epochs"] - r["final_all_but_last"]
            print(f"  [{time.time()-t0:6.1f}s] {tc} {proto}: "
                  f"all-but-last={r['final_all_but_last']:.3f}  "
                  f"all-epochs={r['final_all_epochs']:.3f}  ({d:+.3f})", flush=True)
        OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
