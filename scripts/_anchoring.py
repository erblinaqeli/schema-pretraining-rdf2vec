"""
Shared pipeline for notebooks/anchoring.ipynb.

A dedicated sweep of the **protograph anchoring** regularization strength
``lambda`` (Section "Protograph Anchoring" of the thesis), isolated under one
fixed budget.

Anchoring idea (mirrors the production ``InstanceAnchoringCallback`` in
``scripts/_word2vec.py``): after protograph pretraining, every typed instance is
initialized from its class vector ``v^0_e``. During fine-tuning the instance
embeddings drift away from this schema-informed start. Anchoring counteracts the
drift by, at the **end of EVERY fine-tune epoch (the last included)**, pulling
each class-initialized instance a small step back toward ``v^0_e``:

    v_e  <-  (1 - lambda) * v_e  +  lambda * v^0_e .

This is the closed-form proximal step of the L2 penalty
``lambda * ||v_e - v^0_e||^2`` written in the thesis, applied once per epoch. In
this variant the final epoch is ALSO anchored, so the saved model's anchored
instances are pulled toward their class init before being frozen. ``lambda = 0``
recovers the plain protograph transfer (no anchoring); ``lambda = 1`` resets the
anchored instances to their initialization after every epoch, the last included
(so the saved anchored vectors equal their class init exactly).

Everything except ``lambda`` is held constant (same protograph, same pretrain,
same normalized transfer at TARGET_NORM, same protected fine-tune LR, same
instance-walk corpus, same per-protograph MASCHInE init, same seed), so the only
moving part is the anchoring strength. The sweep is run on the P1, P2 and P3
protographs over the 15 synthetic DLCC test cases ``tc01``--``tc15``. P1/P2 use
the ``most_specific`` MASCHInE rule; P3 uses the ``all_init`` ("classic") rule
that defines the ``p3_classic`` variant elsewhere in the project. A from-scratch
``vanilla`` model and the un-anchored transfer (``lambda = 0``) are the reference
anchors.

For efficiency the protograph pretraining is done **once** per (test case,
protograph); only the cheap instance fine-tune is repeated per ``lambda``. The
cached walk corpora produced by notebooks/synthetic_compare are reused.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gensim.models.callbacks import CallbackAny2Vec  # noqa: E402
from gensim.models.word2vec import LineSentence  # noqa: E402

from _init_strategies import tc_paths  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    ensure_walks,
    make_eval_fn,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
    run_vanilla,
    train_with_eval,
    write_protographs,
)

ROOT = _SCRIPTS_DIR.parent

# tc04 excluded: its instance-walk corpus is ~20x the others (~1.5 h of compute
# for a trivial existence task); the p3_classic experiment excludes it too.
ALL_TCS = [f"tc{i:02d}" for i in range(1, 16) if i != 4]
PROTOGRAPHS = ["p1", "p2", "p3"]

# Init rule per protograph. All three use ``most_specific`` -- the default
# MASCHInE rule: seed each typed instance from the mean of its most-specific
# asserted class codes (no ancestor fallback). Uniform across P1/P2/P3.
PROTO_STRATEGY = {"p1": "most_specific", "p2": "most_specific", "p3": "most_specific"}

# Anchoring strength sweep. 0.0 == plain transfer (no anchoring); 1.0 == hard
# reset to initialization after EVERY epoch (the last included), so the saved
# anchored instances equal their class init exactly.
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# Normal classic transfer recipe, but at the DEFAULT finetune LR so anchoring is
# tested where the model actually drifts from its init instead of being frozen:
#   * finetune LR = 0.025 (gensim default; same as vanilla), NOT the protected
#     0.0025 used in normal classic training -- at 0.0025 the codes barely move
#     and anchoring has nothing to regularize (see synthetic_compare's LR control);
#   * normalize = True, TARGET_NORM = 8 -- the codes ARE L2-normalized to one
#     shared scale, exactly as in normal p1/p2/p3 classic training.
# Everything else matches the init_strategies / synthetic_compare budget.
CFG = dict(
    dim=200,
    walks_per_entity=100,        # instance walks
    proto_walks_per_entity=200,  # protograph walks
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    vanilla_alpha=0.025,
    finetune_alpha=0.025,        # default LR (== vanilla); fair test of anchoring
    min_alpha=0.0001,
    normalize=True,              # normal classic: codes scaled to TARGET_NORM
    target_norm=8.0,             # shared scale of all transferred codes
    ancestor_decay=0.5,
    strategy="most_specific",    # default init rule; all protographs use this via PROTO_STRATEGY
    seed=42,
    workers=16,
)

# Last-epoch-inclusive anchoring variant: outputs are kept in a dedicated
# subfolder so the original "anchor every epoch but the last" thesis artifacts
# (notebooks/anchoring/results.json, new_anchoring.png, the .tex table) are not
# clobbered.
OUT_ROOT = ROOT / "notebooks" / "anchoring" / "anchor_last_epoch"
RESULTS_JSON = OUT_ROOT / "results.json"


class AnchoringCallback(CallbackAny2Vec):
    """Proximal anchoring step at the end of EVERY fine-tune epoch, the last
    included.

    Variant of ``InstanceAnchoringCallback`` from scripts/_word2vec.py that does
    NOT exempt the final epoch: the proximal pull-back is also applied after the
    last epoch, so the saved model's anchored instances are pulled a fraction
    ``alpha`` back toward their class init before being frozen. ``total_epochs``
    is kept for signature compatibility but no longer gates the final epoch.
    """

    def __init__(self, init_vectors: dict[str, np.ndarray], alpha: float, total_epochs: int) -> None:
        self.init_vectors = init_vectors
        self.alpha = float(alpha)
        self.total_epochs = int(total_epochs)
        self._epoch = 0

    def on_epoch_end(self, model) -> None:
        self._epoch += 1
        if self.alpha <= 0.0:
            return
        wv = model.wv
        b, a = 1.0 - self.alpha, self.alpha
        for token, v_init in self.init_vectors.items():
            idx = wv.key_to_index[token]
            wv.vectors[idx] = b * wv.vectors[idx] + a * v_init


def _ensure_corpora(p: dict) -> tuple[Path, dict[str, Path]]:
    """Instance + P1/P2/P3 protograph walk corpora (reused from cache when present)."""
    wd = p["walk_dir"]
    wd.mkdir(parents=True, exist_ok=True)
    proto_paths = write_protographs(p["ontology"], wd)
    proto_walks = {
        kind: ensure_walks(
            proto_paths[kind],
            wd / f"walks_{kind}.txt",
            walks_per_entity=CFG["proto_walks_per_entity"],
            depth=CFG["depth"],
            seed=CFG["seed"],
            ensure_triple_coverage=True,
        )
        for kind in PROTOGRAPHS
    }
    inst_walks = ensure_walks(
        p["graph"],
        wd / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
    )
    return inst_walks, proto_walks


def run_tc(
    tc: str,
    *,
    protographs: list[str] | None = None,
    include_vanilla: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """vanilla + (lambda sweep x protographs) on one test case; returns result rows.

    Pretrains each protograph once and reuses it across the lambda sweep. Pass
    ``protographs`` / ``include_vanilla`` to compute only a subset (used by
    ``run_all`` to extend a cached TC with a newly-added protograph).
    """
    protographs = list(PROTOGRAPHS) if protographs is None else list(protographs)
    p = tc_paths(tc)
    t0 = time.time()
    inst_walks, proto_walks = _ensure_corpora(p)

    eval_fn = make_eval_fn(p["train"], p["test"])
    rows: list[dict] = []

    def record(res: dict) -> None:
        res.pop("model", None)
        res["tc"] = tc
        rows.append(res)
        if verbose:
            accs = " ".join(f"{x:.3f}" for x in res["accs"])
            print(f"  [{time.time() - t0:6.1f}s] {res['variant']:>18}  [{accs}]", flush=True)

    # Reference: from-scratch RDF2Vec.
    if include_vanilla:
        record(run_vanilla(
            inst_walks, p["train"], p["test"],
            dim=CFG["dim"], epochs=CFG["epochs"], alpha=CFG["vanilla_alpha"],
            min_alpha=CFG["min_alpha"], seed=CFG["seed"], workers=CFG["workers"],
        ))

    for proto in protographs:
        strategy = PROTO_STRATEGY[proto]
        # Pretrain the protograph once; reuse the normalized stage-1 codes for
        # every lambda (anchoring only touches the fine-tune stage).
        pre = pretrain_protograph(
            proto_walks[proto], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
            seed=CFG["seed"], workers=CFG["workers"],
        )
        if CFG["normalize"]:
            stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
        else:
            # Raw pretrained codes; no TARGET_NORM scaling (fully-default finetune).
            stage1 = {tok: np.asarray(pre.wv[tok], dtype=np.float32) for tok in pre.wv.index_to_key}
            used_norm = None

        for lam in LAMBDAS:
            t1 = time.time()
            model = new_skipgram_model(
                dim=CFG["dim"], alpha=CFG["finetune_alpha"], min_alpha=CFG["min_alpha"],
                seed=CFG["seed"], workers=CFG["workers"],
            )
            model.build_vocab(LineSentence(str(inst_walks)))
            class_tokens: list[str] = []
            init_stats = protograph_init(
                model, stage1, p["ontology"],
                strategy=strategy, ancestor_decay=CFG["ancestor_decay"],
                target_norm=used_norm, renormalize=CFG["normalize"], noise=0.0, mode="both",
                record_class_init=class_tokens,
            )
            anchor_vecs = {
                tok: np.asarray(model.wv.vectors[model.wv.key_to_index[tok]], dtype=np.float32).copy()
                for tok in class_tokens
            }
            extra = None
            if lam > 0.0 and anchor_vecs:
                extra = [AnchoringCallback(anchor_vecs, lam, total_epochs=CFG["epochs"])]
            accs = train_with_eval(
                model, inst_walks, epochs=CFG["epochs"], alpha=CFG["finetune_alpha"],
                min_alpha=CFG["min_alpha"], eval_fn=eval_fn, extra_callbacks=extra,
            )
            record({
                "variant": f"{proto}_lambda{lam:g}",
                "proto": proto,
                "strategy": strategy,
                "lam": lam,
                "accs": accs,
                "final_acc": accs[-1],
                "init_stats": init_stats,
                "n_anchored": len(anchor_vecs),
                "seconds": round(time.time() - t1, 1),
            })
    return rows


def load_results() -> dict[str, list[dict]]:
    if RESULTS_JSON.is_file():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict[str, list[dict]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def run_all(tcs: list[str] | None = None, *, force: bool = False) -> dict[str, list[dict]]:
    """Run (or load from cache) every TC, saving incrementally to results.json.

    Incremental at the protograph level: a TC already cached with only a subset
    of ``PROTOGRAPHS`` (e.g. an older run before P3 was added) is *extended* --
    only the missing protographs are computed and appended; the cached vanilla
    and prior-protograph rows are kept untouched.
    """
    tcs = tcs or ALL_TCS
    results = {} if force else load_results()
    for tc in tcs:
        cached = results.get(tc, [])
        if force or not cached:
            print(f"{tc}:", flush=True)
            results[tc] = run_tc(tc)
            save_results(results)
            continue
        have = {r.get("proto") for r in cached if r.get("variant") != "vanilla"}
        missing = [pr for pr in PROTOGRAPHS if pr not in have]
        if not missing:
            print(f"{tc}: cached ({len(cached)} variants)", flush=True)
            continue
        print(f"{tc}: extending (+{', '.join(missing)})", flush=True)
        results[tc] = cached + run_tc(tc, protographs=missing, include_vanilla=False)
        save_results(results)
    return results


if __name__ == "__main__":
    run_all()
    print("\nDone.")
