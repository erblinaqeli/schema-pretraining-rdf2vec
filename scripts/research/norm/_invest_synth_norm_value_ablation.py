#!/usr/bin/env python3
"""Synthetic ablation: effect of the *bound-init global rescale norm* (target_norm).

The concept-bound init ends with ONE global rescale so the mean entity norm equals
``target_norm`` (default 8). That norm is the single lever controlling SGNS
sigmoid-saturation "protection": small -> the init drifts under finetuning, large ->
the init freezes (see norm_explanation.md sections 3-4). This sweep isolates that
lever for the p2 bound init, holding everything else fixed.

  p2_bound_norm04   concept-bound init, mean entity norm rescaled to 4
  p2_bound_norm08   concept-bound init, mean entity norm rescaled to 8   (baseline)
  p2_bound_norm16   concept-bound init, mean entity norm rescaled to 16

vanilla (from-scratch RDF2Vec) is norm-independent and recorded once per tc for
reference. All conditions finetune at the protected LR 0.0025 (same as the synthetic
benchmark); only ``target_norm`` varies, so it is not a confound. The same norm is
applied to the stage-1 codes and to protograph_init's non-bound rows so relations and
instances share one scale.

Reuses the cached walks under notebooks/synthetic_compare/<tc>/ (same as the existing
norm ablation). Resumable: a condition with an existing metrics.json is skipped.

Usage:
  .venv/bin/python scripts/_invest_synth_norm_value_ablation.py                 # tc01..tc15
  .venv/bin/python scripts/_invest_synth_norm_value_ablation.py tc07 tc09 tc12  # subset
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _synthetic_compare import (  # noqa: E402
    concept_bound_vectors,
    ensure_walks,
    make_eval_fn,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
    write_protographs,
)

OUT_ROOT = ROOT / "output" / "synth_norm_value_ablation"
WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"
# the relation / qualified-cardinality DLCC tasks the bound init is built for
TCS = [f"tc{i:02d}" for i in range(8, 13)]

# the bound-init global rescale norms to sweep (baseline = 8)
NORMS = [4.0, 8.0, 16.0]
# protograph used for the bound init (the headline P2 fix)
KIND = "p2"

CFG = dict(
    dim=200, walks_per_entity=100, proto_walks_per_entity=200, depth=3,
    epochs=5, pretrain_epochs=5, bound_lr=0.0025,
    vanilla_lr=0.025, min_alpha=0.0001, seed=42,
    workers=int(os.environ.get("SYNTH_WORKERS", "16")),
)

t_global = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t_global:7.0f}s] {msg}", flush=True)


def tc_paths(tc: str) -> dict:
    d = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(ontology=d / "ontology.nt", graph=d / "graph.nt",
               train=d / "1000" / "train_test" / "train.txt",
               test=d / "1000" / "train_test" / "test.txt", walks=WALK_ROOT / tc)


class _Tracker(CallbackAny2Vec):
    def __init__(self, eval_fn, accs):
        self.eval_fn, self.accs = eval_fn, accs

    def on_epoch_end(self, model):
        self.accs.append(self.eval_fn(model))


def finetune(model, walks, eval_fn, lr) -> list[float]:
    accs = [eval_fn(model)]  # epoch 0 (init quality)
    model.train(corpus_iterable=LineSentence(str(walks)), total_examples=model.corpus_count,
                epochs=CFG["epochs"], start_alpha=lr, end_alpha=CFG["min_alpha"],
                callbacks=[_Tracker(eval_fn, accs)])
    return accs


def run_tc(tc: str) -> None:
    p = tc_paths(tc)
    p["walks"].mkdir(parents=True, exist_ok=True)
    proto_paths = write_protographs(p["ontology"], p["walks"])
    proto_walks = ensure_walks(
        proto_paths[KIND], p["walks"] / f"walks_{KIND}.txt",
        walks_per_entity=CFG["proto_walks_per_entity"], depth=CFG["depth"],
        seed=CFG["seed"], ensure_triple_coverage=True)
    inst_walks = ensure_walks(
        p["graph"], p["walks"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"], depth=CFG["depth"], seed=CFG["seed"])
    eval_fn = make_eval_fn(p["train"], p["test"])

    def write(cond: str, row: dict) -> None:
        d = OUT_ROOT / tc / cond
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    def cached(cond: str) -> bool:
        return (OUT_ROOT / tc / cond / "metrics.json").is_file()

    # vanilla reference (norm-independent)
    if not cached("vanilla"):
        m = new_skipgram_model(dim=CFG["dim"], alpha=CFG["vanilla_lr"], min_alpha=CFG["min_alpha"],
                               seed=CFG["seed"], workers=CFG["workers"])
        m.build_vocab(LineSentence(str(inst_walks)))
        accs = finetune(m, inst_walks, eval_fn, CFG["vanilla_lr"])
        write("vanilla", dict(tc=tc, condition="vanilla", family="vanilla", kind=None,
                              target_norm=None, lr=CFG["vanilla_lr"],
                              accs=[round(a, 6) for a in accs], final_acc=round(accs[-1], 6)))
        log(f"{tc} vanilla: {accs[-1]:.3f}")

    # pretrain the protograph codes once (norm-independent); normalize per sweep norm
    pre = None
    for norm in NORMS:
        cond = f"{KIND}_bound_norm{int(norm):02d}"
        if cached(cond):
            log(f"{tc} {cond}: cached")
            continue
        if pre is None:
            pre = pretrain_protograph(proto_walks, dim=CFG["dim"],
                                      epochs=CFG["pretrain_epochs"], seed=CFG["seed"],
                                      workers=CFG["workers"])
        vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=norm)
        m = new_skipgram_model(dim=CFG["dim"], alpha=CFG["bound_lr"], min_alpha=CFG["min_alpha"],
                               seed=CFG["seed"], workers=CFG["workers"])
        m.build_vocab(LineSentence(str(inst_walks)))
        bound_vectors = concept_bound_vectors(p["graph"], p["ontology"], vecs,
                                              target_norm=used_norm)
        stats = protograph_init(m, vecs, p["ontology"], strategy="all_init",
                                target_norm=used_norm, bound_vectors=bound_vectors)
        accs = finetune(m, inst_walks, eval_fn, CFG["bound_lr"])
        write(cond, dict(tc=tc, condition=cond, family="bound", kind=KIND,
                         target_norm=norm, lr=CFG["bound_lr"], init_stats=stats,
                         accs=[round(a, 6) for a in accs], final_acc=round(accs[-1], 6)))
        log(f"{tc} {cond:18s} init {accs[0]:.3f} -> final {accs[-1]:.3f}")


def write_summary() -> None:
    rows = [json.loads(mf.read_text()) for mf in sorted(OUT_ROOT.glob("tc*/*/metrics.json"))]
    (OUT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    log(f"summary: {len(rows)} rows -> {OUT_ROOT / 'summary.json'}")


def main() -> None:
    tcs = [a for a in sys.argv[1:] if not a.startswith("-")] or TCS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"norms: {NORMS}; kind: {KIND}; tcs: {', '.join(tcs)}")
    for tc in tcs:
        run_tc(tc)
    write_summary()
    log("ALL DONE")


if __name__ == "__main__":
    main()
