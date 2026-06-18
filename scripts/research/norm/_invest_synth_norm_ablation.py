#!/usr/bin/env python3
"""Synthetic ablation: effect of the per-component normalization on BOTH init families.

For each protograph p1/p2/p3, four conditions (identical except the named lever),
finetuned + evaluated per epoch on the synthetic DLCC test cases:

  classic_raw    classic init (all_init class-mean), class codes NOT normalized  (default)
  classic_unit   classic init, each class code L2-normalized before averaging
  bound_norm     concept-bound init, components L2-normalized                    (default)
  bound_nonorm   concept-bound init, component normalization disabled

classic_* finetune at LR 0.025, bound_* at the protected LR 0.0025 (same as the
synthetic benchmark) -- the with/without-norm comparison is WITHIN a family, so the
LR is held fixed and is not a confound. vanilla is recorded once for reference.

Reuses the cached walks under notebooks/synthetic_compare/<tc>/. Resumable: a
condition with an existing metrics.json is skipped.

Usage:
  .venv/bin/python scripts/_invest_synth_norm_ablation.py                 # tc01..tc15
  .venv/bin/python scripts/_invest_synth_norm_ablation.py tc07 tc09 tc12  # subset
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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

OUT_ROOT = ROOT / "output" / "synthetic_norm_ablation"
WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"
TCS = [f"tc{i:02d}" for i in range(1, 16)]

CFG = dict(
    dim=200, walks_per_entity=100, proto_walks_per_entity=200, depth=3,
    epochs=5, pretrain_epochs=5, classic_lr=0.025, bound_lr=0.0025,
    vanilla_lr=0.025, min_alpha=0.0001, target_norm=8.0, seed=42,
    workers=int(os.environ.get("SYNTH_WORKERS", "16")),
)

# (condition, protograph kind, family, lr, init kwargs)
CONDITIONS = []
for k in ("p1", "p2", "p3"):
    CONDITIONS += [
        (f"{k}_classic_raw",  k, "classic", CFG["classic_lr"], dict(normalize_class_codes=False)),
        (f"{k}_classic_unit", k, "classic", CFG["classic_lr"], dict(normalize_class_codes=True)),
        (f"{k}_bound_norm",   k, "bound",   CFG["bound_lr"],   dict(normalize_components=True)),
        (f"{k}_bound_nonorm", k, "bound",   CFG["bound_lr"],   dict(normalize_components=False)),
    ]

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
    proto_walks = {
        kind: ensure_walks(path, p["walks"] / f"walks_{kind}.txt",
                           walks_per_entity=CFG["proto_walks_per_entity"], depth=CFG["depth"],
                           seed=CFG["seed"], ensure_triple_coverage=True)
        for kind, path in proto_paths.items()
    }
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

    # vanilla reference
    if not cached("vanilla"):
        m = new_skipgram_model(dim=CFG["dim"], alpha=CFG["vanilla_lr"], min_alpha=CFG["min_alpha"],
                               seed=CFG["seed"], workers=CFG["workers"])
        m.build_vocab(LineSentence(str(inst_walks)))
        accs = finetune(m, inst_walks, eval_fn, CFG["vanilla_lr"])
        write("vanilla", dict(tc=tc, condition="vanilla", family="vanilla", kind=None,
                              lr=CFG["vanilla_lr"], accs=[round(a, 6) for a in accs],
                              final_acc=round(accs[-1], 6)))
        log(f"{tc} vanilla: {accs[-1]:.3f}")

    # cache stage-1 codes per protograph kind
    stage1_cache: dict[str, tuple[dict, float]] = {}

    def stage1(kind: str):
        if kind not in stage1_cache:
            pre = pretrain_protograph(proto_walks[kind], dim=CFG["dim"],
                                      epochs=CFG["pretrain_epochs"], seed=CFG["seed"],
                                      workers=CFG["workers"])
            stage1_cache[kind] = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
        return stage1_cache[kind]

    for cond, kind, family, lr, init_kw in CONDITIONS:
        if cached(cond):
            log(f"{tc} {cond}: cached")
            continue
        vecs, used_norm = stage1(kind)
        m = new_skipgram_model(dim=CFG["dim"], alpha=lr, min_alpha=CFG["min_alpha"],
                               seed=CFG["seed"], workers=CFG["workers"])
        m.build_vocab(LineSentence(str(inst_walks)))
        if family == "bound":
            bound_vectors = concept_bound_vectors(p["graph"], p["ontology"], vecs,
                                                  target_norm=used_norm, **init_kw)
            stats = protograph_init(m, vecs, p["ontology"], strategy="all_init",
                                    target_norm=used_norm, bound_vectors=bound_vectors)
        else:
            stats = protograph_init(m, vecs, p["ontology"], strategy="all_init",
                                    target_norm=used_norm, **init_kw)
        accs = finetune(m, inst_walks, eval_fn, lr)
        write(cond, dict(tc=tc, condition=cond, family=family, kind=kind, lr=lr,
                         init_stats=stats, accs=[round(a, 6) for a in accs],
                         final_acc=round(accs[-1], 6)))
        log(f"{tc} {cond:18s} init {accs[0]:.3f} -> final {accs[-1]:.3f}")


def write_summary() -> None:
    rows = [json.loads(mf.read_text()) for mf in sorted(OUT_ROOT.glob("tc*/*/metrics.json"))]
    (OUT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    log(f"summary: {len(rows)} rows -> {OUT_ROOT / 'summary.json'}")


def main() -> None:
    tcs = [a for a in sys.argv[1:] if not a.startswith("-")] or TCS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"conditions per tc: {len(CONDITIONS)} + vanilla; tcs: {', '.join(tcs)}")
    for tc in tcs:
        run_tc(tc)
    write_summary()
    log("ALL DONE")


if __name__ == "__main__":
    main()
