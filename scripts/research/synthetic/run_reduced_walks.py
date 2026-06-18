"""
Reduced fine-tuning walk budget experiment (thesis Section 3.6 + Chapter 5 revisit).

Re-runs the synthetic_compare variants with *reduced* instance-walk budgets
(50 and 25 walks per entity instead of the full 100 used by
scripts/run_method_proposal.py) so the thesis can compare:

  * vanilla at the FULL budget (taken from method_proposal/data/<tc>.json)
  * vanilla / p1_classic / p2_classic at the reduced budget   (Section 3.6)
  * p1_bound / p2_bound / p3_bound at the reduced budget      (Chapter 5 revisit)

All other hyperparameters match run_method_proposal.py exactly (dim 200,
depth-3 duplicate-free walks, 200 protograph walks per entity, 5 pretrain and
5 finetune epochs, protected LR 0.0025 for pretrained variants, target norm 8,
seed 42), so curves are directly comparable across budgets.

Outputs (resumable; delete a json to re-run that tc/budget):
  method_proposal/data/nw<budget>/<tc>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from run_method_proposal import CFG, TCS, tc_paths, train_tracked  # noqa: E402

OUT_ROOT = ROOT / "method_proposal" / "data"
BUDGETS = (50, 25)


def run_tc_budget(tc: str, budget: int, stage1_cache: dict) -> None:
    out_json = OUT_ROOT / f"nw{budget}" / f"{tc}.json"
    if out_json.is_file():
        print(f"{tc} nw{budget}: cached, skipping", flush=True)
        return

    p = tc_paths(tc)
    t_start = time.time()
    p["out"].mkdir(parents=True, exist_ok=True)
    proto_paths = write_protographs(p["ontology"], p["out"])
    proto_walks = {
        kind: ensure_walks(
            path,
            p["out"] / f"walks_{kind}.txt",
            walks_per_entity=CFG["proto_walks_per_entity"],
            depth=CFG["depth"],
            seed=CFG["seed"],
            ensure_triple_coverage=True,
        )
        for kind, path in proto_paths.items()
    }
    inst_walks = ensure_walks(
        p["graph"],
        p["out"] / f"walks_instance_w{budget}_d{CFG['depth']}.txt",
        walks_per_entity=budget,
        depth=CFG["depth"],
        seed=CFG["seed"],
    )

    eval_fn = make_eval_fn(p["train"], p["test"])
    rows: list[dict] = []

    def record(name, accs, losses, pretrain_s, finetune_s, alpha):
        rows.append(dict(
            tc=tc,
            variant=name,
            walks_per_entity=budget,
            lr=alpha,
            accs=[round(a, 6) for a in accs],
            losses=[round(l, 2) for l in losses],
            final_acc=accs[-1],
            pretrain_s=round(pretrain_s, 1),
            finetune_s=round(finetune_s, 1),
            total_s=round(pretrain_s + finetune_s, 1),
        ))
        print(
            f"  [{time.time()-t_start:6.1f}s] nw{budget} {name:>11}  "
            f"acc=[{' '.join(f'{a:.3f}' for a in accs)}]",
            flush=True,
        )

    # --- vanilla at the reduced budget --------------------------------------
    t0 = time.time()
    model = new_skipgram_model(
        dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
        seed=CFG["seed"], workers=CFG["workers"],
    )
    model.build_vocab(LineSentence(str(inst_walks)))
    vocab_s = time.time() - t0
    accs, losses, train_s = train_tracked(
        model, inst_walks, epochs=CFG["epochs"],
        alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
    )
    record("vanilla", accs, losses, 0.0, vocab_s + train_s, CFG["vanilla_alpha"])

    # --- pretrained codes per protograph (cached across budgets) ------------
    def stage1(kind: str):
        key = (tc, kind)
        if key not in stage1_cache:
            t0 = time.time()
            pre = pretrain_protograph(
                proto_walks[kind], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
                seed=CFG["seed"], workers=CFG["workers"],
            )
            pretrain_s = time.time() - t0
            vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
            stage1_cache[key] = (vecs, used_norm, pretrain_s)
        return stage1_cache[key]

    def run_initialized(name, kind, *, bound: bool, alpha: float):
        vecs, used_norm, pretrain_s = stage1(kind)
        bound_vectors = None
        if bound:
            bound_vectors = concept_bound_vectors(
                p["graph"], p["ontology"], vecs, target_norm=used_norm,
            )
        t0 = time.time()
        model = new_skipgram_model(
            dim=CFG["dim"], alpha=alpha, min_alpha=CFG["min_alpha"],
            seed=CFG["seed"], workers=CFG["workers"],
        )
        model.build_vocab(LineSentence(str(inst_walks)))
        protograph_init(
            model, vecs, p["ontology"], strategy="all_init",
            target_norm=used_norm, bound_vectors=bound_vectors,
        )
        setup_s = time.time() - t0
        accs, losses, train_s = train_tracked(
            model, inst_walks, epochs=CFG["epochs"],
            alpha=alpha, min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
        )
        record(name, accs, losses, pretrain_s, setup_s + train_s, alpha)

    # same LR configuration as run_method_proposal.py
    run_initialized("p1_classic", "p1", bound=False, alpha=CFG["vanilla_alpha"])
    run_initialized("p2_classic", "p2", bound=False, alpha=CFG["finetune_alpha"])
    for kind in ("p1", "p2", "p3"):
        run_initialized(f"{kind}_bound", kind, bound=True, alpha=CFG["finetune_alpha"])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"{tc} nw{budget}: done in {time.time()-t_start:.1f}s -> {out_json}", flush=True)


def main():
    tcs = sys.argv[1:] or TCS
    print(f"Running {len(tcs)} test cases at budgets {BUDGETS}: {', '.join(tcs)}", flush=True)
    stage1_cache: dict = {}
    for tc in tcs:
        print(f"{tc}:", flush=True)
        for budget in BUDGETS:
            run_tc_budget(tc, budget, stage1_cache)
        # protograph pretrains are shared across budgets but not across TCs
        stage1_cache.clear()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
