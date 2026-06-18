"""Fill in the missing p3_classic variant for the reduced-walk-budget table.

Runs p3_classic (protograph-p3 codes, classic all_init transfer, NO concept
bound, finetune LR 0.0025 -- same config as p2_classic) at the full (100) and
reduced (50, 25) instance-walk budgets, then appends a row to each existing
data file so scripts/plot_reduced_walks.py picks it up:

  method_proposal/data/<tc>.json          (100 walks)
  method_proposal/data/nw50/<tc>.json     (50 walks)
  method_proposal/data/nw25/<tc>.json     (25 walks)

Resumable: a (tc, budget) pair whose file already contains a p3_classic row is
skipped. Every other hyperparameter matches run_method_proposal.py /
run_reduced_walks.py exactly so the row is comparable to the rest of the table.
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
    ensure_walks,
    make_eval_fn,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
    write_protographs,
)
from run_method_proposal import CFG, tc_paths, train_tracked  # noqa: E402

DATA = ROOT / "method_proposal" / "data"
TCS = sorted(p.stem for p in (DATA / "nw50").glob("tc*.json"))
BUDGETS = (100, 50, 25)


def data_path(tc: str, budget: int) -> Path:
    return DATA / f"{tc}.json" if budget == 100 else DATA / f"nw{budget}" / f"{tc}.json"


def run_tc(tc: str) -> None:
    p = tc_paths(tc)
    todo = [b for b in BUDGETS
            if not any(r["variant"] == "p3_classic"
                       for r in json.loads(data_path(tc, b).read_text()))]
    if not todo:
        print(f"{tc}: p3_classic present at all budgets, skipping", flush=True)
        return

    t_start = time.time()
    p["out"].mkdir(parents=True, exist_ok=True)

    # protograph-p3 codes (shared across budgets, one pretrain per tc)
    proto_paths = write_protographs(p["ontology"], p["out"])
    proto_walks = ensure_walks(
        proto_paths["p3"], p["out"] / "walks_p3.txt",
        walks_per_entity=CFG["proto_walks_per_entity"], depth=CFG["depth"],
        seed=CFG["seed"], ensure_triple_coverage=True,
    )
    pre = pretrain_protograph(
        proto_walks, dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
        seed=CFG["seed"], workers=CFG["workers"],
    )
    vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
    eval_fn = make_eval_fn(p["train"], p["test"])

    for budget in todo:
        inst_walks = ensure_walks(
            p["graph"],
            p["out"] / f"walks_instance_w{budget}_d{CFG['depth']}.txt",
            walks_per_entity=budget, depth=CFG["depth"], seed=CFG["seed"],
        )
        t0 = time.time()
        model = new_skipgram_model(
            dim=CFG["dim"], alpha=CFG["finetune_alpha"], min_alpha=CFG["min_alpha"],
            seed=CFG["seed"], workers=CFG["workers"],
        )
        model.build_vocab(LineSentence(str(inst_walks)))
        protograph_init(
            model, vecs, p["ontology"], strategy="all_init",
            target_norm=used_norm, bound_vectors=None,
        )
        setup_s = time.time() - t0
        accs, losses, train_s = train_tracked(
            model, inst_walks, epochs=CFG["epochs"],
            alpha=CFG["finetune_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
        )
        row = dict(
            tc=tc, variant="p3_classic", walks_per_entity=budget,
            lr=CFG["finetune_alpha"],
            accs=[round(a, 6) for a in accs],
            losses=[round(l, 2) for l in losses],
            final_acc=accs[-1],
            pretrain_s=round(setup_s, 1),
            finetune_s=round(train_s, 1),
            total_s=round(setup_s + train_s, 1),
        )
        dp = data_path(tc, budget)
        rows = json.loads(dp.read_text())
        rows.append(row)
        dp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(
            f"  [{time.time()-t_start:6.1f}s] nw{budget:>3} p3_classic  "
            f"acc=[{' '.join(f'{a:.3f}' for a in accs)}] -> {dp.name}",
            flush=True,
        )


def main() -> None:
    tcs = sys.argv[1:] or TCS
    print(f"Running p3_classic at budgets {BUDGETS}: {', '.join(tcs)}", flush=True)
    for tc in tcs:
        print(f"{tc}:", flush=True)
        run_tc(tc)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
