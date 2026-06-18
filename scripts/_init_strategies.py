"""
Shared pipeline for notebooks/init_strategies.ipynb.

A dedicated, controlled comparison of the **four class-hierarchy initialization
strategies** of the classic protograph transfer (Section 3.1 of the thesis),
isolated under one fixed budget:

    most_specific     MASCHInE rule: mean of the codes of the instance's most
                      specific classes; uncovered instances stay random.
    all_init          like most_specific, but climb subClassOf upward until a
                      covered ancestor is found (coverage remedy).
    average_hier      mean over the instance's classes AND all transitive
                      superclasses (uniform blend).
    weight_hierarchy  like average_hier, but ancestors at hop d are down-weighted
                      by lambda**d (default lambda = 0.5).

Everything else is held constant across the four strategies (same protograph,
same pretrain, same normalized transfer at TARGET_NORM, same protected finetune
LR, same instance-walk corpus, same seed), so the only moving part is *which*
class vectors seed an instance. The comparison is run on both the P1 and the P2
protograph because the strategies are remedies for the leaf-class coverage gap,
which is wider under P1 (domain/range classes only) than under P2 (+ direct
subclasses). A from-scratch ``vanilla`` model is included as a reference anchor.

Reuses the cached walk corpora produced by notebooks/synthetic_compare so the
expensive instance-walk generation is not repeated.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _synthetic_compare import (  # noqa: E402
    ensure_walks,
    run_protograph_variant,
    run_vanilla,
    write_protographs,
)

ROOT = _SCRIPTS_DIR.parent

# Same TC set as the thesis main results table (tab:synthetic_final): the
# fifteen synthetic DLCC test cases tc01--tc15. tc16 data exists on disk but is
# not part of the thesis benchmark, so it is excluded from the aggregation.
ALL_TCS = [f"tc{i:02d}" for i in range(1, 16)]

STRATEGIES = ["most_specific", "all_init", "average_hier", "weight_hierarchy"]
PROTOGRAPHS = ["p1", "p2", "p3"]

# Same budget for every variant (matches notebooks/synthetic_compare).
CFG = dict(
    dim=200,
    walks_per_entity=100,        # instance walks
    proto_walks_per_entity=200,  # protograph walks
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    vanilla_alpha=0.025,         # from-scratch LR (baseline anchor)
    finetune_alpha=0.0025,       # protected finetune LR for every transfer
    min_alpha=0.0001,
    target_norm=8.0,             # shared scale of all transferred codes
    ancestor_decay=0.5,          # lambda for weight_hierarchy
    seed=42,
)

# Reuse the cached walk corpora from synthetic_compare; fall back to a local dir.
WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"
OUT_ROOT = ROOT / "notebooks" / "init_strategies"
RESULTS_JSON = OUT_ROOT / "results.json"


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    walk_dir = WALK_ROOT / tc if (WALK_ROOT / tc).is_dir() else OUT_ROOT / tc
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        walk_dir=walk_dir,
    )


def _ensure_corpora(p: dict) -> tuple[Path, dict[str, Path]]:
    """Instance + P1/P2 protograph walk corpora (reused from cache when present)."""
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


def run_tc(tc: str, *, verbose: bool = True) -> list[dict]:
    """vanilla + (4 strategies x {P1, P2}) on one test case; returns result rows."""
    p = tc_paths(tc)
    t0 = time.time()
    inst_walks, proto_walks = _ensure_corpora(p)

    shared = dict(dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"])
    rows: list[dict] = []

    def record(res: dict) -> None:
        res.pop("model", None)
        res["tc"] = tc
        rows.append(res)
        if verbose:
            accs = " ".join(f"{a:.3f}" for a in res["accs"])
            print(f"  [{time.time() - t0:6.1f}s] {res['variant']:>20}  [{accs}]", flush=True)

    record(run_vanilla(inst_walks, p["train"], p["test"], alpha=CFG["vanilla_alpha"], **shared))

    transfer_shared = dict(
        normalize=True,
        target_norm=CFG["target_norm"],
        finetune_alpha=CFG["finetune_alpha"],
        pretrain_epochs=CFG["pretrain_epochs"],
        ancestor_decay=CFG["ancestor_decay"],
        **shared,
    )
    for kind in PROTOGRAPHS:
        for strat in STRATEGIES:
            record(run_protograph_variant(
                f"{kind}_{strat}", proto_walks[kind], inst_walks, p["ontology"],
                p["train"], p["test"], strategy=strat, **transfer_shared,
            ))
    return rows


def load_results() -> dict[str, list[dict]]:
    if RESULTS_JSON.is_file():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict[str, list[dict]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def run_all(tcs: list[str] | None = None, *, force: bool = False) -> dict[str, list[dict]]:
    """Run (or load from cache) every TC, saving incrementally to results.json."""
    tcs = tcs or ALL_TCS
    results = {} if force else load_results()
    for tc in tcs:
        if tc in results and not force:
            print(f"{tc}: cached ({len(results[tc])} variants)", flush=True)
            continue
        print(f"{tc}:", flush=True)
        results[tc] = run_tc(tc)
        save_results(results)
    return results


if __name__ == "__main__":
    run_all()
    print("\nDone.")
