"""One-off: add the 4 P3 init-strategy variants to notebooks/init_strategies/results.json.

Reuses the cached P3 protograph walks; trains only p3_{most_specific,all_init,
average_hier,weight_hierarchy} per TC and merges them into the existing rows,
leaving the vanilla/P1/P2 results byte-for-byte unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _init_strategies import (  # noqa: E402
    ALL_TCS, CFG, STRATEGIES, _ensure_corpora, load_results, save_results, tc_paths,
)
from _synthetic_compare import run_protograph_variant  # noqa: E402

import time  # noqa: E402


def main() -> None:
    results = load_results()
    shared = dict(dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"])
    transfer_shared = dict(
        normalize=True, target_norm=CFG["target_norm"], finetune_alpha=CFG["finetune_alpha"],
        pretrain_epochs=CFG["pretrain_epochs"], ancestor_decay=CFG["ancestor_decay"], **shared,
    )
    for tc in ALL_TCS:
        rows = results.get(tc)
        if rows is None:
            print(f"{tc}: no existing rows, skipping", flush=True)
            continue
        have = {r["variant"] for r in rows}
        if all(f"p3_{s}" in have for s in STRATEGIES):
            print(f"{tc}: p3 already present", flush=True)
            continue
        p = tc_paths(tc)
        inst_walks, proto_walks = _ensure_corpora(p)  # p3 walks cached -> fast
        t0 = time.time()
        print(f"{tc}:", flush=True)
        for strat in STRATEGIES:
            res = run_protograph_variant(
                f"p3_{strat}", proto_walks["p3"], inst_walks, p["ontology"],
                p["train"], p["test"], strategy=strat, **transfer_shared,
            )
            res.pop("model", None)
            res["tc"] = tc
            rows.append(res)
            accs = " ".join(f"{a:.3f}" for a in res["accs"])
            print(f"  [{time.time() - t0:6.1f}s] {res['variant']:>20}  [{accs}]", flush=True)
        results[tc] = rows
        save_results(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
