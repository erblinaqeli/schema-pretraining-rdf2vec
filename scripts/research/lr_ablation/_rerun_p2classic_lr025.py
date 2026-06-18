"""
Re-run p2_classic at the default from-scratch finetune LR 0.025 (instead of the
protected 0.0025) for tc01-tc15, using the EXACT synthetic_compare pipeline and
its cached walks, so the numbers are directly comparable to the other columns of
Table tab:synthetic_final (notebooks/synthetic_compare/results.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _synthetic_compare import ensure_walks, run_protograph_variant, write_protographs  # noqa: E402

OUT_ROOT = ROOT / "notebooks" / "synthetic_compare"

CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    finetune_alpha=0.025,   # <-- the change: default LR, not the protected 0.0025
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)

TCS = [f"tc{i:02d}" for i in range(1, 16)]


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=OUT_ROOT / tc,
    )


def main():
    tcs = sys.argv[1:] or TCS
    results = {}
    for tc in tcs:
        p = tc_paths(tc)
        out = p["out"]
        proto_paths = write_protographs(p["ontology"], out)
        proto_walks = {
            kind: ensure_walks(
                path, out / f"walks_{kind}.txt",
                walks_per_entity=CFG["proto_walks_per_entity"], depth=CFG["depth"],
                seed=CFG["seed"], ensure_triple_coverage=True,
            )
            for kind, path in proto_paths.items()
        }
        inst_walks = ensure_walks(
            p["graph"],
            out / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
            walks_per_entity=CFG["walks_per_entity"], depth=CFG["depth"], seed=CFG["seed"],
        )
        res = run_protograph_variant(
            "p2_classic", proto_walks["p2"], inst_walks, p["ontology"],
            p["train"], p["test"],
            strategy="all_init", normalize=True, target_norm=CFG["target_norm"],
            finetune_alpha=CFG["finetune_alpha"], pretrain_epochs=CFG["pretrain_epochs"],
            dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"],
        )
        accs = " ".join(f"{a:.3f}" for a in res["accs"])
        print(f"{tc}: p2_classic@lr0.025 final={res['final_acc']:.3f}  [{accs}]", flush=True)
        results[tc] = res["final_acc"]

    print("\nFINAL:")
    print(json.dumps({tc: round(v, 3) for tc, v in results.items()}, indent=2))
    (OUT_ROOT / "p2_classic_lr025.json").write_text(
        json.dumps({tc: v for tc, v in results.items()}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
