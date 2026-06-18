"""One-off: run the synthetic_compare 6-variant pipeline for tc02 and append to results.json.

Mirrors notebooks/synthetic_compare.ipynb `run_tc`, restricted to tc02. Walks are
cached on disk, so this only does pretraining + finetuning + evaluation.
"""
import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
OUT_ROOT = ROOT / "notebooks" / "synthetic_compare"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _synthetic_compare import (  # noqa: E402
    ensure_walks,
    run_protograph_variant,
    run_vanilla,
    write_protographs,
)

CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    vanilla_alpha=0.025,
    finetune_alpha=0.0025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)

TC = "tc02"


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=OUT_ROOT / tc,
    )


def run_tc(tc: str) -> list[dict]:
    p = tc_paths(tc)
    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    proto_paths = write_protographs(p["ontology"], out)
    proto_walks = {
        kind: ensure_walks(
            path,
            out / f"walks_{kind}.txt",
            walks_per_entity=CFG["proto_walks_per_entity"],
            depth=CFG["depth"],
            seed=CFG["seed"],
            ensure_triple_coverage=True,
        )
        for kind, path in proto_paths.items()
    }
    inst_walks = ensure_walks(
        p["graph"],
        out / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
    )

    shared = dict(dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"])
    rows = []

    def record(res):
        res.pop("model", None)
        res["tc"] = tc
        rows.append(res)
        accs = " ".join(f"{a:.3f}" for a in res["accs"])
        print(f"  [{time.time()-t0:6.1f}s] {res['variant']:>11}  [{accs}]", flush=True)

    record(run_vanilla(inst_walks, p["train"], p["test"], alpha=CFG["vanilla_alpha"], **shared))

    proto_shared = dict(
        strategy="all_init",
        normalize=True,
        target_norm=CFG["target_norm"],
        finetune_alpha=CFG["finetune_alpha"],
        pretrain_epochs=CFG["pretrain_epochs"],
        **shared,
    )
    for kind in ("p1", "p2"):
        record(run_protograph_variant(
            f"{kind}_classic", proto_walks[kind], inst_walks, p["ontology"],
            p["train"], p["test"], **proto_shared,
        ))
    for kind in ("p1", "p2", "p3"):
        record(run_protograph_variant(
            f"{kind}_bound", proto_walks[kind], inst_walks, p["ontology"],
            p["train"], p["test"], bound_graph=p["graph"], **proto_shared,
        ))
    return rows


def main():
    results_json = OUT_ROOT / "results.json"
    results = json.loads(results_json.read_text(encoding="utf-8")) if results_json.is_file() else {}
    print(f"{TC}:")
    results[TC] = run_tc(TC)
    results_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {TC} ({len(results[TC])} variants) to {results_json}")
    for r in results[TC]:
        print(f"  {r['variant']:>11}: {r['final_acc']:.3f}")


if __name__ == "__main__":
    main()
