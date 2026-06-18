"""Run p3_classic on synthetic TCs (one-off helper for notebooks/p3_classic.ipynb)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _synthetic_compare import (  # noqa: E402
    ensure_walks,
    run_protograph_variant,
    write_protographs,
)

OUT_ROOT = ROOT / "notebooks" / "synthetic_compare"
RESULTS_JSON = ROOT / "notebooks" / "p3_classic" / "results.json"

TCS = [f"tc{i:02d}" for i in range(1, 16) if i != 4]

CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    finetune_alpha=0.0025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=OUT_ROOT / tc,
    )


def run_p3_classic(tc: str) -> dict:
    p = tc_paths(tc)
    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    proto_paths = write_protographs(p["ontology"], out)
    proto_walks = ensure_walks(
        proto_paths["p3"],
        out / "walks_p3.txt",
        walks_per_entity=CFG["proto_walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
        ensure_triple_coverage=True,
    )
    inst_walks = ensure_walks(
        p["graph"],
        out / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
    )

    res = run_protograph_variant(
        "p3_classic",
        proto_walks,
        inst_walks,
        p["ontology"],
        p["train"],
        p["test"],
        strategy="all_init",
        normalize=True,
        target_norm=CFG["target_norm"],
        finetune_alpha=CFG["finetune_alpha"],
        pretrain_epochs=CFG["pretrain_epochs"],
        dim=CFG["dim"],
        epochs=CFG["epochs"],
        min_alpha=CFG["min_alpha"],
        seed=CFG["seed"],
    )
    res.pop("model", None)
    res["tc"] = tc
    accs = " ".join(f"{a:.3f}" for a in res["accs"])
    print(f"[{time.time()-t0:6.1f}s] {tc} p3_classic [{accs}]", flush=True)
    return res


def main() -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if RESULTS_JSON.is_file():
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    for tc in TCS:
        if tc in results:
            print(f"{tc}: cached", flush=True)
            continue
        print(f"{tc}:", flush=True)
        results[tc] = run_p3_classic(tc)
        RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
