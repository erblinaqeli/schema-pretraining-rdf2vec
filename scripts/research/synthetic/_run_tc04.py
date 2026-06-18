"""One-off: produce the tc04 row of tab:synthetic_final.

Runs the six synthetic_compare variants (vanilla, p1_classic, p2_classic,
p1_bound, p2_bound, p3_bound) exactly as notebooks/synthetic_compare.ipynb's
`run_tc`, plus p3_classic exactly as scripts/_run_p3_classic.py, and appends
tc04 into both result files:

  notebooks/synthetic_compare/results.json   (six variants)
  notebooks/p3_classic/results.json          (p3_classic)

Idempotent: re-running skips whatever is already present.
"""
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
    run_vanilla,
    write_protographs,
)

TC = "tc04"
COMPARE_ROOT = ROOT / "notebooks" / "synthetic_compare"
COMPARE_JSON = COMPARE_ROOT / "results.json"
P3_JSON = ROOT / "notebooks" / "p3_classic" / "results.json"

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


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=COMPARE_ROOT / tc,
    )


def build_walks(p: dict) -> tuple[dict, Path]:
    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)
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
    return proto_walks, inst_walks


def run_six(p: dict, proto_walks: dict, inst_walks: Path) -> list[dict]:
    """Mirror of notebooks/synthetic_compare.ipynb run_tc."""
    t0 = time.time()
    shared = dict(dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"])
    rows: list[dict] = []

    def record(res):
        res.pop("model", None)
        res["tc"] = TC
        rows.append(res)
        accs = " ".join(f"{a:.3f}" for a in res["accs"])
        print(f"  [{time.time()-t0:7.1f}s] {res['variant']:>11}  [{accs}]", flush=True)

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


def run_p3_classic(p: dict, proto_walks: dict, inst_walks: Path) -> dict:
    """Mirror of scripts/_run_p3_classic.py."""
    t0 = time.time()
    res = run_protograph_variant(
        "p3_classic",
        proto_walks["p3"], inst_walks, p["ontology"], p["train"], p["test"],
        strategy="all_init", normalize=True, target_norm=CFG["target_norm"],
        finetune_alpha=CFG["finetune_alpha"], pretrain_epochs=CFG["pretrain_epochs"],
        dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"],
    )
    res.pop("model", None)
    res["tc"] = TC
    accs = " ".join(f"{a:.3f}" for a in res["accs"])
    print(f"  [{time.time()-t0:7.1f}s] {TC} p3_classic [{accs}]", flush=True)
    return res


def main() -> None:
    p = tc_paths(TC)
    for k in ("ontology", "graph", "train", "test"):
        assert p[k].is_file(), f"missing {k}: {p[k]}"

    compare = json.loads(COMPARE_JSON.read_text(encoding="utf-8")) if COMPARE_JSON.is_file() else {}
    p3 = json.loads(P3_JSON.read_text(encoding="utf-8")) if P3_JSON.is_file() else {}

    need_six = TC not in compare
    need_p3 = TC not in p3
    if not need_six and not need_p3:
        print(f"{TC}: already present in both result files, nothing to do.")
        return

    print(f"{TC}: building walks ...", flush=True)
    proto_walks, inst_walks = build_walks(p)

    if need_six:
        print(f"{TC}: six variants", flush=True)
        compare[TC] = run_six(p, proto_walks, inst_walks)
        COMPARE_JSON.write_text(json.dumps(compare, indent=2) + "\n", encoding="utf-8")
        print(f"  -> wrote {COMPARE_JSON}", flush=True)
    else:
        print(f"{TC}: six variants already present, skipping", flush=True)

    if need_p3:
        print(f"{TC}: p3_classic", flush=True)
        p3[TC] = run_p3_classic(p, proto_walks, inst_walks)
        P3_JSON.write_text(json.dumps(p3, indent=2) + "\n", encoding="utf-8")
        print(f"  -> wrote {P3_JSON}", flush=True)
    else:
        print(f"{TC}: p3_classic already present, skipping", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
