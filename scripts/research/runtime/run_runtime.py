"""Run protograph pipeline runtime benchmarks (walk generation excluded)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _runtime_bench import run_tc_timings  # noqa: E402
from _synthetic_compare import ensure_walks  # noqa: E402

WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"
RESULTS_JSON = ROOT / "notebooks" / "runtime" / "results.json"

TCS = [f"tc{i:02d}" for i in range(1, 17) if i != 4]

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
    workers=16,
)

# Pin the worker count: vanilla and variant timings must use the SAME value or
# their fine-tune wall-clocks are not comparable (see tc09-tc12 cross-machine
# artifact). run_vanilla_timing.py asserts against this constant.
EXPECTED_WORKERS = 16
assert CFG["workers"] == EXPECTED_WORKERS, CFG["workers"]


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=WALK_ROOT / tc,
    )


def load_cached_walks(tc: str) -> tuple[dict[str, Path], Path]:
    """Return existing walk corpora; fail fast if any are missing."""
    p = tc_paths(tc)
    out = p["out"]
    proto_walks = {
        kind: out / f"walks_{kind}.txt"
        for kind in ("p1", "p2", "p3")
    }
    inst_walks = out / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt"
    missing = [str(path) for path in (*proto_walks.values(), inst_walks) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{tc}: missing cached walks (generate via synthetic_compare first):\n"
            + "\n".join(f"  {m}" for m in missing)
        )
    return proto_walks, inst_walks


def main() -> None:
    tcs = sys.argv[1:] or TCS
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if RESULTS_JSON.is_file():
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    print(f"Benchmarking {len(tcs)} TCs (walk generation excluded), workers={CFG['workers']}", flush=True)
    for tc in tcs:
        if tc in results:
            print(f"{tc}: cached", flush=True)
            continue
        print(f"{tc}:", flush=True)
        proto_walks, inst_walks = load_cached_walks(tc)
        payload = run_tc_timings(tc, tc_paths(tc), proto_walks, inst_walks, cfg=CFG)
        payload["config"] = CFG
        results[tc] = payload
        RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        for row in payload["variants"]:
            s = row["stages"]
            print(
                f"  {row['variant']:>11}  total={s['total']:7.1f}s  "
                f"pre={s['pretrain_sgns']:5.1f}  bound={s['concept_bound']:5.1f}  "
                f"init={s['init_vectors']:5.1f}  fine={s['finetune_sgns']:5.1f}",
                flush=True,
            )

    print(f"\nSaved {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
