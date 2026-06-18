"""Measure protograph-creation wall-clock (ontology -> protograph_{p1,p2,p3}.nt).

This stage is excluded from scripts/run_runtime.py. We time it here per kind per
test case and merge the numbers into notebooks/runtime/results.json under a new
"protograph_creation_by_kind" key (seconds, mean over REPEATS).
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

from _protograph_gen import build_p1, load_ontology, triples_to_nt_lines  # noqa: E402
from _synthetic_compare import build_p2, build_p3  # noqa: E402

RESULTS_JSON = ROOT / "notebooks" / "runtime" / "results.json"
TCS = [f"tc{i:02d}" for i in range(1, 17) if i != 4]
REPEATS = 5


def _ont_path(tc: str) -> Path:
    return ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology" / "ontology.nt"


def time_creation(ontology_nt: Path) -> dict[str, float]:
    """Per-kind seconds: load_ontology + build + serialize, self-contained per kind."""
    out: dict[str, list[float]] = {"p1": [], "p2": [], "p3": []}
    for _ in range(REPEATS):
        # p1
        t0 = time.perf_counter()
        domains, ranges, children, _it, _cp = load_ontology(ontology_nt)
        p1 = build_p1(domains, ranges)
        _ = "".join(triples_to_nt_lines(p1))
        out["p1"].append(time.perf_counter() - t0)
        # p2 (needs p1)
        t0 = time.perf_counter()
        domains, ranges, children, _it, _cp = load_ontology(ontology_nt)
        p1 = build_p1(domains, ranges)
        p2 = build_p2(p1, children)
        _ = "".join(triples_to_nt_lines(p2))
        out["p2"].append(time.perf_counter() - t0)
        # p3 (needs p1)
        t0 = time.perf_counter()
        domains, ranges, children, _it, _cp = load_ontology(ontology_nt)
        p1 = build_p1(domains, ranges)
        p3 = build_p3(p1, children)
        _ = "".join(triples_to_nt_lines(p3))
        out["p3"].append(time.perf_counter() - t0)
    return {k: round(sum(v) / len(v), 4) for k, v in out.items()}


def main() -> None:
    tcs = sys.argv[1:] or TCS
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    for tc in tcs:
        ont = _ont_path(tc)
        if not ont.is_file():
            print(f"{tc}: MISSING {ont}", flush=True)
            continue
        timings = time_creation(ont)
        results.setdefault(tc, {})["protograph_creation_by_kind"] = timings
        print(f"{tc}: p1={timings['p1']:.4f}s p2={timings['p2']:.4f}s p3={timings['p3']:.4f}s", flush=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nMerged into {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
