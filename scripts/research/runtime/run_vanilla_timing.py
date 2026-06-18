"""Measure vanilla (from-scratch RDF2Vec) fine-tuning runtime per test case.

Vanilla has no protograph creation, no pretraining and a random (not pretrained)
init, so only the finetune SGNS stage applies. We time build_vocab + finetune on
the same cached instance walks used by scripts/run_runtime.py and merge the
result into notebooks/runtime/results.json under a per-TC "vanilla" key.
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

from gensim.models.word2vec import LineSentence  # noqa: E402

from _synthetic_compare import new_skipgram_model  # noqa: E402

WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"
RESULTS_JSON = ROOT / "notebooks" / "runtime" / "results.json"
TCS = [f"tc{i:02d}" for i in range(1, 17) if i != 4]

# Must match scripts/run_runtime.py CFG["workers"]; vanilla and variant
# fine-tune timings are only comparable when measured with the same workers.
EXPECTED_WORKERS = 16


def _inst_walks(tc: str, cfg: dict) -> Path:
    return WALK_ROOT / tc / f"walks_instance_w{cfg['walks_per_entity']}_d{cfg['depth']}.txt"


def time_vanilla(inst_walks: Path, cfg: dict) -> dict[str, float]:
    # vanilla baseline uses the standard word2vec LR (0.025); LR does not affect wall-clock.
    t0 = time.perf_counter()
    model = new_skipgram_model(
        dim=cfg["dim"], alpha=0.025, min_alpha=cfg["min_alpha"],
        seed=cfg["seed"], workers=cfg["workers"],
    )
    model.build_vocab(LineSentence(str(inst_walks)))
    build_vocab = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    model.train(
        corpus_iterable=LineSentence(str(inst_walks)),
        total_examples=model.corpus_count,
        epochs=cfg["epochs"], start_alpha=0.025, end_alpha=cfg["min_alpha"],
    )
    finetune = round(time.perf_counter() - t0, 3)
    return {"build_vocab": build_vocab, "finetune_sgns": finetune}


def main() -> None:
    tcs = sys.argv[1:] or TCS
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    for tc in tcs:
        if tc not in results:
            print(f"{tc}: no entry in results.json, skipping", flush=True)
            continue
        cfg = results[tc]["config"]
        assert cfg["workers"] == EXPECTED_WORKERS, (
            f"{tc}: variant timings used workers={cfg['workers']}, expected "
            f"{EXPECTED_WORKERS}; re-run scripts/run_runtime.py for this TC."
        )
        inst = _inst_walks(tc, cfg)
        if not inst.is_file():
            print(f"{tc}: MISSING {inst}", flush=True)
            continue
        stages = time_vanilla(inst, cfg)
        results[tc]["vanilla"] = stages
        print(f"{tc}: workers={cfg['workers']} build_vocab={stages['build_vocab']:.2f}s finetune={stages['finetune_sgns']:.2f}s", flush=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nMerged into {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
