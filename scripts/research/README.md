# Research drivers

One-off experiment scripts that reproduce the thesis results. They are grouped by topic here to keep
`scripts/` itself focused on the user-facing CLIs (`train.py`, `search.py`) and the importable library
modules. Figure-by-figure provenance is in [../../docs/REPRODUCE.md](../../docs/REPRODUCE.md).

## How to run

Run from the **repository root** with `uv run`:

```bash
uv run python scripts/research/synthetic/run_synthetic_benchmark.py
uv run python scripts/research/dbpedia/run_dbpedia_compare.py
```

Each driver discovers the repo root by walking up to `pyproject.toml` and puts `scripts/` on `sys.path`,
so it imports the shared library modules (`_pipeline`, `_synthetic_compare`, `_dbpedia_compare`,
`_evaluate`, …) which remain in `scripts/`. Drivers within one topic folder may import each other (e.g.
the synth sweeps import `run_synthetic_benchmark`). Most read/write gitignored data under `v1/`,
`output/`, `notebooks/<name>/`, or `method_proposal/` — see [../../docs/OUTPUT_STRUCTURE.md](../../docs/OUTPUT_STRUCTURE.md).

## Topics

| Folder | Reproduces |
|--------|------------|
| `synthetic/` | Synthetic DLCC benchmark, sweeps, reduced-walk + method-proposal runs, direction grid |
| `dbpedia/` | DBpedia compare + the init/finetune investigation (exp1–6), shortcuts, thesis/schema plots |
| `norm/` | Normalization ablations (classic-norm, synthetic component-norm, bound norm-value) + report builders |
| `bound/` | Clean bound-init full-corpus runs + notebook builder |
| `anchoring/` | Anchoring last-epoch experiment + per-TC LaTeX table export |
| `tc11hard/` | tc11_hard cap-drop investigation + notebook builder |
| `geval/` | GEval init-space sweep |
| `init_strategies/` | P3 init-strategy augmentation + coverage tables |
| `runtime/` | Runtime benchmarks + per-TC / total-lines figures |
| `method_proposal/` | Method-proposal figures, markdown injection, reduced-walks figure |
| `lr_ablation/` | Classic-transfer learning-rate reruns |
| `oneoffs/` | Small utilities (`compare_word2vec`, `seed_walk_cache_from_all_walks`) |
| `legacy/` | Superseded `train_old.py` + its `run_bina_walks` dependency |

## Note

These are research scripts, not a stable API: several need gitignored data or specific cached artifacts
to run, and a few notebooks they feed were assembled by hand (see the notebook index). The library code
they build on lives in `scripts/`.
