# Notebooks

Per-topic result notebooks for the thesis. Each notebook is **tracked**; its paired output directory
(`notebooks/<name>/`, holding regenerable embeddings/walks) is **gitignored** (`notebooks/*/`) and
rebuilt by the driver/builder scripts. Full figure-to-script provenance is in
[../docs/REPRODUCE.md](../docs/REPRODUCE.md).

Run notebooks from the repo root environment (`uv run jupyter lab`). Most read cached results produced
by a driver under `scripts/`; a few were authored by hand.

## Synthetic ontology

| Notebook | Topic | Built / fed by |
|----------|-------|----------------|
| `synthetic.ipynb` | Per-TC synthetic results | `scripts/_synthetic_compare.py` |
| `synthetic_compare.ipynb` | 7-variant DLCC benchmark | `scripts/run_synthetic_benchmark.py`, lib `_synthetic_compare.py` |
| `synthetic_visuals.ipynb` | PCA / LDA / tSNE visuals | `scripts/plot/synthetic_visuals.py` |
| `synth_norm_nonorm_loss_plot.ipynb` | Norm-vs-no-norm loss curves | `scripts/run_synth_nonorm_sweep.py` (relocated from repo root) |

## DBpedia (DLCC)

| Notebook | Topic | Built / fed by |
|----------|-------|----------------|
| `dbpedia.ipynb` | Early DBpedia exploration | — |
| `dbpedia_compare.ipynb` | 6/7-variant compare | `scripts/run_dbpedia_compare.py`, lib `_dbpedia_compare.py` |
| `dbpedia_investigate.ipynb` | Init/finetune investigation (exp1–6) | `scripts/_invest_make_notebook.py` ← `_invest_exp{1..6}_*.py` |
| `dbpedia_shortcuts.ipynb` | Shortcut / leakage analysis | `scripts/_dbpedia_shortcuts_make_notebook.py` ← `run_dbpedia_shortcuts.py` |
| `dbpedia_tc11hard.ipynb` | tc11_hard cap-drop artifact | `scripts/_build_tc11hard_nb.py` ← `_invest_tc11hard.py` |
| `loss_investigate.ipynb` | SGNS loss-curve diagnostic | **hand-authored** (data from `_invest_full_three.py`) |

## Ablations: bound init & normalization

| Notebook | Topic | Built / fed by |
|----------|-------|----------------|
| `clean_bound.ipynb`, `clean_bound_full.ipynb` | Bound init (decay / most_specific × cap) | `scripts/_build_clean_bound_full_nb.py` ← `_invest_clean_bound_full.py` |
| `norm_ablation.ipynb` | Bound norm-value {4,8,16} sweep | `scripts/_build_norm_value_ablation_nb.py` ← `_invest_synth_norm_value_ablation.py` |
| `random_jitter.ipynb` | Random-jitter init control | **hand-authored** (no builder) |

## Other experiments

| Notebook | Topic | Built / fed by |
|----------|-------|----------------|
| `anchoring.ipynb` | Anchoring λ-sweep | lib `scripts/_anchoring.py`; `export_anchoring_pertc_latex.py` |
| `geval_{clustering,classification,regression,analogies}.ipynb` | GEval 4 downstream tasks | lib `scripts/_geval.py` |
| `init_strategies.ipynb` | P2/P3 init strategy comparison | lib `scripts/_init_strategies.py`, `_augment_init_strategies_p3.py` |
| `p3_classic.ipynb` | P3 classic-transfer budgets | `scripts/_run_p3_classic.py`, `run_p3_classic_budgets.py` |
| `direction_aware_cardinality.ipynb` | Direction/cardinality walk grid | `scripts/_direction_grid.py` |
| `5_times.ipynb` | Vanilla 5-seed stability | `scripts/run_dbpedia_vanilla_5x.py` |
| `runtime.ipynb` | Runtime benchmarks | **hand-authored** (data from `run_runtime.py`, `run_vanilla_timing.py`, `run_protograph_creation_timing.py`) |
| `classic_lr_comparison.ipynb` | Classic-transfer LR ablation | **hand-authored** (data from `_rerun_p{1,2,3}classic_lr025.py`) |

## Legacy / scratch

| File | Status |
|------|--------|
| `main.ipynb` | **Superseded** by `scripts/train.py` + `scripts/_pipeline.py`. Early end-to-end scratch notebook; kept because it imports the helper modules below. |
| `build_protographs.py` | Helper imported by `main.ipynb`. ⚠️ Contains a hardcoded `C:/Users/Erblina/...` path — will not run as-is. Superseded by `scripts/_protograph_gen.py`. |
| `resume_graph_train.py` | Helper imported by `main.ipynb`. Superseded by `scripts/_pipeline.py` + `_maschine_init.py`. |
| `embed_protograph.py` | Standalone jRDF2Vec embedding helper. Superseded by `scripts/_jrdf2vec_jar.py`. |
| `build_entity2classes_hier.py` | Original `entity2classes.json` generator. Superseded by `scripts/_protograph_gen.py::build_entity2classes_hier`. |

> The four `.py` helpers are kept here (not relocated) because `main.ipynb` imports them as
> same-directory modules. They are legacy; the canonical code is under `scripts/`.

## Notes

- **Hand-authored notebooks** (`loss_investigate`, `random_jitter`, `runtime`, `classic_lr_comparison`)
  do not regenerate end-to-end from a single script — their cells were assembled manually from the
  driver outputs listed above.
- Local scratch dirs `notebooks/_artifacts/` and `notebooks/_logs/` (parked generated files/logs) are
  gitignored.
