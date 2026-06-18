# Reproducing thesis figures & tables

Each result has a **driver** (runs experiments → caches JSON/npz/embeddings) and often a **builder**
(turns the cache into a notebook or a `latex/assets/*` figure/table). Run everything from the repo root
with `uv run python ...`. Most drivers need gitignored inputs (`v1/`, `dbpedia_graph/`, `walks/`,
`output/`) — see [SETUP.md](SETUP.md).

## Synthetic ontology (DLCC test cases)

| Result | Driver → output | Builder / figure |
|--------|------------------|------------------|
| Synthetic benchmark (7 variants × tc01–15) | `run_synthetic_benchmark.py` → `output/synthetic_benchmark/` | `notebooks/synthetic.ipynb` Part 2 (lib `_synthetic_compare.py`); PCA/LDA via `plot/*` |
| most_specific sweep | `run_synth_ms_sweep.py` | — |
| No-norm sweep | `run_synth_nonorm_sweep.py` | `output/synthetic_benchmark_nonorm/` |
| Reduced fine-tune walk budget | `run_reduced_walks.py`, `run_p3_classic_budgets.py` → `method_proposal/data/` | `plot_reduced_walks.py` → `latex/assets/{nw50,nw_bound}/`, `reduced_walks_table.tex` |
| Method proposal write-up | `run_method_proposal.py` → `method_proposal/data/` | `plot_method_proposal.py`, `build_method_proposal_md.py` |
| tc02 / tc04 rows | `_run_tc02_compare.py`, `_run_tc04.py` | — |
| p3_classic | `_run_p3_classic.py`, `run_p3_classic_budgets.py` | `notebooks/p3_classic.ipynb` |

## DBpedia (DLCC benchmark)

| Result | Driver → output | Builder / figure |
|--------|------------------|------------------|
| 6/7-variant compare | `run_dbpedia_compare.py` → `notebooks/dbpedia_compare/results.json` | `notebooks/dbpedia_compare.ipynb`; `plot_dbpedia6_thesis.py` → `latex/assets/{final_acc,acc_per_epoch_*}.png` |
| Damped "mixed" variant | `run_dbpedia_mixed.py` | — |
| Init/finetune investigation (exp1–6) | `_invest_build_artifacts.py`, `_invest_exp{1..6}_*.py` → `notebooks/dbpedia_investigate/` | `_invest_make_notebook.py` → `notebooks/dbpedia_investigate.ipynb` |
| Full-corpus bound recipes | `_invest_full_three.py` | `notebooks/loss_investigate.ipynb` (hand-assembled) |
| Shortcut/leakage analysis | `run_dbpedia_shortcuts.py` → `notebooks/dbpedia_shortcuts/` | `_dbpedia_shortcuts_make_notebook.py` → `dbpedia_shortcuts.ipynb` |
| tc11_hard cap-drop | `_invest_tc11hard.py` | `_build_tc11hard_nb.py` → `dbpedia_tc11hard.ipynb` |
| Vanilla 5-seed stability | `run_dbpedia_vanilla_5x.py` → `notebooks/dbpedia_vanilla_5x_10pct/` | `notebooks/5_times.ipynb` |
| Schema projection | `plot_dbpedia_schema_projection.py` → `dbpedia_schema_*.png` | — |
| Degree/popularity probe | `scripts/_dbpedia/degree_probe.py` → `notebooks/dbpedia_compare/degree_probe.json` | — |

## Ablations: normalization & bound init

| Result | Driver → output | Builder |
|--------|------------------|---------|
| Classic-norm (DBpedia) | `_invest_classic_norm.py` | `_build_classic_norm_report.py` → `notebooks/dbpedia_investigate/classic_norm/report.ipynb` |
| Synthetic component-norm on/off | `_invest_synth_norm_ablation.py` | `_build_synth_norm_report.py` → `output/synthetic_norm_ablation/report.ipynb` |
| Synthetic bound norm-value {4,8,16} | `_invest_synth_norm_value_ablation.py` → `output/synth_norm_value_ablation/` | `_build_norm_value_ablation_nb.py` → `notebooks/norm_ablation.ipynb` |
| Clean bound (decay / most_specific) | `_invest_clean_bound_full.py` | `_build_clean_bound_full_nb.py` → `notebooks/clean_bound_full.ipynb` |

## Anchoring, GEval, init strategies, runtime

| Result | Driver → output | Builder / figure |
|--------|------------------|------------------|
| Anchoring λ-sweep | lib `_anchoring.py` | `notebooks/anchoring.ipynb`; `export_anchoring_pertc_latex.py` → `latex/assets/anchoring_per_tc_tables.tex` |
| Anchoring last-epoch | `anchoring_lastepoch.py` | reads `notebooks/anchoring/results.json` ⚠️ |
| GEval (clustering, classification, analogies) | lib `_geval.py`, `_geval_init_sweep.py` | `notebooks/geval.ipynb` |
| Init strategies | lib `_init_strategies.py`, `_augment_init_strategies_p3.py` | `notebooks/init_strategies.ipynb` |
| Coverage tables | `_coverage_tables.py` → `output/coverage_tables/` | — |
| Direction-aware grid | `_direction_grid.py` | `notebooks/direction_aware_cardinality.ipynb`; `latex/assets/direction/` |
| Runtime benchmarks | `run_runtime.py`, `run_vanilla_timing.py`, `run_protograph_creation_timing.py` → `notebooks/runtime/results.json` | `plot_runtime_per_tc.py` → `runtime_per_tc_table.tex`; `plot_runtime_total_lines.py` → `runtime_total_lines.pdf`; `notebooks/runtime.ipynb` (hand-assembled) |
| LR ablation | `_rerun_p{1,2,3}classic_lr025.py` | `notebooks/classic_lr_comparison.ipynb`; `latex/assets/lr_ablation*.{png,tex}` |

## Projection figures (`scripts/plot/`)

| Figures | Generator |
|---------|-----------|
| `latex/assets/pca/` | `plot/pca.py` (canonical), `project_from_run.py --method pca`, `regen_class_distribution_pca.py` ⚠️ same path |
| `latex/assets/lda/` | `plot/lda.py`, `project_from_run.py --method lda` |
| `latex/assets/pca_drift/` | `plot/pca_drift.py` (retrain), `pca_drift_from_run.py` (cached) |
| `latex/assets/cosine_drift/` | `plot/cosine_drift_from_run.py` |
| `latex/assets/pca_lda/` | `plot/gen_pca_lda_grids.py` |
| `latex/assets/visuals/` | `plot/synthetic_visuals.py`, `plot/diagnostics_table.py` |

## Notes / caveats

- ⚠️ `export_anchoring_pertc_latex.py` and `anchoring_lastepoch.py` read the **all-epochs**
  `notebooks/anchoring/results.json` (collapses to 0.603 at λ=1), not the production results — keep that
  source intact when moving files.
- **Hand-assembled notebooks with no builder script:** `random_jitter.ipynb`, `runtime.ipynb`,
  `classic_lr_comparison.ipynb`, `loss_investigate.ipynb`, `main.ipynb`. Their data comes from the
  drivers above but the notebook itself was authored manually — they do not regenerate end-to-end.
- Several PCA/LDA scripts write the **same** `latex/assets/pca/` paths; regenerate one at a time to avoid
  clobbering.
