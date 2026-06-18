# Code structure

A catalog of the Python in this repo: what to run, what is importable library code, and what each
research script reproduces. For the data/output side see [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md);
for figure provenance see [REPRODUCE.md](REPRODUCE.md).

## How imports work (read this before moving any file)

Everything lives under `scripts/` and uses a **flat, bare-module import convention**:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # put scripts/ on sys.path
from _pipeline import run_pipeline                          # bare module name
```

Consequences:

- **Run everything from the repository root** with `uv run python scripts/<name>.py`. Paths to
  `v1/`, `walks/`, `output/`, the jar, etc. are resolved relative to the repo root.
- `scripts/_pipeline.py` launches `_walks.py`, `_word2vec.py`, `_evaluate.py`, and `_protograph_gen.py`
  as **subprocesses by path** (not as imports), so those four files must stay in `scripts/`.
- The `scripts/_dbpedia/` sub-package imports as `from _dbpedia.x import ...` (it also relies on
  `scripts/` being on `sys.path`), and `scripts/plot/*` import both `_common` (sibling) and the
  top-level libraries.

There is intentionally **no installed package** — the flat convention is the design. See
`repo_plan.md` §6 for why this is kept as-is.

## Entry points (the things you run)

| Script | What it does |
|--------|--------------|
| `scripts/train.py` | Universal RDF2Vec trainer for synthetic + DBpedia. Modes `p1` / `p2` / `vanilla`; consumes `conf/*.yaml` via `--config`; `--prepare-dbpedia` builds the DBpedia graph. |
| `scripts/search.py` | Hyperparameter search (random / `--grid-search` / `--optuna`). Config YAML is the **first positional arg**; `--tc` and `--training-mode {no_pretrain,p1,p2}` are required. |
| `scripts/_dbpedia/cli.py` | DBpedia data prep CLI: RDFS schema export, instance-subgraph filtering, IRI-NT materialize, SPARQL fetch, v1 URI collection. |
| `scripts/_dbpedia/eval.py` | Evaluate one embedding checkpoint on all DBpedia DLCC splits. |
| `scripts/plot/*.py` | Each is a directly-run figure/table generator (see below). |

## Core library modules (`scripts/_*.py`, imported by entrypoints)

| Module | Responsibility |
|--------|----------------|
| `_pipeline.py` | Top-level synthetic training orchestrator (config → walks → Word2Vec → eval); the glue behind `train.py` / `search.py`. |
| `_kg_io.py` | Config helpers, runtime metrics, output-layout, N-Triples I/O. Foundational leaf (no internal deps). |
| `_walks.py` | Random walks over an `.nt` graph + jRDF2Vec duplicate-free walks. Also runnable as a CLI. |
| `_word2vec.py` | Gensim Word2Vec/SGNS trainer with single-file and two-stage (p1/p2) modes. Library + CLI shim. |
| `_evaluate.py` | Node-classification evaluation of embeddings (`.pt`/`.kv`/`.model`). Widely imported. |
| `_maschine_init.py` | MASCHInE transfer of protograph class vectors to instance vocab; init strategies (classic / bound). |
| `_protograph_gen.py` | Build MASCHInE protographs P1/P2(/P3) from RDF/S schema triples. |
| `_jrdf2vec_jar.py` | Adapter to the jRDF2Vec jar (`-onlyWalks`); `resolve_jar_path()` checks repo root then `jars/`. |

## DBpedia sub-package (`scripts/_dbpedia/`)

`cli.py`, `eval.py`, `degree_probe.py`, plus helpers `graph_build`, `sparql_fetch`, `nt_stream`,
`schema_rdfs`, `entities`, `iri_nt_materialize`, `tc01_walks`, `v1_collect`, `aggregate_results`.
This is the canonical successor of the old `src/dbpedia/` tree.

## Plotting (`scripts/plot/`)

`_common.py` is the shared helper (run-dir resolution, checkpoint IO, label colors, thesis rcParams).
`_bench_runs.py` is the shared loader for the "from-run" drivers (reads the saved
`output/synthetic_benchmark/` checkpoints; owns `VARIANTS`, `scatter_labels`, `bench_embeddings`).
Generators: `accuracy`, `experiment`, `loss`, `pca` (canonical PCA), `lda`, the drift family
(`shifts`, `embedding_drift`, `cosine_drift_from_run`), and the PCA/LDA-grid family
(`project_from_run` (PCA+LDA grid from saved checkpoints), `pca_drift`, `pca_drift_from_run`,
`gen_pca_lda_grids`, `regen_class_distribution_pca`, `diagnostics_table`, `synthetic_visuals`).
Top-level `scripts/plot_*.py` are standalone thesis/DBpedia/runtime figure generators. Most outputs
land in `latex/assets/`.

> Note: several PCA/LDA scripts write the same `latex/assets/pca/` paths and the drift trio overlaps —
> [REPRODUCE.md](REPRODUCE.md) records the canonical generator per figure.

## Experiment-pipeline libraries (imported by the research drivers)

`_synthetic_compare.py` (synthetic vanilla vs P1/P2/P3 variants — the most widely imported experiment
core), `_dbpedia_compare.py` (the DBpedia port with schema/typing cleaning), `_dbpedia_investigate.py`
(hub of the `_invest_*` family), `_init_strategies.py`, `_anchoring.py`, `_geval.py`,
`_runtime_bench.py`, `_dbpedia_shortcuts.py`.

## Research / experiment drivers (run directly; nothing imports them)

These reproduce thesis sections. They live in `scripts/research/<topic>/` (each discovers the repo root
and puts `scripts/` on `sys.path`; see `scripts/research/README.md`):

| Topic | Scripts |
|-------|---------|
| Synthetic benchmark | `run_synthetic_benchmark`, `run_synth_ms_sweep`, `run_synth_nonorm_sweep`, `run_method_proposal`, `run_reduced_walks`, `run_p3_classic_budgets`, `_run_tc02_compare`, `_run_tc04`, `_run_p3_classic` |
| DBpedia compare/investigate | `run_dbpedia_compare`, `run_dbpedia_mixed`, `run_dbpedia_shortcuts`, `run_dbpedia_vanilla_5x`, `_invest_build_artifacts`, `_invest_exp{1..6}_*`, `_invest_full_three`, `_invest_make_notebook`, `_dbpedia_shortcuts_make_notebook`, `plot_dbpedia6_thesis`, `plot_dbpedia_schema_projection` |
| Normalization ablations | `_invest_classic_norm` + `_build_classic_norm_report`, `_invest_synth_norm_ablation` + `_build_synth_norm_report`, `_invest_synth_norm_value_ablation` + `_build_norm_value_ablation_nb` |
| Bound init | `_invest_clean_bound_full` + `_build_clean_bound_full_nb` |
| Anchoring | `anchoring_lastepoch`, `export_anchoring_pertc_latex` |
| tc11_hard | `_invest_tc11hard` + `_build_tc11hard_nb` |
| GEval | `_geval_init_sweep` |
| Init strategies | `_augment_init_strategies_p3`, `_coverage_tables` |
| Runtime | `run_runtime`, `run_vanilla_timing`, `run_protograph_creation_timing` |
| Method proposal | `plot_method_proposal`, `build_method_proposal_md` |
| LR ablation | `_rerun_p{1,2,3}classic_lr025` |
| Utilities | `compare_word2vec`, `seed_walk_cache_from_all_walks` |
| Legacy (superseded) | `train_old.py` (← `train.py`), `run_bina_walks.py` |

## Configs (`conf/`)

`train.py` reads a config via `--config` (defaults: `p1_default.yaml` / `p2_default.yaml` /
`no_pretrain_default.yaml`). `search.py` takes a grid YAML (e.g. `grid_search_p1p2.yaml`,
`grid_search_finetune_stabilization.yaml`) as its first positional argument. The `*_recommended.yaml`
files hold grid-search-winning pretrain settings. `rdf2vec_fixed.yaml` is kept for reference (its old
consumer was removed in the reorg).

## Tests

`tests/test_init_strategy.py` covers the init strategies + finetune CLI plumbing. It imports bare module
names, so run it from the repo root (no pytest config is declared yet).
