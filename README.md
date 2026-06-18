# Knowledge Graphs

Experiments for **RDF2Vec-style walks + Word2Vec/SGNS training** on RDF N-Triples knowledge graphs,
across a synthetic-ontology benchmark (DLCC test cases) and DBpedia. Protograph (MASCHInE) pretraining
initializes instance embeddings from the class schema before fine-tuning on the instance graph.

Run all Python entrypoints with **`uv run python`** from the repository root.

## Install & setup

```bash
uv sync                 # create the env and install pinned deps (Python 3.13)
./setup.sh              # locate benchmark data + the jRDF2Vec jar, then run a prerequisite check
```

A fresh clone needs external artifacts that are **not** committed (the `v1/` benchmark, the jRDF2Vec
jar, and — for DBpedia — a materialized graph). See **[docs/SETUP.md](docs/SETUP.md)** for the full
walkthrough; `setup.sh` automates the parts it can.

## Quick start

```bash
# Train one synthetic test case in P2 mode
uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2

# Train DBpedia in P2 mode (train + in-process DLCC eval)
uv run python scripts/train.py --dataset dbpedia --train-mode p2 --finetune-lr 0.00025

# Hyperparameter search on tc12 (the config YAML is the FIRST positional argument)
uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p2 --limit 5

# PCA comparison plot from existing runs
uv run python scripts/plot/pca.py --mode compare --tc tc01
```

Training modes are `p1`, `p2`, `vanilla`. See `uv run python scripts/train.py --help` for the full flag set.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/SETUP.md](docs/SETUP.md) | Get the data, place the jar, build the DBpedia graph, smoke test |
| [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) | Full script & module catalog + the import convention |
| [docs/OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md) | Layout of `output/`, `notebooks/`, `walks/`, and the data dirs |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Which script/notebook produces each thesis figure & table |
| [repo_plan.md](repo_plan.md) | The repository cleanup plan driving the current reorg |

## Layout

| Path | Role |
|------|------|
| `scripts/train.py` | Main training pipeline (synthetic + DBpedia) |
| `scripts/search.py` | Grid / random / Optuna hyperparameter search |
| `scripts/_*.py` | Importable library modules (pipeline, walks, word2vec, eval, MASCHInE init) |
| `scripts/_dbpedia/` | DBpedia graph build / SPARQL fetch / DLCC eval sub-package |
| `scripts/plot/` | Figure generators from existing runs |
| `scripts/research/` | Thesis experiment drivers, grouped by topic — see [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| `conf/` | YAML hyperparameter configs (consumed by `train.py --config` / `search.py`) |
| `v1/` | Benchmark datasets (synthetic TCs, DBpedia DLCC splits) — *gitignored, see SETUP* |
| `output/` | Experiment runs, `_cache/`, `index/experiments.jsonl` — *gitignored* |
| `walks/`, `dbpedia_graph/` | Generated walk corpora / materialized DBpedia graph — *gitignored* |
| `notebooks/` | Per-topic result notebooks |
| `latex/` | Thesis source (`thesis.tex`), bibliography, figures (`assets/`, `tables/`) |

## Walk generation

Walks are generated automatically during training and cached under `output/_cache/walks/`. To generate
walks manually:

```bash
uv run python scripts/_walks.py v1/synthetic_ontology/tc01/synthetic_ontology/graph.nt walks.txt \
  --mode jrdf2vec-duplicate-free --depth 4 --walks-per-entity 100
```

Modes: `classic` (entity graph) or `jrdf2vec-duplicate-free` (forward-only jRDF2Vec-style chains). The
`jrdf2vec-duplicate-free` mode shells out to the jRDF2Vec jar (see [docs/SETUP.md](docs/SETUP.md)).

## DBpedia graph preparation

```bash
uv run python scripts/train.py --prepare-dbpedia materialize --dir dbpedia_graph
```

See `scripts/train.py --help` and [docs/SETUP.md](docs/SETUP.md) for the other prep subcommands.
