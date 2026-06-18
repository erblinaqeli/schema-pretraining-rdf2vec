# Output & data structure

What every data / artifact directory is, whether it is **input** or **generated**, whether it is
committed, and how to (re)produce it. All of the large directories below are **gitignored** — a fresh
clone obtains them via [SETUP.md](SETUP.md) or by regenerating them.

## Input data

| Path | What | Regenerate / obtain |
|------|------|---------------------|
| `v1/` | Benchmark datasets: `synthetic_ontology/<tc>/synthetic_ontology/graph.nt` + DLCC splits, `dbpedia/<tc>/<domain>/<n>/` entity splits, `geval/`. | External (DLCC). Unzip `v1.zip` (see SETUP). Not regenerable from this repo. |
| `dbpedia_graph/` | `graph.nt` + `ontology.nt` (generated) materialized from downloaded DBpedia `.ttl.bz2` dumps (input). | `scripts/train.py --prepare-dbpedia materialize --dir dbpedia_graph`. Dumps are an external download (SETUP). |
| `seeds/` | `seed_entities.txt`: deduplicated DLCC DBpedia entity URIs in IRI-NT form. | Logic in `scripts/_dbpedia/v1_collect.py` (`collect_dbpedia_resource_uris`). |

## Generated artifacts

| Path | What | Produced by |
|------|------|-------------|
| `output/` | Trained models, `_cache/walks/` (the canonical walk cache), `dbpedia/` + `dbpedia_protograph_only/` eval runs, `logs/`, `index/experiments.jsonl`, comparison PNGs. | `scripts/train.py` and most `run_*` / `_invest_*` drivers. |
| `walks/` | Walk corpora: `all_walks_{005,010,025}.txt` + vocab `.pkl`, `jrdf2vec_g_full/`, `p1/`, `p2/`. | `scripts/_walks.py` + `scripts/_jrdf2vec_jar.py`; p1/p2 via `scripts/run_bina_walks.py`. Cached automatically by `train.py` under `output/_cache/walks/`. |
| `all_walks/` | Pre-baked per-TC walk shards (`tcNN/walk_file_0.txt.gz`) used to seed the cache. | `scripts/seed_walk_cache_from_all_walks.py` consumes these. |
| `method_proposal/` | `data/` + `figures/` for the method-proposal write-up. | `scripts/run_method_proposal.py` (+ `plot_method_proposal.py`, `build_method_proposal_md.py`). |
| `plots/` | Anchor-regularization sweep figures. | `scripts/plot/*.py`. |

## Notebooks (`notebooks/`)

Each topical `.ipynb` is paired with a generated output subdirectory (embeddings `.npy`/`.kv`, walks).
**The `.ipynb` files are tracked; the paired run dirs are gitignored** (`notebooks/*/`) because they are
large and regenerable. The builder script for each notebook is listed in [REPRODUCE.md](REPRODUCE.md).

Themes: synthetic, dbpedia (compare / investigate / shortcuts / tc11hard), bound+norm ablation,
anchoring, geval (4 tasks), init_strategies, stability (5x), runtime, direction-aware, random_jitter.

## LaTeX (`latex/`)

| Path | Tracked? | Notes |
|------|----------|-------|
| `thesis.tex`, `references.bib` | yes | The thesis source + bibliography. |
| `tables/*.tex` | yes | Generated result tables `\input` by the thesis. |
| `assets/` | yes | Generated figures + figure-tables (PNG/PDF/TEX) from the plot scripts. |
| build artifacts (`*.aux`, `*.pdf`, `thesis_old.*`, …) | no | Ignored by the `latex/**` rule. |

## What is gitignored

`v1/`, `output/`, `walks/`, `all_walks/`, `dbpedia_graph/`, `seeds/`, `method_proposal/`, `plots/`,
`notebooks/*/`, all `*.zip` / `*.jar`, `jars/`, `*.log`, and `latex/` build artifacts. See `.gitignore`.
