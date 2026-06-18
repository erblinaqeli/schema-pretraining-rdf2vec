# Setup

What a fresh clone needs in order to run. The synthetic quick start needs only the environment, the
`v1/` benchmark, and the jRDF2Vec jar. DBpedia additionally needs a materialized graph.

`./setup.sh` automates the steps it can (locate data, locate/move the jar, prerequisite check) and
prints what is missing.

## 1. Environment

```bash
uv sync          # Python 3.13; installs gensim, torch, rdflib, scikit-learn, optuna, pyoxigraph, ...
```

Verify:

```bash
uv run python -c "import torch, gensim, rdflib, sklearn; print('env ok')"
```

## 2. Benchmark data (`v1/`)

`v1/` is gitignored. It originates from the DLCC gold-standard benchmark on Zenodo:
<https://zenodo.org/records/6509715>. The repo also ships a ready-to-use copy as a zip at the root
(`v1 (1).zip`, or `v1.zip` after `setup.sh` normalizes the name), already in the expected layout.

Unzip the local copy so that `v1/synthetic_ontology/` and `v1/dbpedia/` exist (if you fetch from Zenodo
instead, arrange the files into the same layout):

```bash
unzip -n "v1 (1).zip"        # or: ./setup.sh
ls v1/synthetic_ontology/tc01/synthetic_ontology/graph.nt   # sanity check
```

`new_tcs.zip` holds additional synthetic test cases; extract them straight into
`v1/synthetic_ontology/`:

```bash
unzip -n new_tcs.zip -d v1/synthetic_ontology/
```

## 3. jRDF2Vec jar

The `jrdf2vec-duplicate-free` walk mode (the default for training) shells out to the jRDF2Vec jar. The
pipeline expects it at `jars/jrdf2vec-1.3-SNAPSHOT_seed.jar`. Download the jar from the upstream
[dwslab/jRDF2Vec `jars` branch](https://github.com/dwslab/jRDF2Vec/blob/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar)
and save it under that name:

```bash
mkdir -p jars
curl -L -o jars/jrdf2vec-1.3-SNAPSHOT_seed.jar \
  https://github.com/dwslab/jRDF2Vec/raw/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar
```

(The `_seed` suffix is the local name for the deterministic-seeded build used in these experiments.)
`scripts/_jrdf2vec_jar.py::resolve_jar_path()` looks in the repo root and in `jars/`; `setup.sh` moves a
root-level jar into `jars/` for you.

The jar also relies on the local walk server in `python-server/` (a Flask helper jRDF2Vec talks to over
HTTP); see `python-server/` if walk generation reports connection errors.

## 4. DBpedia graph (only for `--dataset dbpedia`)

1. **Download the DBpedia dumps** — the DBpedia snapshot 2021-06 collection (instance-types +
   mappingbased-objects `.ttl.bz2`):
   <https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2021-06>
2. **Materialize** the graph:

   ```bash
   uv run python scripts/train.py --prepare-dbpedia materialize --dir dbpedia_graph
   ```

   This writes `dbpedia_graph/graph.nt` + `ontology.nt`. See `scripts/train.py --help` and
   `scripts/_dbpedia/cli.py` for the other prep subcommands (schema export, SPARQL fetch, seed
   collection).

3. (Optional) Regenerate `seeds/seed_entities.txt` from the DLCC entity lists via the logic in
   `scripts/_dbpedia/v1_collect.py`.

## 5. Smoke test

```bash
# fast: wiring only (no training)
uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2 --dry-run

# full: a real short synthetic run (needs v1/ + the jar)
uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2
```

Or run `./setup.sh --smoke` to do the prerequisite check and then the dry-run.

## Quick reference

| Need | Command |
|------|---------|
| Install env | `uv sync` |
| Lay out data + jar, check prereqs | `./setup.sh` |
| Train synthetic | `uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2` |
| Train DBpedia | `uv run python scripts/train.py --dataset dbpedia --train-mode p2 --finetune-lr 0.00025` |
| Search | `uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p2 --limit 5` |
