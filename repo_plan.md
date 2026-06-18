# Repository Cleanup Plan

> Goal: make this repo **easy to share, set up, and run**, with **plain, un-abstracted code** and **real documentation**, while **deleting nothing** (everything is preserved — the work is reorganizing + documenting).

This plan is the product of a full file-by-file inventory and an import-graph analysis of the whole tree (93 tracked files + ~150 untracked scripts/notebooks/assets). It is **a plan only** — no files are moved or deleted by reading it. Execute it in the phases below; each phase is independently shippable.

---

## 1. Guiding principles (decisions already locked in)

1. **Organize in place, delete nothing.** The ~80 research scripts and ~25 notebooks that produced thesis figures are *grouped and labeled*, never removed. Truly obsolete files are *flagged* `SUPERSEDED`, not deleted.
2. **Don't repackage the working core.** The code already works through a flat `scripts/`-on-`sys.path` import convention. Converting `scripts/_*.py` into an installed `src/<pkg>/` package would mean rewriting imports in ~70 files **plus notebook cells with baked-in imports**, re-pointing 6 hardcoded subprocess paths, and re-anchoring `repo_root` — a large, error-prone change that fights both the existing design *and* the "no weird abstraction" goal. **We keep the flat importable core and tidy around it.** (A real package is documented as optional Depth B in §6, not recommended now.)
3. **Big data stays out of git; setup is scripted.** `v1/`, `walks/`, `output/`, `dbpedia_graph/`, the jar, and the zips stay gitignored. We add docs + fetch/regenerate scripts so a fresh clone can rebuild them.
4. **Working notes get a home, not a grave.** Root `*.md` thesis-working-notes move to `notes/` (tracked).

---

## 2. Current state snapshot

| Area | Reality found |
|------|---------------|
| **Entry points** | Exactly **two** user CLIs: `scripts/train.py`, `scripts/search.py`. Both `--help` run cleanly under `uv run`. |
| **Importable core** | `_pipeline, _kg_io, _walks, _word2vec, _evaluate, _maschine_init, _protograph_gen, _jrdf2vec_jar` + the `_dbpedia/` sub-package + `plot/` package. |
| **Research tail** | ~55 *driver* scripts (`run_*`, `_invest_*`, `_build_*`, `_rerun_*`, `_run_*`) that nothing imports — each reproduces one thesis section/figure. |
| **Experiment libs** | `_synthetic_compare`, `_dbpedia_compare`, `_dbpedia_investigate`, `_init_strategies`, `_anchoring`, `_geval`, `_runtime_bench` — imported by the drivers; part of the graph. |
| **Notebooks** | ~25 `.ipynb` (41 GB incl. paired output dirs), grouped by topic; most have a builder script. |
| **Import design** | Flat **bare-module** imports (`from _pipeline import …`) after a per-file `sys.path.insert(0, scripts_dir)`. `repo_root = __file__.parent.parent`. `_pipeline` runs `_walks/_word2vec/_evaluate/_protograph_gen` **as subprocesses by path**. |
| **Sizes** | `walks/` 67 G · `output/` 62 G · `notebooks/` 41 G · `dbpedia_graph/` 4.2 G · `v1/` 240 M · `method_proposal/` 90 M · `plots/` 33 G→33 M · root zips 324 M + 36 M + 6 M · jar 24 M. |

---

## 3. Fix-first issues (bugs found — handle before ANY file move)

These are real defects that will bite during or after a reorg. Do them in **Phase 0/1**.

- [ ] **`.gitignore` whitelists the wrong thesis file.** It allows `latex/seminar.tex` (now renamed to `thesis.tex`), so the **live `latex/thesis.tex` and `latex/tables/*.tex` are currently UNTRACKED** — a fresh clone cannot compile the thesis. Add `!latex/thesis.tex` and `!latex/tables/`.
- [ ] **Big artifacts are NOT gitignored** and a single `git add -A` would balloon the repo: `walks/` (67 G), `all_walks/` + `all_walks.zip` (324 M each), `seeds/` (19 M), `method_proposal/` (90 M), `plots/` (33 M), `new_tcs.zip`, `v1 (1).zip`, `jrdf2vec-*.jar`. **Extend `.gitignore` first.**
- [ ] **`notes/` is gitignored** but is the chosen destination for working docs — moving root `*.md` there would silently untrack them. Decide notes/ is tracked and remove it from `.gitignore` (the move and the ignore-rule must be reconciled together).
- [ ] **README links two missing docs:** `CODE_STRUCTURE.md` and `OUTPUT_STRUCTURE.md` (+ a "§5.2.1" anchor) do not exist. Author them (§9) or drop the links.
- [ ] **`conf/` headers name deleted scripts:** `run_grid_search.py` (→ `search.py`), `run_finetune_epoch_eval.py` (gone), `run_rdf2vec_fixed_e2e.py` (gone). Update the headers; verify each YAML still loads.
- [ ] **`search.py` docstring still calls itself `run_grid_search.py`.** Fix.
- [ ] **`tmp_help.txt`** is git-tracked but deleted on disk — finalize with `git rm`. Same for `latex/seminar.tex` (tracked, deleted on disk).
- [ ] **Commit the in-flight reorg as a checkpoint.** The working tree has staged `src/→scripts/` renames + many deletions + many untracked new files. Commit this WIP first so every later step is reversible.

---

## 4. Proposed target structure

Top level stays shallow and legible. **All importable Python stays flat in `scripts/`** (one import root, no package machinery). Only the *non-imported driver scripts* fold into `scripts/research/<topic>/`.

```
knowledge-graphs/
├── README.md                     # quick start + links into docs/
├── pyproject.toml  uv.lock  .python-version  .gitignore
├── setup.sh                      # one-command: fetch data, place jar, smoke-test (NEW)
├── docs/                         # NEW — real documentation
│   ├── SETUP.md                  #   get v1/, the jar, build the DBpedia graph, smoke test
│   ├── CODE_STRUCTURE.md         #   script/module catalog (README already links this)
│   ├── OUTPUT_STRUCTURE.md       #   output/ + notebooks/ + data dir layout
│   └── REPRODUCE.md              #   which script/notebook makes each thesis figure/table
│
├── scripts/                      # RUN + IMPORTABLE CORE (kept flat — preserves imports)
│   ├── train.py  search.py       #   the two user CLIs
│   ├── _pipeline.py _kg_io.py _walks.py _word2vec.py _evaluate.py
│   │   _maschine_init.py _protograph_gen.py _jrdf2vec_jar.py        # core library
│   ├── _synthetic_compare.py _dbpedia_compare.py _dbpedia_investigate.py
│   │   _init_strategies.py _anchoring.py _geval.py _runtime_bench.py  # experiment libs (imported)
│   ├── _dbpedia/                 #   dbpedia sub-package (+ cli.py, eval.py, degree_probe.py)
│   ├── plot/                     #   figure generators (+ _common.py)
│   └── research/                 # NEW — the thesis experiment DRIVERS, grouped (nothing imports these)
│       ├── synthetic/   dbpedia/   norm/   bound/   anchoring/
│       ├── tc11hard/    geval/    init_strategies/  runtime/
│       ├── method_proposal/   lr_ablation/   oneoffs/   legacy/
│       └── README.md            #   "run from repo root with uv run; what each topic reproduces"
│
├── conf/                         # YAML configs (headers fixed)
├── notebooks/                    # grouped by topic; paired output dirs gitignored
│   └── README.md                 #   topic index + builder-script for each notebook
├── notes/                        # working docs (TRACKED): error_report, Fixes, story, toc, norm_explanation
│
├── data/        (gitignored)     # OPTIONAL consolidation — see §7; default: keep v1/ etc. in place
├── jars/        (gitignored)     # jrdf2vec jar (safe move — already a fallback path) + FETCH note
├── output/      (gitignored)     # generated results + _cache/
├── latex/                        # thesis.tex (now tracked), references.bib, assets/, tables/
├── python-server/                # infra sub-tool (Flask jRDF2Vec walk server)
├── slurm/                        # HPC job (env-specific, flagged)
└── tests/                        # test_init_strategy.py
```

**Why drivers can move but core can't, cheaply:** the `run_*/_invest_*/_build_*` drivers are imported by *nothing* (they're leaf `__main__` scripts). They only need `scripts/` on `sys.path` to import the flat libs. So each moved driver gets **one uniform bootstrap header** (find repo root by walking up to `pyproject.toml`; insert `repo/"scripts"`; set `ROOT = repo`). That *replaces* the current brittle `parents[N]` anchoring with something explicit and depth-independent — an improvement, not an abstraction.

---

## 5. Phased execution plan

Each phase is shippable on its own. **You can stop after Phase 3 and already have a shareable, documented repo.** Phases 4–5 are the deeper (optional) code tidy.

### Phase 0 — Safety net (no moves)
- [ ] Commit current WIP as a checkpoint ("mid-reorg snapshot").
- [ ] Extend `.gitignore` to cover every un-ignored big artifact (§3). Verify `git status` shows nothing huge stageable.
- [ ] Fix the `latex/thesis.tex` + `latex/tables/` whitelist; confirm they're now tracked.
- [ ] Remove `notes/` from `.gitignore` (it will hold tracked docs).
- [ ] `git rm` the tracked-but-deleted `tmp_help.txt`, `latex/seminar.tex`.

### Phase 1 — Documentation & setup (no code moves; biggest share/usability win)
- [ ] Author `docs/CODE_STRUCTURE.md`, `docs/OUTPUT_STRUCTURE.md`, `docs/SETUP.md`, `docs/REPRODUCE.md` (§9).
- [ ] Fix README links + the two entrypoint examples; add a one-command setup pointer.
- [ ] Fix `conf/*.yaml` headers and the `search.py` docstring (§3).
- [ ] Write `setup.sh` (§7): unzip `v1`, place the jar in `jars/`, optional DBpedia materialize, smoke-test `train.py --dataset synthetic --tc tc12`.

### Phase 2 — Move only NON-imported files (near-zero risk)
- [ ] Root working-notes → `notes/`: `error_report.md`, `Fixes.md`, `story.md`, `norm_explanation.md`, `toc.md`. (`toc.md` is named in your memory `thesis-restructure-toc` — fine, it's just relocating.)
- [ ] Root stray notebook → `notebooks/`: `synth_norm_nonorm_loss_plot.ipynb`.
- [ ] Stray `.py` helpers out of `notebooks/` → `scripts/research/protograph/` (legacy): `build_protographs.py` *(has a hardcoded `C:/Users/Erblina/…` path — flag)*, `embed_protograph.py`, `build_entity2classes_hier.py`, `resume_graph_train.py`.
- [ ] Move the duplicate `notebooks/jrdf2vec-*.jar` and `notebooks/python-server/` out (use the canonical `jars/` and top-level `python-server/`).
- [ ] Move the jar → `jars/` (safe: `_jrdf2vec_jar.py` already checks `repo_root()/jars/`). Add a fetch/build note (custom seeded jRDF2Vec fork — no in-repo source today).
- [ ] Park loose `notebooks/` artifacts (`walks_p1.txt`, `word2vec_p1.kv/.model`, `*.log`) under a gitignored `notebooks/_artifacts/` + `_logs/`.

### Phase 3 — Group notebooks + flag obsolete (no risk to code)
- [ ] Group `notebooks/` into topic subfolders (synthetic / dbpedia / ablation_bound_norm / anchoring / geval / init_strategies / stability / runtime). Add `notebooks/README.md` mapping each notebook → its builder script.
- [ ] Confirm paired output dirs are gitignored; note the builder/regenerate command for each in `docs/REPRODUCE.md`.
- [ ] Flag (do not delete) `SUPERSEDED`: `notebooks/main.ipynb` (pre-pipeline scratch), `latex/thesis_old.*`.

### Phase 4 — De-risk the anchors (prerequisite for any code move)
> One focused, well-tested change that makes the codebase robust to relocation.
- [ ] In `_kg_io.py` / `_pipeline.py` / `_jrdf2vec_jar.py`, replace `repo_root = __file__.parent.parent` with **explicit repo-root discovery** (walk up to the dir containing `pyproject.toml`). This removes the depth-fragility that makes every move dangerous.
- [ ] Confirm `_pipeline.py`'s four **subprocess paths** (`_walks.py`, `_word2vec.py`, `_evaluate.py`, `_protograph_gen.py`) resolve via the repo-root anchor, not a relative guess. **These stay in `scripts/` regardless.**
- [ ] Smoke-test gate: `train.py` (synthetic tc12, p2) end-to-end + `search.py --help` + `tests/test_init_strategy.py` all green.

### Phase 5 — Fold the research drivers into `scripts/research/<topic>/` (optional, medium effort)
> Do it topic-by-topic; re-run the smoke test after each topic. Stop anytime.
- [ ] Add the uniform bootstrap header (repo-root discovery + `sys.path.insert(repo/"scripts")` + `ROOT = repo`) to each moved driver.
- [ ] Move by topic per the map in §8. Keep coupled driver-sets together (runtime trio; method_proposal trio; synth benchmark + sweeps).
- [ ] Keep **all imported `_*.py` libs flat in `scripts/`** (they're in the import graph). Only the leaf drivers move.
- [ ] Update `docs/REPRODUCE.md` paths; re-run the smoke test + the full driver for one topic to confirm output paths still resolve.

---

## 6. The packaging question (why we are NOT doing it)

A "proper" `src/kg/` package is tempting but **explicitly out of scope** because it conflicts with your goals and the code's reality:

- **~70 files** use bare `from _pipeline import …` after a `sys.path` insert; **notebook cells bake the same imports into committed JSON.** A package move rewrites all of them.
- `_pipeline.py` spawns `_walks/_word2vec/_evaluate/_protograph_gen` as **subprocesses by path** — a package move silently breaks training at runtime, not import time.
- `_maschine_init` and `plot/pca.py` reach into a **private** `_dbpedia.iri_nt_materialize` helper — a clean package split would have to untangle that.
- It adds machinery (`console_scripts`, package config, relative imports) — i.e. the "weird abstraction" you asked to avoid.

**Recommendation:** keep the flat, working convention; just make `repo_root` explicit (Phase 4). Record "promote to `src/kg/`" as a documented *future option* (Depth B) for if this ever needs to be `pip install`-able — not now.

---

## 7. Setup & reproducibility plan (so a fresh clone can run)

The repo is **not runnable from a bare clone today** (data + jar are external). Wire this up:

| Need | Source today | Plan |
|------|--------------|------|
| `v1/` benchmark (synthetic + DBpedia DLCC) | `v1 (1).zip` (36 M) at root; `new_tcs.zip` (6 M) | Rename to `data/v1.zip`; `setup.sh` unzips → `v1/`. Document provenance (DLCC is external). **Only shippable copy — don't discard the zip.** |
| jRDF2Vec jar (24 M) | `jrdf2vec-*.jar` at root | Move to `jars/`; `setup.sh` checks/places it. **Add a build/download note** — it's a custom *seeded* fork with no in-repo source. |
| `dbpedia_graph/graph.nt` (4.2 G) | `train.py --prepare-dbpedia materialize` from external `.ttl.bz2` dumps | `docs/SETUP.md` splits "download DBpedia dumps" (manual link) vs "materialize" (scripted). |
| `seeds/seed_entities.txt` (19 M) | regenerator `generate_seed_entities.py` was **deleted**; logic survives in `_dbpedia/v1_collect.py` | Re-expose as a `scripts/_dbpedia/cli.py` subcommand; document. |
| `walks/`, `output/_cache/walks/` | auto-generated/cached by `train.py`; or seeded from `all_walks/` via `seed_walk_cache_from_all_walks.py` | Document both paths in `docs/OUTPUT_STRUCTURE.md`. |

**`setup.sh` (one command):** unzip `v1` → place jar in `jars/` → (optional) materialize DBpedia graph → smoke-test `uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2`.

**Data-dir naming:** default is to **keep `v1/`, `output/`, `walks/`, `dbpedia_graph/`, `seeds/` at top level** (they're gitignored) — because dozens of scripts hardcode `ROOT/"v1"`, `ROOT/"walks"`, etc. A `data/` consolidation is possible later but needs a global path sweep (a symlink `data/v1 → ../v1` is the cheap middle ground). The table above's `data/v1.zip` refers only to the transport zip, not a live-path rename.

---

## 8. Disposition map (where everything goes)

> Full per-file detail (purpose, imports, evidence-of-use, status) lives in the inventory artifact at
> `…/tasks/wv22g23ox.output`. Summary by destination:

**`scripts/` (stay flat — entrypoints + importable core):** `train.py`, `search.py`, and all libs `_pipeline _kg_io _walks _word2vec _evaluate _maschine_init _protograph_gen _jrdf2vec_jar`, plus experiment libs `_synthetic_compare _dbpedia_compare _dbpedia_investigate _init_strategies _anchoring _geval _runtime_bench`, plus `_dbpedia/` and `plot/`. *No moves — only Phase-4 repo_root fix.*

**`scripts/research/<topic>/` (move in Phase 5 — leaf drivers):**

| Topic | Drivers (move here) |
|-------|---------------------|
| `synthetic/` | `run_synthetic_benchmark`, `run_synth_ms_sweep`, `run_synth_nonorm_sweep`, `run_method_proposal`†, `run_reduced_walks`, `run_p3_classic_budgets` |
| `dbpedia/` | `run_dbpedia_compare`, `run_dbpedia_mixed`, `run_dbpedia_shortcuts`, `run_dbpedia_vanilla_5x`, `_invest_build_artifacts`, `_invest_exp{1..6}_*`, `_invest_full_three`, `_invest_make_notebook`, `_dbpedia_shortcuts_make_notebook`, `plot_dbpedia6_thesis`‡, `plot_dbpedia_schema_projection`‡ |
| `norm/` | `_invest_classic_norm` + `_build_classic_norm_report`, `_invest_synth_norm_ablation` + `_build_synth_norm_report`, `_invest_synth_norm_value_ablation` + `_build_norm_value_ablation_nb` |
| `bound/` | `_invest_clean_bound_full` + `_build_clean_bound_full_nb` |
| `anchoring/` | `anchoring_lastepoch`, `export_anchoring_pertc_latex` *(memory: reads the all-epochs results.json — keep its source intact)* |
| `tc11hard/` | `_invest_tc11hard` + `_build_tc11hard_nb` |
| `geval/` | `_geval_init_sweep` |
| `init_strategies/` | `_augment_init_strategies_p3`, `_run_p3_classic`, `_coverage_tables` |
| `runtime/` | `run_runtime`, `_runtime_bench`§, `run_vanilla_timing`, `run_protograph_creation_timing`, `run_runtime`-coupled set |
| `method_proposal/` | `plot_method_proposal`, `build_method_proposal_md` (+ `run_method_proposal`†) |
| `lr_ablation/` | `_rerun_p1classic_lr025`, `_rerun_p2classic_lr025`, `_rerun_p3classic_lr025` |
| `oneoffs/` | `_run_tc02_compare`, `_run_tc04`, `compare_word2vec`, `seed_walk_cache_from_all_walks` |
| `legacy/` | `train_old.py` *(SUPERSEDED by `train.py`)*, `run_bina_walks.py` *(its dependency)* |

† `run_method_proposal` is shared between synthetic + method_proposal pipelines — pick one home, cross-link. ‡ named in your memory notes — keep filenames stable to avoid desyncing them. § `_runtime_bench` is *imported* by `run_runtime`; if you want zero path-juggling, keep it flat in `scripts/`.

**`scripts/plot/` keeps its package**, but note redundancy to document (not delete): the PCA/LDA-grid family (`pca_from_run`, `lda_from_run`, `pca_drift`, `pca_drift_from_run`, `gen_pca_lda_grids`, `regen_class_distribution_pca`) overlaps and several write the **same** `latex/assets/pca/` paths (collision risk); the drift trio (`shifts`, `embedding_drift`, `cosine_drift_from_run`) overlaps; the flat `plot_*.py` roll their own rcParams instead of `_common`. Record the canonical one per figure in `docs/REPRODUCE.md`.

**`notes/` (tracked):** `error_report.md`, `Fixes.md`, `story.md`, `norm_explanation.md`, `toc.md`.

**Gitignored data/artifacts (keep in place, just ignore):** `v1/`, `output/`, `walks/`, `all_walks/`, `dbpedia_graph/`, `seeds/`, `method_proposal/`, `plots/`, the three root zips, `jars/`.

**`SUPERSEDED` — flag, keep in history (already deleted in working tree or orphaned):** old `src/` tree (`src/dbpedia/*` → now `scripts/_dbpedia/`; `random_walks→_walks`, `train_word2vec→_word2vec`, `evaluate_embeddings→_evaluate`, `protograph_gen→_protograph_gen`, `maschine_init→_maschine_init`) and the **dropped, zero-reference** modules `walk_sampler`, `walk_sampler_bounded`, `wl_walks`, `train_word2vec_torch`, `convert.py`, `tucker.py`, `plot_rdf2vec_pca.py`. Do **not** re-home these into the live tree — they'd masquerade as current code.

---

## 9. Documentation to author

| File | Contents |
|------|----------|
| `README.md` (fix) | What this is · install (`uv sync`) · `setup.sh` · the 4 quick-start commands · links to `docs/` · layout table. Remove dead links. |
| `docs/SETUP.md` | Get `v1/` (unzip), the jar (`jars/` + build note), DBpedia dumps + materialize, seeds regen, smoke test. |
| `docs/CODE_STRUCTURE.md` | The catalog README already promises: entrypoints, the import convention (flat + `sys.path`), core libs, `_dbpedia/`, `plot/`, and `research/<topic>/` with one line each. |
| `docs/OUTPUT_STRUCTURE.md` | `output/` (+ `_cache/`, `index/`), `notebooks/<topic>/` + paired dirs, `walks/`, `dbpedia_graph/`, `latex/assets/`. |
| `docs/REPRODUCE.md` | Per thesis figure/table → the script or notebook builder that makes it (the inventory already maps most of these). |
| `scripts/research/README.md`, `notebooks/README.md` | Topic indexes + "run from repo root with `uv run`" + builder-per-notebook. |

---

## 10. Risk register (landmines for whoever executes)

1. **Subprocess-by-path coupling (most fragile).** `_pipeline.py` runs `_walks/_word2vec/_evaluate/_protograph_gen` via `SCRIPTS_DIR/"X.py"`. They **must stay in `scripts/`**; the Phase-4 anchor fix must keep these paths correct.
2. **`repo_root = __file__.parent.parent`.** Moving any of `_kg_io/_pipeline/_jrdf2vec_jar` a level deeper silently breaks every absolute path to `v1/ walks/ output/ jar`. Fix to explicit pyproject-based discovery *before* any move.
3. **Bare-module imports + notebook cells.** Don't relocate imported `_*.py`; only leaf drivers move, each with the uniform bootstrap. Notebooks bake imports — leave their import cells working (i.e. keep libs in `scripts/`).
4. **`ROOT`-relative data paths in drivers.** Many drivers hardcode `ROOT/"v1"/…`, `ROOT/"walks"/…`, `ROOT/"output"/…` off `__file__`. Re-anchor `ROOT` to repo root in the bootstrap, and **re-run one full driver per topic** to confirm outputs land in the right place.
5. **Cross-layer leak.** `_maschine_init` + `plot/pca.py` import a private `_dbpedia.iri_nt_materialize` helper — keep `_dbpedia/` importable from `scripts/`.
6. **`plot/` needs `_common` on path** and several plot scripts pull 5–6 core libs; keep `plot/` inside `scripts/`.
7. **Foreign/hardcoded paths to flag (won't run as-is):** `notebooks/build_protographs.py` (`C:/Users/Erblina/…`), `slurm/train_word2vec_p1.slurm` (`/pfs/work9/…ma_eqeli…`), `python-server/python_command.txt` (absolute `.venv`).
8. **Anchoring source subtlety (your memory).** `export_anchoring_pertc_latex.py` / `anchoring_lastepoch.py` read the *all-epochs* `notebooks/anchoring/results.json` (collapses to 0.603 at λ=1), not the production one. Don't silently repoint when moving.
9. **Notebooks with NO builder** (`random_jitter`, `runtime`, `classic_lr_comparison`, `loss_investigate`, `main`) — hand-assembled or built by deleted drivers. Document the manual data chain; don't assume they auto-regenerate.

---

## 11. Open decisions for you

1. **How far to go on code moves?** Phases 0–3 (docs, setup, notes, notebook grouping, gitignore) give you a shareable, documented repo with ~zero risk. Phases 4–5 (anchor fix + folder the drivers) make `scripts/` genuinely tidy but touch ~55 files. *Recommend: do 0–4 now; do 5 topic-by-topic when you have time.*
2. **`data/` consolidation?** Keep `v1/ walks/ output/ …` at top level (cheap, safe) vs. move under `data/` (cleaner, needs a global path sweep or symlinks). *Recommend: keep in place now.*
3. **`scripts/research/` vs top-level `research/`?** Nesting under `scripts/` keeps one import root and a one-line bootstrap. A sibling `research/` reads cleaner but needs two path entries. *Recommend: `scripts/research/`.*
4. **`_runtime_bench`, `compare_word2vec`, `seed_walk_cache_from_all_walks`** are tool-ish, not pure experiments — `research/` or keep as `scripts/` utilities? *Minor; recommend `research/runtime` + `research/oneoffs`.*

---

### TL;DR
Keep the working flat core; fix the `.gitignore`/thesis-tracking/broken-doc bugs; write `docs/` + `setup.sh` so a clone can run; move only notes, notebook strays, and the jar in Phase 2–3; then (optionally) re-anchor `repo_root` and fold the ~55 research drivers into `scripts/research/<topic>/` with a uniform bootstrap, gated by a smoke test. Nothing is deleted — obsolete files are flagged, not removed.
