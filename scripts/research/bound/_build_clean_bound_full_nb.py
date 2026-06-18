#!/usr/bin/env python3
"""Build notebooks/clean_bound_full.ipynb — the FULL-corpus report for the four
cleaner concept-bound variants trained by scripts/_invest_clean_bound_full.py."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NB_PATH = ROOT / "notebooks" / "clean_bound_full.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# Cleaner concept-bound init on **full** DBpedia

Companion to [clean_bound.ipynb](clean_bound.ipynb), which proposed two less-noisy
weightings of the asserted class hierarchy inside the concept-bound init (the
`class_weight_mode` knob on
[`concept_bound_vectors`](../scripts/_dbpedia_compare.py)) and tested them on the
**10 %** DBpedia corpus:

| variant | `class_weight_mode` | what it does |
|---|---|---|
| **decay** | `"decay"` | keep all materialized ancestors, weight class `c` by `0.5**hops` from the leaf (leaf=1, parent=0.5, …) |
| **most-specific** | `"most_specific"` | drop ancestors; use only the most-specific asserted class |

This notebook trains both on the **full 1.16 B-token corpus** (`walks/all_walks.txt`,
133 M walks), each **with** and **without** the per-row norm cap @ 16 that the
[dbpedia_investigate](dbpedia_investigate.ipynb) notebook found unfreezes the hub
rows during finetuning. A clean **2×2** (init mode × norm policy), all on **P2**,
all with the protected finetune LR **0.0025 → 0.0001**:

| run | init mode | norm policy |
|---|---|---|
| `p2_bound_decay` | decay | global rescale (no cap) |
| `p2_bound_specific` | most_specific | global rescale (no cap) |
| `p2_bound_decay_cap16` | decay | per-row cap @ 16 |
| `p2_bound_specific_cap16` | most_specific | per-row cap @ 16 |

Everything else matches the existing full-corpus runs (P2 protograph pretrain,
`target_norm=8`, `rolled` direction tag, skip-gram dim 200, 5+5 epochs, seed 42).
Models + per-epoch accuracies are saved under `output/dbpedia/<run>/`.

**Baselines (apples-to-apples).** All rows below are evaluated under the **same**
`evaluate_models` harness (LogReg, standardize lens, 89 DLCC splits):
- **vanilla** and **p2_bound (uniform, no-cap)** — re-evaluated here from the cached
  full-corpus `dbpedia_compare` `.kv` files, so the lens matches exactly;
- **p2bound_cap16 (uniform)** — the full-corpus uniform-hierarchy run with the *same*
  cap @ 16 + LR 0.0025→0.0001, read from its `results.json`. This is the exact
  uniform counterpart to the two cap16 runs, so the only thing that changes across
  the cap16 trio is the class-weighting.

The question: **does reweighting the hierarchy (decay / most-specific) help over the
uniform base at full scale, under each norm policy?**
""")

# --------------------------------------------------------------------------- #
code(r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path("..").resolve()
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _dbpedia_investigate import Model, evaluate_models, load_eval_splits, split_meta, summarize  # noqa: E402

OUT = ROOT / "output" / "dbpedia"
REPORT_DIR = ROOT / "notebooks" / "clean_bound_full"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LENS = "standardize"     # headline lens (matches dbpedia_investigate)
N_JOBS = 12

# (label, source-kind, locator). order = display order.
#   results : output/dbpedia/<name>/results.json  (own standardize+raw evals)
#   ref_kv  : cached .kv re-evaluated here under the same standardize harness
ROWS = [
    ("vanilla (full ref)",            "ref_kv",  "notebooks/dbpedia_compare/vanilla.kv"),
    ("p2_bound (uniform, no-cap)",    "ref_kv",  "notebooks/dbpedia_compare/p2_bound.kv"),
    ("p2_bound_decay",                "results", "p2_bound_decay"),
    ("p2_bound_specific",             "results", "p2_bound_specific"),
    ("p2bound_cap16 (uniform)",       "results", "p2bound_cap16"),
    ("p2_bound_decay_cap16",          "results", "p2_bound_decay_cap16"),
    ("p2_bound_specific_cap16",       "results", "p2_bound_specific_cap16"),
]
print("rows:", [r[0] for r in ROWS])
""")

# --------------------------------------------------------------------------- #
code(r"""
# Reference re-eval: load the cached full-corpus .kv files and score them under the
# SAME standardize harness as the new runs (cached to ref_evals.json -> cheap re-run).
REF_CACHE = REPORT_DIR / "ref_evals.json"
ref_evals = json.loads(REF_CACHE.read_text()) if REF_CACHE.is_file() else {}

needed = [(lbl, loc) for lbl, kind, loc in ROWS if kind == "ref_kv" and lbl not in ref_evals]
if needed:
    from gensim.models import KeyedVectors
    splits = load_eval_splits(ROOT / "v1" / "dbpedia", k=5000)
    print(f"re-evaluating {len(needed)} reference embedding(s) under '{LENS}' "
          f"({len(splits)} splits) ...", flush=True)
    for lbl, loc in needed:
        kv = KeyedVectors.load(str(ROOT / loc), mmap="r")
        m = Model(lbl, kv=kv, dim=kv.vectors.shape[1])
        accs = evaluate_models(splits, [m], scaling=LENS, n_jobs=N_JOBS, seed=42)
        ref_evals[lbl] = accs
        s = summarize(accs)
        print(f"  {lbl:32s} all {s['all']:.3f} (normal {s['normal']:.3f} hard {s['hard']:.3f})",
              flush=True)
        del kv, m
    REF_CACHE.write_text(json.dumps(ref_evals, indent=2) + "\n")
else:
    print("reference evals: all cached")
""")

# --------------------------------------------------------------------------- #
code(r"""
# Load every row into a common shape: per-split accs (final) + init/final summaries.
def load_row(label, kind, loc):
    if kind == "ref_kv":
        accs = ref_evals.get(label)
        if accs is None:
            return None
        s = summarize(accs)
        return dict(label=label, present=True, init=None, final=s,
                    final_accs=accs, status="ref")
    rj = OUT / loc / "results.json"
    if not rj.is_file():
        return dict(label=label, present=False, init=None, final=None,
                    final_accs=None, status="pending")
    r = json.loads(rj.read_text())
    init = r["per_epoch"][0][LENS]
    final = r["per_epoch"][-1][LENS]
    return dict(label=label, present=True, init=init, final=final,
                final_accs=r["final_accs"][LENS], status="ok",
                seconds=r.get("seconds"), norm_policy=r.get("norm_policy"),
                class_weight_mode=r.get("class_weight_mode"))

loaded = [load_row(*r) for r in ROWS]
for d in loaded:
    if d is None:
        continue
    if not d["present"]:
        print(f"  [pending] {d['label']}  (results.json not written yet)")
print("loaded", sum(1 for d in loaded if d and d['present']), "/", len(ROWS), "rows")
""")

# --------------------------------------------------------------------------- #
md(r"""
## Headline — mean LogReg test accuracy (standardize lens, all DLCC splits)

`init_all` is epoch 0 (the init before any finetuning); `final_all` is epoch 5.
`normal` / `hard` are the final means over the easy / `_hard` split halves.
Best `final_*` per column in **bold green**. (Pending runs are omitted until their
`results.json` is written.)
""")
code(r"""
def headline_row(d):
    if d is None or not d["present"]:
        return None
    f = d["final"]; i = d["init"]
    return {
        "init_all":   (i["all"] if i else np.nan),
        "final_all":  f["all"],
        "normal":     f["normal"],
        "hard":       f["hard"],
        "card_H":     f["cardinality (tc09-12) H"],
        "indiv_H":    f["individual {e} (tc04-06) H"],
    }

hl = {d["label"]: headline_row(d) for d in loaded if d and d["present"]}
df_hl = pd.DataFrame(hl).T[["init_all", "final_all", "normal", "hard", "card_H", "indiv_H"]]
display(df_hl.round(4).style.highlight_max(
    subset=["final_all", "normal", "hard", "card_H", "indiv_H"], axis=0,
    props="font-weight:bold;color:#1a7f37;"))
""")

# --------------------------------------------------------------------------- #
md(r"""
## Category breakdown (final epoch, standardize lens)

DLCC families, normal (`N`) and hard (`H`) means: existence (tc01-03),
individual `{e}` (tc04-06), qualified-existence (tc07-08), cardinality (tc09-12).
""")
code(r"""
FAM_COLS = [
    "existence (tc01-03) N", "existence (tc01-03) H",
    "individual {e} (tc04-06) N", "individual {e} (tc04-06) H",
    "qualified-exist (tc07-08) N", "qualified-exist (tc07-08) H",
    "cardinality (tc09-12) N", "cardinality (tc09-12) H",
]
fam = {d["label"]: {c: d["final"][c] for c in FAM_COLS}
       for d in loaded if d and d["present"]}
df_fam = pd.DataFrame(fam).T[FAM_COLS]
display(df_fam.round(3).style.highlight_max(axis=0, props="font-weight:bold;color:#1a7f37;"))
""")

# --------------------------------------------------------------------------- #
md(r"""
## Per-test-case breakdown (final epoch, standardize lens)

Final LogReg test accuracy per test case (mean over domains), one row per tc with a
separate `_hard` row.
""")
code(r"""
def per_tc(accs):
    if accs is None:
        return {}
    b = {}
    for split, a in accs.items():
        tc, _dom, hard = split_meta(split)
        b.setdefault(f"{tc}_hard" if hard else tc, []).append(a)
    return {k: float(np.mean(v)) for k, v in b.items()}

cols = {d["label"]: per_tc(d["final_accs"]) for d in loaded if d and d["present"]}
df_tc = pd.DataFrame(cols)
tcs = sorted({r[:-5] if r.endswith("_hard") else r for r in df_tc.index},
             key=lambda s: int(s[2:]) if s[2:].isdigit() else 99)
order = []
for tc in tcs:
    if tc in df_tc.index: order.append(tc)
    if f"{tc}_hard" in df_tc.index: order.append(f"{tc}_hard")
df_tc = df_tc.loc[order]
df_tc.loc["MEAN"] = df_tc.mean()
display(df_tc.round(3).style.highlight_max(axis=1, props="font-weight:bold;color:#1a7f37;"))
""")

# --------------------------------------------------------------------------- #
md(r"""
## Reading the table

_Filled in after all four runs finish and this notebook is executed._
""")

# --------------------------------------------------------------------------- #
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", NB_PATH)
