#!/usr/bin/env python3
"""Build the report notebook for the classic-init class-code normalization ablation.
Run once; produces notebooks/dbpedia_investigate/classic_norm/report.ipynb."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = ROOT / "notebooks" / "dbpedia_investigate" / "classic_norm" / "report.ipynb"

md = new_markdown_cell
code = new_code_cell
cells = []

cells.append(md(r"""# Classic init: does unit-normalizing class codes help?

**DBpedia DLCC ablation — 10% corpus, p2 stage-1 codes**

Question: the concept-**bound** init averages *unit-normalized* class codes, while
the **classic** (pure class-mean transfer) init averages *raw* stage-1 class codes
and only renormalizes the final per-entity mean. Does adding the same per-class-code
normalization to the classic init change downstream DLCC accuracy?

**Answer (short): no — the effect is within run-to-run noise (Δ ≤ 0.004).**
The rest of this notebook shows the exact lever, the setup, and the numbers."""))

cells.append(md(r"""## The exact lever

Both builders live in [`scripts/_dbpedia_compare.py`](../../../scripts/_dbpedia_compare.py).

**Bound** — each class code is L2-normalized to unit length *before* it is used, and
the per-entity class mix is normalized again:

```python
# _class_codes(...)  -> "Unit code per class id"
out.append(_unit(np.asarray(v, dtype=np.float32)) if v is not None else None)
# alpha (own-types) term:
mix = _unit(np.mean([cls_codes[c] for c in ids], axis=0).astype(np.float32))
```

**Classic** — uses `_resolved_class_codes` (raw stage-1 vectors, climbing
`subClassOf` to the nearest coded ancestor; **no `_unit`**), averages them, then
renormalizes only the **final** mean to `target_norm`:

```python
parts = [resolved[c] for c in cids if resolved[c] is not None]   # RAW norms
v = np.mean(parts, axis=0).astype(np.float32)
n = float(np.linalg.norm(v))
if n > 0:
    v *= target_norm / n            # only the final mean is normalized
```

So in classic, a multi-typed entity's mean is dominated by whichever class has the
larger pretrain norm. The ablation adds a flag `normalize_class_codes` to
`classic_init`; when `True`:

```python
if normalize_class_codes:
    parts = [_unit(p) for p in parts]   # equal-weight every class code (bound-style)
```"""))

cells.append(md(r"""## Setup

- **Corpus:** 10% instance walks (`walks/all_walks_010.txt`), same testbed as exp3/exp6.
- **Codes:** cached p2 stage-1 protograph codes; `used_norm = 8.0`.
- **Init:** *pure classic transfer* — `bound_vectors=None`, so ~800k entities get the
  class-mean init. The two variants are **identical in every way except the flag**.
- **Finetune:** word2vec SG, dim 200, 5 epochs, LR `0.0025 → 0.0001` linear decay
  (the standard classic recipe, `finetune_alpha=0.0025`).
- **Eval:** DLCC LogReg over 89 splits (tc01–tc12 × domains), reported as
  `all / normal / hard`, under two scaling lenses (`standardize`, `raw`).
- **Sanity:** the raw-codes baseline reproduces the `p2_classic ≈ 0.832` reference.

Driver: [`scripts/_invest_classic_norm.py`](../../../scripts/_invest_classic_norm.py)
(`classic_raw` = flag off, `classic_unit` = flag on)."""))

cells.append(code(r"""import json
from pathlib import Path

import pandas as pd

# locate results.json whether the notebook is run from its folder or the repo root
_cands = [Path("results.json"),
          Path("notebooks/dbpedia_investigate/classic_norm/results.json")]
RES = next(p for p in _cands if p.exists())
HERE = RES.parent
results = json.loads(RES.read_text())

meta = results["classic_raw"]
print(f"corpus      : {meta['corpus']}")
print(f"epochs      : {meta['epochs']}   alpha {meta['alpha_bounds'][0]} -> {meta['alpha_bounds'][-1]}")
print(f"variants    : {list(results)}")
print(f"wall (s)    : " + ", ".join(f"{k} {results[k]['seconds']:.0f}" for k in results))
LENSES = ("standardize", "raw")"""))

cells.append(md("## Per-epoch accuracy"))

cells.append(code(r"""def per_epoch(variant, lens):
    rows = []
    for i, e in enumerate(results[variant]["per_epoch"]):
        s = e[lens]
        rows.append({"epoch": i, "all": s["all"], "normal": s["normal"],
                     "hard": s["hard"], "loss": e["loss"]})
    return pd.DataFrame(rows).set_index("epoch")

def side_by_side(lens):
    a = per_epoch("classic_raw", lens)[["all", "normal", "hard"]]
    b = per_epoch("classic_unit", lens)[["all", "normal", "hard"]]
    df = a.join(b, lsuffix="  raw", rsuffix="  unit")
    df["Δ all"] = (b["all"] - a["all"]).round(3)
    return df.round(3)

print("=== standardize lens ===")
side_by_side("standardize")"""))

cells.append(code(r"""print("=== raw lens ===")
side_by_side("raw")"""))

cells.append(md("## Final-epoch summary (the headline)"))

cells.append(code(r"""def final_summary():
    rows = []
    for lens in LENSES:
        a = results["classic_raw"]["per_epoch"][-1][lens]
        b = results["classic_unit"]["per_epoch"][-1][lens]
        for metric in ("all", "normal", "hard"):
            rows.append({"lens": lens, "metric": metric,
                         "raw codes": round(a[metric], 3),
                         "unit codes": round(b[metric], 3),
                         "Δ": round(b[metric] - a[metric], 3)})
    return pd.DataFrame(rows)

summary = final_summary()
summary"""))

cells.append(code(r"""# largest absolute delta, against the ~0.003 run-to-run std (5-seed stability study)
worst = summary["Δ"].abs().max()
print(f"largest |Δ| across all lens/metric = {worst:.3f}")
print(f"run-to-run std (reference)        = 0.003")
print("verdict:", "WITHIN NOISE" if worst <= 0.005 else "POSSIBLE SIGNAL")"""))

cells.append(md("## Per-epoch curves"))

cells.append(code(r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, lens in zip(axes, LENSES):
    for variant, style in (("classic_raw", "o-"), ("classic_unit", "s--")):
        df = per_epoch(variant, lens)
        ax.plot(df.index, df["all"], style, label=variant)
    ax.set_title(f"{lens} lens — all-splits acc")
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
axes[0].set_ylabel("DLCC accuracy")
fig.suptitle("Classic init: raw vs unit-normalized class codes (10% corpus, p2)")
fig.tight_layout()
fig.savefig(HERE / "classic_norm_curves.png", dpi=130, bbox_inches="tight")
plt.show()
print("saved", (HERE / "classic_norm_curves.png"))"""))

cells.append(md(r"""## Why it barely moves: where do the inits actually differ?

The change only alters the *direction* of multi-(resolved-)class entities, and only
for entities whose resolved classes have unequal norms. Single-class entities are
mathematically unchanged. Checking the **epoch-0 init** accuracy per split shows how
little of the test set is affected."""))

cells.append(code(r"""init_a = results["classic_raw"]["init_accs"]["standardize"]
init_b = results["classic_unit"]["init_accs"]["standardize"]
splits = sorted(init_a)
diffs = [(k, round(init_b[k] - init_a[k], 4)) for k in splits
         if abs(init_b[k] - init_a[k]) > 1e-9]
print(f"DLCC splits total            : {len(splits)}")
print(f"bit-identical at init        : {len(splits) - len(diffs)}")
print(f"splits that move at all      : {len(diffs)}")
pd.DataFrame(diffs, columns=["split", "Δ init acc (unit - raw)"])"""))

cells.append(md(r"""## Conclusion

**Unit-normalizing the class codes in the classic init has no meaningful effect on
DLCC** (final Δ ≤ 0.004, within the ±0.003 run-to-run noise; the sign even flips
across epochs). The raw-codes baseline reproduces the `p2_classic ≈ 0.832` reference,
so the harness is faithful.

**Why** — two compounding reasons:

1. Classic already renormalizes each entity's *final* mean to `target_norm`, so
   class-code normalization can only change the **direction** of the mean, and only
   for **multi-(resolved-)class** entities. Single-class entities are unchanged.
2. The DLCC test entities are almost entirely single-resolved-class: **87 of 89
   splits are bit-identical at init**; only the two `people` splits move (Person
   entities carry multiple types — Person/Agent, plus the Person-is-a-Species quirk),
   and only by ≤ 0.001.

The training **loss curves do differ** (the init genuinely changed for multi-typed
entities globally), but that difference never reaches the test set, and finetuning
converges to within noise either way.

**Contrast with the bound init:** there the vector *norm* carries the
degree/cardinality signal, so per-component normalization is structural and matters.
Classic's final per-row renorm erases that signal, leaving only a negligible
direction tweak.

**Recommendation:** keep `normalize_class_codes=False` (the default). No benefit for
the classic init on DLCC; not worth carrying as a tuned knob."""))

cells.append(md(r"""## Reproduce

```bash
.venv/bin/python scripts/_invest_classic_norm.py            # 10% corpus, p2, ~17 min
# full corpus:
.venv/bin/python scripts/_invest_classic_norm.py --walks walks/all_walks.txt
```

Artifacts in this folder: `results.json`, `classic_raw.kv`, `classic_unit.kv`,
`classic_norm_curves.png`, `run.log`. The flag lives on `classic_init` in
`scripts/_dbpedia_compare.py` (default `False`)."""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", OUT)
