#!/usr/bin/env python3
"""Build the report notebook for the synthetic normalization ablation
(classic class-code normalization vs bound component normalization, p1/p2/p3)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = ROOT / "output" / "synthetic_norm_ablation" / "report.ipynb"

md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# Effect of component normalization on the synthetic inits

**Synthetic DLCC (tc01–tc15), protographs p1 / p2 / p3 — does L2 normalization of
the init building blocks matter, for the *classic* and the *bound* init families?**

Two levers, one per family, each toggled in isolation:

| family | lever | default | ablation |
|---|---|---|---|
| **classic** (`all_init` class-mean) | unit-normalize each class code **before averaging** | off (`classic_raw`) | on (`classic_unit`) |
| **bound** (concept-bound) | unit-normalize components (codes, class mix, r⊙c pairs) | on (`bound_norm`) | off (`bound_nonorm`) |

**Headline:** normalization is immaterial to *final* synthetic accuracy for both
families — but for different reasons. The classic lever is **mathematically inert**;
the bound lever **changes the init slightly but finetuning erases it.**"""))

cells.append(md(r"""## Why the classic lever is inert (the key insight)

The classic init averages class codes, then renormalizes the *final* per-entity
mean to `target_norm`. But `normalized_stage1_vectors` already rescales **every**
protograph code to the *same* norm (`target_norm = 8`). Averaging equal-norm
vectors gives the same *direction* whether or not you unit-normalize them first, and
the final renorm erases the only remaining difference (magnitude). So
`classic_unit ≡ classic_raw` up to float noise — confirmed below: the **epoch-0 init
accuracy is identical**.

The bound init is different: it combines codes via Hadamard products (`r ⊙ c`) of
**differing** magnitudes and **accumulates edge counts**, so per-component
normalization genuinely reshapes the vectors — its init does move.

Code: levers are `normalize_class_codes` (threaded `protograph_init` →
`maschine_embedding_for_token` → class-root averagers in `_maschine_init.py`) and
`normalize_components` on `concept_bound_vectors` in `_synthetic_compare.py`.
Driver: `scripts/_invest_synth_norm_ablation.py`."""))

cells.append(code(r"""import glob, json
from pathlib import Path
import numpy as np, pandas as pd

def _find_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "scripts").is_dir() and (base / "output").is_dir():
            return base
    return Path.cwd()

ROOT = _find_root()
ABL = ROOT / "output" / "synthetic_norm_ablation"
BENCH = ROOT / "output" / "synthetic_benchmark"

def load(root, key):
    out = {}
    for f in glob.glob(str(root / "tc*/*/metrics.json")):
        r = json.loads(Path(f).read_text())
        out[(r["tc"], r[key])] = r
    return out

ab = load(ABL, "condition")
bench = load(BENCH, "variant")
TCS = sorted({tc for (tc, _) in ab})
KINDS = ("p1", "p2", "p3")
print(f"loaded {len(ab)} ablation rows across {len(TCS)} TCs: {', '.join(TCS)}")

def acc(tc, cond, which):
    r = ab.get((tc, cond))
    if not r: return None
    return r["accs"][0] if which == "init" else r["final_acc"]"""))

cells.append(md("## Classic: `classic_unit − classic_raw` (init Δ is the clean signal)"))

cells.append(code(r"""def delta_table(lo, hi):
    rows = []
    for tc in TCS:
        row = {"tc": tc}
        for k in KINDS:
            i_lo, i_hi = acc(tc, f"{k}_{lo}", "init"), acc(tc, f"{k}_{hi}", "init")
            f_lo, f_hi = acc(tc, f"{k}_{lo}", "final"), acc(tc, f"{k}_{hi}", "final")
            row[f"{k} initΔ"] = None if None in (i_lo, i_hi) else round(i_hi - i_lo, 3)
            row[f"{k} finΔ"]  = None if None in (f_lo, f_hi) else round(f_hi - f_lo, 3)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("tc")
    df.loc["MEAN"] = df.mean().round(4)
    return df

classic = delta_table("classic_raw", "classic_unit")
classic"""))

cells.append(md("## Bound: `bound_nonorm − bound_norm` (init Δ is real; does it survive finetuning?)"))

cells.append(code(r"""bound = delta_table("bound_norm", "bound_nonorm")
bound"""))

cells.append(md("## Aggregate effect + run-to-run noise band"))

cells.append(code(r"""def flat(df, sub):
    cols = [c for c in df.columns if sub in c]
    return df.loc[[t for t in df.index if t != "MEAN"], cols].to_numpy(dtype=float).ravel()

# independent replicate: my single-seed run vs the existing benchmark (same code/seed,
# different process) -> the spread IS the run-to-run thread noise for that condition.
def noise(my, bn):
    d = []
    for tc in TCS:
        for k in KINDS:
            a = acc(tc, f"{k}_{my}", "final"); b = bench.get((tc, f"{k}_{bn}"))
            if a is not None and b is not None:
                d.append(a - b["final_acc"])
    return np.array(d)

rows = []
for name, df in (("classic (unit-raw)", classic), ("bound (nonorm-norm)", bound)):
    fi, ff = flat(df, "initΔ"), flat(df, "finΔ")
    fi, ff = fi[~np.isnan(fi)], ff[~np.isnan(ff)]
    rows.append({"comparison": name,
                 "init Δ mean": round(fi.mean(), 4), "init Δ std": round(fi.std(), 3),
                 "final Δ mean": round(ff.mean(), 4), "final Δ std": round(ff.std(), 3)})
agg = pd.DataFrame(rows)
n_cls, n_bnd = noise("classic_raw", "classic"), noise("bound_norm", "bound")
print(f"run-to-run noise (final): classic_raw std={n_cls.std():.3f} (n={len(n_cls)}), "
      f"bound_norm std={n_bnd.std():.3f} (n={len(n_bnd)})")
print("-> any |final Δ| below ~2*std is indistinguishable from noise.")
agg"""))

cells.append(md("## Picture: init Δ (deterministic) vs final Δ (noisy), per family"))

cells.append(code(r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (name, df, noise_std) in zip(
        axes,
        [("classic  (unit − raw)", classic, n_cls.std()),
         ("bound  (nonorm − norm)", bound, n_bnd.std())]):
    fi, ff = flat(df, "initΔ"), flat(df, "finΔ")
    ax.axhspan(-2*noise_std, 2*noise_std, color="grey", alpha=0.15, label="±2σ noise band")
    ax.axhline(0, color="k", lw=0.6)
    ax.scatter(np.zeros_like(fi)+0, fi, s=22, alpha=0.7, label="init Δ (per TC×kind)")
    ax.scatter(np.zeros_like(ff)+1, ff, s=22, alpha=0.7, label="final Δ")
    ax.plot([0, 1], [np.nanmean(fi), np.nanmean(ff)], "r-o", lw=2, label="mean")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["init (ep0)", "final (ep5)"])
    ax.set_title(name); ax.set_ylabel("Δ accuracy"); ax.grid(alpha=0.3)
    ax.set_ylim(-0.09, 0.09); ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Effect of component normalization — synthetic DLCC (p1/p2/p3, all TCs)")
fig.tight_layout()
fig.savefig(ABL / "synth_norm_effect.png", dpi=130, bbox_inches="tight")
plt.show()
print("saved", ABL / "synth_norm_effect.png")"""))

cells.append(md(r"""## Does bound normalization earn its keep on the cardinality tasks (tc09–tc12)?

The all-task average hides the one place normalization is *designed* to matter: the
qualified-cardinality tasks tc09–tc12, which are separable only through the
accumulated-norm (count) signal that per-component `unit()` protects. If the design
rationale is right, removing normalization should hurt *here* specifically."""))

cells.append(code(r"""CARD = ["tc09", "tc10", "tc11", "tc12"]

def card_table():
    rows = []
    for tc in CARD:
        for k in KINDS:
            ni, xi = acc(tc, f"{k}_bound_norm", "init"),  acc(tc, f"{k}_bound_nonorm", "init")
            nf, xf = acc(tc, f"{k}_bound_norm", "final"), acc(tc, f"{k}_bound_nonorm", "final")
            if None in (ni, xi, nf, xf): continue
            rows.append({"tc": tc, "kind": k,
                         "norm init": round(ni, 3), "nonorm init": round(xi, 3), "Δinit": round(xi - ni, 3),
                         "norm fin": round(nf, 3), "nonorm fin": round(xf, 3), "Δfin": round(xf - nf, 3)})
    return pd.DataFrame(rows).set_index(["tc", "kind"])

card_table()"""))

cells.append(code(r"""# cardinality vs the rest: Δ = nonorm - norm (negative => normalization HELPS)
def group_delta(tcs):
    di, df = [], []
    for tc in tcs:
        for k in KINDS:
            ni, xi = acc(tc, f"{k}_bound_norm", "init"),  acc(tc, f"{k}_bound_nonorm", "init")
            nf, xf = acc(tc, f"{k}_bound_norm", "final"), acc(tc, f"{k}_bound_nonorm", "final")
            if None in (ni, xi, nf, xf): continue
            di.append(xi - ni); df.append(xf - nf)
    return np.array(di), np.array(df)

rows = []
for nm, tcs in (("cardinality 09-12", CARD), ("other tasks", [t for t in TCS if t not in CARD])):
    di, df = group_delta(tcs)
    rows.append({"group": nm, "mean Δinit": round(di.mean(), 4),
                 "mean Δfin": round(df.mean(), 4), "Δfin std": round(df.std(), 3), "n": len(df)})
print("sign of mean Δfin FLIPS: normalization helps cardinality, is mild dead-weight elsewhere.")
print("(magnitudes are within the ~0.023 run-to-run noise -> directional, not proven on one seed.)")
pd.DataFrame(rows)"""))

cells.append(md(r"""**Finding:** the sign of the final-accuracy effect **flips by task group** —
removing normalization *hurts* the cardinality tasks (mean Δfin ≈ −0.004) but
slightly *helps* the others (≈ +0.004). **tc10 is the clearest case**: all three
protographs lose accuracy without normalization (−0.012 / −0.032 / −0.010). The
benefit appears *after* finetuning (at init, no-norm is marginally higher), i.e. the
cleaner count geometry aids learning rather than the raw init readout. Magnitudes sit
within the ±0.023 thread-noise band, so this is directionally consistent with the
design intent but needs multiple seeds to be conclusive."""))

cells.append(md(r"""## Conclusion

**Component normalization does not change final synthetic DLCC accuracy for either
family** — both mean final Δ sit inside the ±2σ run-to-run noise band — but the
*mechanism* differs:

- **Classic — inert by construction.** The epoch-0 init is **identical** whether or
  not class codes are unit-normalized (init Δ ≈ 0.000 per TC). Because
  `normalized_stage1_vectors` pre-equalizes every code to a common norm, averaging is
  scale-invariant and the final renorm erases the rest. Any final-accuracy wobble is
  pure `workers>1` thread noise (std ≈ 0.02), not the lever.

- **Bound — a small init effect that finetuning erases.** Dropping component
  normalization *does* move the init (mean init Δ ≈ +0.007, deterministic — and
  slightly in favour of *no* normalization), but the difference is gone by epoch 5
  (mean final Δ ≈ +0.002, within noise).

This mirrors the DBpedia classic result (`classic_init` normalization Δ ≤ 0.004,
within noise) and extends it: even the *bound* family — where normalization is
structural and visibly reshapes the init — shows no durable accuracy effect on the
synthetic benchmark.

**Recommendation:** keep both defaults (`normalize_class_codes=False`,
`normalize_components=True`). Normalization is not a useful accuracy knob here; the
bound default is fine to keep for its cleaner, scale-controlled geometry."""))

cells.append(md(r"""## Reproduce

```bash
# parallel sweep (3 workers, ~30 min), resumable:
SYNTH_WORKERS=8 .venv/bin/python scripts/_invest_synth_norm_ablation.py tc01 tc02 tc03 tc04 tc05 &
SYNTH_WORKERS=8 .venv/bin/python scripts/_invest_synth_norm_ablation.py tc06 tc07 tc08 tc09 tc10 &
SYNTH_WORKERS=8 .venv/bin/python scripts/_invest_synth_norm_ablation.py tc11 tc12 tc13 tc14 tc15 &
# rebuild this report:
.venv/bin/python scripts/_build_synth_norm_report.py
```

Per-condition metrics: `output/synthetic_norm_ablation/<tc>/<condition>/metrics.json`
(accs[0] = init, final_acc = epoch 5). Levers live on `concept_bound_vectors`
(`normalize_components`) and the `protograph_init`→`_maschine_init` chain
(`normalize_class_codes`), both default = current behavior."""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", OUT)
