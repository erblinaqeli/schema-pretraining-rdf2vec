"""Thesis DBpedia figures for 6 variants: vanilla, p1/p2/p3 classic, p2 bound, p2 bound+cap16.

All numbers are LogReg test accuracy under the *raw* lens (no scaling), which is exactly
how `scripts/run_dbpedia_compare.py` evaluates (C=1.0) and matches the exp pipeline's
`raw` lens (C=1.0, max_iter=2000). Aggregation is mean-over-splits via the shared
`summarize`/`per_tc` helpers, so the four compare variants and the two exp variants are
directly comparable. Epoch 0 = init (pre-finetune) in both pipelines.

Data sources (all cached JSON, no embedding recompute):
  vanilla, p1_classic, p2_classic, p2_bound  -> notebooks/dbpedia_compare/results.json
                                                (split_accs[split] = [acc_ep0..acc_ep5])
  p3_classic  -> notebooks/dbpedia_investigate/exp5_p3classic.json
  cap16       -> output/dbpedia/p2bound_cap16/results.json
  (the two exp files carry per_epoch[ep]['raw'] summarize dicts + final_accs['raw'] per-split)

Outputs -> latex/assets/{final_acc,acc_per_epoch_normal_hard,acc_per_epoch_domains}.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _dbpedia_investigate import FAMILIES, per_tc, summarize  # noqa: E402

ASSETS = ROOT / "latex" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- variant registry: key -> (display label, colour) ----------------------
VARIANTS = ["vanilla", "p1_classic", "p2_classic", "p3_classic", "p2_bound", "cap16"]
LABEL = {
    "vanilla": "vanilla",
    "p1_classic": "p1 classic",
    "p2_classic": "p2 classic",
    "p3_classic": "p3 classic",
    "p2_bound": "p2 bound",
    "cap16": "p2 bound + cap16",
}
COLORS = {
    "vanilla": "#444444",
    "p1_classic": "#9ecae1",
    "p2_classic": "#4292c6",
    "p3_classic": "#08519c",
    "p2_bound": "#f16913",
    "cap16": "#6a3d9a",
}
N_EPOCHS = 6

# ---------------------------------------------------------------------------
# Load everything into a uniform shape:
#   summ[variant]      = [summarize-dict for ep 0..5]   (normal/hard/family N,H)
#   final_tc[variant]  = {tc: final normal accuracy (mean over domains)}
# ---------------------------------------------------------------------------
compare = json.loads((ROOT / "notebooks" / "dbpedia_compare" / "results.json").read_text())
p3c = json.loads((ROOT / "notebooks" / "dbpedia_investigate" / "exp5_p3classic.json")
                 .read_text())["p3_classic_FULL"]
cap = json.loads((ROOT / "output" / "dbpedia" / "p2bound_cap16" / "results.json").read_text())

summ: dict[str, list[dict]] = {}
final_tc: dict[str, dict[str, float]] = {}

# --- four compare variants: derive summarize() per epoch from split_accs ----
for v in ("vanilla", "p1_classic", "p2_classic", "p2_bound"):
    sa = compare[v]["split_accs"]
    summ[v] = [summarize({split: vals[ep] for split, vals in sa.items()})
               for ep in range(N_EPOCHS)]
    final_tc[v] = per_tc({split: vals[-1] for split, vals in sa.items()}, hard=False)

# --- p3_classic (exp): per_epoch already carries raw summarize dicts --------
summ["p3_classic"] = [e["raw"] for e in p3c["per_epoch"]]
final_tc["p3_classic"] = per_tc(p3c["final_accs"]["raw"], hard=False)

# --- cap16 (exp) ------------------------------------------------------------
summ["cap16"] = [e["raw"] for e in cap["per_epoch"]]
final_tc["cap16"] = per_tc(cap["final_accs"]["raw"], hard=False)

EPS = np.arange(N_EPOCHS)


# ===========================================================================
# Plot 1 — final accuracy per test case (mean over domains, NORMAL splits)
# ===========================================================================
tcs = sorted(final_tc["vanilla"])  # tc01..tc12
fig, ax = plt.subplots(figsize=(13, 4.4))
x = np.arange(len(tcs))
w = 0.8 / len(VARIANTS)
for i, v in enumerate(VARIANTS):
    ys = [final_tc[v][tc] for tc in tcs]
    ax.bar(x + (i - len(VARIANTS) / 2 + 0.5) * w, ys, w, label=LABEL[v], color=COLORS[v])
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_xticks(x, tcs)
ax.set_ylim(0.45, 1.0)
ax.set_ylabel("final accuracy (mean over domains)")
ax.set_title("DLCC DBpedia (k=5000, normal splits): final accuracy per test case")
ax.legend(ncol=6, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(ASSETS / "final_acc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", ASSETS / "final_acc.png")


# ===========================================================================
# Plot 2 — mean accuracy per epoch (mean over splits), normal & hard
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, (label, key) in zip(axes, (("normal splits", "normal"), ("hard splits", "hard"))):
    for v in VARIANTS:
        ys = [summ[v][ep][key] for ep in range(N_EPOCHS)]
        ax.plot(EPS, ys, marker="o", ms=4, lw=1.5, label=LABEL[v], color=COLORS[v])
    ax.set_xlabel("epoch (0 = init)")
    ax.set_xticks(EPS)
    ax.set_title(f"mean accuracy per epoch — {label}")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("mean LogReg test accuracy")
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(ASSETS / "acc_per_epoch_normal_hard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", ASSETS / "acc_per_epoch_normal_hard.png")


# ===========================================================================
# Plot 3 — per test-case-family accuracy per epoch (mean over domains, HARD)
#   tc07-08 have NO hard splits -> that family is dropped (would be all-NaN).
# ===========================================================================
fam_panels = [(fam, f"{fam} H") for fam in FAMILIES
              if any(not np.isnan(summ["vanilla"][ep].get(f"{fam} H", np.nan))
                     for ep in range(N_EPOCHS))]
fig, axes = plt.subplots(1, len(fam_panels), figsize=(5 * len(fam_panels), 4.2), sharey=True)
for ax, (fam, fkey) in zip(np.atleast_1d(axes), fam_panels):
    for v in VARIANTS:
        ys = [summ[v][ep].get(fkey, np.nan) for ep in range(N_EPOCHS)]
        ax.plot(EPS, ys, marker="o", ms=4, lw=1.5, label=LABEL[v], color=COLORS[v])
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_title(fam, fontsize=11)
    ax.set_xlabel("epoch (0 = init)")
    ax.set_xticks(EPS)
    ax.grid(alpha=0.25)
np.atleast_1d(axes)[0].set_ylabel("mean LogReg test accuracy (hard splits)")
np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Per test-case-family accuracy vs epoch (mean over domains, hard splits)", y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "acc_per_epoch_domains.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", ASSETS / "acc_per_epoch_domains.png")
