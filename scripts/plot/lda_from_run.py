"""LDA (label-aware) projections of test-entity embeddings after full training.

Companion to scripts/plot/pca_from_run.py: same per-TC, all-seven-methods grid
and epoch (epoch 5, end of fine-tuning), but projects with LDA instead of PCA.
PCA shows max-variance directions and often *misses* a label boundary that lives
in a low-variance direction; LDA (LD1 = the discriminant axis LogReg exploits,
plus an orthogonal PC for layout) reveals it. Mirrors the PCA-vs-LDA recipe in
notebooks/synthetic_visuals.ipynb (fit_lda_2d).

Reads the epoch-5 entity-vector checkpoints from scripts/run_synthetic_benchmark.py
(no retraining):

    output/synthetic_benchmark/<tc>/<variant>/ckpt_epoch05.npz
    output/synthetic_benchmark/<tc>/<variant>/keys.json

Output: latex/assets/lda/<tc>_lda.png

Usage:
    python scripts/plot/lda_from_run.py            # all tc01-tc15
    python scripts/plot/lda_from_run.py tc01       # selected TCs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (SCRIPTS, SCRIPTS / "plot"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _common import LABEL_COLORS, fit_lda_2d  # noqa: E402
from _evaluate import load_labeled_txt, tokens_to_embeddings  # noqa: E402
from pca_drift import ALL_TCS, tc_paths  # noqa: E402

BENCH_ROOT = ROOT / "output" / "synthetic_benchmark"
ASSETS_OUT = ROOT / "latex" / "assets" / "lda"
ASSETS_OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCH = 5  # epoch-5 = end of fine-tuning (after full training)
VARIANTS = ["vanilla", "p1_classic", "p2_classic", "p3_classic",
            "p1_bound", "p2_bound", "p3_bound"]
NCOL = 4


def variant_x5(tc: str, variant: str, test_tokens: list) -> np.ndarray:
    vdir = BENCH_ROOT / tc / variant
    keys = json.loads((vdir / "keys.json").read_text())
    w2i = {k: i for i, k in enumerate(keys)}
    vecs = np.load(vdir / f"ckpt_epoch{EPOCH:02d}.npz")["vectors"]
    x, _ = tokens_to_embeddings(test_tokens, vecs, w2i, "", 4096, progress=False)
    return x


def scatter_labels(ax, xy, y, *, title: str) -> None:
    for lab in (0, 1):
        m = y == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.65, color=LABEL_COLORS[lab],
                   lw=0, label=f"label {lab} (n={int(m.sum())})")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)


def plot_tc(tc: str) -> Path:
    test_tokens, y = load_labeled_txt(tc_paths(tc)["test"])
    y = np.asarray(y)
    nrow = int(np.ceil(len(VARIANTS) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(3.4 * NCOL, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, variant in zip(axes, VARIANTS):
        x = variant_x5(tc, variant, test_tokens)
        if len(np.unique(y)) > 1:
            xy, ld1, orth = fit_lda_2d(x, y, seed=SEED)
            sil = silhouette_score(x, y)
        else:
            xy, ld1, orth, sil = np.zeros((len(x), 2)), float("nan"), float("nan"), float("nan")
        scatter_labels(ax, xy, y,
                       title=f"{variant}  sil={sil:.3f}\n"
                             f"LD1 {ld1*100:.1f}%  PC$\\perp$ {orth*100:.1f}%")
        print(f"  {tc}/{variant}: sil={sil:.3f}  LD1={ld1*100:.1f}%  PC_orth={orth*100:.1f}%",
              flush=True)
    for ax in axes[len(VARIANTS):]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0],
               markersize=8, label="label 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1],
               markersize=8, label="label 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(
        f"{tc.upper()}: LDA (label-aware) projection of test-entity embeddings "
        f"after full training (epoch 5) — all seven methods",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = ASSETS_OUT / f"{tc}_lda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{tc}: wrote {out.relative_to(ROOT)}")
    return out


def main(argv: list[str]) -> None:
    tcs = [a for a in argv if a.startswith("tc")] or ALL_TCS
    for tc in tcs:
        plot_tc(tc)


if __name__ == "__main__":
    main(sys.argv[1:])
