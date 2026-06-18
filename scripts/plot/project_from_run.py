"""PCA / LDA projections of test-entity embeddings after full training, from the saved run.

One figure per TC: a 2-D projection of the epoch-5 (end of fine-tuning) test-entity
embeddings for all seven methods (vanilla + p1/p2/p3 x classic/bound), coloured by
label, with a per-panel silhouette score and explained variance. This matches the
thesis figure caption ("entity embeddings ... taken after full training") and the
scatter style of notebooks/synthetic.ipynb (Part 3) Section 10.

Two projections (same per-TC, all-seven-methods grid; epoch 5):

  --method pca   directions of maximum variance (often *misses* a label boundary
                 that lives in a low-variance direction)
  --method lda   LD1 (the label-aware discriminant axis LogReg exploits) plus an
                 orthogonal PC for layout, which reveals that boundary
  --method both  both of the above (default)

Instead of retraining it reads the epoch-5 entity-vector checkpoints written by
scripts/run_synthetic_benchmark.py (see scripts/plot/_bench_runs.py). PCA is written
to latex/assets/pca/<tc>_pca.png, LDA to latex/assets/lda/<tc>_lda.png.

Usage:
    python scripts/plot/project_from_run.py                    # both methods, all TCs
    python scripts/plot/project_from_run.py --method lda tc01  # LDA only, selected TCs
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from _bench_runs import (  # noqa: E402
    LABEL_COLORS,
    ROOT,
    VARIANTS,
    bench_embeddings,
    scatter_labels,
    select_tcs,
    tc_paths,
)
from _common import fit_lda_2d  # noqa: E402
from _evaluate import load_labeled_txt  # noqa: E402

SEED = 42
EPOCH = 5  # epoch-5 = end of fine-tuning (after full training)
NCOL = 4

ASSET_DIRS = {
    "pca": ROOT / "latex" / "assets" / "pca",
    "lda": ROOT / "latex" / "assets" / "lda",
}
SUPTITLE = {
    "pca": "PCA of test-entity embeddings after full training (epoch 5) — all seven methods",
    "lda": "LDA (label-aware) projection of test-entity embeddings after full training "
           "(epoch 5) — all seven methods",
}


def project(method: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Project ``x`` to 2-D; return (xy, panel-title suffix, console suffix)."""
    sil = silhouette_score(x, y) if len(np.unique(y)) > 1 else float("nan")
    if method == "pca":
        pca = PCA(n_components=2, random_state=SEED)
        xy = pca.fit_transform(x)
        ev = pca.explained_variance_ratio_ * 100
        return (xy,
                f"sil={sil:.3f}\nPC1 {ev[0]:.1f}%  PC2 {ev[1]:.1f}%",
                f"sil={sil:.3f}  ev=({ev[0]:.1f}%, {ev[1]:.1f}%)")
    # method == "lda"
    if len(np.unique(y)) > 1:
        xy, ld1, orth = fit_lda_2d(x, y, seed=SEED)
    else:
        xy, ld1, orth = np.zeros((len(x), 2)), float("nan"), float("nan")
    return (xy,
            f"sil={sil:.3f}\nLD1 {ld1*100:.1f}%  PC$\\perp$ {orth*100:.1f}%",
            f"sil={sil:.3f}  LD1={ld1*100:.1f}%  PC_orth={orth*100:.1f}%")


def plot_tc(tc: str, method: str):
    test_tokens, y = load_labeled_txt(tc_paths(tc)["test"])
    y = np.asarray(y)
    nrow = int(np.ceil(len(VARIANTS) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(3.4 * NCOL, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, variant in zip(axes, VARIANTS):
        x = bench_embeddings(tc, variant, EPOCH, test_tokens)
        xy, title_suffix, console = project(method, x, y)
        scatter_labels(ax, xy, y, title=f"{variant}  {title_suffix}")
        print(f"  {tc}/{variant}: {console}", flush=True)
    for ax in axes[len(VARIANTS):]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0],
               markersize=8, label="label 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1],
               markersize=8, label="label 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(f"{tc.upper()}: {SUPTITLE[method]}", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_dir = ASSET_DIRS[method]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tc}_{method}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{tc}: wrote {out.relative_to(ROOT)}")
    return out


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", choices=["pca", "lda", "both"], default="both")
    parser.add_argument("tcs", nargs="*", help="TCs to plot, e.g. tc01 tc09 (default: all)")
    args = parser.parse_args(argv)

    methods = ["pca", "lda"] if args.method == "both" else [args.method]
    for method in methods:
        for tc in select_tcs(args.tcs):
            plot_tc(tc, method)


if __name__ == "__main__":
    main(sys.argv[1:])
