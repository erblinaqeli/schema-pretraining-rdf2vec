#!/usr/bin/env python3
"""Export thesis figures from the cached synthetic_visuals embeddings.

Reads the per-TC embedding snapshots cached by notebooks/synthetic_visuals.ipynb
(epoch-0 and epoch-5 test-entity embeddings for vanilla / p1_classic / p3_bound)
and writes publication figures plus a LaTeX summary table to latex/assets/visuals/.

Run from the repo root:  uv run python scripts/plot/synthetic_visuals.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from _common import (  # noqa: E402
    LABEL_COLORS,
    REPO_ROOT,
    SCRIPTS_DIR,
    apply_plot_style,
    fit_lda_2d,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _evaluate import load_labeled_txt  # noqa: E402
from _protograph_gen import iter_rdf_iris  # noqa: E402

CACHE_ROOT = REPO_ROOT / "notebooks" / "synthetic_visuals" / "cache"
DEFAULT_OUT = REPO_ROOT / "latex" / "assets" / "visuals"

TCS = ("tc07", "tc09", "tc14")
VARIANTS = ("vanilla", "p1_classic", "p3_bound")
SEED = 42


def tc_dir(tc: str) -> Path:
    return REPO_ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"


def load_runs(tc: str) -> dict[str, dict]:
    cache_path = CACHE_ROOT / f"{tc}.npz"
    if not cache_path.is_file():
        raise SystemExit(
            f"Missing cache {cache_path}; run notebooks/synthetic_visuals.ipynb first"
        )
    data = np.load(cache_path, allow_pickle=True)
    runs = {}
    for variant in VARIANTS:
        runs[variant] = dict(
            x0=data[f"{variant}_x0"],
            x5=data[f"{variant}_x5"],
            y=data["y_test"],
            accs=[float(a) for a in data[f"{variant}_accs"]],
            mean_cos=float(data[f"{variant}_mean_cos"]),
            n_distinct=int(data[f"{variant}_n_distinct"]),
        )
    return runs


def scatter_labels(ax, xy: np.ndarray, y: np.ndarray, *, s: float = 16, alpha: float = 0.7) -> None:
    for lab in (0, 1):
        m = y == lab
        ax.scatter(
            xy[m, 0], xy[m, 1], s=s, alpha=alpha, color=LABEL_COLORS[lab],
            label=f"label {lab} (n={int(m.sum())})", lw=0,
        )
    ax.grid(True, alpha=0.25)


def plot_pca_vs_lda(tc: str, runs: dict[str, dict], out: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(len(VARIANTS), 2, figsize=(9.5, 3.4 * len(VARIANTS)))
    for row, variant in enumerate(VARIANTS):
        run = runs[variant]
        x, y = run["x0"], run["y"]
        pca = PCA(n_components=2, random_state=SEED)
        xy_pca = pca.fit_transform(x)
        xy_lda, ld1, orth = fit_lda_2d(x, y, seed=SEED)

        scatter_labels(axes[row, 0], xy_pca, y)
        axes[row, 0].set_title(
            f"{variant} — PCA "
            f"(PC1 {pca.explained_variance_ratio_[0] * 100:.1f} %, "
            f"PC2 {pca.explained_variance_ratio_[1] * 100:.1f} %)",
            fontsize=11,
        )
        axes[row, 0].set_xlabel("PC1")
        axes[row, 0].set_ylabel("PC2")

        scatter_labels(axes[row, 1], xy_lda, y)
        axes[row, 1].set_title(
            f"{variant} — LDA (LD1, acc@init {run['accs'][0]:.3f})", fontsize=11
        )
        axes[row, 1].set_xlabel("LD1")
        axes[row, 1].set_ylabel("PC⊥")
    axes[0, 0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_tsne(tc: str, runs: dict[str, dict], out: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 4.0))
    for ax, variant in zip(axes, VARIANTS):
        run = runs[variant]
        xy = TSNE(
            n_components=2, perplexity=35, learning_rate="auto",
            init="pca", random_state=SEED,
        ).fit_transform(run["x0"])
        scatter_labels(ax, xy, run["y"])
        ax.set_title(
            f"{variant}\n{run['n_distinct']} distinct vectors, acc@init {run['accs'][0]:.3f}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def load_target_prop(tc: str) -> str:
    config = tc_dir(tc) / "configuration.txt"
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("Target property:"):
            return line.split(":", 1)[1].strip().strip("<>")
    raise SystemExit(f"No target property in {config}")


def plot_norm_vs_count(tc: str, runs: dict[str, dict], out: Path, *, dpi: int) -> None:
    target_prop = load_target_prop(tc)
    counts: dict[str, int] = {}
    for s, p, _o in iter_rdf_iris(tc_dir(tc) / "graph.nt"):
        if p == target_prop:
            counts[s] = counts.get(s, 0) + 1

    test_tokens, y = load_labeled_txt(tc_dir(tc) / "1000" / "train_test" / "test.txt")
    count_arr = np.array([counts.get(t.strip("<>"), 0) for t in test_tokens])
    prop_short = target_prop.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 3.8), sharey=True)
    offsets = {0: -0.12, 1: 0.12}
    rng = np.random.default_rng(SEED)
    for ax, variant in zip(axes, VARIANTS):
        run = runs[variant]
        norms = np.linalg.norm(run["x0"], axis=1)
        for lab in (0, 1):
            m = y == lab
            jitter = rng.uniform(-0.04, 0.04, size=int(m.sum()))
            ax.scatter(
                count_arr[m] + offsets[lab] + jitter, norms[m], s=16, alpha=0.6,
                color=LABEL_COLORS[lab], label=f"label {lab}", lw=0,
            )
        ax.set_xticks(sorted(set(count_arr.tolist())))
        ax.set_xlabel(f"outgoing {prop_short} edges")
        ax.set_title(f"{variant} (acc@init {run['accs'][0]:.3f})", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("$\\ell_2$ norm at initialization")
    axes[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_ld1_violins(all_runs: dict[str, dict[str, dict]], out: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(
        len(TCS), len(VARIANTS), figsize=(3.4 * len(VARIANTS), 2.9 * len(TCS))
    )
    for i, tc in enumerate(TCS):
        for j, variant in enumerate(VARIANTS):
            run = all_runs[tc][variant]
            lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
            z = lda.fit_transform(run["x0"], run["y"]).ravel()
            # Standardize so panels share a comparable scale; the raw LD1 scale
            # is arbitrary and degenerates when few distinct vectors exist.
            std = z.std()
            if std > 0:
                z = (z - z.mean()) / std
            data = [z[run["y"] == lab] for lab in (0, 1)]
            parts = axes[i, j].violinplot(data, positions=[0, 1], showmeans=True, showextrema=False)
            for body, lab in zip(parts["bodies"], (0, 1)):
                body.set_facecolor(LABEL_COLORS[lab])
                body.set_alpha(0.65)
            axes[i, j].set_xticks([0, 1], ["label 0", "label 1"])
            axes[i, j].set_title(
                f"{tc} / {variant} (acc@init {run['accs'][0]:.3f})", fontsize=9
            )
            if j == 0:
                axes[i, j].set_ylabel("LD1 projection")
            axes[i, j].grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    all_runs = {tc: load_runs(tc) for tc in TCS}

    plot_pca_vs_lda("tc07", all_runs["tc07"], out_dir / "tc07_pca_vs_lda.png", dpi=args.dpi)
    plot_tsne("tc07", all_runs["tc07"], out_dir / "tc07_tsne.png", dpi=args.dpi)
    plot_norm_vs_count("tc09", all_runs["tc09"], out_dir / "tc09_norm_vs_count.png", dpi=args.dpi)
    plot_ld1_violins(all_runs, out_dir / "ld1_violins.png", dpi=args.dpi)
    # The diagnostics summary tables (diagnostics_summary.tex and its appendix)
    # are generated from output/synthetic_benchmark by
    # scripts/plot/diagnostics_table.py, which covers all seven variants.


if __name__ == "__main__":
    main()
