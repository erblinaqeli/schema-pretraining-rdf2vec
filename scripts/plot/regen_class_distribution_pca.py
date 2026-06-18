#!/usr/bin/env python3
"""Regenerate the 3.2.1 Class Distribution PCA panels for the thesis.

For each synthetic test case, trains the three variants shown in the thesis
(vanilla / p1_classic / p2_classic) with the current notebook recipe and writes
a 3-panel PCA of the final-epoch test-entity embeddings, coloured by label,
to latex/assets/pca/tcXX_pca.png.

This mirrors notebooks/synthetic_compare.ipynb (the `_synthetic_compare` recipe),
so the figures stay in sync with the freshest training pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SCRIPTS / "plot") not in sys.path:
    sys.path.insert(0, str(SCRIPTS / "plot"))

from sklearn.decomposition import PCA  # noqa: E402

from _common import LABEL_COLORS  # noqa: E402
from _evaluate import load_labeled_txt, tokens_to_embeddings  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    ensure_walks,
    run_protograph_variant,
    run_vanilla,
    write_protographs,
)

OUT_ROOT = REPO_ROOT / "notebooks" / "synthetic_compare"
ASSET_DIR = REPO_ROOT / "latex" / "assets" / "pca"

# Same configuration as notebooks/synthetic_compare.ipynb.
CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    vanilla_alpha=0.025,
    finetune_alpha=0.0025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)

# Panel order / labels shown in the thesis figure.
PANELS = ["vanilla", "p1_classic", "p2_classic"]


def tc_paths(tc: str) -> dict:
    tc_dir = REPO_ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=OUT_ROOT / tc,
    )


def train_variants(tc: str) -> dict[str, dict]:
    """Train vanilla, p1_classic, p2_classic; return {variant: result-with-model}."""
    p = tc_paths(tc)
    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)

    proto_paths = write_protographs(p["ontology"], out)
    proto_walks = {
        kind: ensure_walks(
            path,
            out / f"walks_{kind}.txt",
            walks_per_entity=CFG["proto_walks_per_entity"],
            depth=CFG["depth"],
            seed=CFG["seed"],
            ensure_triple_coverage=True,
        )
        for kind, path in proto_paths.items()
    }
    inst_walks = ensure_walks(
        p["graph"],
        out / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
    )

    shared = dict(dim=CFG["dim"], epochs=CFG["epochs"], min_alpha=CFG["min_alpha"], seed=CFG["seed"])
    proto_shared = dict(
        strategy="all_init",
        normalize=True,
        target_norm=CFG["target_norm"],
        finetune_alpha=CFG["finetune_alpha"],
        pretrain_epochs=CFG["pretrain_epochs"],
        **shared,
    )

    results: dict[str, dict] = {}
    results["vanilla"] = run_vanilla(
        inst_walks, p["train"], p["test"], alpha=CFG["vanilla_alpha"], **shared
    )
    for kind in ("p1", "p2"):
        results[f"{kind}_classic"] = run_protograph_variant(
            f"{kind}_classic", proto_walks[kind], inst_walks, p["ontology"],
            p["train"], p["test"], **proto_shared,
        )
    return results


def plot_panel(ax: plt.Axes, X: np.ndarray, labels: np.ndarray, *, seed: int) -> tuple[int, int]:
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)
    counts = {}
    for label in (0, 1):
        mask = labels == label
        counts[label] = int(mask.sum())
        ax.scatter(
            Z[mask, 0], Z[mask, 1], s=14, alpha=0.75, linewidths=0,
            color=LABEL_COLORS[label],
        )
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f} %)")
    ax.grid(True, alpha=0.2)
    return counts[0], counts[1]


def render_tc(tc: str, results: dict[str, dict]) -> Path:
    test_tokens, y_test = load_labeled_txt(tc_paths(tc)["test"])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.0))
    n_neg = n_pos = 0
    for ax, variant in zip(axes, PANELS):
        model = results[variant]["model"]
        emb = np.asarray(model.wv.vectors, dtype=np.float32)
        w2i = dict(model.wv.key_to_index)
        X, _ = tokens_to_embeddings(test_tokens, emb, w2i, "", 4096, progress=False)
        n_neg, n_pos = plot_panel(ax, X, y_test, seed=CFG["seed"])
        ax.set_title(variant, fontsize=12)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=LABEL_COLORS[0], markeredgecolor="none",
               label=f"label 0 (n={n_neg})"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=LABEL_COLORS[1], markeredgecolor="none",
               label=f"label 1 (n={n_pos})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSET_DIR / f"{tc}_pca.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tcs",
        nargs="+",
        default=[f"tc{i:02d}" for i in range(1, 13)],
        help="Test cases to regenerate (default tc01..tc12).",
    )
    args = ap.parse_args()

    for tc in args.tcs:
        t0 = time.time()
        print(f"{tc}: training {', '.join(PANELS)} ...", flush=True)
        results = train_variants(tc)
        accs = "  ".join(f"{v}={results[v]['accs'][-1]:.3f}" for v in PANELS)
        out = render_tc(tc, results)
        print(f"{tc}: [{time.time() - t0:6.1f}s] {accs}  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
