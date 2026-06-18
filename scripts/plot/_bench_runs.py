"""Shared helpers for the synthetic-benchmark "from-run" plot drivers.

These read the per-epoch entity-vector checkpoints written by
scripts/run_synthetic_benchmark.py (no retraining):

    output/synthetic_benchmark/<tc>/<variant>/ckpt_epoch{00..05}.npz
    output/synthetic_benchmark/<tc>/<variant>/keys.json
    output/synthetic_benchmark/<tc>/<variant>/metrics.json

and back the from-run plot drivers:

    project_from_run.py        PCA / LDA projection grid (epoch 5)
    pca_drift_from_run.py      PCA fine-tune drift, epoch 0 -> 5
    cosine_drift_from_run.py   per-epoch cosine drift curves

``ALL_TCS`` and ``tc_paths`` are re-exported from pca_drift, which owns the
single definition of the synthetic test-case layout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (SCRIPTS, SCRIPTS / "plot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import LABEL_COLORS  # noqa: E402  (re-exported)
from _evaluate import tokens_to_embeddings  # noqa: E402
from pca_drift import ALL_TCS, tc_paths  # noqa: E402  (re-exported)

BENCH_ROOT = ROOT / "output" / "synthetic_benchmark"

# Variant order + colours mirror notebooks/synthetic.ipynb (Part 3) (ALL_COLORS):
# vanilla baseline, then p1/p2/p3 for classic and bound init.
VARIANTS = ["vanilla", "p1_classic", "p2_classic", "p3_classic",
            "p1_bound", "p2_bound", "p3_bound"]
VARIANT_COLORS = {
    "vanilla": "#9e9e9e",
    "p1_classic": "#c5b3e6",
    "p2_classic": "#9575cd",
    "p3_classic": "#7b1fa2",
    "p1_bound": "#90caf9",
    "p2_bound": "#1e88e5",
    "p3_bound": "#0d47a1",
}


def variant_dir(tc: str, variant: str) -> Path:
    """Directory holding the saved checkpoints for one TC / variant."""
    return BENCH_ROOT / tc / variant


def variant_w2i(vdir: Path) -> dict[str, int]:
    """Word->index map from a variant's keys.json (vocab order)."""
    keys = json.loads((vdir / "keys.json").read_text())
    return {k: i for i, k in enumerate(keys)}


def bench_embeddings(
    tc: str,
    variant: str,
    epoch: int,
    test_tokens: list,
    w2i: dict[str, int] | None = None,
) -> np.ndarray:
    """Test-entity embeddings for one variant at a given epoch (no retraining).

    Pass ``w2i`` to reuse a vocab map across epochs (avoids re-reading keys.json).
    """
    vdir = variant_dir(tc, variant)
    if w2i is None:
        w2i = variant_w2i(vdir)
    vecs = np.load(vdir / f"ckpt_epoch{epoch:02d}.npz")["vectors"]
    x, _ = tokens_to_embeddings(test_tokens, vecs, w2i, "", 4096, progress=False)
    return x


def scatter_labels(ax, xy: np.ndarray, y: np.ndarray, *, title: str) -> None:
    """Label-coloured scatter of a 2-D projection on a single axis."""
    for lab in (0, 1):
        m = y == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.65, color=LABEL_COLORS[lab],
                   lw=0, label=f"label {lab} (n={int(m.sum())})")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)


def select_tcs(argv: list[str]) -> list[str]:
    """TCs named on the command line (``tcNN`` tokens), or all TCs by default."""
    return [a for a in argv if a.startswith("tc")] or list(ALL_TCS)
