"""Per-epoch cosine-drift plots, sourced from the saved benchmark run.

For each variant and TC, snapshots the test-entity embeddings at epoch 0
(initialization) and after every finetune epoch, then measures the *cosine drift*

    d_i = mean_e ( 1 - |cos(x_e^(0), x_e^(i))| )

— the mean over test entities of 1 - |cosine similarity| between epoch 0 and
epoch i (so d_0 = 0 by construction; |.| treats a sign flip as no drift). This
mirrors Section 11 of notebooks/synthetic_visuals.ipynb, but instead of
retraining it reads the per-epoch entity-vector checkpoints written by
scripts/run_synthetic_benchmark.py:

    output/synthetic_benchmark/<tc>/<variant>/ckpt_epoch{00..05}.npz
    output/synthetic_benchmark/<tc>/<variant>/keys.json

One figure per TC (all seven methods) -> latex/assets/cosine_drift/<tc>_cosine_drift.png.

Usage:
    python scripts/plot/cosine_drift_from_run.py            # all tc01-tc15
    python scripts/plot/cosine_drift_from_run.py tc01       # selected TCs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (SCRIPTS, SCRIPTS / "plot"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _evaluate import load_labeled_txt, tokens_to_embeddings  # noqa: E402
from pca_drift import ALL_TCS, tc_paths  # noqa: E402

BENCH_ROOT = ROOT / "output" / "synthetic_benchmark"
ASSETS_OUT = ROOT / "latex" / "assets" / "cosine_drift"
ASSETS_OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = 5  # ckpt_epoch00 (init) .. ckpt_epoch05 (end of finetune)

# variant order + colours mirror notebooks/synthetic_visuals.ipynb (ALL_COLORS).
ALL_VARIANTS = ["vanilla", "p1_classic", "p2_classic", "p3_classic",
                "p1_bound", "p2_bound", "p3_bound"]
ALL_COLORS = {
    "vanilla": "#9e9e9e",
    "p1_classic": "#c5b3e6",
    "p2_classic": "#9575cd",
    "p3_classic": "#7b1fa2",
    "p1_bound": "#90caf9",
    "p2_bound": "#1e88e5",
    "p3_bound": "#0d47a1",
}


def _mean_cos_drift(x0: np.ndarray, xi: np.ndarray) -> float:
    """mean_e (1 - |cos(x0_e, xi_e)|) over rows with non-zero norm at both epochs."""
    n0, ni = np.linalg.norm(x0, axis=1), np.linalg.norm(xi, axis=1)
    ok = (n0 > 0) & (ni > 0)
    if not ok.any():
        return float("nan")
    cos = (x0[ok] * xi[ok]).sum(1) / (n0[ok] * ni[ok])
    return float((1.0 - np.abs(cos)).mean())


def _emb_at(vdir: Path, epoch: int, test_tokens: list, w2i: dict) -> np.ndarray:
    vecs = np.load(vdir / f"ckpt_epoch{epoch:02d}.npz")["vectors"]
    x, _ = tokens_to_embeddings(test_tokens, vecs, w2i, "", 4096, progress=False)
    return x


def variant_curve(vdir: Path, test_tokens: list) -> list[float]:
    keys = json.loads((vdir / "keys.json").read_text())
    w2i = {k: i for i, k in enumerate(keys)}
    x0 = _emb_at(vdir, 0, test_tokens, w2i)
    curve = [0.0]
    for i in range(1, EPOCHS + 1):
        curve.append(_mean_cos_drift(x0, _emb_at(vdir, i, test_tokens, w2i)))
    return curve


def plot_tc(tc: str) -> Path:
    test_tokens, _y = load_labeled_txt(tc_paths(tc)["test"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for v in ALL_VARIANTS:
        vdir = BENCH_ROOT / tc / v
        if not (vdir / "keys.json").is_file():
            continue
        curve = variant_curve(vdir, test_tokens)
        ax.plot(range(len(curve)), curve, marker="o", ms=4, lw=1.8,
                color=ALL_COLORS[v], label=v)
        print(f"  {tc}/{v}: drift e1..e5 = "
              f"[{' '.join(f'{d:.3f}' for d in curve[1:])}]", flush=True)
    ax.set_xlabel("epoch")
    ax.set_ylabel("$1 - |\\cos(e_0, e_i)|$")
    ax.set_xticks(range(EPOCHS + 1))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"{tc.upper()}: cosine drift from initialization "
        f"(mean over test entities)\nclassic @ LR 0.025, bound @ 0.0025",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = ASSETS_OUT / f"{tc}_cosine_drift.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"{tc}: wrote {out.relative_to(ROOT)}")
    return out


def main(argv: list[str]) -> None:
    tcs = [a for a in argv if a.startswith("tc")] or ALL_TCS
    for tc in tcs:
        plot_tc(tc)


if __name__ == "__main__":
    main(sys.argv[1:])
