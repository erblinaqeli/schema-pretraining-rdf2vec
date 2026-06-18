"""
Figures and markdown tables for method_proposal.md from method_proposal/data/.

Per TC:  acc_<tc>.png, loss_<tc>.png, pca_<tc>.png
Global:  final_accuracy.png, cos_drift.png
Tables:  printed to method_proposal/tables.md (accuracy per epoch + runtime/drift)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DATA = ROOT / "method_proposal" / "data"
FIGS = ROOT / "method_proposal" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

VARIANTS = ["vanilla", "p1_classic", "p2_classic", "p1_bound", "p2_bound", "p3_bound"]
COLORS = {
    "vanilla": "#9e9e9e",
    "p1_classic": "#c5b3e6",
    "p2_classic": "#9575cd",
    "p1_bound": "#90caf9",
    "p2_bound": "#1e88e5",
    "p3_bound": "#0d47a1",
}
LABELS = {v: v for v in VARIANTS}
LABELS["p1_classic"] = "p1_classic (LR 0.025)"

TCS = sorted(p.stem for p in DATA.glob("tc*.json"))


def load(tc):
    rows = json.loads((DATA / f"{tc}.json").read_text())
    return {r["variant"]: r for r in rows}


def per_tc_figures(tc):
    rows = load(tc)

    # accuracy curves
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for v in VARIANTS:
        r = rows[v]
        ax.plot(range(len(r["accs"])), r["accs"], marker="o", ms=3.5,
                color=COLORS[v], label=LABELS[v])
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.grid(alpha=0.3)
    ax.set_xlabel("epoch (0 = initialization)")
    ax.set_ylabel("LogReg test accuracy")
    ax.set_title(f"{tc} — per-epoch test accuracy")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIGS / f"acc_{tc}.png", dpi=150)
    plt.close(fig)

    # loss curves
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for v in VARIANTS:
        r = rows[v]
        ax.plot(range(1, len(r["losses"]) + 1), r["losses"], marker="o", ms=3.5,
                color=COLORS[v], label=LABELS[v])
    ax.grid(alpha=0.3)
    ax.set_xlabel("finetune epoch")
    ax.set_ylabel("SGNS training loss (per epoch)")
    ax.set_title(f"{tc} — training loss on instance walks")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIGS / f"loss_{tc}.png", dpi=150)
    plt.close(fig)

    # PCA epoch0 vs epoch5, shared projection per variant row
    npz = np.load(DATA / f"{tc}_emb.npz")
    y = npz["y_test"]
    fig, axes = plt.subplots(len(VARIANTS), 2, figsize=(8.0, 2.6 * len(VARIANTS)))
    for (ax0, ax5), v in zip(axes, VARIANTS):
        r = rows[v]
        x0, x5 = npz[f"{v}_x0"], npz[f"{v}_x5"]
        n_distinct = int(np.unique(np.round(x0, 5), axis=0).shape[0])
        xy = PCA(n_components=2, random_state=0).fit_transform(np.vstack([x0, x5]))
        xy0, xy5 = xy[: len(x0)], xy[len(x0):]
        for ax, pts, title in (
            (ax0, xy0, f"epoch 0 — acc {r['accs'][0]:.3f}, {n_distinct}/{len(y)} distinct"),
            (ax5, xy5, f"epoch 5 — acc {r['accs'][-1]:.3f}, cos(e0,e5) = {r['mean_cos_e0_e5']:.3f}"),
        ):
            for lab, color in ((0, "#e53935"), (1, "#1e88e5")):
                m = y == lab
                ax.scatter(pts[m, 0], pts[m, 1], s=8, alpha=0.5, color=color,
                           label=f"y={lab}", lw=0)
            ax.set_title(f"{LABELS[v]}\n{title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{tc}: PCA of test-entity embeddings (shared projection per row)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIGS / f"pca_{tc}.png", dpi=130)
    plt.close(fig)


def global_figures():
    all_rows = {tc: load(tc) for tc in TCS}

    # final accuracy bars
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(TCS))
    w = 0.13
    for i, v in enumerate(VARIANTS):
        vals = [all_rows[tc][v]["final_acc"] for tc in TCS]
        ax.bar(x + (i - 2.5) * w, vals, w, label=LABELS[v], color=COLORS[v])
    ax.set_xticks(x, TCS)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_ylabel("LogReg test accuracy (epoch 5)")
    ax.set_title("Final accuracy per test case")
    ax.legend(ncol=6, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "final_accuracy.png", dpi=150)
    plt.close(fig)

    # cosine drift bars
    fig, ax = plt.subplots(figsize=(13, 4))
    for i, v in enumerate(VARIANTS):
        vals = [all_rows[tc][v]["mean_cos_e0_e5"] for tc in TCS]
        ax.bar(x + (i - 2.5) * w, vals, w, label=LABELS[v], color=COLORS[v])
    ax.set_xticks(x, TCS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean cos(epoch 0, epoch 5)")
    ax.set_title("Embedding drift during finetuning (1.0 = frozen, 0.0 = fully rewritten)")
    ax.legend(ncol=6, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "cos_drift.png", dpi=150)
    plt.close(fig)


def tables():
    lines = []
    for tc in TCS:
        rows = load(tc)
        n_ep = len(rows["vanilla"]["accs"])
        lines.append(f"### {tc}\n")
        lines.append("**Accuracy per epoch**\n")
        header = "| variant | " + " | ".join(f"ep{e}" for e in range(n_ep)) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (n_ep + 1))
        for v in VARIANTS:
            r = rows[v]
            lines.append(
                f"| {LABELS[v]} | " + " | ".join(f"{a:.4f}" for a in r["accs"]) + " |"
            )
        lines.append("")
        lines.append("**Runtime and drift**\n")
        lines.append("| variant | pretrain W2V (s) | finetune W2V (s) | total (s) | mean cos(e0, e5) |")
        lines.append("|---|---|---|---|---|")
        for v in VARIANTS:
            r = rows[v]
            pre = "—" if v == "vanilla" else f"{r['pretrain_s']:.1f}"
            lines.append(
                f"| {LABELS[v]} | {pre} | {r['finetune_s']:.1f} | {r['total_s']:.1f} "
                f"| {r['mean_cos_e0_e5']:.3f} |"
            )
        lines.append("")
    (ROOT / "method_proposal" / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    print(f"TCs: {TCS}")
    for tc in TCS:
        per_tc_figures(tc)
        print(f"figures for {tc} done", flush=True)
    global_figures()
    tables()
    print("all figures + tables written")
