#!/usr/bin/env python3
"""Plot per-epoch test accuracy: vanilla vs P1 vs P2 from finetune_epoch_eval sweeps."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    ACCURACY_MODE_PLOT_STYLE,
    ACCURACY_PLOT_TITLE,
    ACCURACY_PLOT_YLABEL,
    EPOCH_ACCURACY_MARKER_EDGE_WIDTH,
    EPOCH_ACCURACY_MARKER_SIZE,
    MODE_COLORS,
    REPO_ROOT,
    SCRIPTS_DIR,
    apply_plot_style,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODE_LABELS = {
    mode: ACCURACY_MODE_PLOT_STYLE[mode if mode != "no_pretrain" else "vanilla"][0]
    for mode in ("p1", "p2", "no_pretrain")
}
MODE_ORDER = ("p1", "p2", "no_pretrain")


def load_accuracy_rows(csv_path: Path) -> list[dict[str, str | float | int]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pivot_by_tc_epoch(
    rows: list[dict[str, str | float | int]],
) -> tuple[list[str], list[int], dict[str, dict[str, dict[int, float]]]]:
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    epochs_set: set[int] = set()
    tcs_set: set[str] = set()
    for row in rows:
        tc = str(row["tc"])
        mode = str(row["training_mode"])
        epoch = int(row["finetune_epoch"])
        acc = float(row["accuracy"])
        tcs_set.add(tc)
        epochs_set.add(epoch)
        data[tc][mode][epoch] = acc
    tcs = sorted(tcs_set, key=lambda t: int(t[2:]) if t[2:].isdigit() else t)
    epochs = sorted(epochs_set)
    return tcs, epochs, dict(data)


def write_summary_csv(
    path: Path,
    *,
    tcs: list[str],
    epochs: list[int],
    data: dict[str, dict[str, dict[int, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "finetune_epoch",
                "mean_accuracy_vanilla",
                "mean_accuracy_p1",
                "mean_accuracy_p2",
                "n_tc_p1_beats_vanilla",
                "n_tc_p2_beats_vanilla",
                "n_tc_best_is_p1",
                "n_tc_best_is_p2",
                "n_tc_best_is_vanilla",
            ]
        )
        for epoch in epochs:
            means: dict[str, float] = {}
            for mode in MODE_ORDER:
                vals = [data[tc][mode][epoch] for tc in tcs if epoch in data[tc].get(mode, {})]
                means[mode] = float(np.mean(vals)) if vals else float("nan")
            p1_beats_v = p2_beats_v = best_p1 = best_p2 = best_v = 0
            for tc in tcs:
                by_mode = {
                    m: data[tc][m][epoch]
                    for m in MODE_ORDER
                    if epoch in data[tc].get(m, {})
                }
                if len(by_mode) < 3:
                    continue
                v, p1, p2 = by_mode["no_pretrain"], by_mode["p1"], by_mode["p2"]
                if p1 > v:
                    p1_beats_v += 1
                if p2 > v:
                    p2_beats_v += 1
                best = max(by_mode, key=by_mode.get)
                if best == "p1":
                    best_p1 += 1
                elif best == "p2":
                    best_p2 += 1
                else:
                    best_v += 1
            w.writerow(
                [
                    epoch,
                    f"{means['no_pretrain']:.6f}",
                    f"{means['p1']:.6f}",
                    f"{means['p2']:.6f}",
                    p1_beats_v,
                    p2_beats_v,
                    best_p1,
                    best_p2,
                    best_v,
                ]
            )


def plot_per_tc_grid(
    *,
    tcs: list[str],
    epochs: list[int],
    data: dict[str, dict[str, dict[int, float]]],
    out_path: Path,
) -> None:
    ncols, nrows = 3, 4
    if len(tcs) != ncols * nrows:
        raise ValueError(f"Expected {ncols * nrows} TCs, got {len(tcs)}: {tcs}")

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 13), sharex=True)
    axes_flat = axes.ravel()
    x = np.array(epochs, dtype=float)

    for ax, tc in zip(axes_flat, tcs, strict=True):
        for mode in MODE_ORDER:
            if mode not in data[tc]:
                continue
            y = [data[tc][mode].get(e, float("nan")) for e in epochs]
            color = MODE_COLORS[mode]
            ax.plot(
                x,
                y,
                label=MODE_LABELS[mode],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=EPOCH_ACCURACY_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=EPOCH_ACCURACY_MARKER_EDGE_WIDTH,
                alpha=0.95,
            )
        ax.set_title(ACCURACY_PLOT_TITLE.format(tc=tc))
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", framealpha=0.92, fontsize=8)

    for ax in axes_flat[-3:]:
        ax.set_xlabel("Epoch")
    for row in axes:
        row[0].set_ylabel(ACCURACY_PLOT_YLABEL)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mean_accuracy(
    *,
    tcs: list[str],
    epochs: list[int],
    data: dict[str, dict[str, dict[int, float]]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(epochs, dtype=float)
    for mode in MODE_ORDER:
        ys = []
        for epoch in epochs:
            vals = [data[tc][mode][epoch] for tc in tcs if epoch in data[tc].get(mode, {})]
            ys.append(float(np.mean(vals)) if vals else float("nan"))
        color = MODE_COLORS[mode]
        ax.plot(
            x,
            ys,
            marker="o",
            markersize=EPOCH_ACCURACY_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=EPOCH_ACCURACY_MARKER_EDGE_WIDTH,
            linewidth=2.2,
            label=MODE_LABELS[mode],
            color=color,
        )
    ax.set_xlabel("finetune epoch (0 = post-pretrain init for P1/P2)")
    ax.set_ylabel("mean test accuracy across TCs")
    ax.set_xticks(epochs)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_win_counts(summary_csv: Path, out_path: Path) -> None:
    epochs: list[int] = []
    p1_wins: list[int] = []
    p2_wins: list[int] = []
    v_wins: list[int] = []
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["finetune_epoch"]))
            p1_wins.append(int(row["n_tc_best_is_p1"]))
            p2_wins.append(int(row["n_tc_best_is_p2"]))
            v_wins.append(int(row["n_tc_best_is_vanilla"]))
    x = np.arange(len(epochs))
    width = 0.55
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, p1_wins, width, label="P1 best", color=MODE_COLORS["p1"])
    ax.bar(x, p2_wins, width, bottom=p1_wins, label="P2 best", color=MODE_COLORS["p2"])
    bottom = np.array(p1_wins) + np.array(p2_wins)
    ax.bar(x, v_wins, width, bottom=bottom, label="vanilla best", color=MODE_COLORS["no_pretrain"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_xlabel("finetune epoch")
    ax.set_ylabel("# test cases with highest accuracy")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare vanilla / P1 / P2 test accuracy per finetune epoch.",
    )
    ap.add_argument(
        "sweep_root",
        type=Path,
        nargs="?",
        default=None,
        help="finetune_epoch_eval sweep dir (default: latest under output/finetune_epoch_eval/)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Override input CSV (default: <sweep_root>/finetune_epoch_accuracy_all.csv)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for summary CSV and extra plots (default: sweep_root)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/p1_p2_vanilla_finetune_epoch_accuracy_comparison.png"),
        help="Output path for the 3x4 TC grid image",
    )
    ap.add_argument(
        "--extras",
        action="store_true",
        help="Also write mean-accuracy line plot and per-epoch win-count bar chart",
    )
    args = ap.parse_args()

    if args.sweep_root is not None:
        sweep_root = args.sweep_root.resolve()
    else:
        base = REPO_ROOT / "output" / "finetune_epoch_eval"
        if not base.is_dir():
            raise SystemExit(f"No sweep directory: {base}")
        sweeps = sorted(p for p in base.iterdir() if p.is_dir())
        if not sweeps:
            raise SystemExit(f"No sweeps under {base}")
        sweep_root = sweeps[-1]

    csv_path = (
        args.csv.resolve()
        if args.csv is not None
        else sweep_root / "finetune_epoch_accuracy_all.csv"
    )
    if not csv_path.is_file():
        raise SystemExit(
            f"Missing {csv_path}. Run finetune epoch eval first "
            f"(train.py --save-epoch-checkpoints)."
        )

    out_dir = args.out_dir.resolve() if args.out_dir else sweep_root
    rows = load_accuracy_rows(csv_path)
    tcs, epochs, data = pivot_by_tc_epoch(rows)

    apply_plot_style()
    summary_csv = out_dir / "finetune_epoch_accuracy_summary.csv"
    write_summary_csv(summary_csv, tcs=tcs, epochs=epochs, data=data)

    grid_path = args.out.resolve()
    plot_per_tc_grid(tcs=tcs, epochs=epochs, data=data, out_path=grid_path)

    print(f"Input:  {csv_path}")
    print(f"Wrote:  {grid_path}")
    print(f"        {summary_csv}")

    if args.extras:
        plot_mean_accuracy(
            tcs=tcs,
            epochs=epochs,
            data=data,
            out_path=out_dir / "finetune_epoch_accuracy_mean.png",
        )
        plot_epoch_win_counts(
            summary_csv,
            out_path=out_dir / "finetune_epoch_accuracy_win_counts.png",
        )
        print(f"        {out_dir / 'finetune_epoch_accuracy_mean.png'}")
        print(f"        {out_dir / 'finetune_epoch_accuracy_win_counts.png'}")


if __name__ == "__main__":
    main()
