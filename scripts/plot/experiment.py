#!/usr/bin/env python3
"""Plot P1 / P2 / Vanilla comparison grids for one experiment slug."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    ACCURACY_CSV_NAME,
    ACCURACY_MODE_PLOT_STYLE,
    ACCURACY_PLOT_TITLE,
    ACCURACY_PLOT_YLABEL,
    EPOCH_ACCURACY_MARKER_EDGE_WIDTH,
    EPOCH_ACCURACY_MARKER_SIZE,
    EPOCH_LOSS_MARKER_EDGE_WIDTH,
    EPOCH_LOSS_MARKER_SIZE,
    LOSS_CSV_NAMES,
    MODE_ORDER,
    MODE_PLOT_STYLE,
    SCRIPTS_DIR,
    ExperimentRun,
    apply_plot_style,
    discover_experiment_runs,
    experiment_slug_root,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_epoch_loss_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("epoch", "loss"):
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}")
    return df[["epoch", "loss"]].sort_values("epoch")


def load_epoch_eval_csv(path: Path, *, y_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    x_col = "finetune_epoch"
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}")
    return df[[x_col, y_col]].sort_values(x_col)


def load_epoch_accuracy_csv(path: Path) -> pd.DataFrame:
    return load_epoch_eval_csv(path, y_col="accuracy")


def load_epoch_f1_csv(path: Path) -> pd.DataFrame:
    return load_epoch_eval_csv(path, y_col="f1")


def average_series(
    frames: list[pd.DataFrame],
    *,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    if not frames:
        raise ValueError("Need at least one series to average")
    if len(frames) == 1:
        return frames[0][[x_col, y_col]].copy()
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)


def load_averaged_series(
    run: ExperimentRun,
    *,
    mode: str,
    csv_name: str,
    load_series: Callable[[Path], pd.DataFrame],
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    frames = [
        load_series(run_dirs[mode] / csv_name)
        for ts in run.timestamps
        for run_dirs in [run.run_dirs_by_timestamp[ts]]
    ]
    return average_series(frames, x_col=x_col, y_col=y_col)


def compute_grid_shape(n_tcs: int) -> tuple[int, int]:
    if n_tcs <= 0:
        raise ValueError("Need at least one TC to plot")
    ncols = math.ceil(math.sqrt(n_tcs))
    nrows = math.ceil(n_tcs / ncols)
    return nrows, ncols


def plot_run_on_ax(
    ax: plt.Axes,
    run: ExperimentRun,
    *,
    load_series: Callable[[Path], pd.DataFrame],
    x_col: str,
    y_col: str,
    mode_plot_style: dict[str, tuple[str, str]] | None = None,
    show_title: bool = True,
) -> None:
    is_accuracy = y_col == "accuracy"
    style = mode_plot_style or (ACCURACY_MODE_PLOT_STYLE if is_accuracy else MODE_PLOT_STYLE)
    for mode in MODE_ORDER:
        label, color = style[mode]
        csv_name = LOSS_CSV_NAMES[mode] if y_col == "loss" else ACCURACY_CSV_NAME
        df = load_averaged_series(
            run,
            mode=mode,
            csv_name=csv_name,
            load_series=load_series,
            x_col=x_col,
            y_col=y_col,
        )
        if y_col == "loss":
            marker_size = EPOCH_LOSS_MARKER_SIZE
            marker_edge = EPOCH_LOSS_MARKER_EDGE_WIDTH
            marker_face = color
        elif y_col == "accuracy":
            marker_size = EPOCH_ACCURACY_MARKER_SIZE
            marker_edge = EPOCH_ACCURACY_MARKER_EDGE_WIDTH
            marker_face = color
        else:
            marker_size = 3.0
            marker_edge = 0.9
            marker_face = "white"
        ax.plot(
            df[x_col],
            df[y_col],
            label=label,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=marker_size,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=marker_edge,
            alpha=0.95,
        )
    if show_title:
        title = ACCURACY_PLOT_TITLE.format(tc=run.tc) if is_accuracy else run.tc.upper()
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right" if is_accuracy else "best",
        framealpha=0.92,
        fontsize=8,
    )


def plot_single_tc(
    *,
    run: ExperimentRun,
    ylabel: str,
    out_path: Path,
    load_series: Callable[[Path], pd.DataFrame],
    x_col: str,
    y_col: str,
    mode_plot_style: dict[str, tuple[str, str]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_run_on_ax(
        ax,
        run,
        load_series=load_series,
        x_col=x_col,
        y_col=y_col,
        mode_plot_style=mode_plot_style,
        show_title=y_col == "accuracy",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tc_grid(
    *,
    runs: list[ExperimentRun],
    ylabel: str,
    out_path: Path,
    load_series: Callable[[Path], pd.DataFrame],
    x_col: str,
    y_col: str,
    sharey: bool = True,
    mode_plot_style: dict[str, tuple[str, str]] | None = None,
) -> None:
    nrows, ncols = compute_grid_shape(len(runs))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.1 * nrows),
        sharex=True,
        sharey=sharey,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, run in zip(axes_flat, runs, strict=False):
        plot_run_on_ax(
            ax,
            run,
            load_series=load_series,
            x_col=x_col,
            y_col=y_col,
            mode_plot_style=mode_plot_style,
        )

    for ax in axes_flat[len(runs) :]:
        ax.set_visible(False)

    for ax_idx, ax in enumerate(axes_flat[: len(runs)]):
        row = ax_idx // ncols
        col = ax_idx % ncols
        if row == nrows - 1:
            ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel(ylabel)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def repeat_label_for_runs(runs: list[ExperimentRun], pinned_timestamp: str | None) -> str:
    if pinned_timestamp:
        return f"timestamp={pinned_timestamp}"
    repeat_counts = {run.n_repeats for run in runs}
    if len(repeat_counts) == 1:
        n = next(iter(repeat_counts))
        if n == 1:
            return f"timestamp={runs[0].timestamps[0]}"
        return f"averaged over {n} runs"
    min_repeats = min(repeat_counts)
    max_repeats = max(repeat_counts)
    if min_repeats == max_repeats:
        return f"averaged over {min_repeats} runs"
    return f"averaged over {min_repeats}-{max_repeats} runs per TC"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot P1 / P2 / Vanilla loss, accuracy, and F1 grids for one experiment slug.",
    )
    ap.add_argument("--slug", default="default", help="P1/P2 experiment slug (default: default)")
    ap.add_argument("--dataset", default="synthetic", help="Dataset under output/ (default: synthetic)")
    ap.add_argument("--output-root", type=Path, default=Path("output"), help="Output root directory")
    ap.add_argument("--out-dir", type=Path, default=Path("plots"), help="Directory for PNG outputs")
    ap.add_argument(
        "--timestamp",
        default=None,
        help="Use only this run timestamp (default: average all aligned repeats per TC)",
    )
    ap.add_argument(
        "--absolute-loss",
        action="store_true",
        help="Plot raw finetune loss without normalization or multi-run averaging "
        "(uses --timestamp if set, otherwise the latest aligned timestamp per TC)",
    )
    args = ap.parse_args()

    output_root = args.output_root.resolve()
    slug_root = experiment_slug_root(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
    )
    if not slug_root.is_dir():
        raise SystemExit(f"Experiment slug directory not found: {slug_root}")

    all_tcs = sorted(
        (p.name for p in slug_root.iterdir() if p.is_dir() and p.name.startswith("tc")),
        key=lambda t: int(t[2:]) if t[2:].isdigit() else t,
    )
    runs = discover_experiment_runs(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
        timestamp=args.timestamp,
        latest_only=args.absolute_loss and args.timestamp is None,
    )
    found_tcs = {run.tc for run in runs}
    for tc in all_tcs:
        if tc not in found_tcs:
            print(f"Warning: skipping {tc} — no aligned P1/P2/Vanilla run with required CSVs", flush=True)

    if not runs:
        raise SystemExit(f"No complete runs found under {slug_root}")

    repeat_label = repeat_label_for_runs(runs, args.timestamp)
    apply_plot_style()
    out_dir = args.out_dir.resolve()
    slug_dir = out_dir / args.slug

    loss_path = out_dir / f"{args.slug}_training_loss.png"
    loss_ylabel = "Training loss"
    plot_tc_grid(
        runs=runs,
        ylabel=loss_ylabel,
        out_path=loss_path,
        load_series=load_epoch_loss_csv,
        x_col="epoch",
        y_col="loss",
        sharey=False,
    )

    accuracy_path = out_dir / f"{args.slug}_accuracy.png"
    plot_tc_grid(
        runs=runs,
        ylabel=ACCURACY_PLOT_YLABEL,
        out_path=accuracy_path,
        load_series=load_epoch_accuracy_csv,
        x_col="finetune_epoch",
        y_col="accuracy",
    )

    f1_path = out_dir / f"{args.slug}_f1.png"
    plot_tc_grid(
        runs=runs,
        ylabel="F1",
        out_path=f1_path,
        load_series=load_epoch_f1_csv,
        x_col="finetune_epoch",
        y_col="f1",
    )

    per_tc_loss_paths: list[Path] = []
    per_tc_accuracy_paths: list[Path] = []
    per_tc_f1_paths: list[Path] = []
    for run in runs:
        tc_loss_path = slug_dir / f"{run.tc}_training_loss.png"
        plot_single_tc(
            run=run,
            ylabel=loss_ylabel,
            out_path=tc_loss_path,
            load_series=load_epoch_loss_csv,
            x_col="epoch",
            y_col="loss",
        )
        per_tc_loss_paths.append(tc_loss_path)

        tc_accuracy_path = slug_dir / f"{run.tc}_accuracy.png"
        plot_single_tc(
            run=run,
            ylabel=ACCURACY_PLOT_YLABEL,
            out_path=tc_accuracy_path,
            load_series=load_epoch_accuracy_csv,
            x_col="finetune_epoch",
            y_col="accuracy",
        )
        per_tc_accuracy_paths.append(tc_accuracy_path)

        tc_f1_path = slug_dir / f"{run.tc}_f1.png"
        plot_single_tc(
            run=run,
            ylabel="F1",
            out_path=tc_f1_path,
            load_series=load_epoch_f1_csv,
            x_col="finetune_epoch",
            y_col="f1",
        )
        per_tc_f1_paths.append(tc_f1_path)

    print(f"Discovered {len(runs)} TCs: {', '.join(run.tc for run in runs)}", flush=True)
    for run in runs:
        if run.n_repeats == 1:
            print(f"  {run.tc}: 1 run ({run.timestamps[0]})", flush=True)
        else:
            print(f"  {run.tc}: {run.n_repeats} runs averaged ({', '.join(run.timestamps)})", flush=True)
    print(f"Aggregation: {repeat_label}", flush=True)
    print(f"Wrote {loss_path}", flush=True)
    print(f"Wrote {accuracy_path}", flush=True)
    print(f"Wrote {f1_path}", flush=True)
    print(f"Wrote {len(per_tc_loss_paths)} per-TC loss plots under {slug_dir}", flush=True)
    print(f"Wrote {len(per_tc_accuracy_paths)} per-TC accuracy plots under {slug_dir}", flush=True)
    print(f"Wrote {len(per_tc_f1_paths)} per-TC F1 plots under {slug_dir}", flush=True)


if __name__ == "__main__":
    main()
