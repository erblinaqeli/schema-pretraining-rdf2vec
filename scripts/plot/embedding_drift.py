#!/usr/bin/env python3
"""Plot 1 − |cosine similarity| of mean embeddings vs finetune epoch 0 for P1 / P2 / Vanilla."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    ACCURACY_MODE_PLOT_STYLE,
    DRIFT_PLOT_TITLE,
    DRIFT_PLOT_YLABEL,
    EPOCH_ACCURACY_MARKER_EDGE_WIDTH,
    EPOCH_ACCURACY_MARKER_SIZE,
    MODE_ORDER,
    SCRIPTS_DIR,
    ExperimentRun,
    apply_plot_style,
    discover_checkpoint_runs,
    experiment_slug_root,
    list_finetune_epoch_checkpoints,
    load_checkpoint,
    avg_embedding_cosine_drift_from_baseline,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def compute_drift_series(run_dir: Path) -> pd.DataFrame:
    """Return epoch and 1 − cos_sim(mean embedding, epoch-0 mean) for one run directory."""
    checkpoints = list_finetune_epoch_checkpoints(run_dir)
    if not checkpoints:
        raise ValueError(f"No finetune epoch checkpoints under {run_dir / 'ckpt'}")
    baseline_emb, _ = load_checkpoint(checkpoints[0][1])
    rows: list[dict[str, float | int]] = []
    for epoch, path in checkpoints:
        emb, _ = load_checkpoint(path)
        drift = avg_embedding_cosine_drift_from_baseline(baseline_emb, emb)
        rows.append({"epoch": epoch, "cosine_drift": drift})
    return pd.DataFrame(rows).sort_values("epoch")


def average_drift_series(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("Need at least one drift series to average")
    if len(frames) == 1:
        return frames[0][["epoch", "cosine_drift"]].copy()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("epoch", as_index=False)["cosine_drift"]
        .mean()
        .sort_values("epoch")
    )


def load_averaged_drift_series(run: ExperimentRun, *, mode: str) -> pd.DataFrame:
    frames = [
        compute_drift_series(run.run_dirs_by_timestamp[ts][mode])
        for ts in run.timestamps
    ]
    return average_drift_series(frames)


def build_drift_table(run: ExperimentRun) -> pd.DataFrame:
    """Wide table: epoch, p1, p2, vanilla 1 − cos_sim drift values."""
    merged: pd.DataFrame | None = None
    for mode in MODE_ORDER:
        series = load_averaged_drift_series(run, mode=mode)
        series = series.rename(columns={"cosine_drift": mode})
        merged = series if merged is None else merged.merge(series, on="epoch", how="outer")
    if merged is None:
        raise ValueError(f"No drift data for {run.tc}")
    return merged.sort_values("epoch")


def compute_grid_shape(n_tcs: int) -> tuple[int, int]:
    if n_tcs <= 0:
        raise ValueError("Need at least one TC to plot")
    ncols = math.ceil(math.sqrt(n_tcs))
    nrows = math.ceil(n_tcs / ncols)
    return nrows, ncols


def plot_run_on_ax(ax: plt.Axes, run: ExperimentRun, *, show_title: bool = True) -> None:
    for mode in MODE_ORDER:
        label, color = ACCURACY_MODE_PLOT_STYLE[mode]
        df = load_averaged_drift_series(run, mode=mode)
        ax.plot(
            df["epoch"],
            df["cosine_drift"],
            label=label,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=EPOCH_ACCURACY_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=EPOCH_ACCURACY_MARKER_EDGE_WIDTH,
            alpha=0.95,
        )
    if show_title:
        ax.set_title(DRIFT_PLOT_TITLE.format(tc=run.tc))
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8)


def plot_single_tc(
    *,
    run: ExperimentRun,
    out_path: Path,
    csv_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_run_on_ax(ax, run, show_title=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(DRIFT_PLOT_YLABEL)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    drift_table = build_drift_table(run)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    drift_table.to_csv(csv_path, index=False)


def plot_tc_grid(
    *,
    runs: list[ExperimentRun],
    out_path: Path,
) -> None:
    nrows, ncols = compute_grid_shape(len(runs))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.1 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, run in zip(axes_flat, runs, strict=False):
        plot_run_on_ax(ax, run)

    for ax in axes_flat[len(runs) :]:
        ax.set_visible(False)

    for ax_idx, ax in enumerate(axes_flat[: len(runs)]):
        row = ax_idx // ncols
        col = ax_idx % ncols
        if row == nrows - 1:
            ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel(DRIFT_PLOT_YLABEL)

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
        description=(
            "Plot 1 − |cosine similarity| of mean embeddings vs finetune epoch 0 for P1 / P2 / Vanilla."
        ),
    )
    ap.add_argument("--slug", default="default", help="P1/P2 experiment slug (default: default)")
    ap.add_argument("--dataset", default="synthetic", help="Dataset under output/ (default: synthetic)")
    ap.add_argument("--output-root", type=Path, default=Path("output"), help="Output root directory")
    ap.add_argument("--out-dir", type=Path, default=Path("plots"), help="Directory for PNG/CSV outputs")
    ap.add_argument(
        "--timestamp",
        default=None,
        help="Use only this run timestamp (default: average all aligned repeats per TC)",
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
    runs = discover_checkpoint_runs(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
        timestamp=args.timestamp,
    )
    found_tcs = {run.tc for run in runs}
    for tc in all_tcs:
        if tc not in found_tcs:
            print(
                f"Warning: skipping {tc} — no aligned P1/P2/Vanilla run with epoch checkpoints",
                flush=True,
            )

    if not runs:
        raise SystemExit(f"No complete checkpoint runs found under {slug_root}")

    repeat_label = repeat_label_for_runs(runs, args.timestamp)
    apply_plot_style()
    out_dir = args.out_dir.resolve()
    slug_dir = out_dir / args.slug

    grid_path = out_dir / f"{args.slug}_embedding_drift.png"
    plot_tc_grid(
        runs=runs,
        out_path=grid_path,
    )

    per_tc_paths: list[Path] = []
    per_tc_csv_paths: list[Path] = []
    for run in runs:
        tc_plot_path = slug_dir / f"{run.tc}_embedding_drift.png"
        tc_csv_path = slug_dir / f"{run.tc}_embedding_drift.csv"
        plot_single_tc(
            run=run,
            out_path=tc_plot_path,
            csv_path=tc_csv_path,
        )
        per_tc_paths.append(tc_plot_path)
        per_tc_csv_paths.append(tc_csv_path)

    print(f"Discovered {len(runs)} TCs: {', '.join(run.tc for run in runs)}", flush=True)
    for run in runs:
        drift_table = build_drift_table(run)
        epoch0 = drift_table.loc[drift_table["epoch"] == 0, MODE_ORDER]
        epoch0_msg = ""
        if not epoch0.empty:
            max_epoch0 = float(epoch0[list(MODE_ORDER)].to_numpy().max())
            epoch0_msg = f", epoch-0 max drift={max_epoch0:.2e}"
        if run.n_repeats == 1:
            print(f"  {run.tc}: 1 run ({run.timestamps[0]}){epoch0_msg}", flush=True)
        else:
            print(
                f"  {run.tc}: {run.n_repeats} runs averaged "
                f"({', '.join(run.timestamps)}){epoch0_msg}",
                flush=True,
            )
    print(f"Aggregation: {repeat_label}", flush=True)
    print(f"Wrote {grid_path}", flush=True)
    print(f"Wrote {len(per_tc_paths)} per-TC plots under {slug_dir}", flush=True)
    print(f"Wrote {len(per_tc_csv_paths)} per-TC CSVs under {slug_dir}", flush=True)


if __name__ == "__main__":
    main()
