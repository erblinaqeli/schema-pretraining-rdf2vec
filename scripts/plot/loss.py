#!/usr/bin/env python3
"""Consolidated loss curve comparisons."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from _common import (
    EPOCH_LOSS_MARKER_EDGE_WIDTH,
    EPOCH_LOSS_MARKER_SIZE,
    SCRIPTS_DIR,
    apply_plot_style,
    latest_timestamped_run,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_steps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("step", "interval_loss"):
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}")
    return df[["step", "interval_loss"]].copy()


def load_epoch_loss_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("epoch", "loss"):
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}")
    return df[["epoch", "loss"]].sort_values("epoch")


def find_latest_best_run(output_dir: Path, tc: str, mode: str) -> Path | None:
    sweep_root = output_dir / f"{tc}_grid_search_{mode}"
    if not sweep_root.is_dir():
        return None
    runs = [p for p in sweep_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not runs:
        return None
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    best_path = latest / "best_run.json"
    return best_path if best_path.is_file() else None


def discover_tcs(output_dir: Path) -> list[str]:
    pattern = re.compile(r"^(tc\d+)_grid_search_p[12]$")
    tcs: set[str] = set()
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            tcs.add(m.group(1))
    return sorted(tcs, key=lambda t: int(t[2:]))


def smooth_series(y: np.ndarray, window: int = 7) -> np.ndarray:
    n = len(y)
    if n < 3:
        return y.astype(float)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y.astype(float)
    return savgol_filter(y.astype(float), window_length=w, polyorder=2)


def plot_per_step_tc(
    tc_id: str,
    paths: dict[str, Path],
    out_path: Path,
) -> None:
    series = {
        "V (no pretrain)": ("#2563eb", paths["v"]),
        "P1": ("#c026d3", paths["p1"]),
        "P2": ("#ea580c", paths["p2"]),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    for label, (color, csv_path) in series.items():
        df = load_steps_csv(csv_path).sort_values("step")
        ax.plot(
            df["step"],
            df["interval_loss"],
            label=label,
            color=color,
            linewidth=2.15,
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            alpha=0.95,
        )
    ax.set_xlabel("Training step (batch index)")
    ax.set_ylabel("Interval loss")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def mode_per_step(args: argparse.Namespace) -> None:
    apply_plot_style()
    output_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve()

    tcs = [args.tc] if args.tc else ["07", "08", "09", "10"]
    for tc_suffix in tcs:
        tc_name = tc_suffix if tc_suffix.startswith("tc") else f"tc{tc_suffix}"
        v_root = output_root / f"{tc_name}_rdf2vec"
        p1_root = output_root / f"{tc_name}_rdf2vec_p1"
        p2_root = output_root / f"{tc_name}_rdf2vec_p2"

        v_run = latest_timestamped_run(v_root)
        p1_run = latest_timestamped_run(p1_root)
        p2_run = latest_timestamped_run(p2_root)
        if v_run is None or p1_run is None or p2_run is None:
            raise SystemExit(
                f"Missing run directory for {tc_name}: v={v_run}, p1={p1_run}, p2={p2_run}"
            )

        v_csv = v_run / "rdf2vec_word2vec_loss_steps.csv"
        p1_csv = p1_run / "finetune_loss_steps.csv"
        p2_csv = p2_run / "finetune_loss_steps.csv"
        for p in (v_csv, p1_csv, p2_csv):
            if not p.is_file():
                raise SystemExit(f"Missing CSV: {p}")

        out_png = out_dir / f"{tc_name}_loss_per_step_v_p1_p2.png"
        plot_per_step_tc(tc_name, {"v": v_csv, "p1": p1_csv, "p2": p2_csv}, out_png)
        print(f"Wrote {out_png}")


def resolve_finetune_loss_paths(output_dir: Path, tc: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for mode in ("p1", "p2"):
        best_path = find_latest_best_run(output_dir, tc, mode)
        if best_path is None:
            raise FileNotFoundError(f"No best_run.json for {tc} {mode}")
        data = json.loads(best_path.read_text())
        run_dir = Path(data["best"]["run_dir"])
        csv_path = run_dir / "finetune_loss.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing finetune loss CSV: {csv_path}")
        paths[mode] = csv_path

    vanilla_root = output_dir / f"{tc}_rdf2vec"
    vanilla_run = latest_timestamped_run(vanilla_root)
    if vanilla_run is None:
        raise FileNotFoundError(f"No vanilla run directory under {vanilla_root}")
    vanilla_csv = vanilla_run / "rdf2vec_word2vec_loss.csv"
    if not vanilla_csv.is_file():
        raise FileNotFoundError(f"Missing vanilla loss CSV: {vanilla_csv}")
    paths["vanilla"] = vanilla_csv
    return paths


def mode_finetune_epoch(args: argparse.Namespace) -> None:
    apply_plot_style()
    output_dir = args.output_dir.resolve()
    tcs = discover_tcs(output_dir)
    if not tcs:
        raise SystemExit(f"No grid search directories under {output_dir}")

    series_style = {
        "P1": ("#8b7a96", "p1"),
        "P2": ("#b8956e", "p2"),
        "Vanilla": ("#2563eb", "vanilla"),
    }

    fig, axes = plt.subplots(4, 3, figsize=(14, 13), sharex=True)
    axes_flat = axes.ravel()
    for ax, tc in zip(axes_flat, tcs, strict=True):
        paths = resolve_finetune_loss_paths(output_dir, tc)
        for label, (color, key) in series_style.items():
            df = load_epoch_loss_csv(paths[key])
            ax.plot(
                df["epoch"],
                df["loss"],
                label=label,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=EPOCH_LOSS_MARKER_SIZE,
                markerfacecolor=color,
                markeredgewidth=EPOCH_LOSS_MARKER_EDGE_WIDTH,
                alpha=0.95,
            )
        ax.set_title(tc.upper())
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", framealpha=0.92, fontsize=8)

    for ax in axes_flat[-3:]:
        ax.set_xlabel("Epoch")
    for row in axes:
        row[0].set_ylabel("Training loss")

    fig.tight_layout()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_best_params(path: Path) -> dict:
    data = json.loads(path.read_text())
    params = data["best"]["params"]
    walks = params["pretrain_walks"]
    w2v = params["pretrain_word2vec"]
    return {
        "depth": walks["depth"],
        "walks_per_entity": walks["walks_per_entity"],
        "pretrain_epochs": w2v["pretrain_epochs"],
        "pretrain_lr": w2v["pretrain_lr"],
    }


def collect_grid_series(
    output_dir: Path,
    tcs: list[str],
) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    tc_nums = [int(tc[2:]) for tc in tcs]
    series: dict[str, dict[str, list[float]]] = {
        "p1": {k: [] for k in ("depth", "pretrain_epochs", "walks_per_entity", "pretrain_lr")},
        "p2": {k: [] for k in ("depth", "pretrain_epochs", "walks_per_entity", "pretrain_lr")},
    }
    for tc in tcs:
        for mode in ("p1", "p2"):
            path = find_latest_best_run(output_dir, tc, mode)
            if path is None:
                raise FileNotFoundError(f"No best_run.json for {tc} {mode}")
            vals = load_best_params(path)
            for key in series[mode]:
                series[mode][key].append(vals[key])
    return tc_nums, series


def mode_grid_configs(args: argparse.Namespace) -> None:
    apply_plot_style()
    output_dir = args.output_dir.resolve()
    tcs = discover_tcs(output_dir)
    if not tcs:
        raise SystemExit(f"No grid search directories under {output_dir}")

    tc_nums, series = collect_grid_series(output_dir, tcs)
    x_labels = [f"tc{n:02d}" for n in tc_nums]
    metrics = [
        ("depth", "Pretrain walk depth", [4, 8, 12]),
        ("pretrain_epochs", "Pretrain epochs", [5, 7, 10]),
        ("walks_per_entity", "Walks per entity", [100, 200, 300, 400]),
        ("pretrain_lr", "Pretrain learning rate", [0.001, 0.01, 0.025]),
    ]
    bar_colors = {"p1": "#c4b5d4", "p2": "#e8c4a8"}
    line_colors = {"p1": "#8b7a96", "p2": "#b8956e"}
    bar_alpha = 0.55
    n_tcs = len(tc_nums)
    x = np.arange(n_tcs)
    bar_width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes_flat = axes.ravel()
    for ax, (key, ylabel, yticks) in zip(axes_flat, metrics, strict=True):
        p1_vals = np.array(series["p1"][key], dtype=float)
        p2_vals = np.array(series["p2"][key], dtype=float)
        ax.bar(x - bar_width / 2, p1_vals, bar_width, color=bar_colors["p1"], edgecolor="#ffffff", linewidth=0.5, alpha=bar_alpha, zorder=1)
        ax.bar(x + bar_width / 2, p2_vals, bar_width, color=bar_colors["p2"], edgecolor="#ffffff", linewidth=0.5, alpha=bar_alpha, zorder=1)
        for mode, x_offset, label in (("p1", -bar_width / 2, "P1"), ("p2", bar_width / 2, "P2")):
            vals = p1_vals if mode == "p1" else p2_vals
            ax.plot(
                x + x_offset,
                smooth_series(vals),
                color=line_colors[mode],
                linewidth=2.4,
                label=label,
                zorder=3,
                solid_capstyle="round",
            )
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_yticks(yticks)
        if key == "pretrain_lr":
            ax.set_yticklabels([f"{v:g}" for v in yticks])
        step = min(b - a for a, b in zip(yticks, yticks[1:]))
        pad = step * 0.35
        ax.set_ylim(yticks[0] - pad, yticks[-1] + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", framealpha=0.9)

    for ax in axes_flat[2:]:
        ax.set_xlabel("Test case")
    fig.tight_layout()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


LOSS_MODES = ("per-step", "finetune-epoch", "grid-configs")
MODE_HANDLERS = {
    "per-step": mode_per_step,
    "finetune-epoch": mode_finetune_epoch,
    "grid-configs": mode_grid_configs,
}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", required=True, choices=LOSS_MODES)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=LOSS_MODES)
    if pre_args.mode == "per-step":
        parser.add_argument("--output-root", type=Path, default=Path("output"))
        parser.add_argument("--out-dir", type=Path, default=Path("latex/assets"))
        parser.add_argument("--tc", type=str, default=None, help="Single TC (e.g. tc12 or 12); default: 07-10")
    elif pre_args.mode == "finetune-epoch":
        parser.add_argument("--output-dir", type=Path, default=Path("output"))
        parser.add_argument("--out", type=Path, default=Path("output/p1_p2_vanilla_finetune_loss_comparison.png"))
    elif pre_args.mode == "grid-configs":
        parser.add_argument("--output-dir", type=Path, default=Path("output"))
        parser.add_argument("--out", type=Path, default=Path("output/p1_p2_grid_search_config_comparison.png"))

    args = parser.parse_args(["--mode", pre_args.mode, *remaining])
    MODE_HANDLERS[args.mode](args)


if __name__ == "__main__":
    main()
