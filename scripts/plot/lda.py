#!/usr/bin/env python3
"""LDA visualizations for labeled test-set embeddings (label-aware projection)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _common import (  # noqa: E402
    LABEL_COLORS,
    MODE_ORDER,
    MODE_PLOT_STYLE,
    SCRIPTS_DIR,
    ExperimentRun,
    apply_plot_style,
    choose_test_points,
    discover_embedding_runs,
    experiment_slug_root,
    final_checkpoint_path,
    fit_lda_2d,
    is_synthetic_test_set_run,
    load_checkpoint,
    plot_labeled_lda_panel,
    shorten_entity,
    synthetic_test_path,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _evaluate import load_labeled_txt  # noqa: E402


def plot_slug_compare_lda_panels(
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    output: Path,
    *,
    suptitle: str,
    seed: int,
    dpi: int,
) -> list[tuple[str, float, float]]:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), sharex=False, sharey=False)
    if len(panels) == 1:
        axes = [axes]

    explained: list[tuple[str, float, float]] = []
    legend_labels = panels[0][2]
    for ax, (mode_label, X, labels) in zip(axes, panels, strict=True):
        ld1, orth = plot_labeled_lda_panel(ax, X, labels, title=mode_label, seed=seed)
        ax.set_title(f"{mode_label}\nLD1 {ld1 * 100:.1f}%, PC\u22a5 {orth * 100:.1f}%")
        explained.append((mode_label, ld1, orth))

    unique_labels = sorted(int(v) for v in set(legend_labels.tolist()))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            color=LABEL_COLORS.get(label, "#666666"),
            label=f"label {label} (n={int((legend_labels == label).sum())})",
        )
        for label in unique_labels
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(handles), fontsize=9)
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return explained


def plot_test_lda(
    X: np.ndarray,
    tokens: list[str],
    labels: np.ndarray,
    output: Path,
    *,
    title: str,
    dpi: int,
    annotate: bool,
    seed: int,
) -> tuple[float, float]:
    fig, ax = plt.subplots(figsize=(10, 8))
    ld1, orth = plot_labeled_lda_panel(
        ax,
        X,
        labels,
        title=title,
        seed=seed,
        show_legend=True,
    )
    if annotate:
        z, _, _ = fit_lda_2d(X, labels, seed=seed)
        for x, y, token in zip(z[:, 0], z[:, 1], tokens, strict=True):
            ax.annotate(shorten_entity(token), (x, y), fontsize=6, alpha=0.75)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ld1, orth


def _resolve_slug_compare_run(
    run: ExperimentRun,
    *,
    timestamp: str | None,
) -> tuple[str, dict[str, Path]]:
    if timestamp:
        if timestamp not in run.run_dirs_by_timestamp:
            raise SystemExit(f"{run.tc}: timestamp {timestamp!r} not found in discovered runs")
        return timestamp, run.run_dirs_by_timestamp[timestamp]
    if not run.timestamps:
        raise SystemExit(f"{run.tc}: no aligned runs discovered")
    ts = run.timestamps[0]
    return ts, run.run_dirs_by_timestamp[ts]


def _require_both_labels(labels: np.ndarray, *, context: str) -> None:
    unique = sorted(int(v) for v in set(labels.tolist()))
    if len(unique) < 2:
        raise SystemExit(f"{context}: LDA requires both label 0 and label 1; found {unique}")


def mode_slug_compare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    slug_root = experiment_slug_root(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
    )
    if not slug_root.is_dir():
        raise SystemExit(f"Experiment slug directory not found: {slug_root}")

    requested_tcs = list(dict.fromkeys(args.tc))
    runs = discover_embedding_runs(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
        timestamp=args.timestamp,
        latest_only=args.timestamp is None,
    )
    runs_by_tc = {run.tc: run for run in runs}

    for tc in requested_tcs:
        if tc not in runs_by_tc:
            print(
                f"Warning: skipping {tc} — no aligned P1/P2/Vanilla run with final checkpoints",
                flush=True,
            )

    selected_runs = [runs_by_tc[tc] for tc in requested_tcs if tc in runs_by_tc]
    if not selected_runs:
        raise SystemExit(f"No complete runs found for requested TCs under {slug_root}")

    out_dir = (args.out_dir / args.slug).resolve()
    seed = int(args.seed)
    dpi = int(args.dpi)

    if args.dry_run:
        print(f"Slug: {args.slug}")
        print(f"Output directory: {out_dir}")
        for run in selected_runs:
            ts, run_dirs = _resolve_slug_compare_run(run, timestamp=args.timestamp)
            test_path = synthetic_test_path(run.tc)
            print(f"\n{run.tc} (timestamp={ts})")
            print(f"  test: {test_path}")
            for mode in MODE_ORDER:
                run_dir = run_dirs[mode]
                ckpt = final_checkpoint_path(run_dir, mode)
                print(f"  {mode}: {ckpt}")
            print(f"  output: {out_dir / f'{run.tc}_lda.png'}")
        return

    apply_plot_style()
    written: list[Path] = []
    for run in selected_runs:
        ts, run_dirs = _resolve_slug_compare_run(run, timestamp=args.timestamp)
        test_path = synthetic_test_path(run.tc)
        if not test_path.is_file():
            raise SystemExit(f"Test set not found for {run.tc}: {test_path}")

        tokens, all_labels = load_labeled_txt(test_path)
        panel_data: list[tuple[str, np.ndarray, np.ndarray]] = []
        oov_by_mode: dict[str, int] = {}

        for mode in MODE_ORDER:
            run_dir = run_dirs[mode]
            ckpt = final_checkpoint_path(run_dir, mode)
            if not ckpt.is_file():
                raise SystemExit(f"{run.tc}/{mode}: checkpoint not found: {ckpt}")

            mode_label, _color = MODE_PLOT_STYLE[mode]
            embeddings, word2idx = load_checkpoint(ckpt)
            rows, _kept_tokens, kept_labels, oov = choose_test_points(
                tokens,
                all_labels,
                word2idx,
                max_entities=0,
                seed=seed,
            )
            oov_by_mode[mode] = oov
            if len(rows) < 2:
                raise SystemExit(
                    f"{run.tc}/{mode}: need at least 2 in-vocabulary entities for LDA; "
                    f"found {len(rows)} (OOV: {oov} / {len(tokens)})"
                )
            _require_both_labels(kept_labels, context=f"{run.tc}/{mode}")
            X = embeddings[np.asarray(rows, dtype=np.int64)]
            panel_data.append((mode_label, X, kept_labels))

        oov_values = set(oov_by_mode.values())
        if len(oov_values) > 1:
            print(
                f"Warning: {run.tc} OOV counts differ across modes: "
                + ", ".join(f"{mode}={oov_by_mode[mode]}" for mode in MODE_ORDER),
                flush=True,
            )

        out_path = out_dir / f"{run.tc}_lda.png"
        suptitle = f"{run.tc.upper()} — slug={args.slug}, timestamp={ts}"
        explained = plot_slug_compare_lda_panels(
            panel_data,
            out_path,
            suptitle=suptitle,
            seed=seed,
            dpi=dpi,
        )
        written.append(out_path)
        counts = Counter(int(v) for v in panel_data[0][2].tolist())
        print(f"Wrote {out_path}")
        print(f"  Plotted entities: {len(panel_data[0][2])} / {len(tokens)} test rows")
        print(f"  Label counts: {', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}")
        for mode_label, ld1, orth in explained:
            print(f"  {mode_label} LDA LD1 ratio: {ld1:.4f}, orthogonal PC ratio: {orth:.4f}")

    print(f"\nGenerated {len(written)} LDA plot(s) under {out_dir}")


def mode_test_set(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.test.is_file():
        raise SystemExit(f"Test set not found: {args.test}")
    synthetic = is_synthetic_test_set_run(args.checkpoint, args.test)
    max_entities = (
        0
        if args.max_entities is None and synthetic
        else (100 if args.max_entities is None else int(args.max_entities))
    )
    if max_entities < 0:
        raise SystemExit("--max-entities must be >= 0")

    annotate = bool(args.annotate) and not synthetic

    embeddings, word2idx = load_checkpoint(args.checkpoint)
    tokens, labels = load_labeled_txt(args.test)
    rows, kept_tokens, kept_labels, oov = choose_test_points(
        tokens,
        labels,
        word2idx,
        max_entities=max_entities,
        seed=int(args.seed),
    )
    if len(rows) < 2:
        raise SystemExit(
            f"Need at least 2 in-vocabulary test entities for LDA; found {len(rows)} "
            f"(OOV rows: {oov} / {len(tokens)})."
        )
    _require_both_labels(kept_labels, context="test-set")

    X = embeddings[np.asarray(rows, dtype=np.int64)]
    title = args.title or f"LDA of test-set entity embeddings ({args.test.name})"
    ld1, orth = plot_test_lda(
        X,
        kept_tokens,
        kept_labels,
        args.out,
        title=title,
        dpi=int(args.dpi),
        annotate=annotate,
        seed=int(args.seed),
    )
    counts = Counter(int(v) for v in kept_labels.tolist())
    print(f"Wrote {args.out}")
    print(f"Plotted entities: {len(rows)} / {len(tokens)} test rows")
    print(f"OOV skipped: {oov}")
    print(f"Label counts: {', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}")
    print(f"LDA LD1 ratio: {ld1:.4f}, orthogonal PC ratio: {orth:.4f}")


def add_mode_args(parser: argparse.ArgumentParser, mode: str) -> None:
    if mode == "slug-compare":
        parser.add_argument("--slug", default="default", help="P1/P2 experiment slug (default: default)")
        parser.add_argument("--tc", nargs="+", required=True, help="TC list, e.g. tc03 tc09")
        parser.add_argument("--dataset", default="synthetic", help="Dataset under output/ (default: synthetic)")
        parser.add_argument("--output-root", type=Path, default=Path("output"), help="Output root directory")
        parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="Directory for PNG outputs")
        parser.add_argument(
            "--timestamp",
            default=None,
            help="Use only this run timestamp (default: latest aligned per TC)",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--dry-run", action="store_true")
    elif mode == "test-set":
        parser.add_argument("--checkpoint", "-c", type=Path, required=True)
        parser.add_argument("--test", "-t", type=Path, required=True)
        parser.add_argument("--out", "-o", type=Path, default=Path("test_set_lda.png"))
        parser.add_argument("--max-entities", type=int, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--title", type=str, default=None)
        parser.add_argument("--no-annotate", dest="annotate", action="store_false")
        parser.set_defaults(annotate=True)


LDA_MODES = ("slug-compare", "test-set")

MODE_HANDLERS = {
    "slug-compare": mode_slug_compare,
    "test-set": mode_test_set,
}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", required=True, choices=LDA_MODES)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=LDA_MODES)
    add_mode_args(parser, pre_args.mode)
    args = parser.parse_args(["--mode", pre_args.mode, *remaining])
    MODE_HANDLERS[args.mode](args)


if __name__ == "__main__":
    main()
