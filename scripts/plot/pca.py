#!/usr/bin/env python3
"""Consolidated PCA visualizations for DBpedia and synthetic runs."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from _common import (  # noqa: E402
    LABEL_COLORS,
    MODE_ORDER,
    MODE_PLOT_STYLE,
    REPO_ROOT,
    SCRIPTS_DIR,
    DomainSplit,
    ExperimentRun,
    PlotPoint,
    apply_plot_style,
    choose_test_points,
    collect_points,
    discover_domain_splits,
    discover_checkpoint_runs,
    discover_embedding_runs,
    domain_color_map,
    experiment_slug_root,
    final_checkpoint_path,
    finetune_epoch_checkpoint_path,
    is_synthetic_test_set_run,
    jitter_pca_coords,
    load_checkpoint,
    _sorted_tc_names,
    resolve_tc_dir,
    shorten_entity,
    split_suffix,
    synthetic_test_path,
    vocab_index_for_token,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _evaluate import load_labeled_txt  # noqa: E402
from _maschine_init import (  # noqa: E402
    MASCHINE_CLASS_INIT_STRATEGIES,
    OWL_CLASS,
    RDF_TYPE,
    _most_specific_types,
    build_entity_to_class_tokens,
    load_ontology_mapping,
    maschine_embedding_for_token,
    merge_instance_rdf_types,
    token_inner_iri,
)
from _protograph_gen import RDFS_SUBCLASS, iter_nt_iris  # noqa: E402
from _walks import nt_term  # noqa: E402
from _word2vec import Word2VecWithStepLoss  # noqa: F401, E402

POSITIVE_COLOR = LABEL_COLORS[1]
NEGATIVE_COLOR = LABEL_COLORS[0]
CLASS_COLOR = "#4a7ab8"
PREDICATE_COLOR = "#666666"
DBO_NS = "http://dbpedia.org/ontology/"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
EXTRA_CLASS_RE = re.compile(r"^EXTRA_I_FOR_CLASS_(C_tc\d+_\d+)_")


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------


def plot_domain_pca(
    embeddings: np.ndarray,
    points: list[PlotPoint],
    output: Path,
    *,
    title: str,
    dpi: int,
    annotate: bool,
    seed: int,
) -> tuple[float, float]:
    rows = np.asarray([p.row for p in points], dtype=np.int64)
    domains = np.asarray([p.domain for p in points], dtype=object)
    labels = np.asarray([p.label for p in points], dtype=np.int64)

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(embeddings[rows])

    unique_domains = sorted(set(domains.tolist()))
    colors = domain_color_map(unique_domains)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    for domain in unique_domains:
        mask = domains == domain
        color = colors[domain]
        if set(labels.tolist()) <= {1}:
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                s=24,
                alpha=0.88,
                linewidths=0.35,
                edgecolors="#222222",
                color=color,
                label=f"{domain} (n={int(mask.sum())})",
            )
            continue
        pos_mask = mask & (labels == 1)
        neg_mask = mask & (labels == 0)
        if pos_mask.any():
            ax.scatter(
                Z[pos_mask, 0],
                Z[pos_mask, 1],
                s=24,
                alpha=0.88,
                linewidths=0.35,
                edgecolors="#222222",
                color=color,
                marker="o",
                label=f"{domain} + (n={int(pos_mask.sum())})",
            )
        if neg_mask.any():
            ax.scatter(
                Z[neg_mask, 0],
                Z[neg_mask, 1],
                s=20,
                alpha=0.45,
                linewidths=0.8,
                color=color,
                marker="x",
                label=f"{domain} - (n={int(neg_mask.sum())})",
            )

    if annotate:
        for (x, y), point in zip(Z, points, strict=True):
            ax.annotate(shorten_entity(point.token), (x, y), fontsize=6, alpha=0.7)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=11, markerscale=1.4, frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def mode_domains(args: argparse.Namespace) -> None:
    if args.max_per_domain < 0:
        raise SystemExit("--max-per-domain must be >= 0")
    if args.hard and args.split_dir not in (None, "train_test_hard"):
        raise SystemExit("--hard cannot be combined with --split-dir other than train_test_hard")

    dbpedia_root = args.dbpedia_root.resolve()
    tc_dir = resolve_tc_dir(args.tc, dbpedia_root)
    split_dir = "train_test_hard" if args.hard else (args.split_dir or "train_test")
    splits = discover_domain_splits(tc_dir, int(args.k), split_dir)
    if not splits:
        raise SystemExit(
            f"No test splits found under {tc_dir} with k={args.k} and split-dir={split_dir}"
        )

    if args.dry_run:
        for split in splits:
            print(f"{split.domain}\t{split.split_dir}\t{split.test_path}")
        print(f"\n{len(splits)} split(s)")
        return

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    embeddings, word2idx = load_checkpoint(checkpoint)
    points, oov_by_domain, skipped_by_domain = collect_points(
        splits,
        word2idx,
        label_filter=args.label_filter,
        max_per_domain=int(args.max_per_domain),
        seed=int(args.seed),
    )
    if len(points) < 2:
        raise SystemExit(
            f"Need at least 2 in-vocabulary entities for PCA; found {len(points)} "
            f"(OOV skipped: {sum(oov_by_domain.values())})."
        )

    out_path = (
        args.out.resolve()
        if args.out is not None
        else Path("output")
        / (
            f"{tc_dir.name}_domains_pca.png"
            if split_dir == "train_test"
            else f"{tc_dir.name}_domains_{split_dir.removeprefix('train_test_')}_pca.png"
        )
    )
    title = (
        args.title
        or f"{tc_dir.name} {split_dir} entities by domain ({checkpoint.stem}, {args.label_filter})"
    )
    ev1, ev2 = plot_domain_pca(
        embeddings,
        points,
        out_path,
        title=title,
        dpi=int(args.dpi),
        annotate=bool(args.annotate),
        seed=int(args.seed),
    )

    plotted_counts = Counter(point.domain for point in points)
    print(f"Wrote {out_path}")
    print(f"Discovered splits: {len(splits)}")
    print(f"Plotted entities: {len(points)}")
    for domain, count in sorted(plotted_counts.items()):
        print(
            f"  {domain}: plotted={count}, "
            f"oov={oov_by_domain[domain]}, label-filter-skipped={skipped_by_domain[domain]}"
        )
    print(f"PCA explained variance: {ev1:.4f}, {ev2:.4f}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


def _scatter_checkpoint_panel(
    ax: plt.Axes,
    spec: CheckpointSpec,
    points: list[PlotPoint],
    z: np.ndarray,
    colors: dict[str, object],
    *,
    ev1: float,
    ev2: float,
    jitter: float = 0.0,
    seed: int = 42,
) -> None:
    domains = np.asarray([p.domain for p in points], dtype=object)
    labels = np.asarray([p.label for p in points], dtype=np.int64)
    z = jitter_pca_coords(z, scale=jitter, rng=np.random.default_rng(seed))

    only_positive = set(labels.tolist()) <= {1}
    for domain in sorted(colors):
        mask = domains == domain
        if not mask.any():
            continue
        color = colors[domain]
        if only_positive:
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=18,
                alpha=0.86,
                linewidths=0.3,
                edgecolors="#222222",
                color=color,
            )
            continue
        pos_mask = mask & (labels == 1)
        neg_mask = mask & (labels == 0)
        if pos_mask.any():
            ax.scatter(
                z[pos_mask, 0],
                z[pos_mask, 1],
                s=18,
                alpha=0.86,
                linewidths=0.3,
                edgecolors="#222222",
                color=color,
                marker="o",
            )
        if neg_mask.any():
            ax.scatter(
                z[neg_mask, 0],
                z[neg_mask, 1],
                s=18,
                alpha=0.45,
                linewidths=0.8,
                color=color,
                marker="x",
            )

    ax.set_title(f"{spec.label}\nPC1 {ev1 * 100:.1f}%, PC2 {ev2 * 100:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)


def plot_checkpoint_panel(
    ax: plt.Axes,
    spec: CheckpointSpec,
    points: list[PlotPoint],
    embeddings: np.ndarray,
    colors: dict[str, object],
    *,
    seed: int,
    jitter: float = 0.0,
    z: np.ndarray | None = None,
    explained: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if z is None:
        rows = np.asarray([p.row for p in points], dtype=np.int64)
        pca = PCA(n_components=2, random_state=seed)
        z = pca.fit_transform(embeddings[rows])
        ev1 = float(pca.explained_variance_ratio_[0])
        ev2 = float(pca.explained_variance_ratio_[1])
    else:
        if explained is None:
            raise ValueError("explained variance required when z is precomputed")
        ev1, ev2 = explained

    _scatter_checkpoint_panel(
        ax,
        spec,
        points,
        z,
        colors,
        ev1=ev1,
        ev2=ev2,
        jitter=jitter,
        seed=seed,
    )
    return ev1, ev2


def _joint_pca_coords_for_compare(
    panel_data: list[tuple[CheckpointSpec, np.ndarray, list[PlotPoint], Counter[str], Counter[str]]],
    *,
    seed: int,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Fit one PCA on stacked embeddings (same entity order) and split per panel."""
    ref_points = panel_data[0][2]
    tokens = [point.token for point in ref_points]
    stacked: list[np.ndarray] = []
    for _spec, embeddings, points, _oov, _skipped in panel_data:
        token_to_row = {point.token: point.row for point in points}
        missing = [token for token in tokens if token not in token_to_row]
        if missing:
            raise SystemExit(
                "Joint PCA requires the same in-vocabulary entities in every checkpoint; "
                f"missing {len(missing)} tokens in {_spec.label} (e.g. {missing[0]!r})"
            )
        rows = np.asarray([token_to_row[token] for token in tokens], dtype=np.int64)
        stacked.append(embeddings[rows])

    joint = np.vstack(stacked)
    pca = PCA(n_components=2, random_state=seed)
    z_joint = pca.fit_transform(joint)
    n = len(tokens)
    coords = [z_joint[i * n : (i + 1) * n] for i in range(len(panel_data))]
    explained = (
        float(pca.explained_variance_ratio_[0]),
        float(pca.explained_variance_ratio_[1]),
    )
    return coords, explained


def mode_compare(args: argparse.Namespace) -> None:
    if args.max_per_domain < 0:
        raise SystemExit("--max-per-domain must be >= 0")
    if args.hard and args.split_dir not in (None, "train_test_hard"):
        raise SystemExit("--hard cannot be combined with --split-dir other than train_test_hard")

    dbpedia_root = args.dbpedia_root.resolve()
    tc_dir = resolve_tc_dir(args.tc, dbpedia_root)
    split_dir = "train_test_hard" if args.hard else (args.split_dir or "train_test")
    splits = discover_domain_splits(tc_dir, int(args.k), split_dir)
    if not splits:
        raise SystemExit(
            f"No test splits found under {tc_dir} with k={args.k} and split-dir={split_dir}"
        )

    if args.checkpoint:
        specs = [CheckpointSpec(label, Path(path).resolve()) for label, path in args.checkpoint]
    else:
        specs = [
            CheckpointSpec("no_pretrain", args.ckpt_no_pretrain.resolve()),
            CheckpointSpec("p1", args.ckpt_p1.resolve()),
            CheckpointSpec("p2", args.ckpt_p2.resolve()),
        ]
    jitter_labels = set(args.jitter_for)

    if args.dry_run:
        print("Splits:")
        for split in splits:
            print(f"  {split.domain}\t{split.split_dir}\t{split.test_path}")
        print("\nCheckpoints:")
        for spec in specs:
            print(f"  {spec.label}\t{spec.path}")
        return

    for spec in specs:
        if not spec.path.is_file():
            raise SystemExit(f"{spec.label} checkpoint not found: {spec.path}")

    panel_data: list[tuple[CheckpointSpec, np.ndarray, list[PlotPoint], Counter[str], Counter[str]]] = []
    all_domains: set[str] = set()
    for spec in specs:
        embeddings, word2idx = load_checkpoint(spec.path)
        points, oov_by_domain, skipped_by_domain = collect_points(
            splits,
            word2idx,
            label_filter=args.label_filter,
            max_per_domain=int(args.max_per_domain),
            seed=int(args.seed),
        )
        if len(points) < 2:
            raise SystemExit(
                f"{spec.label}: need at least 2 in-vocabulary entities for PCA; found {len(points)}"
            )
        all_domains.update(point.domain for point in points)
        panel_data.append((spec, embeddings, points, oov_by_domain, skipped_by_domain))

    colors = domain_color_map(sorted(all_domains))
    share_axes = bool(getattr(args, "aligned", False))
    fig, axes = plt.subplots(
        1,
        len(panel_data),
        figsize=(19, 6.5),
        sharex=share_axes,
        sharey=share_axes,
    )
    if len(panel_data) == 1:
        axes = [axes]

    joint_coords: list[np.ndarray] | None = None
    joint_explained: tuple[float, float] | None = None
    if share_axes:
        joint_coords, joint_explained = _joint_pca_coords_for_compare(
            panel_data,
            seed=int(args.seed),
        )

    explained: list[tuple[str, float, float]] = []
    for panel_idx, (ax, (spec, embeddings, points, _oov, _skipped)) in enumerate(
        zip(axes, panel_data, strict=True)
    ):
        z = joint_coords[panel_idx] if joint_coords is not None else None
        ev1, ev2 = plot_checkpoint_panel(
            ax,
            spec,
            points,
            embeddings,
            colors,
            seed=int(args.seed),
            jitter=float(args.jitter_scale) if spec.label in jitter_labels else 0.0,
            z=z,
            explained=joint_explained,
        )
        explained.append((spec.label, ev1, ev2))

    align_note = "joint PCA basis" if share_axes else "independent PCA per panel"
    title = (
        args.title
        or f"{tc_dir.name} {split_dir} PCA by domain ({args.label_filter}, {align_note})"
    )
    fig.suptitle(title, fontsize=16)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[domain],
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            markersize=9,
            label=domain,
        )
        for domain in sorted(colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(max(len(legend_handles), 1), 6),
        fontsize=12,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    default_name = (
        f"{tc_dir.name}_domains{split_suffix(split_dir)}_compare_pca"
        + ("_aligned" if share_axes else "")
        + ".png"
    )
    out_path = args.out.resolve() if args.out is not None else Path("output") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Discovered splits: {len(splits)}")
    for spec, _embeddings, points, oov_by_domain, skipped_by_domain in panel_data:
        plotted_counts = Counter(point.domain for point in points)
        print(f"\n{spec.label}: plotted entities={len(points)}")
        for domain, count in sorted(plotted_counts.items()):
            print(
                f"  {domain}: plotted={count}, "
                f"oov={oov_by_domain[domain]}, "
                f"label-filter-skipped={skipped_by_domain[domain]}"
            )
    print("\nPCA explained variance:")
    for label, ev1, ev2 in explained:
        print(f"  {label}: {ev1:.4f}, {ev2:.4f}")


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelSpec:
    panel_key: str
    domain: str
    split_dir: str
    k: int
    test_path: Path


def panel_key_for(domain: str, split_dir: str) -> str:
    if split_dir == "train_test_hard":
        return f"{domain}_hard"
    return domain


def discover_panels(tc_dir: Path) -> list[PanelSpec]:
    wanted = ("train_test", "train_test_hard")
    panels: list[PanelSpec] = []
    for domain_dir in sorted(tc_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        k_values = sorted(
            int(child.name)
            for child in domain_dir.iterdir()
            if child.is_dir() and child.name.isdigit()
        )
        if not k_values:
            continue
        max_k = k_values[-1]
        k_dir = domain_dir / str(max_k)
        for split_dir in wanted:
            test_path = k_dir / split_dir / "test.txt"
            if not test_path.is_file():
                continue
            panels.append(
                PanelSpec(
                    panel_key=panel_key_for(domain_dir.name, split_dir),
                    domain=domain_dir.name,
                    split_dir=split_dir,
                    k=max_k,
                    test_path=test_path,
                )
            )
    return panels


def collect_panel_points(
    panel: PanelSpec,
    word2idx: dict[str, int],
) -> tuple[list[PlotPoint], int, int]:
    tokens, labels = load_labeled_txt(panel.test_path)
    points: list[PlotPoint] = []
    seen: set[str] = set()
    oov = 0
    duplicates = 0
    for token, label_raw in zip(tokens, labels, strict=True):
        label = int(label_raw)
        idx = vocab_index_for_token(token, word2idx)
        if idx is None:
            oov += 1
            continue
        if token in seen:
            duplicates += 1
            continue
        seen.add(token)
        points.append(
            PlotPoint(token=token, domain=panel.panel_key, label=label, row=idx)
        )
    return points, oov, duplicates


def subplot_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def plot_domain_panel(
    ax: plt.Axes,
    embeddings: np.ndarray,
    points: list[PlotPoint],
    *,
    title: str,
    seed: int,
) -> tuple[float, float]:
    rows = np.asarray([p.row for p in points], dtype=np.int64)
    labels = np.asarray([p.label for p in points], dtype=np.int64)
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(embeddings[rows])
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.any():
        ax.scatter(
            Z[pos_mask, 0],
            Z[pos_mask, 1],
            s=16,
            alpha=0.82,
            linewidths=0.25,
            edgecolors="#222222",
            color=POSITIVE_COLOR,
            marker="o",
        )
    if neg_mask.any():
        ax.scatter(
            Z[neg_mask, 0],
            Z[neg_mask, 1],
            s=16,
            alpha=0.55,
            linewidths=0.7,
            color=NEGATIVE_COLOR,
            marker="x",
        )
    ev1 = float(pca.explained_variance_ratio_[0])
    ev2 = float(pca.explained_variance_ratio_[1])
    ax.set_title(f"{title}\n+={int(pos_mask.sum())}, -={int(neg_mask.sum())} | PC1 {ev1 * 100:.1f}%, PC2 {ev2 * 100:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    return ev1, ev2


def mode_panels(args: argparse.Namespace) -> None:
    dbpedia_root = args.dbpedia_root.resolve()
    tc_dir = resolve_tc_dir(args.tc, dbpedia_root)
    panels = discover_panels(tc_dir)
    if not panels:
        raise SystemExit(f"No panels found under {tc_dir}")

    if args.dry_run:
        for panel in panels:
            print(f"{panel.panel_key}\tk={panel.k}\t{panel.split_dir}\t{panel.test_path}")
        print(f"\n{len(panels)} panel(s)")
        return

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    embeddings, word2idx = load_checkpoint(checkpoint)
    panel_data: list[tuple[PanelSpec, list[PlotPoint], int, int]] = []
    for panel in panels:
        points, oov, duplicates = collect_panel_points(panel, word2idx)
        if len(points) < 2:
            print(
                f"Skipping {panel.panel_key}: need >=2 in-vocabulary entities, "
                f"found {len(points)} (oov={oov}, duplicates={duplicates})"
            )
            continue
        panel_data.append((panel, points, oov, duplicates))

    if not panel_data:
        raise SystemExit("No panels with enough in-vocabulary entities for PCA.")

    nrows, ncols = subplot_grid(len(panel_data))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.0 * nrows), squeeze=False)
    explained: list[tuple[str, float, float]] = []
    for ax, (panel, points, _oov, _duplicates) in zip(axes.ravel(), panel_data):
        ev1, ev2 = plot_domain_panel(
            ax,
            embeddings,
            points,
            title=f"{panel.panel_key} (k={panel.k})",
            seed=int(args.seed),
        )
        explained.append((panel.panel_key, ev1, ev2))
    for ax in axes.ravel()[len(panel_data) :]:
        ax.axis("off")

    suptitle = args.title or f"{tc_dir.name} domain PCA by class ({checkpoint.stem})"
    fig.suptitle(suptitle, fontsize=14)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=POSITIVE_COLOR,
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            markersize=8,
            label="positive (1)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=NEGATIVE_COLOR,
            markeredgewidth=1.0,
            markersize=8,
            linestyle="None",
            label="negative (0)",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=11, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    out_path = (
        args.out.resolve()
        if args.out is not None
        else Path("output") / f"{tc_dir.name}_{checkpoint.parent.name}_domain_panels.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Panels: {len(panel_data)}")
    for panel, points, oov, duplicates in panel_data:
        counts = Counter(point.label for point in points)
        print(
            f"  {panel.panel_key} (k={panel.k}): plotted={len(points)}, "
            f"pos={counts.get(1, 0)}, neg={counts.get(0, 0)}, "
            f"oov={oov}, duplicates={duplicates}"
        )


# ---------------------------------------------------------------------------
# protograph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtographSpec:
    label: str
    path: Path


def load_stage1_vectors(pretrained_path: Path) -> dict[str, np.ndarray]:
    if not pretrained_path.is_file():
        raise FileNotFoundError(pretrained_path)
    model = Word2Vec.load(str(pretrained_path))
    return {w: np.asarray(model.wv[w], dtype=np.float32).copy() for w in model.wv.index_to_key}


def collect_plot_tokens(
    splits: list[DomainSplit],
    *,
    label_filter: str,
    max_per_domain: int,
    seed: int,
) -> tuple[list[tuple[str, str, int]], Counter[str], Counter[str]]:
    from _common import keep_label

    rows_by_domain: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    seen_by_domain: dict[str, set[str]] = defaultdict(set)
    oov_by_domain: Counter[str] = Counter()
    skipped_by_domain: Counter[str] = Counter()

    for split in splits:
        tokens, labels = load_labeled_txt(split.test_path)
        domain_key = split.domain if split.split_dir == "train_test" else f"{split.domain}_hard"
        for token, label_raw in zip(tokens, labels, strict=True):
            label = int(label_raw)
            if not keep_label(label, label_filter):
                skipped_by_domain[domain_key] += 1
                continue
            if token in seen_by_domain[domain_key]:
                continue
            seen_by_domain[domain_key].add(token)
            rows_by_domain[domain_key].append((token, domain_key, label))

    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str, int]] = []
    for domain in sorted(rows_by_domain):
        domain_rows = rows_by_domain[domain]
        if max_per_domain > 0 and len(domain_rows) > max_per_domain:
            pick = rng.choice(len(domain_rows), size=max_per_domain, replace=False)
            domain_rows = [domain_rows[i] for i in pick.tolist()]
        rows.extend(domain_rows)
    return rows, oov_by_domain, skipped_by_domain


def iter_instance_type_triples(path: Path, subjects: set[str]):
    from _dbpedia.iri_nt_materialize import _guess_format, _open_binary_maybe_compressed
    from pyoxigraph import NamedNode, parse

    with _open_binary_maybe_compressed(path) as bio:
        for t in parse(input=bio, format=_guess_format(path), lenient=True):
            if not isinstance(t.subject, NamedNode):
                continue
            if not isinstance(t.predicate, NamedNode):
                continue
            if not isinstance(t.object, NamedNode):
                continue
            s, p, o = t.subject.value, t.predicate.value, t.object.value
            if p != RDF_TYPE or s not in subjects:
                continue
            yield s, p, o


def build_maschine_maps(
    ontology_path: Path,
    instance_types_path: Path | None,
    subjects: set[str],
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    parents, entity_types = load_ontology_mapping(ontology_path)
    if instance_types_path is not None:
        if not instance_types_path.is_file():
            raise FileNotFoundError(instance_types_path)
        n = merge_instance_rdf_types(
            parents,
            entity_types,
            instance_types_path,
            subject_filter=subjects,
            triple_iter=iter_instance_type_triples(instance_types_path, subjects),
        )
        print(f"Merged rdf:type for {n} instance subjects from {instance_types_path}")
    return parents, build_entity_to_class_tokens(parents, entity_types)


def embeddings_for_rows(
    rows: list[tuple[str, str, int]],
    stage1_vectors: dict[str, np.ndarray],
    parents: dict[str, set[str]],
    ent_to_classes: dict[str, list[str]],
    *,
    strategy: str,
    ancestor_decay: float,
) -> tuple[np.ndarray, list[PlotPoint], Counter[str]]:
    vectors: list[np.ndarray] = []
    points: list[PlotPoint] = []
    missing_by_domain: Counter[str] = Counter()
    dim = next(iter(stage1_vectors.values())).shape[0] if stage1_vectors else 0
    for token, domain, label in rows:
        vec = maschine_embedding_for_token(
            token,
            stage1_vectors,
            parents,
            ent_to_classes,
            strategy=strategy,
            ancestor_decay=ancestor_decay,
        )
        if vec is None:
            missing_by_domain[domain] += 1
            continue
        vectors.append(vec)
        points.append(PlotPoint(token=token, domain=domain, label=label, row=len(vectors) - 1))
    if not vectors:
        return np.zeros((0, dim), dtype=np.float32), points, missing_by_domain
    return np.stack(vectors, axis=0), points, missing_by_domain


def plot_protograph_panel(
    ax: plt.Axes,
    spec: ProtographSpec,
    points: list[PlotPoint],
    embeddings: np.ndarray,
    colors: dict[str, object],
    *,
    seed: int,
    point_size: float,
    jitter: float,
) -> tuple[float, float]:
    rows = np.asarray([p.row for p in points], dtype=np.int64)
    domains = np.asarray([p.domain for p in points], dtype=object)
    labels = np.asarray([p.label for p in points], dtype=np.int64)
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(embeddings[rows])
    Z = jitter_pca_coords(Z, scale=jitter, rng=np.random.default_rng(seed))
    only_positive = set(labels.tolist()) <= {1}
    fill_alpha = 0.72 if jitter > 0.0 else 0.86
    for domain in sorted(colors):
        mask = domains == domain
        if not mask.any():
            continue
        color = colors[domain]
        if only_positive:
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                s=point_size,
                alpha=fill_alpha,
                linewidths=0.4,
                edgecolors="#222222",
                color=color,
            )
            continue
        pos_mask = mask & (labels == 1)
        neg_mask = mask & (labels == 0)
        if pos_mask.any():
            ax.scatter(
                Z[pos_mask, 0],
                Z[pos_mask, 1],
                s=point_size,
                alpha=fill_alpha,
                linewidths=0.4,
                edgecolors="#222222",
                color=color,
                marker="o",
            )
        if neg_mask.any():
            ax.scatter(
                Z[neg_mask, 0],
                Z[neg_mask, 1],
                s=point_size * 1.15,
                alpha=0.5 if jitter > 0.0 else 0.45,
                linewidths=1.0,
                color=color,
                marker="x",
            )
    ev1 = float(pca.explained_variance_ratio_[0])
    ev2 = float(pca.explained_variance_ratio_[1])
    ax.set_title(f"{spec.label} (protograph)\nPC1 {ev1 * 100:.1f}%, PC2 {ev2 * 100:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    return ev1, ev2


def mode_protograph(args: argparse.Namespace) -> None:
    if args.max_per_domain < 0:
        raise SystemExit("--max-per-domain must be >= 0")
    if args.hard and args.split_dir not in (None, "train_test_hard"):
        raise SystemExit("--hard cannot be combined with --split-dir other than train_test_hard")
    if args.maschine_strategy == "ancestor_weighted" and not (
        0.0 < args.maschine_ancestor_decay <= 1.0
    ):
        raise SystemExit("--maschine-ancestor-decay must satisfy 0 < R <= 1 for ancestor_weighted")

    dbpedia_root = args.dbpedia_root.resolve()
    tc_dir = resolve_tc_dir(args.tc, dbpedia_root)
    split_dir = "train_test_hard" if args.hard else (args.split_dir or "train_test")
    splits = discover_domain_splits(tc_dir, int(args.k), split_dir)
    if not splits:
        raise SystemExit(
            f"No test splits found under {tc_dir} with k={args.k} and split-dir={split_dir}"
        )

    specs = [
        ProtographSpec("p1", args.ckpt_p1.resolve()),
        ProtographSpec("p2", args.ckpt_p2.resolve()),
    ]

    if args.dry_run:
        print("Splits:")
        for split in splits:
            print(f"  {split.domain}\t{split.split_dir}\t{split.test_path}")
        print("\nProtograph checkpoints:")
        for spec in specs:
            print(f"  {spec.label}\t{spec.path}")
        return

    ontology_path = args.ontology.resolve()
    if not ontology_path.is_file():
        raise SystemExit(f"Ontology not found: {ontology_path}")
    for spec in specs:
        if not spec.path.is_file():
            raise SystemExit(f"{spec.label} checkpoint not found: {spec.path}")

    token_rows, _oov_placeholder, skipped_by_domain = collect_plot_tokens(
        splits,
        label_filter=args.label_filter,
        max_per_domain=int(args.max_per_domain),
        seed=int(args.seed),
    )
    if not token_rows:
        raise SystemExit("No entities selected for plotting after label filtering.")

    subjects = {token_inner_iri(token) for token, _domain, _label in token_rows}
    if args.instance_types is None:
        print(
            "Warning: --instance-types not set; DBpedia resource URIs usually have no rdf:type in "
            "the schema-only ontology, so MASCHInE init will miss them.",
            file=sys.stderr,
        )
        parents, ent_to_classes = build_maschine_maps(ontology_path, None, subjects)
    else:
        parents, ent_to_classes = build_maschine_maps(
            ontology_path,
            args.instance_types.resolve(),
            subjects,
        )

    panel_data: list[
        tuple[ProtographSpec, np.ndarray, list[PlotPoint], Counter[str], Counter[str]]
    ] = []
    all_domains: set[str] = set()
    for spec in specs:
        stage1_vectors = load_stage1_vectors(spec.path)
        embeddings, points, missing_by_domain = embeddings_for_rows(
            token_rows,
            stage1_vectors,
            parents,
            ent_to_classes,
            strategy=args.maschine_strategy,
            ancestor_decay=float(args.maschine_ancestor_decay),
        )
        if len(points) < 2:
            raise SystemExit(
                f"{spec.label}: need at least 2 entities with MASCHInE/protograph vectors for PCA; "
                f"found {len(points)} (missing={sum(missing_by_domain.values())}). "
                "Pass --instance-types for DBpedia resources."
            )
        all_domains.update(point.domain for point in points)
        panel_data.append((spec, embeddings, points, missing_by_domain, skipped_by_domain))

    colors = domain_color_map(sorted(all_domains))
    fig, axes = plt.subplots(1, len(panel_data), figsize=(19, 6.5), sharex=False, sharey=False)
    if len(panel_data) == 1:
        axes = [axes]

    explained: list[tuple[str, float, float]] = []
    for ax, (spec, embeddings, points, _missing, _skipped) in zip(axes, panel_data, strict=True):
        ev1, ev2 = plot_protograph_panel(
            ax,
            spec,
            points,
            embeddings,
            colors,
            seed=int(args.seed),
            point_size=float(args.point_size),
            jitter=float(args.jitter),
        )
        explained.append((spec.label, ev1, ev2))

    title = (
        args.title
        or f"{tc_dir.name} {split_dir} protograph PCA ({args.label_filter}, init={args.maschine_strategy})"
    )
    fig.suptitle(title, fontsize=16)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[domain],
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            markersize=9,
            label=domain,
        )
        for domain in sorted(colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(max(len(legend_handles), 1), 6),
        fontsize=12,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    out_path = (
        args.out.resolve()
        if args.out is not None
        else Path("output")
        / f"{tc_dir.name}_domains{split_suffix(split_dir)}_protograph_compare_pca.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")
    for spec, _embeddings, points, missing_by_domain, skipped_by_domain in panel_data:
        plotted_counts = Counter(point.domain for point in points)
        print(f"\n{spec.label}: plotted entities={len(points)}")
        for domain, count in sorted(plotted_counts.items()):
            print(
                f"  {domain}: plotted={count}, "
                f"no-embedding={missing_by_domain[domain]}, "
                f"label-filter-skipped={skipped_by_domain[domain]}"
            )


# ---------------------------------------------------------------------------
# protograph-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassPanelPoint:
    row: int
    kind: str
    inner_iri: str
    display_name: str


def token_inner(token: str) -> str | None:
    if len(token) >= 2 and token[0] == "<" and token[-1] == ">":
        return token[1:-1]
    return None


def local_display_name(inner_iri: str) -> str:
    if inner_iri.startswith(DBO_NS):
        return inner_iri[len(DBO_NS) :]
    return inner_iri


def is_class_iri(inner: str) -> bool:
    if not inner.startswith(DBO_NS):
        return False
    local = inner.rsplit("/", 1)[-1]
    return bool(local) and local[0].isupper()


def load_protograph_class_iris(protograph_nt: Path) -> set[str]:
    classes: set[str] = set()
    for s, _p, o in iter_nt_iris(protograph_nt):
        if s.startswith(DBO_NS):
            classes.add(s)
        if o.startswith(DBO_NS):
            classes.add(o)
    return classes


def collect_class_panel_points(
    word2idx: dict[str, int],
    *,
    class_iris: set[str] | None,
    include_predicates: bool,
    max_points: int,
    seed: int,
) -> list[ClassPanelPoint]:
    points: list[ClassPanelPoint] = []
    for token, row in word2idx.items():
        inner = token_inner(token)
        if inner is None:
            continue
        if class_iris is not None and inner not in class_iris:
            if not include_predicates:
                continue
            kind = "predicate"
        elif is_class_iri(inner):
            kind = "class"
        else:
            if inner.startswith(XSD_NS) or not include_predicates:
                continue
            kind = "predicate"
        points.append(
            ClassPanelPoint(
                row=int(row),
                kind=kind,
                inner_iri=inner,
                display_name=local_display_name(inner),
            )
        )
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(points), size=max_points, replace=False)
        points = [points[i] for i in pick.tolist()]
    return points


def mode_protograph_only(args: argparse.Namespace) -> None:
    default_out = REPO_ROOT / "output" / "dbpedia_protograph_only"
    out_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    def _resolve_out(path: Path | None, default_name: str) -> Path:
        if path is None:
            return out_dir / default_name
        return path if path.is_absolute() else REPO_ROOT / path

    out_p1 = _resolve_out(args.p1_output, "pca_p1.png")
    out_p2 = _resolve_out(args.p2_output, "pca_p2.png")
    ckpt_p1 = args.ckpt_p1 if args.ckpt_p1.is_absolute() else REPO_ROOT / args.ckpt_p1
    ckpt_p2 = args.ckpt_p2 if args.ckpt_p2.is_absolute() else REPO_ROOT / args.ckpt_p2

    def _class_iris(proto_arg: Path) -> set[str] | None:
        proto = proto_arg if proto_arg.is_absolute() else REPO_ROOT / proto_arg
        return load_protograph_class_iris(proto) if proto.is_file() else None

    panel_specs: list[tuple[str, Path, Path, Path | None, set[str] | None]] = []
    if args.modes in ("both", "p1"):
        names_p1 = None if args.no_names_file else out_dir / "pca_p1_point_names.txt"
        panel_specs.append(("P1", ckpt_p1, out_p1, names_p1, _class_iris(args.protograph_p1)))
    if args.modes in ("both", "p2"):
        names_p2 = None if args.no_names_file else out_dir / "pca_p2_point_names.txt"
        panel_specs.append(("P2", ckpt_p2, out_p2, names_p2, _class_iris(args.protograph_p2)))

    suptitle = args.title or "Protograph-only RDF2Vec — PCA"
    for label, ckpt, png_path, names_path, class_iris in panel_specs:
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        emb, word2idx = load_checkpoint(ckpt)
        points = collect_class_panel_points(
            word2idx,
            class_iris=class_iris,
            include_predicates=args.include_predicates,
            max_points=args.max_points,
            seed=args.seed,
        )
        if len(points) < 2:
            raise SystemExit(f"{label}: need at least 2 points for PCA, got {len(points)}")

        rows = np.asarray([p.row for p in points], dtype=np.int64)
        pca = PCA(n_components=2, random_state=args.seed)
        Z = pca.fit_transform(emb[rows])
        class_mask = np.array([p.kind == "class" for p in points])
        pred_mask = np.array([p.kind == "predicate" for p in points])

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        if class_mask.any():
            ax.scatter(
                Z[class_mask, 0],
                Z[class_mask, 1],
                s=args.point_size,
                alpha=0.75,
                linewidths=0.2,
                edgecolors="#333333",
                color=CLASS_COLOR,
                label="class",
            )
        if pred_mask.any():
            ax.scatter(
                Z[pred_mask, 0],
                Z[pred_mask, 1],
                s=args.point_size * 0.6,
                alpha=0.45,
                linewidths=0,
                color=PREDICATE_COLOR,
                marker="x",
                label="predicate",
            )
        if not args.no_annotate:
            for i, p in enumerate(points):
                if p.kind == "class" or args.annotate_predicates:
                    ax.annotate(p.display_name, (Z[i, 0], Z[i, 1]), fontsize=7, alpha=0.9)
        ax.set_title(f"{label}\nPC1 {pca.explained_variance_ratio_[0] * 100:.1f}%, PC2 {pca.explained_variance_ratio_[1] * 100:.1f}%")
        if suptitle:
            fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png_path}")
        if names_path is not None:
            class_names = sorted(p.display_name for p in points if p.kind == "class")
            names_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
            print(f"Wrote point names: {names_path}")


# ---------------------------------------------------------------------------
# test-set
# ---------------------------------------------------------------------------


def plot_labeled_pca_panel(
    ax: plt.Axes,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    seed: int,
    label_colors: dict[int, str] | None = None,
    show_legend: bool = False,
) -> tuple[float, float]:
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)
    colors = label_colors or LABEL_COLORS
    unique_labels = sorted(int(v) for v in set(labels.tolist()))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    fallback_cmap = plt.colormaps["tab10" if len(unique_labels) <= 10 else "tab20"].resampled(
        max(len(unique_labels), 1)
    )
    norm = max(len(unique_labels) - 1, 1)
    for label in unique_labels:
        mask = labels == label
        color = colors.get(label, fallback_cmap(label_to_idx[label] / norm))
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=18,
            alpha=0.75,
            linewidths=0,
            edgecolors="none",
            color=color,
            label=f"label {label} (n={int(mask.sum())})" if show_legend else None,
        )
    ev1 = float(pca.explained_variance_ratio_[0])
    ev2 = float(pca.explained_variance_ratio_[1])
    ax.set_xlabel(f"PC1 ({ev1 * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({ev2 * 100:.1f} %)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=True)
    return ev1, ev2


def plot_test_pca(
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
    ev1, ev2 = plot_labeled_pca_panel(
        ax,
        X,
        labels,
        title=title,
        seed=seed,
        show_legend=True,
    )
    if annotate:
        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(X)
        for x, y, token in zip(Z[:, 0], Z[:, 1], tokens, strict=True):
            ax.annotate(shorten_entity(token), (x, y), fontsize=6, alpha=0.75)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ev1, ev2


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
            f"Need at least 2 in-vocabulary test entities for PCA; found {len(rows)} "
            f"(OOV rows: {oov} / {len(tokens)})."
        )

    X = embeddings[np.asarray(rows, dtype=np.int64)]
    title = args.title or f"PCA of test-set entity embeddings ({args.test.name})"
    ev1, ev2 = plot_test_pca(
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
    print(f"PCA explained variance: {ev1:.4f}, {ev2:.4f}")


# ---------------------------------------------------------------------------
# slug-compare
# ---------------------------------------------------------------------------


def plot_slug_compare_panels(
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    output: Path,
    *,
    seed: int,
    dpi: int,
) -> list[tuple[str, float, float]]:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), sharex=False, sharey=False)
    if len(panels) == 1:
        axes = [axes]

    explained: list[tuple[str, float, float]] = []
    legend_labels = panels[0][2]
    for ax, (mode_label, X, labels) in zip(axes, panels, strict=True):
        ev1, ev2 = plot_labeled_pca_panel(ax, X, labels, title="", seed=seed)
        explained.append((mode_label, ev1, ev2))

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
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return explained


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


def _slug_compare_checkpoint_path(run_dir: Path, mode: str, *, epoch: int | None) -> Path:
    if epoch is not None:
        return finetune_epoch_checkpoint_path(run_dir, epoch)
    return final_checkpoint_path(run_dir, mode)


def mode_slug_compare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    slug_root = experiment_slug_root(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
    )
    if not slug_root.is_dir():
        raise SystemExit(f"Experiment slug directory not found: {slug_root}")

    epoch: int | None = args.epoch
    discover_runs = discover_checkpoint_runs if epoch is not None else discover_embedding_runs
    missing_msg = (
        "no aligned P1/P2/Vanilla run with finetune epoch checkpoints"
        if epoch is not None
        else "no aligned P1/P2/Vanilla run with final checkpoints"
    )
    runs = discover_runs(
        output_root=output_root,
        dataset=args.dataset,
        slug=args.slug,
        timestamp=args.timestamp,
        latest_only=args.timestamp is None,
    )
    runs_by_tc = {run.tc: run for run in runs}

    if args.tc:
        requested_tcs = list(dict.fromkeys(args.tc))
        for tc in requested_tcs:
            if tc not in runs_by_tc:
                print(
                    f"Warning: skipping {tc} — {missing_msg}",
                    flush=True,
                )
        selected_runs = [runs_by_tc[tc] for tc in requested_tcs if tc in runs_by_tc]
    else:
        all_tcs = _sorted_tc_names(slug_root)
        found_tcs = {run.tc for run in runs}
        for tc in all_tcs:
            if tc not in found_tcs:
                print(
                    f"Warning: skipping {tc} — {missing_msg}",
                    flush=True,
                )
        selected_runs = runs

    if not selected_runs:
        raise SystemExit(f"No complete runs found under {slug_root}")

    out_dir = (args.out_dir / args.slug).resolve()
    if epoch is not None:
        out_dir = out_dir / f"epoch{epoch:02d}"
    seed = int(args.seed)
    dpi = int(args.dpi)

    if args.dry_run:
        print(f"Slug: {args.slug}")
        if epoch is not None:
            print(f"Epoch: {epoch}")
        print(f"Output directory: {out_dir}")
        for run in selected_runs:
            ts, run_dirs = _resolve_slug_compare_run(run, timestamp=args.timestamp)
            test_path = synthetic_test_path(run.tc)
            print(f"\n{run.tc} (timestamp={ts})")
            print(f"  test: {test_path}")
            for mode in MODE_ORDER:
                run_dir = run_dirs[mode]
                ckpt = _slug_compare_checkpoint_path(run_dir, mode, epoch=epoch)
                print(f"  {mode}: {ckpt}")
            print(f"  output: {out_dir / f'{run.tc}_pca.png'}")
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
            ckpt = _slug_compare_checkpoint_path(run_dir, mode, epoch=epoch)
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
                    f"{run.tc}/{mode}: need at least 2 in-vocabulary entities for PCA; "
                    f"found {len(rows)} (OOV: {oov} / {len(tokens)})"
                )
            X = embeddings[np.asarray(rows, dtype=np.int64)]
            panel_data.append((mode_label, X, kept_labels))

        oov_values = set(oov_by_mode.values())
        if len(oov_values) > 1:
            print(
                f"Warning: {run.tc} OOV counts differ across modes: "
                + ", ".join(f"{mode}={oov_by_mode[mode]}" for mode in MODE_ORDER),
                flush=True,
            )

        out_path = out_dir / f"{run.tc}_pca.png"
        explained = plot_slug_compare_panels(
            panel_data,
            out_path,
            seed=seed,
            dpi=dpi,
        )
        written.append(out_path)
        counts = Counter(int(v) for v in panel_data[0][2].tolist())
        print(f"Wrote {out_path}")
        print(f"  Plotted entities: {len(panel_data[0][2])} / {len(tokens)} test rows")
        print(f"  Label counts: {', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}")
        for mode_label, ev1, ev2 in explained:
            print(f"  {mode_label} PCA explained variance: {ev1:.4f}, {ev2:.4f}")

    print(f"\nGenerated {len(written)} PCA plot(s) under {out_dir}")


# ---------------------------------------------------------------------------
# rdf-types
# ---------------------------------------------------------------------------


def load_subclass_parents(ontology_path: Path) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    for s, p, o in iter_nt_iris(ontology_path):
        if p == RDFS_SUBCLASS:
            parents[s].add(o)
    return dict(parents)


def load_rdf_types_from_graph(graph_path: Path) -> dict[str, set[str]]:
    raw: dict[str, set[str]] = defaultdict(set)
    for s, p, o in iter_nt_iris(graph_path):
        if p != RDF_TYPE:
            continue
        if o == OWL_CLASS:
            continue
        raw[s].add(o)
    return dict(raw)


def entity_class_label(types: set[str], parents: dict[str, set[str]]) -> str | None:
    if not types:
        return None
    mst = _most_specific_types(types, parents)
    if not mst:
        return sorted(types)[0]
    return sorted(mst)[0]


def shorten_iri(iri: str, max_len: int = 48) -> str:
    if len(iri) <= max_len:
        return iri
    return iri[: max_len - 3] + "..."


def resolve_vocab_key(entity_inner: str, word2idx: dict[str, int]) -> str | None:
    angled = nt_term(entity_inner)
    if angled in word2idx:
        return angled
    if entity_inner in word2idx:
        return entity_inner
    return None


def infer_class_from_token(inner: str) -> str | None:
    m = EXTRA_CLASS_RE.match(inner)
    if m:
        return m.group(1)
    return None


def mode_rdf_types(args: argparse.Namespace) -> None:
    if not args.graph.is_file():
        raise SystemExit(f"Graph not found: {args.graph}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.ontology is not None and not args.ontology.is_file():
        raise SystemExit(f"Ontology not found: {args.ontology}")

    parents = load_subclass_parents(args.ontology) if args.ontology is not None else {}
    raw_types = load_rdf_types_from_graph(args.graph)
    emb, word2idx = load_checkpoint(args.checkpoint)

    entity_to_label: dict[str, str] = {}
    for ent, types in raw_types.items():
        lab = entity_class_label(types, parents)
        if lab is not None:
            entity_to_label[ent] = lab

    rows: list[int] = []
    labels: list[str] = []
    for inner, lab in entity_to_label.items():
        key = resolve_vocab_key(inner, word2idx)
        if key is None:
            continue
        rows.append(word2idx[key])
        labels.append(lab)

    if args.extra_i_fallback:
        for token in word2idx:
            if len(token) < 2 or token[0] != "<" or token[-1] != ">":
                continue
            inner = token[1:-1]
            if inner in entity_to_label:
                continue
            c = infer_class_from_token(inner)
            if c is None:
                continue
            lab = entity_class_label({c}, parents)
            if lab is None:
                continue
            rows.append(word2idx[token])
            labels.append(lab)

    if not rows:
        raise SystemExit(
            "No embeddings matched entities with rdf:type in --graph. "
            "Check graph.nt and vocabulary alignment."
        )

    rng = np.random.default_rng(args.seed)
    n = len(rows)
    if args.max_points > 0 and n > args.max_points:
        pick = rng.choice(n, size=args.max_points, replace=False)
        rows = [rows[i] for i in pick]
        labels = [labels[i] for i in pick]

    label_counts = Counter(labels)
    if args.max_classes > 0 and len(label_counts) > args.max_classes:
        keep = {lab for lab, _ in label_counts.most_common(args.max_classes)}
        filt_r: list[int] = []
        filt_l: list[str] = []
        for r, lab in zip(rows, labels, strict=True):
            filt_r.append(r)
            filt_l.append(lab if lab in keep else "__other__")
        rows, labels = filt_r, filt_l

    X = emb[np.asarray(rows, dtype=np.int64)]
    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(X)
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    cmap = plt.colormaps["tab20"].resampled(max(n_classes, 1))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    norm = max(n_classes - 1, 1)
    point_colors = [cmap(label_to_idx[lab] / norm) for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(Z[:, 0], Z[:, 1], c=point_colors, s=8, alpha=0.65, linewidths=0, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
    ax.set_title("RDF2Vec embeddings — PCA by class (rdf:type)")
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(label_to_idx[lab] / norm),
            markersize=8,
            label=shorten_iri(lab),
        )
        for lab in unique_labels
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"Points: {len(rows)}  |  Classes: {n_classes}")
    print(
        f"PCA explained variance: {pca.explained_variance_ratio_[0]:.4f}, "
        f"{pca.explained_variance_ratio_[1]:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_mode_args(parser: argparse.ArgumentParser, mode: str) -> None:
    if mode == "domains":
        parser.add_argument("--checkpoint", "-c", type=Path, required=True)
        parser.add_argument("--tc", type=Path, required=True)
        parser.add_argument("--dbpedia-root", type=Path, default=Path("v1/dbpedia"))
        parser.add_argument("--k", type=int, default=5000)
        parser.add_argument("--split-dir", choices=("train_test", "train_test_hard", "both"), default=None)
        parser.add_argument("--hard", action="store_true")
        parser.add_argument("--label-filter", choices=("positive", "negative", "all"), default="positive")
        parser.add_argument("--max-per-domain", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--out", "-o", type=Path, default=None)
        parser.add_argument("--title", type=str, default=None)
        parser.add_argument("--annotate", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
    elif mode == "compare":
        parser.add_argument("--tc", type=Path, required=True)
        parser.add_argument("--dbpedia-root", type=Path, default=Path("v1/dbpedia"))
        parser.add_argument("--k", type=int, default=5000)
        parser.add_argument("--split-dir", choices=("train_test", "train_test_hard", "both"), default=None)
        parser.add_argument("--hard", action="store_true")
        parser.add_argument("--label-filter", choices=("positive", "negative", "all"), default="positive")
        parser.add_argument("--max-per-domain", type=int, default=0)
        parser.add_argument("--ckpt-no-pretrain", type=Path, default=Path("output/dbpedia_no_pretrain/rdf2vec_final.pt"))
        parser.add_argument("--ckpt-p1", type=Path, default=Path("output/dbpedia_p1/rdf2vec_final.pt"))
        parser.add_argument("--ckpt-p2", type=Path, default=Path("output/dbpedia_p2/rdf2vec_final.pt"))
        parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("LABEL", "PATH"))
        parser.add_argument("--jitter-for", action="append", default=[], metavar="LABEL")
        parser.add_argument("--jitter-scale", type=float, default=0.06)
        parser.add_argument(
            "--aligned",
            action="store_true",
            help="Fit one joint PCA on all checkpoints (same coordinate system). "
            "Exposes rotation differences hidden by default independent PCA per panel.",
        )
        parser.add_argument("--out", "-o", type=Path, default=None)
        parser.add_argument("--title", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--dry-run", action="store_true")
    elif mode == "panels":
        parser.add_argument("--checkpoint", "-c", type=Path, required=True)
        parser.add_argument("--tc", type=Path, required=True)
        parser.add_argument("--dbpedia-root", type=Path, default=Path("v1/dbpedia"))
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--out", "-o", type=Path, default=None)
        parser.add_argument("--title", type=str, default=None)
        parser.add_argument("--dry-run", action="store_true")
    elif mode == "protograph":
        parser.add_argument("--tc", type=Path, required=True)
        parser.add_argument("--dbpedia-root", type=Path, default=Path("v1/dbpedia"))
        parser.add_argument("--k", type=int, default=5000)
        parser.add_argument("--split-dir", choices=("train_test", "train_test_hard", "both"), default=None)
        parser.add_argument("--hard", action="store_true")
        parser.add_argument("--label-filter", choices=("positive", "negative", "all"), default="positive")
        parser.add_argument("--max-per-domain", type=int, default=0)
        parser.add_argument("--ckpt-p1", type=Path, default=Path("output/dbpedia_p1/rdf2vec_pretrained.model"))
        parser.add_argument("--ckpt-p2", type=Path, default=Path("output/dbpedia_p2/rdf2vec_pretrained.model"))
        parser.add_argument("--ontology", type=Path, default=Path("data/ontology--DEV_type=parsed.nt"))
        parser.add_argument("--instance-types", type=Path, default=None)
        parser.add_argument("--maschine-strategy", choices=sorted(MASCHINE_CLASS_INIT_STRATEGIES), default="most_specific")
        parser.add_argument("--maschine-ancestor-decay", type=float, default=0.5)
        parser.add_argument("--out", "-o", type=Path, default=None)
        parser.add_argument("--title", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--point-size", type=float, default=40.0)
        parser.add_argument("--jitter", type=float, default=0.06)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--dry-run", action="store_true")
    elif mode == "protograph-only":
        default_out = REPO_ROOT / "output" / "dbpedia_protograph_only"
        parser.add_argument("--ckpt-p1", type=Path, default=default_out / "p1" / "rdf2vec_pretrained.pt")
        parser.add_argument("--ckpt-p2", type=Path, default=default_out / "p2" / "rdf2vec_pretrained.pt")
        parser.add_argument("--protograph-p1", type=Path, default=default_out / "protograph_p1.nt")
        parser.add_argument("--protograph-p2", type=Path, default=default_out / "protograph_p2.nt")
        parser.add_argument("--output-dir", type=Path, default=default_out)
        parser.add_argument("--p1-output", type=Path, default=None)
        parser.add_argument("--p2-output", type=Path, default=None)
        parser.add_argument("--modes", choices=("both", "p1", "p2"), default="both")
        parser.add_argument("--include-predicates", action="store_true")
        parser.add_argument("--max-points", type=int, default=0)
        parser.add_argument("--point-size", type=float, default=10.0)
        parser.add_argument("--no-annotate", action="store_true")
        parser.add_argument("--annotate-fontsize", type=float, default=None)
        parser.add_argument("--label-jitter", type=float, default=0.035)
        parser.add_argument("--annotate-predicates", action="store_true")
        parser.add_argument("--no-names-file", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--title", type=str, default=None)
    elif mode == "test-set":
        parser.add_argument("--checkpoint", "-c", type=Path, required=True)
        parser.add_argument("--test", "-t", type=Path, required=True)
        parser.add_argument("--out", "-o", type=Path, default=Path("test_set_pca.png"))
        parser.add_argument("--max-entities", type=int, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--title", type=str, default=None)
        parser.add_argument("--no-annotate", dest="annotate", action="store_false")
        parser.set_defaults(annotate=True)
    elif mode == "slug-compare":
        parser.add_argument("--slug", default="default", help="P1/P2 experiment slug (default: default)")
        parser.add_argument(
            "--tc",
            nargs="+",
            default=None,
            help="TC list to plot (default: all TCs under the slug with complete runs)",
        )
        parser.add_argument("--dataset", default="synthetic", help="Dataset under output/ (default: synthetic)")
        parser.add_argument("--output-root", type=Path, default=Path("output"), help="Output root directory")
        parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="Directory for PNG outputs")
        parser.add_argument(
            "--timestamp",
            default=None,
            help="Use only this run timestamp (default: latest aligned per TC)",
        )
        parser.add_argument(
            "--epoch",
            type=int,
            default=None,
            help="Finetune epoch checkpoint (0 = post-pretrain init); default: final checkpoint",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--dry-run", action="store_true")
    elif mode == "rdf-types":
        parser.add_argument("--graph", type=Path, required=True)
        parser.add_argument("--ontology", type=Path, default=None)
        parser.add_argument("-c", "--checkpoint", type=Path, required=True)
        parser.add_argument("-o", "--output", type=Path, default=Path("rdf2vec_pca.png"))
        parser.add_argument("--max-points", type=int, default=8000)
        parser.add_argument("--max-classes", type=int, default=40)
        parser.add_argument("--dpi", type=int, default=150)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--extra-i-fallback", action="store_true")



MODE_HANDLERS = {
    "domains": mode_domains,
    "compare": mode_compare,
    "panels": mode_panels,
    "protograph": mode_protograph,
    "protograph-only": mode_protograph_only,
    "test-set": mode_test_set,
    "slug-compare": mode_slug_compare,
    "rdf-types": mode_rdf_types,
}


PCA_MODES = (
    "domains",
    "compare",
    "panels",
    "protograph",
    "protograph-only",
    "test-set",
    "slug-compare",
    "rdf-types",
)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", required=True, choices=PCA_MODES)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=PCA_MODES)
    add_mode_args(parser, pre_args.mode)
    args = parser.parse_args(["--mode", pre_args.mode, *remaining])
    MODE_HANDLERS[args.mode](args)


if __name__ == "__main__":
    main()
