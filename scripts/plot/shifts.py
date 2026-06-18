#!/usr/bin/env python3
"""
Plot embedding movement from an untrained baseline checkpoint or random init.

Aligns checkpoints by token, samples entities, and plots L2 distance from baseline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (
    SCRIPTS_DIR,
    load_checkpoint,
    make_random_baseline,
    vocab_index_for_token,
)
from _evaluate import load_labeled_txt

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.name or path.stem, path
    label, _, raw_path = spec.partition("=")
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty checkpoint label in: {spec}")
    return label, Path(raw_path)


def shorten_token(token: str, max_len: int = 32) -> str:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        token = token[1:-1]
    token = token.rstrip("/").rsplit("/", 1)[-1]
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."


def tokens_from_test(
    test_path: Path,
    word2idx_by_checkpoint: dict[str, dict[str, int]],
    *,
    label_filter: str,
) -> tuple[list[str], int]:
    tokens, labels = load_labeled_txt(test_path)
    kept: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for token, label_raw in zip(tokens, labels, strict=True):
        label = int(label_raw)
        if label_filter == "positive" and label != 1:
            skipped += 1
            continue
        if label_filter == "negative" and label != 0:
            skipped += 1
            continue
        if token in seen:
            continue
        if all(vocab_index_for_token(token, w2i) is not None for w2i in word2idx_by_checkpoint.values()):
            kept.append(token)
            seen.add(token)
        else:
            skipped += 1
    return kept, skipped


def tokens_from_vocab(word2idx_by_checkpoint: dict[str, dict[str, int]]) -> list[str]:
    common = set.intersection(*(set(w2i) for w2i in word2idx_by_checkpoint.values()))
    return sorted(common)


def sample_tokens(tokens: list[str], *, max_points: int, seed: int) -> list[str]:
    if max_points < 0:
        raise SystemExit("--max-points must be >= 0")
    if max_points == 0 or len(tokens) <= max_points:
        return tokens
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tokens), size=max_points, replace=False)
    return [tokens[i] for i in pick.tolist()]


def embedding_row(token: str, emb: np.ndarray, word2idx: dict[str, int]) -> np.ndarray:
    idx = vocab_index_for_token(token, word2idx)
    if idx is None:
        raise KeyError(token)
    return emb[int(idx)]


def write_csv(
    path: Path,
    tokens: list[str],
    shifts_by_label: dict[str, np.ndarray],
    cosine_by_label: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["token"]
        for label in shifts_by_label:
            header.append(f"{label}_l2_shift")
            header.append(f"{label}_cosine_distance")
        writer.writerow(header)
        for i, token in enumerate(tokens):
            row: list[str | float] = [token]
            for label in shifts_by_label:
                row.append(f"{float(shifts_by_label[label][i]):.8f}")
                row.append(f"{float(cosine_by_label[label][i]):.8f}")
            writer.writerow(row)


def plot_shifts(
    path: Path,
    tokens: list[str],
    shifts_by_label: dict[str, np.ndarray],
    *,
    title: str,
    dpi: int,
) -> None:
    x = np.arange(len(tokens))
    labels = list(shifts_by_label)
    offsets = np.linspace(-0.24, 0.24, num=max(len(labels), 1))
    fig, ax = plt.subplots(figsize=(13, 7))
    for offset, label in zip(offsets, labels, strict=True):
        shifts = shifts_by_label[label]
        x_shifted = x + offset
        ax.scatter(
            x_shifted,
            shifts,
            s=38,
            alpha=0.88,
            edgecolors="#222222",
            linewidths=0.25,
            label=f"{label} (n={len(shifts)}, mean={float(np.mean(shifts)):.3f})",
        )
        ax.plot(x_shifted, shifts, linewidth=0.8, alpha=0.28)

    ax.set_xlabel("Sampled token")
    ax.set_ylabel("L2 distance from baseline vector")
    ax.set_title(f"{title}\nAll plotted checkpoints share the same {len(tokens)} sampled tokens")
    ax.set_xticks(x)
    ax.set_xticklabels([shorten_token(t) for t in tokens], rotation=90, fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_baseline(
    *,
    use_random: bool,
    baseline_checkpoint: Path | None,
    checkpoints: list[tuple[str, Path]],
    seed: int,
) -> tuple[np.ndarray, dict[str, int], str]:
    if use_random:
        if not checkpoints:
            raise SystemExit("--checkpoint required when --baseline random")
        first_label, first_path = checkpoints[0]
        ref_emb, ref_w2i = load_checkpoint(first_path)
        baseline_emb, baseline_w2i = make_random_baseline(ref_w2i, ref_emb.shape[1], seed)
        return baseline_emb, baseline_w2i, f"random init (seed={seed}, vocab from {first_label})"

    baseline_path = baseline_checkpoint or Path("output/dbedia_untrained/rdf2vec_final.pt")
    if not baseline_path.is_file():
        raise SystemExit(f"Baseline checkpoint not found: {baseline_path}")
    emb, w2i = load_checkpoint(baseline_path)
    return emb, w2i, str(baseline_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=("random",),
        default=None,
        help="Use random-init baseline instead of a checkpoint file",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Untrained baseline checkpoint (default: output/dbedia_untrained/rdf2vec_final.pt)",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        default=None,
        metavar="LABEL=PATH",
        help="Trained checkpoint to compare; repeatable",
    )
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument(
        "--label-filter",
        choices=("positive", "negative", "all"),
        default="all",
    )
    parser.add_argument("--max-points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/dbpedia_compare/embedding_shifts_vs_untrained.png"),
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.baseline == "random" and args.baseline_checkpoint is not None:
        raise SystemExit("Use either --baseline random or --baseline-checkpoint, not both")

    checkpoints = args.checkpoint or [
        ("no_pretrain", Path("output/dbpedia_no_pretrain/rdf2vec_final.pt")),
        ("p1", Path("output/dbpedia_p1/rdf2vec_final.pt")),
        ("p2", Path("output/dbpedia_p2/rdf2vec_final.pt")),
    ]

    for label, path in checkpoints:
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found for {label}: {path}")
    if args.test is not None and not args.test.is_file():
        raise SystemExit(f"Test set not found: {args.test}")

    baseline_emb, baseline_w2i, baseline_desc = build_baseline(
        use_random=args.baseline == "random",
        baseline_checkpoint=args.baseline_checkpoint,
        checkpoints=checkpoints,
        seed=int(args.seed),
    )

    embeddings_by_label: dict[str, np.ndarray] = {}
    word2idx_by_label: dict[str, dict[str, int]] = {"baseline": baseline_w2i}
    for label, path in checkpoints:
        emb, w2i = load_checkpoint(path)
        if emb.shape[1] != baseline_emb.shape[1]:
            raise SystemExit(
                f"Dimension mismatch for {label}: {emb.shape[1]} vs baseline {baseline_emb.shape[1]}"
            )
        if label in embeddings_by_label:
            raise SystemExit(f"Duplicate checkpoint label: {label}")
        embeddings_by_label[label] = emb
        word2idx_by_label[label] = w2i

    if args.test is not None:
        candidates, skipped = tokens_from_test(
            args.test,
            word2idx_by_label,
            label_filter=args.label_filter,
        )
        source = str(args.test)
    else:
        candidates = tokens_from_vocab(word2idx_by_label)
        skipped = 0
        source = "common checkpoint vocabulary"

    if not candidates:
        raise SystemExit("No common in-vocabulary tokens found to plot.")
    sampled = sample_tokens(candidates, max_points=int(args.max_points), seed=int(args.seed))
    if not sampled:
        raise SystemExit("No sampled tokens to plot.")

    base_rows = np.stack([embedding_row(token, baseline_emb, baseline_w2i) for token in sampled])
    base_norm = np.linalg.norm(base_rows, axis=1)
    shifts_by_label: dict[str, np.ndarray] = {}
    cosine_by_label: dict[str, np.ndarray] = {}
    for label, emb in embeddings_by_label.items():
        w2i = word2idx_by_label[label]
        rows = np.stack([embedding_row(token, emb, w2i) for token in sampled])
        deltas = rows - base_rows
        shifts_by_label[label] = np.linalg.norm(deltas, axis=1)
        denom = np.linalg.norm(rows, axis=1) * base_norm
        cosine = np.sum(rows * base_rows, axis=1) / np.maximum(denom, 1e-12)
        cosine_by_label[label] = 1.0 - np.clip(cosine, -1.0, 1.0)

    title = args.title or f"Embedding shift from baseline ({len(sampled)} sampled tokens)"
    plot_shifts(args.out, sampled, shifts_by_label, title=title, dpi=int(args.dpi))
    csv_path = args.csv if args.csv is not None else args.out.with_suffix(".csv")
    write_csv(csv_path, sampled, shifts_by_label, cosine_by_label)

    print(f"Wrote {args.out}")
    print(f"Wrote {csv_path}")
    print(f"Baseline: {baseline_desc}")
    print(f"Token source: {source}")
    print(f"Candidates: {len(candidates)}  |  plotted: {len(sampled)}  |  skipped/OOV/filter: {skipped}")
    for label, shifts in shifts_by_label.items():
        print(
            f"{label}: mean L2={float(np.mean(shifts)):.6f}, "
            f"median L2={float(np.median(shifts)):.6f}, max L2={float(np.max(shifts)):.6f}"
        )


if __name__ == "__main__":
    main()
