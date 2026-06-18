"""Shared helpers for scripts/plot/*.py (not runnable)."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _dbpedia.eval import Split, discover_splits  # noqa: E402
from _evaluate import load_embeddings_checkpoint  # noqa: E402

MODE_PLOT_STYLE: dict[str, tuple[str, str]] = {
    "p1": ("P1", "#F97316"),
    "p2": ("P2", "#16A34A"),
    "vanilla": ("Vanilla", "#3B8BBD"),
}
ACCURACY_MODE_PLOT_STYLE: dict[str, tuple[str, str]] = {
    "p1": ("resume_graph-p1", "#F97316"),
    "p2": ("resume_graph-p2", "#16A34A"),
    "vanilla": ("vanilla_epochwise", "#3B8BBD"),
}
ACCURACY_PLOT_TITLE = "Accuracy per epoch — {tc}"
ACCURACY_PLOT_YLABEL = "Test accuracy"
DRIFT_PLOT_TITLE = "Embedding drift per epoch — {tc}"
DRIFT_PLOT_YLABEL = "1 − |cosine similarity|"
MODE_COLORS: dict[str, str] = {
    mode: color for mode, (_label, color) in MODE_PLOT_STYLE.items()
}
MODE_COLORS["no_pretrain"] = MODE_PLOT_STYLE["vanilla"][1]
MODE_ORDER = ("p1", "p2", "vanilla")
VANILLA_DEFAULT_SLUG = "vanilla_default"
LOSS_CSV_NAMES = {
    "p1": "finetune_loss.csv",
    "p2": "finetune_loss.csv",
    "vanilla": "rdf2vec_word2vec_loss.csv",
}
ACCURACY_CSV_NAME = "epoch_eval_accuracy.csv"

HIGH_CONTRAST_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
    "#CF1C90",
    "#F97B72",
    "#4B4B8F",
    "#A5AA99",
)


@dataclass(frozen=True)
class DomainSplit:
    domain: str
    split_dir: str
    test_path: Path


@dataclass(frozen=True)
class PlotPoint:
    token: str
    domain: str
    label: int
    row: int


def load_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load embedding matrix and token->row index (.pt, .kv, or .model)."""
    return load_embeddings_checkpoint(path)


def latest_timestamped_run(
    run_root: Path,
    *,
    config_slug: str | None = None,
    mode: str | None = None,
    tc: str | None = None,
    dataset: str = "synthetic",
    output_root: Path | None = None,
) -> Path | None:
    """Return the newest timestamped run directory under ``run_root`` or derived paths.

    When ``config_slug``, ``mode``, and ``tc`` are all set, also probes new vs legacy
    synthetic layouts before falling back to ``run_root``.
    """
    if config_slug and mode and tc and output_root is not None:
        resolved = resolve_run_dir(
            output_root=output_root,
            dataset=dataset,
            slug=config_slug,
            tc=tc,
            mode=mode,
        )
        if resolved is not None:
            return resolved

    if not run_root.is_dir():
        return None
    subs = [p for p in run_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


def _timestamp_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(
        (p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
    )


def resolve_run_dir(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    tc: str | None = None,
    mode: str | None = None,
    timestamp: str | None = None,
    pick_latest: bool = True,
) -> Path | None:
    """Resolve one experiment run directory (new layout first, then legacy)."""
    output_root = output_root.resolve()
    mode_norm = None if mode in (None, "vanilla", "no_pretrain") else mode

    def _exists(path: Path) -> Path | None:
        return path.resolve() if path.is_dir() else None

    if timestamp:
        candidates: list[Path] = []
        if tc:
            if mode_norm:
                candidates.append(
                    output_root / dataset / "protograph" / slug / tc / timestamp / mode_norm
                )
                candidates.append(output_root / dataset / slug / tc / timestamp / mode_norm)
                candidates.append(output_root / dataset / mode_norm / slug / tc / timestamp)
            else:
                candidates.append(output_root / dataset / "vanilla" / slug / tc / timestamp)
                candidates.append(output_root / dataset / slug / tc / timestamp)
                if mode in ("vanilla", "no_pretrain"):
                    candidates.append(output_root / dataset / mode / slug / tc / timestamp)
        else:
            if mode_norm:
                candidates.append(
                    output_root / dataset / "protograph" / slug / timestamp / mode_norm
                )
                candidates.append(output_root / dataset / slug / timestamp / mode_norm)
                candidates.append(output_root / dataset / mode_norm / slug / timestamp)
            else:
                candidates.append(output_root / dataset / "vanilla" / slug / timestamp)
                candidates.append(output_root / dataset / slug / timestamp)
        for candidate in candidates:
            found = _exists(candidate)
            if found is not None:
                return found
        return None

    search_roots: list[Path] = []
    if tc:
        if mode_norm:
            search_roots.extend(
                [
                    output_root / dataset / "protograph" / slug / tc,
                    output_root / dataset / slug / tc,
                    output_root / dataset / mode_norm / slug / tc,
                ]
            )
        else:
            search_roots.extend(
                [
                    output_root / dataset / "vanilla" / slug / tc,
                    output_root / dataset / slug / tc,
                ]
            )
            if mode in ("vanilla", "no_pretrain"):
                search_roots.append(output_root / dataset / mode / slug / tc)
    elif mode_norm:
        search_roots.extend(
            [
                output_root / dataset / "protograph" / slug,
                output_root / dataset / slug,
                output_root / dataset / mode_norm / slug,
            ]
        )
    else:
        search_roots.extend(
            [
                output_root / dataset / "vanilla" / slug,
                output_root / dataset / slug,
            ]
        )

    latest: Path | None = None
    latest_mtime = -1.0
    for root in search_roots:
        for ts_dir in _timestamp_dirs(root):
            candidate = ts_dir / mode_norm if (mode_norm and tc is None) else ts_dir
            if mode_norm and tc:
                candidate = ts_dir / mode_norm
                if not candidate.is_dir():
                    candidate = ts_dir
            elif mode_norm and not tc:
                candidate = ts_dir / mode_norm if (ts_dir / mode_norm).is_dir() else ts_dir
            if not candidate.is_dir():
                continue
            mtime = candidate.stat().st_mtime
            if pick_latest and mtime > latest_mtime:
                latest_mtime = mtime
                latest = candidate.resolve()
    return latest


_SLUG_PARAM_KEYS: tuple[str, ...] = (
    "anchor_regularization",
    "epochs",
    "finetune_depth",
    "finetune_dim",
    "finetune_epochs",
    "finetune_lr",
    "finetune_min_count",
    "finetune_negative",
    "finetune_walks_per_entity",
    "finetune_window",
    "init_relations",
    "initialization_noise",
    "init_strategy",
    "pretrain_depth",
    "pretrain_epochs",
    "pretrain_lr",
    "pretrain_walks_per_entity",
)
_PRETRAIN_SLUG_KEYS: frozenset[str] = frozenset(
    {"pretrain_depth", "pretrain_epochs", "pretrain_lr", "pretrain_walks_per_entity"}
)


def _parse_slug_segments(slug: str) -> list[tuple[str, str]]:
    """Parse a config slug into ``(param_key, raw_value)`` pairs (sorted key order)."""
    if slug == "default":
        return []
    segments: list[tuple[str, str]] = []
    pos = 0
    for key in _SLUG_PARAM_KEYS:
        if pos >= len(slug):
            break
        prefix = key + "_"
        if not slug.startswith(prefix, pos):
            continue
        value_start = pos + len(prefix)
        value_end = len(slug)
        for other_key in _SLUG_PARAM_KEYS:
            if other_key == key:
                continue
            marker = "_" + other_key + "_"
            idx = slug.find(marker, value_start)
            if idx != -1:
                value_end = min(value_end, idx)
        segments.append((key, slug[value_start:value_end]))
        pos = value_end + 1 if value_end < len(slug) else value_end
    if not segments:
        raise ValueError(f"Unrecognized experiment slug: {slug!r}")
    return segments


def _build_slug_segments(segments: list[tuple[str, str]]) -> str:
    if not segments:
        return "default"
    return "_".join(f"{key}_{value}" for key, value in segments)


def vanilla_slug_for_experiment(slug: str) -> str:
    """Map a P1/P2 experiment slug to the corresponding vanilla run slug."""
    if slug == "default":
        return VANILLA_DEFAULT_SLUG
    segments = _parse_slug_segments(slug)
    vanilla_segments = [(key, value) for key, value in segments if key not in _PRETRAIN_SLUG_KEYS]
    mapped = _build_slug_segments(vanilla_segments)
    return VANILLA_DEFAULT_SLUG if mapped == "default" else mapped


def _resolve_vanilla_run_dir(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    tc: str | None,
    timestamp: str | None,
) -> Path | None:
    """Resolve vanilla/no_pretrain run dir; fall back across slug and timestamp."""
    candidate_slugs: list[str] = []
    for candidate in (vanilla_slug_for_experiment(slug), slug):
        if candidate not in candidate_slugs:
            candidate_slugs.append(candidate)

    for mode_slug in candidate_slugs:
        for ts in (timestamp, None) if timestamp else (None,):
            run_dir = resolve_run_dir(
                output_root=output_root,
                dataset=dataset,
                slug=mode_slug,
                tc=tc,
                mode="vanilla",
                timestamp=ts,
            )
            if run_dir is not None:
                return run_dir
            if mode_slug == VANILLA_DEFAULT_SLUG:
                run_dir = resolve_run_dir(
                    output_root=output_root,
                    dataset=dataset,
                    slug="default",
                    tc=tc,
                    mode="vanilla",
                    timestamp=ts,
                )
                if run_dir is not None:
                    return run_dir
    return None


def resolve_run_dirs(
    *,
    output_root: Path,
    dataset: str = "synthetic",
    slug: str = "default",
    tc: str | None = None,
    modes: Iterable[str] = ("p1", "p2", "vanilla"),
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Map training mode -> resolved run directory."""
    resolved: dict[str, Path] = {}
    for mode in modes:
        if mode in ("vanilla", "no_pretrain"):
            run_dir = _resolve_vanilla_run_dir(
                output_root=output_root,
                dataset=dataset,
                slug=slug,
                tc=tc,
                timestamp=timestamp,
            )
        else:
            run_dir = resolve_run_dir(
                output_root=output_root,
                dataset=dataset,
                slug=slug,
                tc=tc,
                mode=mode,
                timestamp=timestamp,
            )
        if run_dir is not None:
            resolved[mode] = run_dir
    return resolved


def protograph_slug_root(*, output_root: Path, dataset: str, slug: str) -> Path:
    """Root for P1/P2 runs of one experiment slug under the protograph family."""
    return output_root / dataset / "protograph" / slug


def experiment_slug_root(*, output_root: Path, dataset: str, slug: str) -> Path:
    """Directory listing TC subdirs for a P1/P2 experiment slug (new layout, then legacy)."""
    new_root = protograph_slug_root(output_root=output_root, dataset=dataset, slug=slug)
    if new_root.is_dir():
        return new_root
    return output_root / dataset / slug


def experiment_tc_dir(*, output_root: Path, dataset: str, slug: str, tc: str) -> Path:
    """Timestamp parent directory for one TC in a P1/P2 experiment slug."""
    return experiment_slug_root(output_root=output_root, dataset=dataset, slug=slug) / tc


def run_has_artifacts(run_dirs: dict[str, Path]) -> bool:
    """Return True when P1, P2, and vanilla runs have loss and accuracy CSVs."""
    for mode in MODE_ORDER:
        run_dir = run_dirs.get(mode)
        if run_dir is None or not run_dir.is_dir():
            return False
        if not (run_dir / LOSS_CSV_NAMES[mode]).is_file():
            return False
        if not (run_dir / ACCURACY_CSV_NAME).is_file():
            return False
    return True


def final_checkpoint_path(run_dir: Path, mode: str) -> Path:
    """Return the final finetuned embedding checkpoint for one training mode."""
    if mode in ("vanilla", "no_pretrain"):
        return run_dir / "rdf2vec_word2vec.pt"
    return run_dir / "rdf2vec_final.pt"


def run_has_final_checkpoints(run_dirs: dict[str, Path]) -> bool:
    """Return True when P1, P2, and vanilla runs have final embedding checkpoints."""
    for mode in MODE_ORDER:
        run_dir = run_dirs.get(mode)
        if run_dir is None or not run_dir.is_dir():
            return False
        if not final_checkpoint_path(run_dir, mode).is_file():
            return False
    return True


def list_finetune_epoch_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (epoch, path) pairs from ``run_dir/ckpt/finetune_epoch_*.pt``."""
    ckpt_dir = run_dir / "ckpt"
    if not ckpt_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("finetune_epoch_*.pt"):
        stem = path.stem.removeprefix("finetune_epoch_")
        if not stem.isdigit():
            continue
        out.append((int(stem), path.resolve()))
    return sorted(out, key=lambda item: item[0])


def finetune_epoch_checkpoint_path(run_dir: Path, epoch: int) -> Path:
    """Return ``run_dir/ckpt/finetune_epoch_{epoch:02d}.pt`` (epoch 0 = post-pretrain init)."""
    return run_dir / "ckpt" / f"finetune_epoch_{epoch:02d}.pt"


def run_has_epoch_checkpoints(run_dir: Path, *, min_epochs: int = 2) -> bool:
    """True when ``ckpt/`` contains at least ``min_epochs`` checkpoints including epoch 0."""
    checkpoints = list_finetune_epoch_checkpoints(run_dir)
    if len(checkpoints) < min_epochs:
        return False
    return checkpoints[0][0] == 0


def avg_embedding_cosine_drift_from_baseline(baseline: np.ndarray, emb: np.ndarray) -> float:
    """Return ``1 - |cos_sim|`` between the mean embedding at epoch 0 and the mean at ``emb``."""
    if baseline.shape != emb.shape:
        raise ValueError(
            f"Embedding shape mismatch: baseline {baseline.shape} vs emb {emb.shape}"
        )
    baseline_mean = baseline.mean(axis=0)
    emb_mean = emb.mean(axis=0)
    base_norm = float(np.linalg.norm(baseline_mean))
    emb_norm = float(np.linalg.norm(emb_mean))
    denom = max(base_norm * emb_norm, 1e-12)
    cosine_sim = float(np.clip(float(np.dot(baseline_mean, emb_mean) / denom), -1.0, 1.0))
    return 1.0 - abs(cosine_sim)


def run_dirs_have_epoch_checkpoints(run_dirs: dict[str, Path], *, min_epochs: int = 2) -> bool:
    """Return True when P1, P2, and vanilla runs have finetune epoch checkpoints."""
    for mode in MODE_ORDER:
        run_dir = run_dirs.get(mode)
        if run_dir is None or not run_has_epoch_checkpoints(run_dir, min_epochs=min_epochs):
            return False
    return True


@dataclass(frozen=True)
class ExperimentRun:
    tc: str
    timestamps: tuple[str, ...]
    run_dirs_by_timestamp: dict[str, dict[str, Path]]

    @property
    def n_repeats(self) -> int:
        return len(self.timestamps)


def _sorted_tc_names(slug_root: Path) -> list[str]:
    if not slug_root.is_dir():
        return []
    return sorted(
        (p.name for p in slug_root.iterdir() if p.is_dir() and p.name.startswith("tc")),
        key=lambda t: int(t[2:]) if t[2:].isdigit() else t,
    )


def _sorted_timestamps(tc_dir: Path, *, reverse: bool = True) -> list[str]:
    if not tc_dir.is_dir():
        return []
    timestamps = [
        p.name for p in tc_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(timestamps, key=lambda ts: (tc_dir / ts).stat().st_mtime, reverse=reverse)


def _aligned_timestamps(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    tc: str,
    timestamp: str | None,
) -> list[str]:
    tc_dir = experiment_tc_dir(output_root=output_root, dataset=dataset, slug=slug, tc=tc)
    candidates = [timestamp] if timestamp else _sorted_timestamps(tc_dir)
    aligned: list[str] = []
    for ts in candidates:
        if ts is None:
            continue
        run_dirs = resolve_run_dirs(
            output_root=output_root,
            dataset=dataset,
            slug=slug,
            tc=tc,
            timestamp=ts,
        )
        if run_has_artifacts(run_dirs):
            aligned.append(ts)
    return aligned


def _aligned_final_checkpoint_timestamps(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    tc: str,
    timestamp: str | None,
) -> list[str]:
    tc_dir = experiment_tc_dir(output_root=output_root, dataset=dataset, slug=slug, tc=tc)
    candidates = [timestamp] if timestamp else _sorted_timestamps(tc_dir)
    aligned: list[str] = []
    for ts in candidates:
        if ts is None:
            continue
        run_dirs = resolve_run_dirs(
            output_root=output_root,
            dataset=dataset,
            slug=slug,
            tc=tc,
            timestamp=ts,
        )
        if run_has_final_checkpoints(run_dirs):
            aligned.append(ts)
    return aligned


def discover_embedding_runs(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    timestamp: str | None = None,
    latest_only: bool = False,
) -> list[ExperimentRun]:
    """Discover TC runs with aligned P1/P2/vanilla timestamps and final checkpoints."""
    output_root = output_root.resolve()
    slug_root = experiment_slug_root(output_root=output_root, dataset=dataset, slug=slug)
    runs: list[ExperimentRun] = []
    for tc in _sorted_tc_names(slug_root):
        timestamps = _aligned_final_checkpoint_timestamps(
            output_root=output_root,
            dataset=dataset,
            slug=slug,
            tc=tc,
            timestamp=timestamp,
        )
        if latest_only and timestamps and timestamp is None:
            timestamps = [timestamps[0]]
        if not timestamps:
            continue
        run_dirs_by_timestamp = {
            ts: resolve_run_dirs(
                output_root=output_root,
                dataset=dataset,
                slug=slug,
                tc=tc,
                timestamp=ts,
            )
            for ts in timestamps
        }
        runs.append(
            ExperimentRun(
                tc=tc,
                timestamps=tuple(timestamps),
                run_dirs_by_timestamp=run_dirs_by_timestamp,
            )
        )
    return runs


def discover_experiment_runs(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    timestamp: str | None = None,
    latest_only: bool = False,
) -> list[ExperimentRun]:
    """Discover TC runs with aligned P1/P2/vanilla timestamps and required CSVs."""
    output_root = output_root.resolve()
    slug_root = experiment_slug_root(output_root=output_root, dataset=dataset, slug=slug)
    runs: list[ExperimentRun] = []
    for tc in _sorted_tc_names(slug_root):
        timestamps = _aligned_timestamps(
            output_root=output_root,
            dataset=dataset,
            slug=slug,
            tc=tc,
            timestamp=timestamp,
        )
        if latest_only and timestamps and timestamp is None:
            timestamps = [timestamps[0]]
        if not timestamps:
            continue
        run_dirs_by_timestamp = {
            ts: resolve_run_dirs(
                output_root=output_root,
                dataset=dataset,
                slug=slug,
                tc=tc,
                timestamp=ts,
            )
            for ts in timestamps
        }
        runs.append(
            ExperimentRun(
                tc=tc,
                timestamps=tuple(timestamps),
                run_dirs_by_timestamp=run_dirs_by_timestamp,
            )
        )
    return runs


def _aligned_checkpoint_timestamps(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    tc: str,
    timestamp: str | None,
) -> list[str]:
    tc_dir = experiment_tc_dir(output_root=output_root, dataset=dataset, slug=slug, tc=tc)
    candidates = [timestamp] if timestamp else _sorted_timestamps(tc_dir)
    aligned: list[str] = []
    for ts in candidates:
        if ts is None:
            continue
        run_dirs = resolve_run_dirs(
            output_root=output_root,
            dataset=dataset,
            slug=slug,
            tc=tc,
            timestamp=ts,
        )
        if run_dirs_have_epoch_checkpoints(run_dirs):
            aligned.append(ts)
    return aligned


def discover_checkpoint_runs(
    *,
    output_root: Path,
    dataset: str,
    slug: str,
    timestamp: str | None = None,
    latest_only: bool = False,
) -> list[ExperimentRun]:
    """Discover TC runs with aligned P1/P2/vanilla timestamps and finetune checkpoints."""
    output_root = output_root.resolve()
    slug_root = experiment_slug_root(output_root=output_root, dataset=dataset, slug=slug)
    runs: list[ExperimentRun] = []
    for tc in _sorted_tc_names(slug_root):
        timestamps = _aligned_checkpoint_timestamps(
            output_root=output_root,
            dataset=dataset,
            slug=slug,
            tc=tc,
            timestamp=timestamp,
        )
        if latest_only and timestamps and timestamp is None:
            timestamps = [timestamps[0]]
        if not timestamps:
            continue
        run_dirs_by_timestamp = {
            ts: resolve_run_dirs(
                output_root=output_root,
                dataset=dataset,
                slug=slug,
                tc=tc,
                timestamp=ts,
            )
            for ts in timestamps
        }
        runs.append(
            ExperimentRun(
                tc=tc,
                timestamps=tuple(timestamps),
                run_dirs_by_timestamp=run_dirs_by_timestamp,
            )
        )
    return runs


def discover_dbpedia_splits(
    dbpedia_root: Path,
    k: int,
    *,
    tc: str | None = None,
    split_dir: str | None = None,
) -> list[Split]:
    """Discover DLCC splits (wraps ``_dbpedia.eval.discover_splits``)."""
    splits = discover_splits(dbpedia_root.resolve(), int(k))
    if tc is not None:
        tc_name = Path(tc).name
        splits = [s for s in splits if s.tc == tc_name]
    if split_dir and split_dir != "both":
        splits = [s for s in splits if s.split_dir == split_dir]
    return splits


def resolve_tc_dir(tc: Path, dbpedia_root: Path) -> Path:
    """Resolve either a TC directory path or a TC name under ``dbpedia_root``."""
    if tc.is_dir():
        return tc.resolve()
    candidate = dbpedia_root / tc
    if candidate.is_dir():
        return candidate.resolve()
    raise SystemExit(f"TC directory not found: {tc} (also tried {candidate})")


def discover_domain_splits(tc_dir: Path, k: int, split_dir: str) -> list[DomainSplit]:
    """Discover domain test files under one DBpedia TC directory."""
    k_name = str(int(k))
    wanted = ("train_test", "train_test_hard") if split_dir == "both" else (split_dir,)
    splits: list[DomainSplit] = []

    for domain_dir in sorted(tc_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        k_dir = domain_dir / k_name
        if not k_dir.is_dir():
            continue
        for name in wanted:
            test_path = k_dir / name / "test.txt"
            if test_path.is_file():
                splits.append(
                    DomainSplit(
                        domain=domain_dir.name,
                        split_dir=name,
                        test_path=test_path,
                    )
                )
    return splits


def vocab_index_for_token(token: str, word2idx: dict[str, int]) -> int | None:
    """Resolve plain or angled RDF entity tokens to a checkpoint vocabulary row."""
    token = token.strip()
    if not token:
        return None
    idx = word2idx.get(token)
    if idx is not None:
        return idx
    if token.startswith("<") and token.endswith(">") and len(token) > 2:
        return word2idx.get(token[1:-1])
    return word2idx.get(f"<{token}>")


def shorten_entity(entity: str, max_len: int = 42) -> str:
    """Compact entity labels for optional annotations."""
    entity = entity.strip()
    if entity.startswith("<") and entity.endswith(">"):
        entity = entity[1:-1]
    entity = entity.rstrip("/").rsplit("/", 1)[-1]
    if len(entity) <= max_len:
        return entity
    return entity[: max_len - 3] + "..."


def keep_label(label: int, label_filter: str) -> bool:
    if label_filter == "all":
        return True
    if label_filter == "positive":
        return label == 1
    if label_filter == "negative":
        return label == 0
    raise ValueError(f"Unknown label filter: {label_filter}")


def collect_points(
    splits: list[DomainSplit],
    word2idx: dict[str, int],
    *,
    label_filter: str,
    max_per_domain: int,
    seed: int,
) -> tuple[list[PlotPoint], Counter[str], Counter[str]]:
    """Load test rows, resolve vocabulary rows, and optionally sample by domain."""
    from _evaluate import load_labeled_txt

    points_by_domain: dict[str, list[PlotPoint]] = defaultdict(list)
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

            idx = vocab_index_for_token(token, word2idx)
            if idx is None:
                oov_by_domain[domain_key] += 1
                continue

            if token in seen_by_domain[domain_key]:
                continue
            seen_by_domain[domain_key].add(token)

            points_by_domain[domain_key].append(
                PlotPoint(token=token, domain=domain_key, label=label, row=idx)
            )

    rng = np.random.default_rng(seed)
    points: list[PlotPoint] = []
    for domain in sorted(points_by_domain):
        domain_points = points_by_domain[domain]
        if max_per_domain > 0 and len(domain_points) > max_per_domain:
            pick = rng.choice(len(domain_points), size=max_per_domain, replace=False)
            domain_points = [domain_points[i] for i in sorted(pick.tolist())]
        points.extend(domain_points)

    return points, oov_by_domain, skipped_by_domain


def split_suffix(split_dir: str) -> str:
    if split_dir == "train_test":
        return ""
    return f"_{split_dir.removeprefix('train_test_')}"


def domain_color_map(domains: list[str]) -> dict[str, object]:
    import matplotlib.pyplot as plt

    fallback_cmap = plt.colormaps["gist_rainbow"].resampled(max(len(domains), 1))
    denom = max(len(domains) - 1, 1)
    return {
        domain: (
            HIGH_CONTRAST_COLORS[i]
            if i < len(HIGH_CONTRAST_COLORS)
            else fallback_cmap(i / denom)
        )
        for i, domain in enumerate(domains)
    }


def jitter_pca_coords(
    coords: np.ndarray,
    *,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add axis-wise Gaussian jitter as a fraction of each PC's value range."""
    if scale <= 0.0 or coords.shape[0] == 0:
        return coords
    out = coords.astype(np.float64, copy=True)
    span = np.ptp(out, axis=0)
    span = np.where(span > 0.0, span, 1.0)
    out += rng.normal(0.0, 1.0, size=out.shape) * span * scale
    return out


def apply_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.9,
        }
    )


def make_random_baseline(
    word2idx: dict[str, int],
    dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Gensim-like uniform random init for shift plots (--baseline random)."""
    rng = np.random.default_rng(seed)
    half_range = 0.5 / max(dim, 1)
    n = max(word2idx.values()) + 1 if word2idx else 0
    emb = rng.uniform(-half_range, half_range, size=(n, dim)).astype(np.float32)
    return emb, word2idx


POSITIVE_COLOR = "#009E73"
NEGATIVE_COLOR = "#D55E00"
LABEL_COLORS = {0: NEGATIVE_COLOR, 1: POSITIVE_COLOR}

EPOCH_LOSS_MARKER_SIZE = 4.5
EPOCH_LOSS_MARKER_EDGE_WIDTH = 1.2
EPOCH_ACCURACY_MARKER_SIZE = 6.0
EPOCH_ACCURACY_MARKER_EDGE_WIDTH = 1.0


def synthetic_test_path(tc: str) -> Path:
    return (
        REPO_ROOT
        / "v1"
        / "synthetic_ontology"
        / tc
        / "synthetic_ontology"
        / "1000"
        / "train_test"
        / "test.txt"
    )


def is_synthetic_test_set_run(checkpoint: Path, test: Path) -> bool:
    for path in (checkpoint, test):
        normalized = str(path.resolve()).replace("\\", "/")
        if "synthetic_ontology" in normalized or "/output/synthetic/" in normalized:
            return True
    return False


def choose_test_points(
    tokens: list[str],
    labels: np.ndarray,
    word2idx: dict[str, int],
    *,
    max_entities: int,
    seed: int,
) -> tuple[list[int], list[str], np.ndarray, int]:
    rows: list[int] = []
    kept_tokens: list[str] = []
    kept_labels: list[int] = []
    oov = 0
    for token, label in zip(tokens, labels, strict=True):
        idx = vocab_index_for_token(token, word2idx)
        if idx is None:
            oov += 1
            continue
        rows.append(idx)
        kept_tokens.append(token)
        kept_labels.append(int(label))
    if max_entities > 0 and len(rows) > max_entities:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(rows), size=max_entities, replace=False)
        rows = [rows[i] for i in pick]
        kept_tokens = [kept_tokens[i] for i in pick]
        kept_labels = [kept_labels[i] for i in pick]
    return rows, kept_tokens, np.asarray(kept_labels, dtype=np.int64), oov


def fit_lda_2d(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Project embeddings to 2D: LD1 (label-aware) + orthogonal PC for layout."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError("LDA requires at least 2 classes")

    lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
    z1 = lda.fit_transform(X, labels).ravel()

    w = lda.scalings_[:, 0]
    w_norm = float(np.linalg.norm(w))
    if w_norm > 0.0:
        w = w / w_norm
    x_res = X - np.outer(X @ w, w)
    pca = PCA(n_components=1, random_state=seed)
    z2 = pca.fit_transform(x_res).ravel()

    z = np.column_stack([z1, z2])
    ld1_ratio = float(lda.explained_variance_ratio_[0])
    orth_pc_ratio = float(pca.explained_variance_ratio_[0])
    return z, ld1_ratio, orth_pc_ratio


def plot_labeled_lda_panel(
    ax,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    seed: int,
    label_colors: dict[int, str] | None = None,
    show_legend: bool = False,
) -> tuple[float, float]:
    import matplotlib.pyplot as plt

    z, ld1_ratio, orth_pc_ratio = fit_lda_2d(X, labels, seed=seed)
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
            z[mask, 0],
            z[mask, 1],
            s=18,
            alpha=0.75,
            linewidths=0,
            edgecolors="none",
            color=color,
            label=f"label {label} (n={int(mask.sum())})" if show_legend else None,
        )
    ax.set_xlabel(f"LD1 ({ld1_ratio * 100:.1f} %)")
    ax.set_ylabel(f"PC\u22a5 ({orth_pc_ratio * 100:.1f} %)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=True)
    return ld1_ratio, orth_pc_ratio
