#!/usr/bin/env python3
"""
Evaluate embeddings on a labeled entity file (node classification).

Checkpoint: PyTorch ``.pt`` from ``train_word2vec.py`` (``embeddings`` + ``word2idx``);
gensim ``KeyedVectors`` ``.kv``; or gensim ``Word2Vec`` ``.model`` (e.g. two-stage
``rdf2vec_pretrained.model``). RDF2Vec stage-2 output is ``rdf2vec_final.pt``.

Each line: <entity_id>\\t<label> (tab-separated). By default trains LogReg on train.txt
embeddings and reports test metrics (optional bootstrap standard errors). Pass
``--all-classifiers`` to also fit NB, SVM, and MLP and surface the accuracy-best model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm.auto import tqdm

CLASSIFIER_ORDER = ("NB", "SVM", "MLP", "LogReg")
DEFAULT_CLASSIFIER = "LogReg"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def classifiers_to_fit(*, all_classifiers: bool) -> tuple[str, ...]:
    """Return classifier names to train for this evaluation run."""
    if all_classifiers:
        return CLASSIFIER_ORDER
    return (DEFAULT_CLASSIFIER,)
COVERAGE_LABELED_SPLITS = ("test", "train")
COVERAGE_POOL_SPLITS = ("positives", "negatives")
COVERAGE_SPLIT_PREFIXES = COVERAGE_LABELED_SPLITS + COVERAGE_POOL_SPLITS


def token_candidates(token: str) -> tuple[str, ...]:
    """Return checkpoint vocab key candidates for a labeled entity token."""
    token = token.strip()
    if not token:
        return ("",)
    if token.startswith("<") and token.endswith(">") and len(token) > 2:
        return (token, token[1:-1])
    return (token, f"<{token}>")


def vocab_key_for_token(token: str, word2idx: dict[str, int]) -> str | None:
    """Resolve a labeled entity token to the matching checkpoint vocab key."""
    for cand in token_candidates(token):
        if cand in word2idx:
            return cand
    return None


CLASS_BREAKDOWN_SIDES = ("positive", "negative")
CLASS_BREAKDOWN_BUCKETS = (
    "total",
    "oov",
    "in_vocab",
    "maschine_initialized",
    "random_initialized",
    "stage1_pretrained",
    "unknown",
)
SPLIT_COVERAGE_BUCKETS = (
    "oov",
    "in_vocab",
    "maschine_initialized",
    "random_initialized",
    "stage1_pretrained",
    "unknown",
)


def _coverage_result_fields(
    split: str,
    counts: dict[str, int | dict[str, dict[str, int]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"n_{split}": counts["total"],
        f"n_{split}_oov": counts["oov"],
        f"n_{split}_in_vocab": counts["in_vocab"],
        f"n_{split}_maschine_initialized": counts["maschine_initialized"],
        f"n_{split}_random_initialized": counts["random_initialized"],
        f"n_{split}_stage1_pretrained": counts["stage1_pretrained"],
        f"n_{split}_unknown": counts["unknown"],
    }
    by_class = counts.get("by_class")
    if isinstance(by_class, dict):
        out[f"{split}_class_breakdown"] = by_class
        for side in CLASS_BREAKDOWN_SIDES:
            side_counts = by_class[side]
            for bucket in CLASS_BREAKDOWN_BUCKETS:
                out[f"n_{split}_{side}_{bucket}"] = side_counts[bucket]
    return out


def default_train_path(test_path: Path) -> Path:
    return test_path.parent / "train.txt"


def default_pool_paths(test_path: Path) -> tuple[Path, Path]:
    """Return ``positives.txt`` and ``negatives.txt`` beside ``.../train_test/``."""
    pool_dir = test_path.parent.parent
    return pool_dir / "positives.txt", pool_dir / "negatives.txt"


def load_entity_list_txt(path: Path) -> list[str]:
    """Load one entity id per line (no label column)."""
    tokens: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    if not tokens:
        raise ValueError(f"No entity rows in {path}")
    return tokens


def load_labeled_txt(path: Path) -> tuple[list[str], np.ndarray]:
    tokens: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            tokens.append(parts[0].strip())
            labels.append(int(parts[1].strip()))
    if not tokens:
        raise ValueError(f"No labeled rows in {path}")
    return tokens, np.asarray(labels, dtype=np.int64)


def tokens_to_embeddings(
    tokens: list[str],
    emb: np.ndarray,
    word2idx: dict[str, int],
    desc: str,
    chunk_size: int,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, int]:
    """Map tokens to embedding rows; OOV tokens get a zero vector. Returns (matrix, n_oov)."""
    n = len(tokens)
    d = emb.shape[1]
    out = np.zeros((n, d), dtype=np.float32)
    oov = 0
    starts = range(0, n, chunk_size)
    for start in tqdm(
        starts,
        desc=desc,
        dynamic_ncols=True,
        colour="green",
        leave=False,
        unit="chunk",
        disable=not progress,
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        end = min(start + chunk_size, n)
        for i, t in enumerate(tokens[start:end]):
            j = None
            for cand in token_candidates(t):
                j = word2idx.get(cand)
                if j is not None:
                    break
            if j is None:
                oov += 1
                continue
            out[start + i] = emb[j]
    return out, oov


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    seed: int,
    *,
    progress: bool = True,
) -> dict[str, tuple[float, float]]:
    """Return metric -> (point_estimate, bootstrap_std). Point estimate on full test set."""
    rng = np.random.default_rng(seed)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    point = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    if n_boot <= 0:
        return {k: (v, 0.0) for k, v in point.items()}

    n = len(y_true)
    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []

    for _ in tqdm(
        range(n_boot),
        desc="Bootstrap (test resamples)",
        dynamic_ncols=True,
        colour="cyan",
        unit="draw",
        disable=not progress,
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs.append(float(accuracy_score(yt, yp)))
        precs.append(float(precision_score(yt, yp, average="binary", zero_division=0)))
        recs.append(float(recall_score(yt, yp, average="binary", zero_division=0)))
        f1s.append(float(f1_score(yt, yp, average="binary", zero_division=0)))

    def _std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return float(np.std(xs, ddof=1))

    return {
        "accuracy": (acc, _std(accs)),
        "precision": (prec, _std(precs)),
        "recall": (rec, _std(recs)),
        "f1": (f1, _std(f1s)),
    }


def load_embeddings_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Return (embedding matrix [vocab, dim], token -> row index)."""
    suffix = path.suffix.lower()
    if suffix == ".kv":
        wv = KeyedVectors.load(str(path), mmap="r")
        emb = np.asarray(wv.vectors, dtype=np.float32)
        word2idx: dict[str, int] = dict(wv.key_to_index)
        return emb, word2idx

    if suffix == ".model":
        import sys

        from _word2vec import Word2VecWithStepLoss

        # Checkpoints from `train_word2vec.py` pickle the class as __main__.Word2VecWithStepLoss.
        sys.modules["__main__"].Word2VecWithStepLoss = Word2VecWithStepLoss  # type: ignore[attr-defined]

        model = Word2Vec.load(str(path))
        wv = model.wv
        emb = np.asarray(wv.vectors, dtype=np.float32)
        word2idx = dict(wv.key_to_index)
        return emb, word2idx

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "embeddings" not in ckpt or "word2idx" not in ckpt:
        raise ValueError(
            "Checkpoint must contain 'embeddings' and 'word2idx', or use a .kv / .model file."
        )
    emb_t = ckpt["embeddings"]
    if emb_t.dim() != 2:
        raise ValueError("'embeddings' must be 2D [vocab, dim].")
    emb = emb_t.detach().cpu().numpy().astype(np.float32, copy=False)
    return emb, ckpt["word2idx"]


def make_classifiers(*, max_iter: int, seed: int) -> dict[str, Any]:
    """Return sklearn classifiers keyed by stable display name."""
    return {
        "NB": GaussianNB(),
        "SVM": SVC(random_state=seed),
        "MLP": MLPClassifier(max_iter=max_iter, random_state=seed),
        "LogReg": LogisticRegression(max_iter=max_iter, random_state=seed),
    }


def _metrics_from_bootstrap(
    raw: dict[str, tuple[float, float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        val, std = raw[name]
        out[name] = val
        out[f"{name}_std"] = std
    return out


def select_best_model(models: dict[str, dict[str, float]]) -> str:
    """Pick the model with highest accuracy; tie-break alphabetically by name."""
    return min(models, key=lambda name: (-models[name]["accuracy"], name))


def format_eval_metrics_lines(res: dict[str, Any]) -> list[str]:
    """Format per-model metrics (and best-model summary when multiple classifiers ran)."""
    lines = [
        "Test metrics (binary classification, positive class = 1)",
        "─" * 52,
        "",
    ]
    bootstrap = int(res.get("bootstrap", 0))
    models: dict[str, dict[str, float]] = res["models"]
    model_names = [name for name in CLASSIFIER_ORDER if name in models]

    for model_name in model_names:
        model_metrics = models[model_name]
        lines.append(model_name)
        for name in METRIC_NAMES:
            val = model_metrics[name]
            if bootstrap > 0:
                std = model_metrics.get(f"{name}_std", 0.0)
                lines.append(f"  {name:12s}  {val:.4f}  ±  {std:.4f}  (bootstrap std)")
            else:
                lines.append(f"  {name:12s}  {val:.4f}")
        lines.append("")

    if len(model_names) > 1:
        best_model = res["best_model"]
        best_acc = models[best_model]["accuracy"]
        lines.extend(
            [
                "─" * 52,
                f"Best model: {best_model}  (selected by accuracy = {best_acc:.4f})",
                "─" * 52,
            ]
        )
        for name in METRIC_NAMES:
            val = res[name]
            if bootstrap > 0:
                std = res.get(f"{name}_std", 0.0)
                lines.append(f"  {name:12s}  {val:.4f}  ±  {std:.4f}  (bootstrap std)")
            else:
                lines.append(f"  {name:12s}  {val:.4f}")
    lines.append("─" * 52)

    if res.get("init_labels_available"):
        _append_coverage_section(lines, res, "test", "Test entity coverage")
        breakdown = res.get("test_class_breakdown")
        if isinstance(breakdown, dict):
            lines.extend(
                [
                    "",
                    "Test entity coverage by class (positive=1, negative=0)",
                    "─" * 52,
                ]
            )
            _append_class_breakdown_lines(lines, breakdown)
            lines.append("─" * 52)
        for split, title in (
            ("positives", "Positives pool coverage (positives.txt)"),
            ("negatives", "Negatives pool coverage (negatives.txt)"),
        ):
            if f"n_{split}" in res:
                _append_coverage_section(lines, res, split, title)

    return lines


def _append_coverage_section(
    lines: list[str],
    res: dict[str, Any],
    split: str,
    title: str,
) -> None:
    lines.extend(["", title, "─" * 52])
    coverage_rows = [
        ("total", res[f"n_{split}"]),
        ("in_vocab", res[f"n_{split}_in_vocab"]),
        ("maschine_initialized", res[f"n_{split}_maschine_initialized"]),
        ("random_initialized", res[f"n_{split}_random_initialized"]),
        ("stage1_pretrained", res[f"n_{split}_stage1_pretrained"]),
        ("oov", res[f"n_{split}_oov"]),
    ]
    for label, value in coverage_rows:
        lines.append(f"  {label:22s}  {value}")
    lines.append("─" * 52)


def _append_class_breakdown_lines(
    lines: list[str],
    breakdown: dict[str, dict[str, int]],
) -> None:
    for side in CLASS_BREAKDOWN_SIDES:
        side_counts = breakdown[side]
        lines.append(f"  {side}:")
        lines.append(f"    {'total':20s}  {side_counts['total']}")
        lines.append(
            f"    {'protograph_init':20s}  {side_counts['maschine_initialized']}"
        )
        lines.append(f"    {'random_init':20s}  {side_counts['random_initialized']}")
        lines.append(f"    {'oov':20s}  {side_counts['oov']}")


def _split_coverage_from_res(res: dict[str, Any], split: str) -> dict[str, Any] | None:
    """Build nested entity-coverage section for *split*."""
    total_key = f"n_{split}"
    if total_key not in res:
        return None
    section: dict[str, Any] = {"total": res[total_key]}
    for bucket in SPLIT_COVERAGE_BUCKETS:
        key = f"n_{split}_{bucket}"
        if key in res:
            section[bucket] = res[key]
    breakdown_key = f"{split}_class_breakdown"
    breakdown = res.get(breakdown_key)
    if isinstance(breakdown, dict):
        section["by_class"] = breakdown
    return section


def _vocabulary_coverage_from_res(res: dict[str, Any]) -> dict[str, Any] | None:
    """Build checkpoint vocabulary init summary from a run_evaluation result."""
    vocab = res.get("vocab_coverage")
    if not isinstance(vocab, dict):
        return None
    return {
        "vocab_size": vocab["total"],
        "vocab_initialized": vocab["maschine_initialized"] + vocab["stage1_pretrained"],
        "vocab_initialized_entities": vocab["initialized_instances"],
        "vocab_initialized_relations": vocab["initialized_relations"],
        "random_init": vocab["random_initialized"],
        "not_in_vocab": vocab["unknown"],
    }


def build_eval_coverage_payload(res: dict[str, Any]) -> dict[str, Any]:
    """Build JSON-serializable eval coverage record from a run_evaluation result."""
    init_labels_available = bool(res.get("init_labels_available", False))
    payload: dict[str, Any] = {
        "test_path": res["test_path"],
        "train_path": res["train_path"],
        "checkpoint_path": res["checkpoint_path"],
        "init_labels_available": init_labels_available,
        "init_labels_path": res.get("init_labels_path"),
        "positives_path": res.get("positives_path"),
        "negatives_path": res.get("negatives_path"),
        "accuracy": res.get("accuracy"),
        "f1": res.get("f1"),
        "best_model": res.get("best_model"),
        "oov": {
            "train": res.get("oov_train"),
            "test": res.get("oov_test"),
        },
    }
    if init_labels_available:
        payload["vocabulary"] = _vocabulary_coverage_from_res(res)
        for split in COVERAGE_SPLIT_PREFIXES:
            section = _split_coverage_from_res(res, split)
            if section is not None:
                payload[split] = section
    else:
        payload["vocabulary"] = None
        for split in COVERAGE_SPLIT_PREFIXES:
            payload[split] = None
    init_strategy = res.get("init_strategy")
    if init_strategy is not None:
        payload["init_strategy"] = init_strategy
    all_init_fallback = res.get("all_init_fallback")
    if all_init_fallback is not None:
        payload["all_init_fallback"] = all_init_fallback
    return payload


def write_eval_coverage_json(path: Path, res: dict[str, Any]) -> None:
    """Write eval coverage statistics to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_eval_coverage_payload(res), indent=2) + "\n",
        encoding="utf-8",
    )


def run_evaluation(
    test_path: Path,
    checkpoint_path: Path,
    *,
    train_path: Path | None = None,
    bootstrap: int = 0,
    seed: int = 42,
    max_iter: int = 1000,
    chunk_size: int = 2048,
    progress: bool = True,
    verbose: bool = False,
    run_dir: Path | None = None,
    init_labels_path: Path | None = None,
    all_classifiers: bool = False,
) -> dict[str, Any]:
    """
    Train classifiers on train embeddings; return test metrics (+ bootstrap std).

    By default only LogReg is fit. With ``all_classifiers=True``, also fit NB, SVM, and MLP;
    top-level accuracy/precision/recall/f1 then reflect the accuracy-best model.

    Raises FileNotFoundError if paths are missing, ValueError for invalid labeled data or
    checkpoint format.
    """
    test_path = test_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    tr_path = (
        train_path.resolve()
        if train_path is not None
        else default_train_path(test_path).resolve()
    )
    if not tr_path.is_file():
        raise FileNotFoundError(f"Training file not found: {tr_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    emb, word2idx = load_embeddings_checkpoint(checkpoint_path)
    train_tokens, y_train = load_labeled_txt(tr_path)
    test_tokens, y_test = load_labeled_txt(test_path)

    if verbose:
        tqdm.write(
            f"Train rows: {len(train_tokens)}  |  Test rows: {len(test_tokens)}"
        )
        tqdm.write(f"Embedding matrix: {emb.shape[0]} x {emb.shape[1]}")

    X_train, oov_tr = tokens_to_embeddings(
        train_tokens,
        emb,
        word2idx,
        "Embed train entities",
        chunk_size,
        progress=progress,
    )
    X_test, oov_te = tokens_to_embeddings(
        test_tokens,
        emb,
        word2idx,
        "Embed test entities",
        chunk_size,
        progress=progress,
    )
    if verbose and (oov_tr or oov_te):
        tqdm.write(
            f"Note: OOV tokens (zero vector): train={oov_tr}, test={oov_te}"
        )

    from _maschine_init import (  # noqa: PLC0415
        count_entity_init_coverage,
        count_vocab_init_coverage,
        load_finetune_init_stats,
        load_finetune_token_init,
        resolve_finetune_init_stats_path,
        resolve_finetune_token_init_path,
    )

    resolved_init_path = resolve_finetune_token_init_path(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        init_labels_path=init_labels_path,
    )
    init_labels_available = resolved_init_path is not None
    resolved_init_stats_path = resolve_finetune_init_stats_path(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
    )
    init_stats_payload: dict[str, object] | None = None
    if resolved_init_stats_path is not None:
        init_stats_payload = load_finetune_init_stats(resolved_init_stats_path)
    pos_path, neg_path = default_pool_paths(test_path)
    coverage_fields: dict[str, Any] = {
        "init_labels_available": init_labels_available,
        "init_labels_path": str(resolved_init_path.resolve())
        if resolved_init_path is not None
        else None,
        "positives_path": str(pos_path.resolve()) if pos_path.is_file() else None,
        "negatives_path": str(neg_path.resolve()) if neg_path.is_file() else None,
    }
    if init_stats_payload is not None:
        strategy = init_stats_payload.get("init_strategy")
        if isinstance(strategy, str):
            coverage_fields["init_strategy"] = strategy
        fallback = init_stats_payload.get("all_init_fallback")
        if isinstance(fallback, dict):
            coverage_fields["all_init_fallback"] = fallback
    if init_labels_available:
        token_init_labels = load_finetune_token_init(resolved_init_path)
        coverage_fields["vocab_coverage"] = count_vocab_init_coverage(
            word2idx,
            token_init_labels,
        )
        for split, tokens, labels in (
            ("train", train_tokens, y_train),
            ("test", test_tokens, y_test),
        ):
            counts = count_entity_init_coverage(
                tokens,
                word2idx,
                token_init_labels,
                class_labels=labels,
                vocab_key_resolver=vocab_key_for_token,
            )
            coverage_fields.update(_coverage_result_fields(split, counts))
        for split, pool_path in (
            ("positives", pos_path),
            ("negatives", neg_path),
        ):
            if not pool_path.is_file():
                continue
            pool_tokens = load_entity_list_txt(pool_path)
            counts = count_entity_init_coverage(
                pool_tokens,
                word2idx,
                token_init_labels,
                vocab_key_resolver=vocab_key_for_token,
            )
            coverage_fields.update(_coverage_result_fields(split, counts))
        if verbose:
            tqdm.write(
                "Test MASCHInE init coverage: "
                f"{coverage_fields['n_test_maschine_initialized']}/"
                f"{coverage_fields['n_test']} "
                f"(in_vocab={coverage_fields['n_test_in_vocab']}, "
                f"oov={coverage_fields['n_test_oov']})"
            )
    else:
        for split in COVERAGE_SPLIT_PREFIXES:
            coverage_fields[f"n_{split}_maschine_initialized"] = None

    classifier_names = classifiers_to_fit(all_classifiers=all_classifiers)
    if verbose:
        if all_classifiers:
            tqdm.write("Fitting classifiers on train embeddings…")
        else:
            tqdm.write(f"Fitting {DEFAULT_CLASSIFIER} on train embeddings…")

    models: dict[str, dict[str, float]] = {}
    all_clfs = make_classifiers(max_iter=max_iter, seed=seed)
    for model_name in classifier_names:
        clf = all_clfs[model_name]
        if verbose:
            tqdm.write(f"  {model_name}…")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        raw = bootstrap_metrics(y_test, y_pred, bootstrap, seed, progress=progress)
        models[model_name] = _metrics_from_bootstrap(raw)

    best_model = select_best_model(models)
    best = models[best_model]

    out: dict[str, Any] = {
        "test_path": str(test_path),
        "train_path": str(tr_path),
        "checkpoint_path": str(checkpoint_path),
        "n_train": len(train_tokens),
        "n_test": len(test_tokens),
        "emb_vocab": int(emb.shape[0]),
        "emb_dim": int(emb.shape[1]),
        "oov_train": oov_tr,
        "oov_test": oov_te,
        "bootstrap": bootstrap,
        "seed": seed,
        "best_model": best_model,
        "models": models,
        **coverage_fields,
    }
    for name in METRIC_NAMES:
        out[name] = best[name]
        out[f"{name}_std"] = best[f"{name}_std"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate embedding checkpoint on labeled test.txt (node classification)."
    )
    p.add_argument(
        "test",
        type=Path,
        help="Labeled test file: <entity>\\t<label> per line",
    )
    p.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help="Embeddings: .pt (train_word2vec / rdf2vec_final.pt), gensim .kv, or Word2Vec .model",
    )
    p.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Labeled train file (default: train.txt next to test file)",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for metric std (0 = skip, point estimates only)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")
    p.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="LogReg / MLP max_iter",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Rows per progress chunk when building embedding matrices",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Printed before the metrics table (e.g. TC + strategy) so batch logs stay readable.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory (used to locate finetune_token_init.json).",
    )
    p.add_argument(
        "--init-labels",
        type=Path,
        default=None,
        help="Explicit path to finetune_token_init.json for MASCHInE init coverage.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write eval_coverage.json under this directory.",
    )
    p.add_argument(
        "--all-classifiers",
        action="store_true",
        help="Also fit NB, SVM, and MLP and report the accuracy-best model "
        f"(default: {DEFAULT_CLASSIFIER} only)",
    )
    args = p.parse_args()

    train_path = args.train if args.train is not None else default_train_path(args.test)
    if not train_path.is_file():
        raise SystemExit(
            f"Training file not found: {train_path}\n"
            "Pass --train /path/to/train.txt or place train.txt beside test.txt."
        )
    if not args.test.is_file():
        raise SystemExit(f"Test file not found: {args.test}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    try:
        res = run_evaluation(
            args.test,
            args.checkpoint,
            train_path=train_path,
            bootstrap=args.bootstrap,
            seed=args.seed,
            max_iter=args.max_iter,
            chunk_size=args.chunk_size,
            progress=True,
            run_dir=args.run_dir,
            init_labels_path=args.init_labels,
            all_classifiers=args.all_classifiers,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    tqdm.write(
        f"Train rows: {res['n_train']}  |  Test rows: {res['n_test']}"
    )
    tqdm.write(f"Embedding matrix: {res['emb_vocab']} x {res['emb_dim']}")
    if args.all_classifiers:
        tqdm.write("Fitted classifiers on train embeddings.")
    else:
        tqdm.write(f"Fitted {DEFAULT_CLASSIFIER} on train embeddings.")
    if res["oov_train"] or res["oov_test"]:
        tqdm.write(
            f"Note: OOV tokens (zero vector): train={res['oov_train']}, test={res['oov_test']}"
        )
    if res.get("init_labels_available"):
        tqdm.write(
            "Test MASCHInE init coverage: "
            f"{res['n_test_maschine_initialized']}/{res['n_test']} "
            f"(in_vocab={res['n_test_in_vocab']}, oov={res['n_test_oov']})"
        )
    elif args.run_dir is not None or args.init_labels is not None:
        tqdm.write("Note: finetune_token_init.json not found; MASCHInE init coverage skipped.")

    if args.out_dir is not None:
        out_path = args.out_dir / "eval_coverage.json"
        write_eval_coverage_json(out_path, res)
        tqdm.write(f"Wrote {out_path}")

    print()
    if args.label:
        print("─" * 52)
        print(args.label)
        print("─" * 52)
    for line in format_eval_metrics_lines(res):
        print(line)
    if args.bootstrap > 0:
        print(f"Bootstrap draws: {args.bootstrap}  |  seed: {args.seed}")


if __name__ == "__main__":
    main()
