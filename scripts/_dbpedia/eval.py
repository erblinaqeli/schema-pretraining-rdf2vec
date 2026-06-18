#!/usr/bin/env python3
"""
Evaluate one embedding checkpoint on all DBpedia DLCC splits under ``v1/dbpedia``.

For each ``tc*`` folder, every object domain (books, cities, people, …) is evaluated
at a fixed positive/negative count (default ``5000``). When both ``train_test`` and
``train_test_hard`` exist, both are evaluated (table rows ``people`` and ``people_hard``).

Example:

  uv run scripts/dbpedia/eval_all_dbpedia.py \\
    --checkpoint output/dbpedia_run/rdf2vec_final.pt \\
    --out output/dbpedia_eval_all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

from _dbpedia.aggregate_results import (
    aggregate_tc_domain_means,
    format_averaged_results_table,
    write_averaged_results_table,
)
from _evaluate import run_evaluation


class _TeeIO:
    """Write to multiple streams (terminal + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


@contextmanager
def _tee_stdio(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        out = _TeeIO(sys.__stdout__, log_file)
        err = _TeeIO(sys.__stderr__, log_file)
        prev_out, prev_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout, sys.stderr = prev_out, prev_err


@dataclass(frozen=True)
class Split:
    name: str
    tc: str
    domain: str
    obj: str
    k: int
    split_dir: str
    test_path: Path


@dataclass(frozen=True)
class EvalRow:
    split: str
    tc: str
    domain: str
    obj: str
    k: int
    split_dir: str
    checkpoint_path: str
    test_path: str
    train_path: str
    n_train: int
    n_test: int
    emb_vocab: int
    emb_dim: int
    oov_train: int
    oov_test: int
    best_model: str
    accuracy: float
    accuracy_std: float
    precision: float
    precision_std: float
    recall: float
    recall_std: float
    f1: float
    f1_std: float


def _obj_label(domain: str, split_dir: str) -> str:
    if split_dir == "train_test_hard":
        return f"{domain}_hard"
    return domain


def discover_splits(dbpedia_root: Path, k: int) -> list[Split]:
    """Return all evaluable splits under ``v1/dbpedia/<tc>/<domain>/<k>/``."""
    k_name = str(int(k))
    splits: list[Split] = []

    for tc_dir in sorted(dbpedia_root.glob("tc*")):
        if not tc_dir.is_dir():
            continue
        tc = tc_dir.name
        for domain_dir in sorted(tc_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            k_dir = domain_dir / k_name
            if not k_dir.is_dir():
                continue

            domain = domain_dir.name
            candidates: list[tuple[str, Path]] = []
            normal_test = k_dir / "train_test" / "test.txt"
            hard_test = k_dir / "train_test_hard" / "test.txt"
            if normal_test.is_file():
                candidates.append(("train_test", normal_test))
            if hard_test.is_file():
                candidates.append(("train_test_hard", hard_test))
            if not candidates:
                continue

            for split_dir, test_path in candidates:
                obj = _obj_label(domain, split_dir)
                name = f"{tc}/{domain}/{k_name}/{split_dir}"
                splits.append(
                    Split(
                        name=name,
                        tc=tc,
                        domain=domain,
                        obj=obj,
                        k=int(k),
                        split_dir=split_dir,
                        test_path=test_path,
                    )
                )

    return splits


def write_results_table(rows: list[EvalRow], path: Path) -> None:
    """Write a compact TC × Obj accuracy table (CSV)."""
    ordered = sorted(rows, key=lambda r: (r.tc, r.obj))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["TC", "Obj", "Accuracy"])
        w.writeheader()
        for r in ordered:
            w.writerow(
                {
                    "TC": r.tc,
                    "Obj": r.obj,
                    "Accuracy": f"{r.accuracy:.4f}",
                }
            )


def format_results_table(rows: list[EvalRow]) -> str:
    """Human-readable fixed-width table for logs."""
    ordered = sorted(rows, key=lambda r: (r.tc, r.obj))
    tc_w = max(len("TC"), max((len(r.tc) for r in ordered), default=2))
    obj_w = max(len("Obj"), max((len(r.obj) for r in ordered), default=3))
    lines = [
        f"{'TC':<{tc_w}}  {'Obj':<{obj_w}}  Accuracy",
        f"{'-' * tc_w}  {'-' * obj_w}  --------",
    ]
    for r in ordered:
        lines.append(f"{r.tc:<{tc_w}}  {r.obj:<{obj_w}}  {r.accuracy:.4f}")
    return "\n".join(lines)


def eval_all_splits(
    checkpoint: Path,
    out_dir: Path,
    *,
    dbpedia_root: Path,
    k: int = 5000,
    bootstrap: int = 0,
    seed: int = 42,
    max_iter: int = 1000,
    chunk_size: int = 2048,
    log_path: Path | None = None,
    dry_run: bool = False,
    all_classifiers: bool = False,
) -> list[EvalRow]:
    """Evaluate checkpoint on all DLCC splits; write metrics under *out_dir*."""
    root = dbpedia_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"DBpedia root not found: {root}")

    checkpoint = checkpoint.resolve()
    if not dry_run and not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    splits = discover_splits(root, int(k))
    if not splits:
        raise SystemExit(
            f"No splits found under {root} with k={k} "
            "(expected **/<k>/train_test/test.txt or train_test_hard/test.txt)"
        )

    if dry_run:
        for s in splits:
            print(s.test_path)
        print(f"\n{len(splits)} split(s)", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "metrics.csv"
    out_jsonl = out_dir / "metrics.jsonl"
    results_table = out_dir / "results_table.csv"
    results_table_averaged = out_dir / "results_table_averaged.csv"
    log_file = log_path.resolve() if log_path is not None else out_dir / "eval.log"

    rows: list[EvalRow] = []
    split_models: dict[str, dict[str, dict[str, float]]] = {}
    with _tee_stdio(log_file):
        started = datetime.now(timezone.utc).isoformat()
        print(f"Eval started at {started}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Output directory: {out_dir}")
        print(f"Full log (stdout+stderr): {log_file}")
        print(f"Splits: {len(splits)}  |  k={k}  |  bootstrap={bootstrap}")
        print()

        for split in splits:
            print(f"== {split.tc} / {split.obj} ({split.name})")
            res = run_evaluation(
                split.test_path,
                checkpoint,
                train_path=None,
                bootstrap=int(bootstrap),
                seed=int(seed),
                max_iter=int(max_iter),
                chunk_size=int(chunk_size),
                progress=True,
                verbose=False,
                all_classifiers=all_classifiers,
            )
            print(
                f"   best={res['best_model']}  "
                f"accuracy={res['accuracy']:.4f}±{res['accuracy_std']:.4f}  "
                f"precision={res['precision']:.4f}±{res['precision_std']:.4f}  "
                f"recall={res['recall']:.4f}±{res['recall_std']:.4f}  "
                f"f1={res['f1']:.4f}±{res['f1_std']:.4f}  "
                f"(oov train={res['oov_train']} test={res['oov_test']})"
            )
            rows.append(
                EvalRow(
                    split=split.name,
                    tc=split.tc,
                    domain=split.domain,
                    obj=split.obj,
                    k=split.k,
                    split_dir=split.split_dir,
                    checkpoint_path=str(checkpoint),
                    test_path=res["test_path"],
                    train_path=res["train_path"],
                    n_train=int(res["n_train"]),
                    n_test=int(res["n_test"]),
                    emb_vocab=int(res["emb_vocab"]),
                    emb_dim=int(res["emb_dim"]),
                    oov_train=int(res["oov_train"]),
                    oov_test=int(res["oov_test"]),
                    best_model=str(res["best_model"]),
                    accuracy=float(res["accuracy"]),
                    accuracy_std=float(res["accuracy_std"]),
                    precision=float(res["precision"]),
                    precision_std=float(res["precision_std"]),
                    recall=float(res["recall"]),
                    recall_std=float(res["recall_std"]),
                    f1=float(res["f1"]),
                    f1_std=float(res["f1_std"]),
                )
            )
            split_models[split.name] = res["models"]

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

        with out_jsonl.open("w", encoding="utf-8") as f:
            for row, split in zip(rows, splits, strict=True):
                payload = asdict(row)
                payload["models"] = split_models[split.name]
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        write_results_table(rows, results_table)
        averaged = aggregate_tc_domain_means(rows)
        write_averaged_results_table(averaged, results_table_averaged)

        print()
        print("Results table (per domain):")
        print(format_results_table(rows))
        print()
        print("Results table (domain-averaged, Table 3 style):")
        print(format_averaged_results_table(averaged))
        print()
        print(f"Wrote {out_csv} ({len(rows)} splits)")
        print(f"Wrote {results_table}")
        print(f"Wrote {results_table_averaged}")
        print(f"Wrote {out_jsonl}")
        print(f"Wrote {log_file}")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Embedding checkpoint (.pt, .kv, or .model)",
    )
    ap.add_argument(
        "--dbpedia-root",
        type=Path,
        default=Path("v1/dbpedia"),
        help="Root containing DBpedia splits (default: v1/dbpedia)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5000,
        help="Positive/negative count folder (default: 5000)",
    )
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples (0 = skip)")
    ap.add_argument("--seed", type=int, default=42, help="Seed for bootstrap + classifier")
    ap.add_argument("--max-iter", type=int, default=1000, help="LogisticRegression max_iter")
    ap.add_argument("--chunk-size", type=int, default=2048, help="Embedding lookup chunk size")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for metrics.csv, results_table.csv, "
        "results_table_averaged.csv, and metrics.jsonl "
        "(default: output/dbpedia_eval_all_<checkpoint_stem>)",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file for stdout/stderr (default: <out>/eval.log)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List splits that would be evaluated and exit",
    )
    ap.add_argument(
        "--all-classifiers",
        action="store_true",
        help="Also fit NB, SVM, and MLP and pick the best by accuracy (default: LogReg only)",
    )
    args = ap.parse_args()

    root = args.dbpedia_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"DBpedia root not found: {root}")

    checkpoint = args.checkpoint.resolve()
    if not args.dry_run and not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    splits = discover_splits(root, int(args.k))
    if not splits:
        raise SystemExit(
            f"No splits found under {root} with k={args.k} "
            "(expected **/<k>/train_test/test.txt or train_test_hard/test.txt)"
        )

    if args.dry_run:
        eval_all_splits(
            checkpoint,
            out_dir if args.out is not None else Path("."),
            dbpedia_root=root,
            k=int(args.k),
            dry_run=True,
        )
        return

    out_dir = (
        args.out.resolve()
        if args.out is not None
        else Path("output") / f"dbpedia_eval_all_{checkpoint.stem}"
    )
    log_path = args.log.resolve() if args.log is not None else out_dir / "eval.log"
    eval_all_splits(
        checkpoint,
        out_dir,
        dbpedia_root=root,
        k=int(args.k),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        chunk_size=int(args.chunk_size),
        log_path=log_path,
        all_classifiers=args.all_classifiers,
    )


if __name__ == "__main__":
    main()
