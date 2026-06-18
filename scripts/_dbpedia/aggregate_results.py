"""
Aggregate per-domain DBpedia eval rows into TC-level means (DLCC Table 3 style).

For each test case and split type (standard vs hard), take the best accuracy per
domain (max over duplicate rows, e.g. multiple classifiers later), then average
across all domains present for that TC.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence


class _EvalRowLike(Protocol):
    tc: str
    domain: str
    split_dir: str
    accuracy: float


@dataclass(frozen=True)
class AveragedTcRow:
    """One row in the domain-averaged results table."""

    tc: str
    hard: bool
    label: str
    n_domains: int
    accuracy: float
    domains: tuple[str, ...]


def _tc_sort_key(tc: str) -> tuple[int, str]:
    digits = "".join(c for c in tc if c.isdigit())
    return (int(digits) if digits else 0, tc)


def aggregate_tc_domain_means(rows: Iterable[_EvalRowLike]) -> list[AveragedTcRow]:
    """
    Average best-per-domain accuracies for each (tc, standard|hard) group.

    Standard rows use ``split_dir == "train_test"``; hard rows use
    ``split_dir == "train_test_hard"``. Hard rows are omitted when no domain
    has a hard split for that TC.
    """
    best: dict[tuple[str, bool], dict[str, float]] = defaultdict(dict)
    for row in rows:
        is_hard = row.split_dir == "train_test_hard"
        key = (row.tc, is_hard)
        prev = best[key].get(row.domain)
        if prev is None or row.accuracy > prev:
            best[key][row.domain] = float(row.accuracy)

    out: list[AveragedTcRow] = []
    for (tc, is_hard) in sorted(best.keys(), key=lambda k: (_tc_sort_key(k[0]), k[1])):
        per_domain = best[(tc, is_hard)]
        domains = tuple(sorted(per_domain.keys()))
        mean_acc = sum(per_domain[d] for d in domains) / len(domains)
        label = f"{tc} hard" if is_hard else tc
        out.append(
            AveragedTcRow(
                tc=tc,
                hard=is_hard,
                label=label,
                n_domains=len(domains),
                accuracy=mean_acc,
                domains=domains,
            )
        )
    return out


def write_averaged_results_table(rows: Sequence[AveragedTcRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["TC", "Accuracy", "N_domains", "Domains"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "TC": r.label,
                    "Accuracy": f"{r.accuracy:.4f}",
                    "N_domains": r.n_domains,
                    "Domains": ";".join(r.domains),
                }
            )


def format_averaged_results_table(rows: Sequence[AveragedTcRow]) -> str:
    ordered = list(rows)
    tc_w = max(len("TC"), max((len(r.label) for r in ordered), default=2))
    lines = [
        f"{'TC':<{tc_w}}  Accuracy  (domains)",
        f"{'-' * tc_w}  --------  --------",
    ]
    for r in ordered:
        dom_note = f"n={r.n_domains}"
        lines.append(f"{r.label:<{tc_w}}  {r.accuracy:.4f}  ({dom_note})")
    return "\n".join(lines)


def load_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class MetricsCsvRow:
    tc: str
    domain: str
    split_dir: str
    accuracy: float


def metrics_csv_rows(records: Iterable[Mapping[str, str]]) -> list[MetricsCsvRow]:
    out: list[MetricsCsvRow] = []
    for rec in records:
        out.append(
            MetricsCsvRow(
                tc=rec["tc"],
                domain=rec["domain"],
                split_dir=rec["split_dir"],
                accuracy=float(rec["accuracy"]),
            )
        )
    return out


def load_averaged_results_csv(path: Path) -> dict[str, float]:
    """Return TC label -> accuracy from ``results_table_averaged.csv``."""
    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records:
        return {}
    if "TC" not in records[0] or "Accuracy" not in records[0]:
        raise ValueError(f"{path}: expected columns TC and Accuracy")
    return {row["TC"]: float(row["Accuracy"]) for row in records}


def merge_averaged_results(
    tables: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Merge several averaged tables into wide rows keyed by TC label.

    ``tables`` maps column name (e.g. ``p1``) -> {TC label: accuracy}.
    """
    tc_order: list[str] = []
    seen: set[str] = set()
    for table in tables.values():
        for label in table:
            if label not in seen:
                seen.add(label)
                tc_order.append(label)

    def _sort_key(label: str) -> tuple[int, int, str]:
        hard = 1 if label.endswith(" hard") else 0
        tc_part = label.removesuffix(" hard")
        return (_tc_sort_key(tc_part)[0], hard, label)

    tc_order.sort(key=_sort_key)
    columns = list(tables.keys())

    rows: list[dict[str, str]] = []
    for tc_label in tc_order:
        row: dict[str, str] = {"TC": tc_label}
        values: list[tuple[str, float]] = []
        for col in columns:
            acc = tables[col].get(tc_label)
            if acc is None:
                row[col] = ""
            else:
                row[col] = f"{acc:.4f}"
                values.append((col, acc))
        if values:
            best_col, best_val = max(values, key=lambda x: x[1])
            row["best"] = best_col
            row["best_accuracy"] = f"{best_val:.4f}"
        else:
            row["best"] = ""
            row["best_accuracy"] = ""
        rows.append(row)
    return columns, rows


def write_comparison_table(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    path: Path,
    *,
    include_best: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["TC", *columns]
    if include_best:
        fieldnames.extend(["best", "best_accuracy"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def format_comparison_table(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> str:
    cols = list(columns)
    tc_w = max(len("TC"), max((len(str(r["TC"])) for r in rows), default=2))
    col_w = max(8, max((len(c) for c in cols), default=8))
    header = f"{'TC':<{tc_w}}  " + "  ".join(f"{c:>{col_w}}" for c in cols)
    sep = f"{'-' * tc_w}  " + "  ".join("-" * col_w for _ in cols)
    lines = [header, sep]
    for row in rows:
        cells: list[str] = []
        best_col = row.get("best", "")
        for c in cols:
            val = row.get(c, "")
            if val and c == best_col:
                cells.append(f"{val:>{col_w}}*")
            else:
                cells.append(f"{val:>{col_w}}" if val else f"{'':>{col_w}}")
        lines.append(f"{row['TC']:<{tc_w}}  " + "  ".join(cells))
    return "\n".join(lines)
