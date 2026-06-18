"""Generate the per-TC total-runtime line plot (latex/assets/runtime_total_lines.pdf).

X-axis: test case (tc01..tc12). Y-axis: end-to-end Total runtime (seconds).
Three lines:
  vanilla  -- the from-scratch baseline (single variant)
  classic  -- mean of {p1,p2,p3}_classic Total
  bound    -- mean of {p1,p2,p3}_bound  Total

Total per (tc, variant) is pretrain_s + init_s + finetune_s, matching the
``Total`` column of Table~\\ref{tab:runtime_per_tc} (runtime_per_tc_table.tex).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
RESULTS_JSON = ROOT / "output" / "synthetic_benchmark" / "summary.json"
OUT_PDF = ROOT / "latex" / "assets" / "runtime_total_lines.pdf"

# tc01..tc12, excluding tc04 (a ~15x runtime outlier whose much larger walk
# corpus dominates the scale and flattens every other test case).
TCS = [f"tc{i:02d}" for i in range(1, 13) if i != 4]


def total_s(row: dict) -> float:
    return row["pretrain_s"] + row["init_s"] + row["finetune_s"]


def main() -> None:
    rows = json.loads(RESULTS_JSON.read_text())
    # index by (tc, variant) -> total seconds
    tot = {(r["tc"], r["variant"]): total_s(r) for r in rows}

    vanilla = [tot[(tc, "vanilla")] for tc in TCS]
    classic = [
        sum(tot[(tc, f"{p}_classic")] for p in ("p1", "p2", "p3")) / 3 for tc in TCS
    ]
    bound = [
        sum(tot[(tc, f"{p}_bound")] for p in ("p1", "p2", "p3")) / 3 for tc in TCS
    ]

    x = list(range(len(TCS)))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(x, vanilla, marker="o", label="vanilla")
    ax.plot(x, classic, marker="s", label="classic (mean p1/p2/p3)")
    ax.plot(x, bound, marker="^", label="bound (mean p1/p2/p3)")

    ax.set_xticks(x)
    ax.set_xticklabels(TCS, rotation=45, ha="right")
    ax.set_xlabel("Test case")
    ax.set_ylabel("Total runtime (s)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
