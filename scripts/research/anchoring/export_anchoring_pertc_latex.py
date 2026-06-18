"""Export the full per-test-case anchoring tables (one per protograph) to LaTeX.

Reads notebooks/anchoring/results.json (the default-LR, most_specific sweep) and
writes latex/assets/anchoring_per_tc_tables.tex: three booktabs tables (P1/P2/P3),
each with rows = test cases, columns = vanilla baseline + every anchoring strength
lambda, and a mean row. The best lambda value in each row is bolded.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
RESULTS = ROOT / "notebooks" / "anchoring" / "results.json"
OUT = ROOT / "latex" / "assets" / "anchoring_per_tc_tables.tex"

PROTOS = ["p1", "p2", "p3"]
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAM_HEAD = ["$0$", "$0.1$", "$0.25$", "$0.5$", "$0.75$", "$1.0$"]


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    tcs = sorted(data)

    # final_acc[proto][tc][lam], vanilla[tc]
    final: dict[str, dict[str, dict[float, float]]] = {p: {} for p in PROTOS}
    vanilla: dict[str, float] = {}
    for tc, rows in data.items():
        for r in rows:
            if r["variant"] == "vanilla":
                vanilla[tc] = r["final_acc"]
            else:
                final[r["proto"]].setdefault(tc, {})[r["lam"]] = r["final_acc"]

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def fmt_row(label: str, van: float, lam_vals: list[float]) -> str:
        best = max(lam_vals)
        cells = [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-9 else f"{v:.3f}"
                 for v in lam_vals]
        return f"{label} & {van:.3f} & " + " & ".join(cells) + r" \\"

    blocks: list[str] = []
    for proto in PROTOS:
        P = proto.upper()
        lines = [
            r"\begin{table}[p]",
            r"\centering",
            r"\small",
            (r"\caption{Per-test-case final LogReg test accuracy for the \textbf{"
             + P + r"} transfer (\texttt{most\_specific} initialization, normalized "
             r"codes): \texttt{vanilla} baseline, "
             r"the un-anchored transfer ($\lambda=0$), and each anchoring strength "
             r"$\lambda$. The best $\lambda$ value in each row is in bold; the bottom "
             r"row is the mean over the fourteen test cases "
             r"(\texttt{tc01}--\texttt{tc15}, excl.\ \texttt{tc04}).}"),
            r"\label{tab:anchoring_pertc_" + proto + "}",
            r"\begin{tabular}{l r rrrrrr}",
            r"\toprule",
            r" & & \multicolumn{6}{c}{anchoring strength $\lambda$} \\",
            r"\cmidrule(lr){3-8}",
            r"TC & \texttt{vanilla} & " + " & ".join(LAM_HEAD) + r" \\",
            r"\midrule",
        ]
        col_means_lam = {lam: [] for lam in LAMBDAS}
        col_mean_van = []
        for tc in tcs:
            lam_vals = [final[proto][tc][lam] for lam in LAMBDAS]
            van = vanilla[tc]
            lines.append(fmt_row(f"\\texttt{{{tc}}}", van, lam_vals))
            for lam, v in zip(LAMBDAS, lam_vals):
                col_means_lam[lam].append(v)
            col_mean_van.append(van)
        lines.append(r"\midrule")
        mean_lam = [mean(col_means_lam[lam]) for lam in LAMBDAS]
        lines.append(fmt_row(r"\textbf{mean}", mean(col_mean_van), mean_lam))
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        blocks.append("\n".join(lines))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {OUT} ({len(tcs)} TCs x {len(PROTOS)} protographs)")


if __name__ == "__main__":
    main()
