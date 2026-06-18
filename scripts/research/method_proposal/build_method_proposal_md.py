"""
Injects the generated summary table and per-TC result sections into
method_proposal.md, replacing everything between/after the markers:

  <!-- SUMMARY:BEGIN --> ... <!-- SUMMARY:END -->
  <!-- RESULTS:BEGIN -->  ... (to end of file)

Run after scripts/run_method_proposal.py and scripts/plot_method_proposal.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DATA = ROOT / "method_proposal" / "data"
MD = ROOT / "method_proposal.md"
FIG = "method_proposal/figures"
# Pandoc PDF uses markdown images + layout grids (HTML <img> is dropped by --pdf-engine).
FIG_WIDTH = {"acc": "32%", "loss": "32%", "pca": "100%"}
PLOT_GRID_COLS = 3
LAYOUT_COL = "0.32"


def fig_img(kind: str, tc: str) -> str:
    alt = f"{tc} {kind}"
    return f"![{alt}]({FIG}/{kind}_{tc}.png){{width={FIG_WIDTH[kind]}}}"


def plot_grid(kind: str, tcs: list[str]) -> str:
    blocks = []
    for i in range(0, len(tcs), PLOT_GRID_COLS):
        row = tcs[i : i + PLOT_GRID_COLS]
        cols = " ".join([LAYOUT_COL] * len(row))
        imgs = "\n".join(fig_img(kind, tc) for tc in row)
        blocks.append(f":::: {{layout=\"{cols}\"}}\n{imgs}\n::::")
    return "\n\n".join(blocks)


def plot_grids_section() -> str:
    return "\n".join(
        [
            "### Per-epoch accuracy (all test cases)\n",
            plot_grid("acc", TCS) + "\n",
            "### Per-epoch training loss (all test cases)\n",
            plot_grid("loss", TCS) + "\n",
        ]
    )


VARIANTS = ["vanilla", "p1_classic", "p2_classic", "p1_bound", "p2_bound", "p3_bound"]
LABELS = {v: f"`{v}`" for v in VARIANTS}
LABELS["p1_classic"] = "`p1_classic` (LR 0.025)"

TCS = sorted(p.stem for p in DATA.glob("tc*.json"))


def load(tc):
    rows = json.loads((DATA / f"{tc}.json").read_text())
    return {r["variant"]: r for r in rows}


ALL = {tc: load(tc) for tc in TCS}


def summary_block() -> str:
    lines = [
        "| tc | " + " | ".join(LABELS[v] for v in VARIANTS) + " | best bound − vanilla |",
        "|---|" + "---|" * (len(VARIANTS) + 1),
    ]
    deltas = []
    for tc in TCS:
        rows = ALL[tc]
        vals = [rows[v]["final_acc"] for v in VARIANTS]
        best_bound = max(rows[v]["final_acc"] for v in ("p1_bound", "p2_bound", "p3_bound"))
        delta = best_bound - rows["vanilla"]["final_acc"]
        deltas.append(delta)
        best = max(vals)
        cells = [f"**{x:.3f}**" if x == best else f"{x:.3f}" for x in vals]
        lines.append(f"| {tc} | " + " | ".join(cells) + f" | {delta:+.3f} |")
    means = [sum(ALL[tc][v]["final_acc"] for tc in TCS) / len(TCS) for v in VARIANTS]
    lines.append(
        "| **mean** | " + " | ".join(f"{m:.3f}" for m in means)
        + f" | {sum(deltas)/len(deltas):+.3f} |"
    )
    return "\n".join(lines)


def runtime_summary_block() -> str:
    lines = [
        "| variant | mean pretrain W2V (s) | mean finetune W2V (s) | mean total (s) |",
        "|---|---|---|---|",
    ]
    for v in VARIANTS:
        pre = sum(ALL[tc][v]["pretrain_s"] for tc in TCS) / len(TCS)
        fin = sum(ALL[tc][v]["finetune_s"] for tc in TCS) / len(TCS)
        pre_s = "—" if v == "vanilla" else f"{pre:.1f}"
        lines.append(f"| {LABELS[v]} | {pre_s} | {fin:.1f} | {pre + fin:.1f} |")
    return "\n".join(lines)


def tc_section(tc: str) -> str:
    rows = ALL[tc]
    n_ep = len(rows["vanilla"]["accs"])
    out = [f"### {tc}\n"]

    out.append("**Accuracy per epoch** (epoch 0 = right after initialization)\n")
    out.append("| variant | " + " | ".join(f"ep {e}" for e in range(n_ep)) + " |")
    out.append("|---|" + "---|" * n_ep)
    for v in VARIANTS:
        accs = rows[v]["accs"]
        best = max(accs)
        cells = [f"**{a:.4f}**" if a == best else f"{a:.4f}" for a in accs]
        out.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(fig_img("pca", tc) + "\n")

    out.append("**Runtime and embedding drift**\n")
    out.append("| variant | pretrain W2V (s) | finetune W2V (s) | total (s) | mean cos(ep 0, ep 5) |")
    out.append("|---|---|---|---|---|")
    for v in VARIANTS:
        r = rows[v]
        pre = "—" if v == "vanilla" else f"{r['pretrain_s']:.1f}"
        out.append(
            f"| {LABELS[v]} | {pre} | {r['finetune_s']:.1f} | {r['total_s']:.1f} "
            f"| {r['mean_cos_e0_e5']:.3f} |"
        )
    out.append("")
    return "\n".join(out)


def main():
    text = MD.read_text(encoding="utf-8")

    s_beg, s_end = "<!-- SUMMARY:BEGIN -->", "<!-- SUMMARY:END -->"
    i, j = text.index(s_beg) + len(s_beg), text.index(s_end)
    text = text[:i] + "\n" + summary_block() + "\n" + text[j:]

    r_beg, r_end = "<!-- RUNTIME:BEGIN -->", "<!-- RUNTIME:END -->"
    i, j = text.index(r_beg) + len(r_beg), text.index(r_end)
    text = text[:i] + "\n" + runtime_summary_block() + "\n" + text[j:]

    marker = "<!-- RESULTS:BEGIN -->"
    i = text.index(marker) + len(marker)
    sections = plot_grids_section() + "\n" + "\n".join(tc_section(tc) for tc in TCS)
    text = text[:i] + "\n\n" + sections
    MD.write_text(text, encoding="utf-8")
    print(f"wrote {MD} ({len(TCS)} TC sections)")


if __name__ == "__main__":
    main()
