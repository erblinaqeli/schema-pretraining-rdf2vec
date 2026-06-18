"""PCA fine-tune drift plots (epoch 0 -> epoch 5) sourced from the saved benchmark run.

Same figure as scripts/plot/pca_drift.py (vanilla baseline top-left, classic/bound rows x
p1/p2/p3 cols, epoch-0 faint / epoch-5 bold, arrows e0 -> e5) but instead of *retraining* each
variant it reads the per-epoch entity-vector checkpoints produced by
scripts/run_synthetic_benchmark.py:

    output/synthetic_benchmark/<tc>/<variant>/ckpt_epoch00.npz   (epoch 0 = post-init)
    output/synthetic_benchmark/<tc>/<variant>/ckpt_epoch05.npz   (epoch 5 = end finetune)
    output/synthetic_benchmark/<tc>/<variant>/keys.json          (vocab order)
    output/synthetic_benchmark/<tc>/<variant>/metrics.json       (per-epoch accuracy)

The plotting itself reuses ``plot_tc`` from scripts/plot/pca_drift.py, so the
output is identical in format to the thesis figure.

Usage:
    python scripts/plot/pca_drift_from_run.py            # all tc01-tc15
    python scripts/plot/pca_drift_from_run.py tc01       # selected TCs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (SCRIPTS, SCRIPTS / "plot"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _evaluate import load_labeled_txt, tokens_to_embeddings  # noqa: E402
from pca_drift import ALL_TCS, VARIANTS, plot_tc, tc_paths  # noqa: E402

BENCH_ROOT = ROOT / "output" / "synthetic_benchmark"


def load_run(tc: str, variant: str, test_tokens: list, y_test: np.ndarray) -> dict:
    vdir = BENCH_ROOT / tc / variant
    keys = json.loads((vdir / "keys.json").read_text())
    w2i = {k: i for i, k in enumerate(keys)}
    emb0 = np.load(vdir / "ckpt_epoch00.npz")["vectors"]
    emb5 = np.load(vdir / "ckpt_epoch05.npz")["vectors"]
    x0, _ = tokens_to_embeddings(test_tokens, emb0, w2i, "", 4096, progress=False)
    x5, _ = tokens_to_embeddings(test_tokens, emb5, w2i, "", 4096, progress=False)

    n0, n5 = np.linalg.norm(x0, axis=1), np.linalg.norm(x5, axis=1)
    ok = (n0 > 0) & (n5 > 0)
    mean_cos = (
        float(((x0[ok] * x5[ok]).sum(1) / (n0[ok] * n5[ok])).mean())
        if ok.any() else float("nan")
    )
    accs = json.loads((vdir / "metrics.json").read_text())["accs"]
    return dict(x0=x0, x5=x5, y=y_test, accs=list(accs), mean_cos=mean_cos)


def load_tc(tc: str) -> dict[str, dict]:
    test_tokens, y_test = load_labeled_txt(tc_paths(tc)["test"])
    runs = {}
    for variant, *_ in VARIANTS:
        runs[variant] = load_run(tc, variant, test_tokens, np.asarray(y_test))
        r = runs[variant]
        print(
            f"  {tc}/{variant}: acc {r['accs'][0]:.3f}->{r['accs'][-1]:.3f}, "
            f"cos={r['mean_cos']:.3f}",
            flush=True,
        )
    return runs


def main(argv: list[str]) -> None:
    tcs = [a for a in argv if a.startswith("tc")] or ALL_TCS
    for tc in tcs:
        runs = load_tc(tc)
        plot_tc(tc, runs)


if __name__ == "__main__":
    main(sys.argv[1:])
