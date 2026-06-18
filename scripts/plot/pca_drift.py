"""PCA fine-tune drift plots (epoch 0 -> epoch 5) for tc01-tc15, vanilla + p1/p2/p3 classic + bound.

Mirrors Section 2 of notebooks/synthetic_visuals.ipynb ("Finetune drift") but extended to
every DLCC synthetic test case and the seven variants requested for the thesis:

    vanilla                                 (no protograph init, from-scratch LR 0.025)
    p1_classic  p2_classic  p3_classic   (init=classic, finetune LR 0.025)
    p1_bound    p2_bound    p3_bound     (init=bound,   finetune LR 0.0025)

IMPORTANT: classic variants are fine-tuned at the from-scratch LR 0.025 (CFG["vanilla_alpha"]),
matching the lr=0.025 classic checkpoints used elsewhere in the thesis; only *_bound variants use
the protected 0.0025 LR; vanilla is the from-scratch baseline (no init, LR 0.025). Test-entity
embeddings are snapshotted at epoch 0 (initialization) and epoch 5 (end of fine-tuning), then
drawn with a shared per-panel PCA basis and arrows e0 -> e5.

Snapshots are cached under notebooks/synthetic_visuals/cache/drift/<tc>.npz so re-runs are fast.
One figure per TC (vanilla baseline top-left, then classic/bound rows x p1/p2/p3 cols) is written
to latex/assets/pca_drift/.

Usage:
    python scripts/plot/pca_drift.py            # all tc01-tc15
    python scripts/plot/pca_drift.py tc07 tc09  # selected TCs
    python scripts/plot/pca_drift.py --force    # ignore cache, retrain
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from gensim.models.word2vec import LineSentence  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
COMPARE_ROOT = ROOT / "notebooks" / "synthetic_compare"
CACHE_ROOT = ROOT / "notebooks" / "synthetic_visuals" / "cache" / "drift"
ASSETS_OUT = ROOT / "latex" / "assets" / "pca_drift"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
ASSETS_OUT.mkdir(parents=True, exist_ok=True)

for p in (SCRIPTS, SCRIPTS / "plot"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _common import LABEL_COLORS  # noqa: E402
from _evaluate import load_labeled_txt, tokens_to_embeddings  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    concept_bound_vectors,
    make_eval_fn,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
    train_with_eval,
)

CFG = dict(
    dim=200,
    walks_per_entity=100,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    vanilla_alpha=0.025,
    finetune_alpha=0.0025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)

ALL_TCS = [f"tc{i:02d}" for i in range(1, 16)]

# (variant, init mode, protograph, finetune LR). vanilla -> no init @ 0.025, classic -> 0.025,
# bound -> 0.0025.
VARIANTS = [
    ("vanilla", "none", "p1", CFG["vanilla_alpha"]),
    ("p1_classic", "classic", "p1", CFG["vanilla_alpha"]),
    ("p2_classic", "classic", "p2", CFG["vanilla_alpha"]),
    ("p3_classic", "classic", "p3", CFG["vanilla_alpha"]),
    ("p1_bound", "bound", "p1", CFG["finetune_alpha"]),
    ("p2_bound", "bound", "p2", CFG["finetune_alpha"]),
    ("p3_bound", "bound", "p3", CFG["finetune_alpha"]),
]

_stage1_cache: dict[tuple[str, str], tuple] = {}


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=COMPARE_ROOT / tc,
    )


def _stage1(proto_walks: Path) -> tuple:
    key = (str(proto_walks), str(CFG["target_norm"]))
    if key not in _stage1_cache:
        pre = pretrain_protograph(
            proto_walks, dim=CFG["dim"], epochs=CFG["pretrain_epochs"], seed=CFG["seed"]
        )
        _stage1_cache[key] = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
    return _stage1_cache[key]


def snapshot_variant(tc: str, init: str, proto: str, alpha: float) -> dict:
    sp = tc_paths(tc)
    inst_walks = sp["out"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt"
    if not inst_walks.is_file():
        raise FileNotFoundError(f"Missing instance walks for {tc}: {inst_walks}")

    model = new_skipgram_model(
        dim=CFG["dim"], alpha=alpha, min_alpha=CFG["min_alpha"], seed=CFG["seed"], workers=16
    )
    model.build_vocab(LineSentence(str(inst_walks)))

    # vanilla (init="none") trains from random init with no protograph pretrain.
    if init != "none":
        proto_walks = sp["out"] / f"walks_{proto}.txt"
        if not proto_walks.is_file():
            raise FileNotFoundError(f"Missing protograph walks for {tc}: {proto_walks}")
        stage1, used_norm = _stage1(proto_walks)
        bound = None
        if init == "bound":
            bound = concept_bound_vectors(sp["graph"], sp["ontology"], stage1, target_norm=used_norm)
        protograph_init(
            model, stage1, sp["ontology"], strategy="all_init",
            target_norm=used_norm, bound_vectors=bound,
        )

    test_tokens, y_test = load_labeled_txt(sp["test"])
    emb0 = np.asarray(model.wv.vectors, dtype=np.float32)
    w2i = dict(model.wv.key_to_index)
    x0, _ = tokens_to_embeddings(test_tokens, emb0, w2i, "", 4096, progress=False)

    accs = train_with_eval(
        model, inst_walks, epochs=CFG["epochs"], alpha=alpha, min_alpha=CFG["min_alpha"],
        eval_fn=make_eval_fn(sp["train"], sp["test"]),
    )
    emb5 = np.asarray(model.wv.vectors, dtype=np.float32)
    x5, _ = tokens_to_embeddings(test_tokens, emb5, w2i, "", 4096, progress=False)

    n0, n5 = np.linalg.norm(x0, axis=1), np.linalg.norm(x5, axis=1)
    ok = (n0 > 0) & (n5 > 0)
    mean_cos = float(((x0[ok] * x5[ok]).sum(1) / (n0[ok] * n5[ok])).mean()) if ok.any() else float("nan")

    return dict(x0=x0, x5=x5, y=y_test, accs=list(accs), mean_cos=mean_cos)


def load_or_train_tc(tc: str, *, force: bool = False) -> dict[str, dict]:
    cache_path = CACHE_ROOT / f"{tc}.npz"
    if cache_path.is_file() and not force:
        data = np.load(cache_path, allow_pickle=True)
        if all(f"{variant}_x0" in data for variant, *_ in VARIANTS):
            runs = {}
            for variant, *_ in VARIANTS:
                runs[variant] = dict(
                    x0=data[f"{variant}_x0"],
                    x5=data[f"{variant}_x5"],
                    y=data["y_test"],
                    accs=list(data[f"{variant}_accs"]),
                    mean_cos=float(data[f"{variant}_mean_cos"]),
                )
            print(f"{tc}: loaded cache ({cache_path.name})")
            return runs
        print(f"{tc}: cache missing variants, retraining")

    t0 = time.time()
    runs, save = {}, {}
    for variant, init, proto, alpha in VARIANTS:
        print(f"  training {tc} / {variant} ...", flush=True)
        r = snapshot_variant(tc, init, proto, alpha)
        runs[variant] = r
        save["y_test"] = r["y"]
        save[f"{variant}_x0"] = r["x0"]
        save[f"{variant}_x5"] = r["x5"]
        save[f"{variant}_accs"] = np.asarray(r["accs"])
        save[f"{variant}_mean_cos"] = r["mean_cos"]
    np.savez_compressed(cache_path, **save)
    print(f"{tc}: trained + cached in {time.time() - t0:.1f}s")
    return runs


# Panel grid: vanilla baseline top-left, then classic across the top row and bound across the
# bottom row for p1/p2/p3. The bottom-left cell has no bound counterpart and is left blank.
PANEL_POS = {
    "vanilla": (0, 0),
    "p1_classic": (0, 1),
    "p2_classic": (0, 2),
    "p3_classic": (0, 3),
    "p1_bound": (1, 1),
    "p2_bound": (1, 2),
    "p3_bound": (1, 3),
}


def plot_tc(tc: str, runs: dict[str, dict]) -> Path:
    y = next(iter(runs.values()))["y"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[1, 0].axis("off")  # vanilla has no classic/bound counterpart
    for variant, *_ in VARIANTS:
        r, c = PANEL_POS[variant]
        ax = axes[r, c]
        run = runs[variant]
        x0, x5 = run["x0"], run["x5"]
        xy = PCA(n_components=2, random_state=CFG["seed"]).fit_transform(np.vstack([x0, x5]))
        xy0, xy5 = xy[: len(x0)], xy[len(x0):]
        for lab, color in LABEL_COLORS.items():
            m = y == lab
            ax.scatter(xy0[m, 0], xy0[m, 1], s=12, alpha=0.35, color=color, lw=0)
            ax.scatter(xy5[m, 0], xy5[m, 1], s=18, alpha=0.75, color=color, lw=0)
        step = max(1, len(xy0) // 80)
        for i in range(0, len(xy0), step):
            ax.annotate(
                "", xy=(xy5[i, 0], xy5[i, 1]), xytext=(xy0[i, 0], xy0[i, 1]),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.6, alpha=0.5),
            )
        ax.set_title(
            f"{variant}\nacc {run['accs'][0]:.3f}$\\rightarrow${run['accs'][-1]:.3f}, "
            f"cos={run['mean_cos']:.3f}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0], markersize=7,
               label="label 0 (e0 faint / e5 bold)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1], markersize=7,
               label="label 1 (e0 faint / e5 bold)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(
        f"{tc.upper()}: PCA fine-tune drift, epoch 0 $\\rightarrow$ 5 "
        f"(vanilla/classic @ LR 0.025, bound @ 0.0025)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = ASSETS_OUT / f"{tc}_pca_drift.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"{tc}: wrote {out.relative_to(ROOT)}")
    return out


def main(argv: list[str]) -> None:
    force = "--force" in argv
    tcs = [a for a in argv if a.startswith("tc")] or ALL_TCS
    for tc in tcs:
        runs = load_or_train_tc(tc, force=force)
        plot_tc(tc, runs)


if __name__ == "__main__":
    main(sys.argv[1:])
