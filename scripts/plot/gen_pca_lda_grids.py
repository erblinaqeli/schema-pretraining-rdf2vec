"""Generate PCA vs LDA grids (one per test case) into latex/assets/pca_lda/.

For every DLCC synthetic test case tc01-tc15 and every protograph variant
(p1/p2/p3 x classic/bound) this trains the RDF2Vec model to epoch 5 on the
cached synthetic_compare walk corpora and projects the *trained* (epoch-5)
test-entity embeddings two ways:

  * PCA  - directions of maximum variance (often misses the label boundary)
  * LDA  - LD1 (label-aware) + an orthogonal PC for layout

Classic variants are finetuned at the from-scratch LR 0.025 (vanilla_alpha);
only the *_bound variants use the protected finetune LR 0.0025. This mirrors
notebooks/synthetic_visuals.ipynb. Trained embeddings are cached under
notebooks/synthetic_visuals/cache/pca_lda/<tc>.npz so re-runs are fast.

Usage:
    python scripts/plot/gen_pca_lda_grids.py            # all tc01-tc15
    python scripts/plot/gen_pca_lda_grids.py tc07 tc09  # subset
    FORCE=1 python scripts/plot/gen_pca_lda_grids.py     # ignore cache
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from gensim.models.word2vec import LineSentence  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
COMPARE_ROOT = ROOT / "notebooks" / "synthetic_compare"
CACHE_ROOT = ROOT / "notebooks" / "synthetic_visuals" / "cache" / "pca_lda"
OUT_DIR = ROOT / "latex" / "assets" / "pca_lda"

for p in (str(SCRIPTS), str(SCRIPTS / "plot")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import fit_lda_2d, LABEL_COLORS  # noqa: E402
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
    vanilla_alpha=0.025,   # classic finetune LR
    finetune_alpha=0.0025,  # bound finetune LR (protected)
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)

# (variant label, init mode, protograph, finetune alpha)
VARIANTS = [
    ("p1_classic", "classic", "p1", CFG["vanilla_alpha"]),
    ("p2_classic", "classic", "p2", CFG["vanilla_alpha"]),
    ("p3_classic", "classic", "p3", CFG["vanilla_alpha"]),
    ("p1_bound", "bound", "p1", CFG["finetune_alpha"]),
    ("p2_bound", "bound", "p2", CFG["finetune_alpha"]),
    ("p3_bound", "bound", "p3", CFG["finetune_alpha"]),
]

ALL_TCS = [f"tc{i:02d}" for i in range(1, 16)]


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=COMPARE_ROOT / tc,
    )


_stage1_cache: dict[str, tuple] = {}


def _stage1(proto_walks: Path) -> tuple:
    key = str(proto_walks)
    if key not in _stage1_cache:
        pre = pretrain_protograph(
            proto_walks, dim=CFG["dim"], epochs=CFG["pretrain_epochs"], seed=CFG["seed"]
        )
        _stage1_cache[key] = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
    return _stage1_cache[key]


def train_variant_x5(tc: str, init: str, proto: str, alpha: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train one variant to epoch 5; return (x5 test embeddings, y_test, accs)."""
    sp = tc_paths(tc)
    inst_walks = sp["out"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt"
    proto_walks = sp["out"] / f"walks_{proto}.txt"
    if not inst_walks.is_file() or not proto_walks.is_file():
        raise FileNotFoundError(f"Missing walks for {tc}; run synthetic_compare first")

    model = new_skipgram_model(
        dim=CFG["dim"], alpha=alpha, min_alpha=CFG["min_alpha"], seed=CFG["seed"], workers=16
    )
    model.build_vocab(LineSentence(str(inst_walks)))

    stage1, used_norm = _stage1(proto_walks)
    bound = None
    if init == "bound":
        bound = concept_bound_vectors(sp["graph"], sp["ontology"], stage1, target_norm=used_norm)
    protograph_init(
        model, stage1, sp["ontology"], strategy="all_init",
        target_norm=used_norm, bound_vectors=bound,
    )

    accs = train_with_eval(
        model, inst_walks, epochs=CFG["epochs"], alpha=alpha, min_alpha=CFG["min_alpha"],
        eval_fn=make_eval_fn(sp["train"], sp["test"]),
    )

    test_tokens, y_test = load_labeled_txt(sp["test"])
    emb5 = np.asarray(model.wv.vectors, dtype=np.float32)
    w2i = dict(model.wv.key_to_index)
    x5, _ = tokens_to_embeddings(test_tokens, emb5, w2i, "", 4096, progress=False)
    return x5, np.asarray(y_test), accs


def load_or_train(tc: str, *, force: bool = False) -> dict:
    cache_path = CACHE_ROOT / f"{tc}.npz"
    if cache_path.is_file() and not force:
        data = np.load(cache_path, allow_pickle=True)
        runs = {"y": data["y_test"]}
        for variant, _, _, _ in VARIANTS:
            runs[variant] = dict(
                x5=data[f"{variant}_x5"], accs=list(data[f"{variant}_accs"]),
            )
        print(f"{tc}: loaded cache")
        return runs

    t0 = time.time()
    runs: dict = {}
    save: dict = {}
    for variant, init, proto, alpha in VARIANTS:
        print(f"  training {tc} / {variant} ...", flush=True)
        x5, y, accs = train_variant_x5(tc, init, proto, alpha)
        runs[variant] = dict(x5=x5, accs=accs)
        runs["y"] = y
        save["y_test"] = y
        save[f"{variant}_x5"] = x5
        save[f"{variant}_accs"] = np.asarray(accs)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **save)
    print(f"{tc}: trained + cached in {time.time() - t0:.1f}s")
    return runs


def scatter_labels(ax, xy, y, *, title: str, s=14):
    for lab in (0, 1):
        m = y == lab
        ax.scatter(
            xy[m, 0], xy[m, 1], s=s, alpha=0.65, color=LABEL_COLORS[lab],
            label=f"label {lab} (n={int(m.sum())})", lw=0,
        )
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_tc_grid(tc: str, runs: dict):
    y = runs["y"]
    fig, axes = plt.subplots(len(VARIANTS), 2, figsize=(10, 3.6 * len(VARIANTS)))
    for row, (variant, _, _, _) in enumerate(VARIANTS):
        x = runs[variant]["x5"]
        acc = runs[variant]["accs"][-1]
        sil = silhouette_score(x, y) if len(np.unique(y)) > 1 else float("nan")
        pca = PCA(n_components=2, random_state=CFG["seed"])
        xy_pca = pca.fit_transform(x)
        xy_lda, ld1, orth = fit_lda_2d(x, y, seed=CFG["seed"])
        scatter_labels(
            axes[row, 0], xy_pca, y,
            title=f"{variant} - PCA  acc={acc:.3f}  sil={sil:.3f}\n"
                  f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, "
                  f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%",
        )
        scatter_labels(
            axes[row, 1], xy_lda, y,
            title=f"{variant} - LDA  LD1 {ld1*100:.1f}%, PC⊥ {orth*100:.1f}%",
        )
        axes[row, 0].set_ylabel(variant)
    for ax in axes[-1]:
        ax.set_xlabel("component 1")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{tc}: PCA (variance) vs LDA (label-aware) at epoch 5  "
        "(classic @ LR 0.025, bound @ 0.0025)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tc}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"{tc}: saved {out_path.relative_to(ROOT)}")


def main(argv: list[str]):
    force = bool(os.environ.get("FORCE"))
    tcs = argv or ALL_TCS
    for tc in tcs:
        runs = load_or_train(tc, force=force)
        plot_tc_grid(tc, runs)


if __name__ == "__main__":
    main(sys.argv[1:])
