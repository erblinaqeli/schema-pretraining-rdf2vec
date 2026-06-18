"""
Experiment runner for method_proposal.md.

Re-runs the six synthetic_compare variants on every DLCC test case (minus tc04)
and records, per variant:

  * per-epoch LogReg test accuracy (epoch 0 = right after initialization),
  * per-epoch SGNS training loss on the instance-walk corpus,
  * Word2Vec wall-clock runtimes: pretrain (protograph) and finetune (instance),
    with the per-epoch evaluation overhead subtracted from the finetune time,
  * test-entity embedding snapshots at epoch 0 and epoch 5 (for PCA and the
    cosine-drift summary).

Per user request, `p1_classic` is finetuned at the from-scratch LR 0.025
(the "control" configuration from the notebook) instead of the protected
0.0025 used by every other initialized variant.

Outputs (resumable; delete a TC's json to re-run it):
  method_proposal/data/<tc>.json        - metrics
  method_proposal/data/<tc>_emb.npz     - x0/x5 test embeddings per variant + y_test
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _evaluate import load_labeled_txt, tokens_to_embeddings  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    concept_bound_vectors,
    ensure_walks,
    make_eval_fn,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
    write_protographs,
)

OUT_ROOT = ROOT / "method_proposal" / "data"
WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"  # reuse cached walks

TCS = ["tc01", "tc02", "tc03"] + [f"tc{i:02d}" for i in range(5, 17)]

CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    vanilla_alpha=0.025,
    finetune_alpha=0.0025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
    workers=16,
)


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        out=WALK_ROOT / tc,
    )


class _EpochTracker(CallbackAny2Vec):
    """Per-epoch accuracy + per-epoch SGNS loss (cumulative loss diffed)."""

    def __init__(self, eval_fn, accs: list[float], losses: list[float]):
        self.eval_fn = eval_fn
        self.accs = accs
        self.losses = losses
        self.eval_seconds = 0.0
        self._prev_cum = 0.0

    def on_epoch_end(self, model):
        cum = float(model.get_latest_training_loss())
        self.losses.append(cum - self._prev_cum)
        self._prev_cum = cum
        t0 = time.time()
        self.accs.append(self.eval_fn(model))
        self.eval_seconds += time.time() - t0


def train_tracked(model, walks_path, *, epochs, alpha, min_alpha, eval_fn):
    """Train with loss tracking; returns (accs, losses, train_seconds_excl_eval)."""
    accs: list[float] = [eval_fn(model)]
    losses: list[float] = []
    tracker = _EpochTracker(eval_fn, accs, losses)
    t0 = time.time()
    model.train(
        corpus_iterable=LineSentence(str(walks_path)),
        total_examples=model.corpus_count,
        epochs=epochs,
        start_alpha=alpha,
        end_alpha=min_alpha,
        compute_loss=True,
        callbacks=[tracker],
    )
    train_s = time.time() - t0 - tracker.eval_seconds
    return accs, losses, train_s


def snapshot_test_embeddings(model, test_tokens):
    emb = np.asarray(model.wv.vectors, dtype=np.float32)
    w2i = dict(model.wv.key_to_index)
    x, _ = tokens_to_embeddings(test_tokens, emb, w2i, "", 4096, progress=False)
    return np.asarray(x, dtype=np.float32)


def mean_cos(x0: np.ndarray, x5: np.ndarray) -> float:
    n0 = np.linalg.norm(x0, axis=1)
    n5 = np.linalg.norm(x5, axis=1)
    ok = (n0 > 0) & (n5 > 0)
    return float(((x0[ok] * x5[ok]).sum(1) / (n0[ok] * n5[ok])).mean())


def run_tc(tc: str) -> None:
    p = tc_paths(tc)
    out_json = OUT_ROOT / f"{tc}.json"
    out_npz = OUT_ROOT / f"{tc}_emb.npz"
    if out_json.is_file() and out_npz.is_file():
        print(f"{tc}: cached, skipping", flush=True)
        return

    t_start = time.time()
    p["out"].mkdir(parents=True, exist_ok=True)
    proto_paths = write_protographs(p["ontology"], p["out"])
    proto_walks = {
        kind: ensure_walks(
            path,
            p["out"] / f"walks_{kind}.txt",
            walks_per_entity=CFG["proto_walks_per_entity"],
            depth=CFG["depth"],
            seed=CFG["seed"],
            ensure_triple_coverage=True,
        )
        for kind, path in proto_paths.items()
    }
    inst_walks = ensure_walks(
        p["graph"],
        p["out"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
    )

    eval_fn = make_eval_fn(p["train"], p["test"])
    test_tokens, y_test = load_labeled_txt(p["test"])

    rows: list[dict] = []
    emb_arrays: dict[str, np.ndarray] = {"y_test": np.asarray(y_test)}

    def record(name, accs, losses, pretrain_s, finetune_s, alpha, x0, x5):
        rows.append(dict(
            tc=tc,
            variant=name,
            lr=alpha,
            accs=[round(a, 6) for a in accs],
            losses=[round(l, 2) for l in losses],
            final_acc=accs[-1],
            pretrain_s=round(pretrain_s, 1),
            finetune_s=round(finetune_s, 1),
            total_s=round(pretrain_s + finetune_s, 1),
            mean_cos_e0_e5=round(mean_cos(x0, x5), 4),
        ))
        emb_arrays[f"{name}_x0"] = x0
        emb_arrays[f"{name}_x5"] = x5
        print(
            f"  [{time.time()-t_start:6.1f}s] {name:>11}  "
            f"acc=[{' '.join(f'{a:.3f}' for a in accs)}]  "
            f"pre={pretrain_s:.1f}s fine={finetune_s:.1f}s",
            flush=True,
        )

    # --- vanilla -----------------------------------------------------------
    t0 = time.time()
    model = new_skipgram_model(
        dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
        seed=CFG["seed"], workers=CFG["workers"],
    )
    model.build_vocab(LineSentence(str(inst_walks)))
    vocab_s = time.time() - t0
    x0 = snapshot_test_embeddings(model, test_tokens)
    accs, losses, train_s = train_tracked(
        model, inst_walks, epochs=CFG["epochs"],
        alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
    )
    x5 = snapshot_test_embeddings(model, test_tokens)
    record("vanilla", accs, losses, 0.0, vocab_s + train_s, CFG["vanilla_alpha"], x0, x5)

    # --- pretrained codes per protograph (shared across classic/bound) ------
    stage1_cache: dict[str, tuple[dict, float, float]] = {}

    def stage1(kind: str):
        if kind not in stage1_cache:
            t0 = time.time()
            pre = pretrain_protograph(
                proto_walks[kind], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
                seed=CFG["seed"], workers=CFG["workers"],
            )
            pretrain_s = time.time() - t0
            vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
            stage1_cache[kind] = (vecs, used_norm, pretrain_s)
        return stage1_cache[kind]

    def run_initialized(name, kind, *, bound: bool, alpha: float):
        vecs, used_norm, pretrain_s = stage1(kind)
        bound_vectors = None
        if bound:
            bound_vectors = concept_bound_vectors(
                p["graph"], p["ontology"], vecs, target_norm=used_norm,
            )
        t0 = time.time()
        model = new_skipgram_model(
            dim=CFG["dim"], alpha=alpha, min_alpha=CFG["min_alpha"],
            seed=CFG["seed"], workers=CFG["workers"],
        )
        model.build_vocab(LineSentence(str(inst_walks)))
        protograph_init(
            model, vecs, p["ontology"], strategy="all_init",
            target_norm=used_norm, bound_vectors=bound_vectors,
        )
        setup_s = time.time() - t0
        x0 = snapshot_test_embeddings(model, test_tokens)
        accs, losses, train_s = train_tracked(
            model, inst_walks, epochs=CFG["epochs"],
            alpha=alpha, min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
        )
        x5 = snapshot_test_embeddings(model, test_tokens)
        record(name, accs, losses, pretrain_s, setup_s + train_s, alpha, x0, x5)

    # p1_classic at the from-scratch LR 0.025 (user-requested control config)
    run_initialized("p1_classic", "p1", bound=False, alpha=CFG["vanilla_alpha"])
    run_initialized("p2_classic", "p2", bound=False, alpha=CFG["finetune_alpha"])
    for kind in ("p1", "p2", "p3"):
        run_initialized(f"{kind}_bound", kind, bound=True, alpha=CFG["finetune_alpha"])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **emb_arrays)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"{tc}: done in {time.time()-t_start:.1f}s -> {out_json}", flush=True)


def main():
    tcs = sys.argv[1:] or TCS
    print(f"Running {len(tcs)} test cases: {', '.join(tcs)}", flush=True)
    for tc in tcs:
        print(f"{tc}:", flush=True)
        run_tc(tc)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
