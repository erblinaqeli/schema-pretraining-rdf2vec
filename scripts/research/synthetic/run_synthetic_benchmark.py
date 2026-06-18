"""
Full synthetic benchmark runner.

Re-runs the synthetic DLCC benchmark for 7 variants on every test case tc01-tc15
and records, per variant:

  * per-epoch LogReg test accuracy  (epoch 0 = right after initialization)
  * per-epoch SGNS finetune loss
  * per-epoch entity-embedding checkpoint (wv.vectors, vectors-only)
  * stage runtimes:  Pretrain, Init, Finetuning, Total
        - Pretrain   : protograph SGNS pretrain + stage1 normalization (0 for vanilla)
        - Init       : building the finetune init from the pretrained codes
                       (concept-bound vector build for *_bound + protograph_init);
                       0 for vanilla
        - Finetuning : vocab build + instance-walk SGNS training
                       (per-epoch eval + checkpoint-save overhead subtracted)
        - Total      : Pretrain + Init + Finetuning

Variants (7):
    vanilla
    p1_classic, p2_classic, p3_classic     -> finetune LR 0.025  (user-requested)
    p1_bound,   p2_bound,   p3_bound       -> finetune LR 0.0025 (protected init)

Outputs (resumable; delete a variant's dir to re-run it):
  output/synthetic_benchmark/<tc>/<variant>/metrics.json
  output/synthetic_benchmark/<tc>/<variant>/keys.json        (vocab, written once)
  output/synthetic_benchmark/<tc>/<variant>/ckpt_epoch00.npz ... ckpt_epoch05.npz
  output/synthetic_benchmark/summary.json                    (all rows flattened)
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

from _evaluate import load_labeled_txt  # noqa: E402
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

OUT_ROOT = ROOT / "output" / "synthetic_benchmark"
WALK_ROOT = ROOT / "notebooks" / "synthetic_compare"  # reuse cached walks

TCS = [f"tc{i:02d}" for i in range(1, 16)]  # tc01 .. tc15

CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    classic_alpha=0.025,    # all *_classic variants (user request)
    bound_alpha=0.0025,     # protected LR for *_bound variants
    vanilla_alpha=0.025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
    workers=16,
)

# (name, protograph kind, bound?, finetune alpha)
VARIANTS = [
    ("p1_classic", "p1", False, CFG["classic_alpha"]),
    ("p2_classic", "p2", False, CFG["classic_alpha"]),
    ("p3_classic", "p3", False, CFG["classic_alpha"]),
    ("p1_bound", "p1", True, CFG["bound_alpha"]),
    ("p2_bound", "p2", True, CFG["bound_alpha"]),
    ("p3_bound", "p3", True, CFG["bound_alpha"]),
]


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        walks=WALK_ROOT / tc,
    )


def save_ckpt(model, path: Path) -> None:
    """Save entity vectors only (vocab order matches keys.json)."""
    np.savez_compressed(path, vectors=np.asarray(model.wv.vectors, dtype=np.float32))


class _EpochTracker(CallbackAny2Vec):
    """Per-epoch accuracy, per-epoch SGNS loss, and per-epoch vector checkpoint.

    Eval + checkpoint wall-time is accumulated so it can be subtracted from the
    measured finetune time.
    """

    def __init__(self, eval_fn, ckpt_dir: Path, accs: list, losses: list, ckpts: list):
        self.eval_fn = eval_fn
        self.ckpt_dir = ckpt_dir
        self.accs = accs
        self.losses = losses
        self.ckpts = ckpts
        self.overhead_seconds = 0.0
        self._prev_cum = 0.0
        self._epoch = 0

    def on_epoch_end(self, model):
        self._epoch += 1
        cum = float(model.get_latest_training_loss())
        self.losses.append(cum - self._prev_cum)
        self._prev_cum = cum
        t0 = time.time()
        self.accs.append(self.eval_fn(model))
        path = self.ckpt_dir / f"ckpt_epoch{self._epoch:02d}.npz"
        save_ckpt(model, path)
        self.ckpts.append(path.name)
        self.overhead_seconds += time.time() - t0


def finetune_tracked(model, walks_path, ckpt_dir, *, epochs, alpha, min_alpha, eval_fn):
    """Train with per-epoch acc/loss/checkpoint tracking.

    epoch 0 acc + ckpt are recorded before training. Returns
    (accs, losses, ckpts, train_seconds_excl_overhead).
    """
    accs = [eval_fn(model)]
    save_ckpt(model, ckpt_dir / "ckpt_epoch00.npz")
    ckpts = ["ckpt_epoch00.npz"]
    losses: list = []
    tracker = _EpochTracker(eval_fn, ckpt_dir, accs, losses, ckpts)
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
    train_s = time.time() - t0 - tracker.overhead_seconds
    return accs, losses, ckpts, train_s


def write_metrics(var_dir: Path, row: dict, keys: list[str]) -> None:
    (var_dir / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    (var_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def run_tc(tc: str) -> None:
    p = tc_paths(tc)
    p["walks"].mkdir(parents=True, exist_ok=True)

    # --- walks (cached on disk; regenerated only if missing) ---------------
    proto_paths = write_protographs(p["ontology"], p["walks"])
    proto_walks = {
        kind: ensure_walks(
            path,
            p["walks"] / f"walks_{kind}.txt",
            walks_per_entity=CFG["proto_walks_per_entity"],
            depth=CFG["depth"],
            seed=CFG["seed"],
            ensure_triple_coverage=True,
        )
        for kind, path in proto_paths.items()
    }
    inst_walks = ensure_walks(
        p["graph"],
        p["walks"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"],
        depth=CFG["depth"],
        seed=CFG["seed"],
    )

    eval_fn = make_eval_fn(p["train"], p["test"])
    t_start = time.time()

    def maybe_run(name: str, fn) -> None:
        var_dir = OUT_ROOT / tc / name
        if (var_dir / "metrics.json").is_file():
            print(f"  {name:>11}: cached, skipping", flush=True)
            return
        var_dir.mkdir(parents=True, exist_ok=True)
        row = fn(var_dir)
        write_metrics(var_dir, row, list_keys)
        print(
            f"  [{time.time()-t_start:6.1f}s] {name:>11}  "
            f"acc=[{' '.join(f'{a:.3f}' for a in row['accs'])}]  "
            f"pre={row['pretrain_s']:.1f}s init={row['init_s']:.1f}s "
            f"fine={row['finetune_s']:.1f}s total={row['total_s']:.1f}s",
            flush=True,
        )

    # vocab is identical across variants (same instance corpus); capture once.
    _probe = new_skipgram_model(
        dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
        seed=CFG["seed"], workers=CFG["workers"],
    )
    _probe.build_vocab(LineSentence(str(inst_walks)))
    list_keys = list(_probe.wv.index_to_key)
    del _probe

    # --- vanilla -----------------------------------------------------------
    def run_vanilla(var_dir: Path) -> dict:
        t0 = time.time()
        model = new_skipgram_model(
            dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
            seed=CFG["seed"], workers=CFG["workers"],
        )
        model.build_vocab(LineSentence(str(inst_walks)))
        vocab_s = time.time() - t0
        accs, losses, ckpts, train_s = finetune_tracked(
            model, inst_walks, var_dir, epochs=CFG["epochs"],
            alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
        )
        return _row(tc, "vanilla", "vanilla", False, CFG["vanilla_alpha"],
                    accs, losses, ckpts,
                    pretrain_s=0.0, init_s=0.0, finetune_s=vocab_s + train_s)

    maybe_run("vanilla", run_vanilla)

    # --- pretrained codes per protograph (shared across classic/bound) -----
    stage1_cache: dict[str, tuple[dict, float, float]] = {}

    def stage1(kind: str):
        if kind not in stage1_cache:
            t0 = time.time()
            pre = pretrain_protograph(
                proto_walks[kind], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
                seed=CFG["seed"], workers=CFG["workers"],
            )
            vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
            pretrain_s = time.time() - t0
            stage1_cache[kind] = (vecs, used_norm, pretrain_s)
        return stage1_cache[kind]

    def make_initialized(name, kind, bound, alpha):
        def run(var_dir: Path) -> dict:
            vecs, used_norm, pretrain_s = stage1(kind)
            # --- Init: build the finetune init from pretrained codes -------
            t_init = time.time()
            bound_vectors = None
            if bound:
                bound_vectors = concept_bound_vectors(
                    p["graph"], p["ontology"], vecs, target_norm=used_norm,
                )
            model = new_skipgram_model(
                dim=CFG["dim"], alpha=alpha, min_alpha=CFG["min_alpha"],
                seed=CFG["seed"], workers=CFG["workers"],
            )
            t_vocab = time.time()
            model.build_vocab(LineSentence(str(inst_walks)))
            vocab_s = time.time() - t_vocab
            protograph_init(
                model, vecs, p["ontology"], strategy="all_init",
                target_norm=used_norm, bound_vectors=bound_vectors,
            )
            init_s = time.time() - t_init - vocab_s  # concept-bound build + protograph_init
            # --- Finetune --------------------------------------------------
            accs, losses, ckpts, train_s = finetune_tracked(
                model, inst_walks, var_dir, epochs=CFG["epochs"],
                alpha=alpha, min_alpha=CFG["min_alpha"], eval_fn=eval_fn,
            )
            return _row(tc, name, kind, bound, alpha, accs, losses, ckpts,
                        pretrain_s=pretrain_s, init_s=init_s,
                        finetune_s=vocab_s + train_s)
        return run

    for name, kind, bound, alpha in VARIANTS:
        maybe_run(name, make_initialized(name, kind, bound, alpha))

    print(f"{tc}: done in {time.time()-t_start:.1f}s", flush=True)


def _row(tc, variant, kind, bound, lr, accs, losses, ckpts,
         *, pretrain_s, init_s, finetune_s) -> dict:
    total_s = pretrain_s + init_s + finetune_s
    return dict(
        tc=tc,
        variant=variant,
        kind=kind,
        bound=bound,
        lr=lr,
        accs=[round(a, 6) for a in accs],
        losses=[round(l, 2) for l in losses],
        final_acc=round(accs[-1], 6),
        checkpoints=ckpts,
        pretrain_s=round(pretrain_s, 2),
        init_s=round(init_s, 2),
        finetune_s=round(finetune_s, 2),
        total_s=round(total_s, 2),
    )


def write_summary() -> None:
    rows = []
    for mf in sorted(OUT_ROOT.glob("tc*/*/metrics.json")):
        rows.append(json.loads(mf.read_text(encoding="utf-8")))
    (OUT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {len(rows)} variant-runs -> {OUT_ROOT/'summary.json'}", flush=True)


def main():
    tcs = [a for a in sys.argv[1:] if not a.startswith("-")] or TCS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(tcs)} test cases: {', '.join(tcs)}", flush=True)
    for tc in tcs:
        print(f"{tc}:", flush=True)
        run_tc(tc)
    write_summary()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
