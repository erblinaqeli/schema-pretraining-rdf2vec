"""No-norm synthetic-benchmark sweep: vanilla + p1/p2/p3 classic (OLD raw-mean init), tc01-tc15.

Identical pipeline & OUTPUT FORMAT to run_synthetic_benchmark.py (reuses its
finetune_tracked / write_metrics / _row), so every variant dir gets:
    keys.json
    ckpt_epoch00.npz ... ckpt_epoch05.npz   (entity vectors per epoch)
    metrics.json                            (accs, losses, checkpoints, timings)

The ONLY change vs the benchmark is the *_classic init:
    benchmark : normalized_stage1_vectors(target_norm=8) + protograph_init(mode="both", renormalize=True)
    here      : raw pretrain vectors           + protograph_init(mode="vectors_only", renormalize=False)
i.e. old mina_resume_graphtrain.py raw np.mean(class_vecs) into wv.vectors only.

Bounds are excluded (they are intrinsically norm-based). Resumable: skips any
variant whose metrics.json already exists. Output dir is SEPARATE from the norm
benchmark so nothing is clobbered.

Usage:  uv run python scripts/run_synth_nonorm_sweep.py [tc01 tc05 ...]
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models.word2vec import LineSentence

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

# reuse the benchmark's exact tracking/output machinery
from run_synthetic_benchmark import CFG, tc_paths, finetune_tracked, write_metrics, _row  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    ensure_walks, make_eval_fn, new_skipgram_model, pretrain_protograph,
    protograph_init, write_protographs,
)

OUT_ROOT = ROOT / "output" / "synthetic_benchmark_nonorm"
TCS = [f"tc{i:02d}" for i in range(1, 16)]
KINDS = ("p1", "p2", "p3")
COLORS = {"vanilla": "black", "p1": "#c026d3", "p2": "#ea580c", "p3": "#2563eb"}


def run_tc(tc: str):
    p = tc_paths(tc)
    p["walks"].mkdir(parents=True, exist_ok=True)
    proto_paths = write_protographs(p["ontology"], p["walks"])
    proto_walks = {k: ensure_walks(proto_paths[k], p["walks"] / f"walks_{k}.txt",
                                   walks_per_entity=CFG["proto_walks_per_entity"], depth=CFG["depth"],
                                   seed=CFG["seed"], ensure_triple_coverage=True) for k in KINDS}
    inst_walks = ensure_walks(p["graph"],
                              p["walks"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
                              walks_per_entity=CFG["walks_per_entity"], depth=CFG["depth"], seed=CFG["seed"])
    eval_fn = make_eval_fn(p["train"], p["test"])

    # vocab keys (identical across variants) — written into each variant's keys.json
    probe = new_skipgram_model(dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
                               seed=CFG["seed"], workers=CFG["workers"])
    probe.build_vocab(LineSentence(str(inst_walks)))
    list_keys = list(probe.wv.index_to_key)
    del probe

    def maybe_run(name, fn):
        var_dir = OUT_ROOT / tc / name
        if (var_dir / "metrics.json").is_file():
            print(f"  {tc} {name} cached", flush=True); return
        var_dir.mkdir(parents=True, exist_ok=True)
        row = fn(var_dir)
        write_metrics(var_dir, row, list_keys)   # writes keys.json + metrics.json
        print(f"  {tc} {name} done  acc={row['accs'][-1]:.3f}", flush=True)

    # --- vanilla (no protograph init) ---
    def run_vanilla(var_dir):
        t0 = time.time()
        model = new_skipgram_model(dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
                                   seed=CFG["seed"], workers=CFG["workers"])
        model.build_vocab(LineSentence(str(inst_walks)))
        vocab_s = time.time() - t0
        accs, losses, ckpts, train_s = finetune_tracked(
            model, inst_walks, var_dir, epochs=CFG["epochs"],
            alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn)
        return _row(tc, "vanilla", "vanilla", False, CFG["vanilla_alpha"], accs, losses, ckpts,
                    pretrain_s=0.0, init_s=0.0, finetune_s=vocab_s + train_s)
    maybe_run("vanilla", run_vanilla)

    # --- p1/p2/p3 classic, NO-NORM raw-mean init ---
    stage1_cache: dict = {}

    def stage1(kind):
        if kind not in stage1_cache:
            t0 = time.time()
            pre = pretrain_protograph(proto_walks[kind], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
                                      seed=CFG["seed"], workers=CFG["workers"])
            raw_vecs = {tok: np.asarray(pre.wv[tok], dtype=np.float32) for tok in pre.wv.index_to_key}
            stage1_cache[kind] = (raw_vecs, time.time() - t0)
        return stage1_cache[kind]

    def make_classic(name, kind):
        def run(var_dir):
            raw_vecs, pretrain_s = stage1(kind)
            t_init = time.time()
            model = new_skipgram_model(dim=CFG["dim"], alpha=CFG["classic_alpha"], min_alpha=CFG["min_alpha"],
                                       seed=CFG["seed"], workers=CFG["workers"])
            t_vocab = time.time()
            model.build_vocab(LineSentence(str(inst_walks)))
            vocab_s = time.time() - t_vocab
            protograph_init(model, raw_vecs, p["ontology"], strategy="all_init",
                            target_norm=None, renormalize=False, mode="vectors_only")
            init_s = time.time() - t_init - vocab_s
            accs, losses, ckpts, train_s = finetune_tracked(
                model, inst_walks, var_dir, epochs=CFG["epochs"],
                alpha=CFG["classic_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn)
            return _row(tc, name, kind, False, CFG["classic_alpha"], accs, losses, ckpts,
                        pretrain_s=pretrain_s, init_s=init_s, finetune_s=vocab_s + train_s)
        return run

    for kind in KINDS:
        maybe_run(f"{kind}_classic", make_classic(f"{kind}_classic", kind))


def plot_tc(tc: str):
    ep = range(1, CFG["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, kind in [("vanilla", "vanilla")] + [(f"{k}_classic", k) for k in KINDS]:
        mf = OUT_ROOT / tc / variant / "metrics.json"
        if not mf.is_file():
            continue
        ls = json.loads(mf.read_text())["losses"]
        style = dict(linestyle="--", marker="s") if kind == "vanilla" else dict(linestyle="-", marker="o")
        ax.plot(ep, ls, color=COLORS[kind], lw=2, label=variant, **style)
    ax.set_title(f"{tc} — SGNS finetune loss per epoch (no-norm raw-mean init)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SGNS loss (per-epoch)")
    ax.set_xticks(list(ep)); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_ROOT / f"{tc}_loss.png", dpi=150); plt.close(fig)


def main():
    tcs = [a for a in sys.argv[1:] if not a.startswith("-")] or TCS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for tc in tcs:
        print(f"{tc}: ({time.time()-t0:.0f}s elapsed)", flush=True)
        try:
            run_tc(tc)
            plot_tc(tc)
        except Exception:
            print(f"  {tc} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
