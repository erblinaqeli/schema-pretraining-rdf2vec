"""most_specific sweep: vanilla + p1/p2/p3 classic, BOTH norm=8 and no-norm, tc01-tc15.

strategy = "most_specific" (NO ancestor fallback, unlike the all_init runs).
Two configs share the per-(tc,kind) pretrain step:
  ms_norm   : normalized_stage1_vectors(target_norm=8) + protograph_init(renormalize=True,  mode="both")
  ms_nonorm : raw pretrain vectors                     + protograph_init(renormalize=False, mode="vectors_only")

Full parity with synthetic_benchmark output (keys.json + ckpt_epoch00-05.npz + metrics.json).
Resumable (skips variants whose metrics.json exists). Bounds excluded. Originals untouched.

Usage: uv run python scripts/run_synth_ms_sweep.py [tc01 tc05 ...]
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

from run_synthetic_benchmark import CFG, tc_paths, finetune_tracked, write_metrics, _row  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    ensure_walks, make_eval_fn, new_skipgram_model, normalized_stage1_vectors,
    pretrain_protograph, protograph_init, write_protographs,
)

STRATEGY = "most_specific"
DIR_NORM = ROOT / "output" / "synthetic_benchmark_ms_norm"
DIR_NONORM = ROOT / "output" / "synthetic_benchmark_ms_nonorm"
CONFIGS = [("norm", DIR_NORM), ("nonorm", DIR_NONORM)]
TCS = [f"tc{i:02d}" for i in range(1, 16)]
KINDS = ("p1", "p2")  # p3 omitted: most_specific == all_init for P3 (no OOV classes) -> copy it
COLORS = {"vanilla": "black", "p1_classic": "#c026d3", "p2_classic": "#ea580c", "p3_classic": "#2563eb"}


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

    probe = new_skipgram_model(dim=CFG["dim"], alpha=CFG["vanilla_alpha"], min_alpha=CFG["min_alpha"],
                               seed=CFG["seed"], workers=CFG["workers"])
    probe.build_vocab(LineSentence(str(inst_walks)))
    list_keys = list(probe.wv.index_to_key)
    del probe

    def finish(var_dir, row):
        write_metrics(var_dir, row, list_keys)
        print(f"  {tc} {var_dir.parent.name}/{var_dir.name} done acc={row['accs'][-1]:.3f}", flush=True)

    # vanilla & p3 are intentionally NOT run here (copied from existing runs:
    # vanilla is init-independent; p3's most_specific == all_init since P3 has no OOV classes).

    # --- classic: pretrain ONCE per kind, derive both norm & no-norm most_specific inits ---
    for kind in KINDS:
        name = f"{kind}_classic"
        need = [(c, base) for c, base in CONFIGS if not (base / tc / name / "metrics.json").is_file()]
        if not need:
            print(f"  {tc} {name} cached (both)", flush=True); continue
        t_pre = time.time()
        pre = pretrain_protograph(proto_walks[kind], dim=CFG["dim"], epochs=CFG["pretrain_epochs"],
                                  seed=CFG["seed"], workers=CFG["workers"])
        pretrain_s = time.time() - t_pre

        for cfg, base in need:
            vd = base / tc / name
            vd.mkdir(parents=True, exist_ok=True)
            t_init = time.time()
            if cfg == "norm":
                vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
                init_kw = dict(target_norm=used_norm, renormalize=True, mode="both")
            else:
                vecs = {tok: np.asarray(pre.wv[tok], dtype=np.float32) for tok in pre.wv.index_to_key}
                init_kw = dict(target_norm=None, renormalize=False, mode="vectors_only")
            model = new_skipgram_model(dim=CFG["dim"], alpha=CFG["classic_alpha"], min_alpha=CFG["min_alpha"],
                                       seed=CFG["seed"], workers=CFG["workers"])
            t_vocab = time.time()
            model.build_vocab(LineSentence(str(inst_walks)))
            vocab_s = time.time() - t_vocab
            protograph_init(model, vecs, p["ontology"], strategy=STRATEGY, **init_kw)
            init_s = time.time() - t_init - vocab_s
            accs, losses, ckpts, train_s = finetune_tracked(
                model, inst_walks, vd, epochs=CFG["epochs"],
                alpha=CFG["classic_alpha"], min_alpha=CFG["min_alpha"], eval_fn=eval_fn)
            finish(vd, _row(tc, name, kind, False, CFG["classic_alpha"], accs, losses, ckpts,
                            pretrain_s=pretrain_s, init_s=init_s, finetune_s=vocab_s + train_s))


def plot_tc(tc: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for (cfg, base), ax in zip(CONFIGS, axes):
        for variant in ["vanilla"] + [f"{k}_classic" for k in KINDS]:
            mf = base / tc / variant / "metrics.json"
            if not mf.is_file():
                continue
            ls = json.loads(mf.read_text())["losses"]; ep = range(1, len(ls) + 1)
            style = dict(linestyle="--", marker="s") if variant == "vanilla" else dict(linestyle="-", marker="o")
            ax.plot(ep, ls, color=COLORS[variant], lw=2, label=variant, **style)
        ax.set_title(f"{tc} — most_specific, {'norm=8' if cfg=='norm' else 'no-norm'}")
        ax.set_xlabel("Epoch"); ax.set_xticks(list(range(1, 6))); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel("SGNS loss (per-epoch)")
    fig.tight_layout()
    (ROOT / "output").mkdir(exist_ok=True)
    fig.savefig(DIR_NONORM.parent / f"{tc}_ms_loss_norm_vs_nonorm.png", dpi=150); plt.close(fig)


def main():
    tcs = [a for a in sys.argv[1:] if not a.startswith("-")] or TCS
    for _, base in CONFIGS:
        base.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for tc in tcs:
        print(f"{tc}: ({time.time()-t0:.0f}s elapsed)", flush=True)
        try:
            run_tc(tc)
            plot_tc(tc)
        except Exception:
            print(f"  {tc} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
