#!/usr/bin/env python3
"""
Train Word2Vec with Gensim (fast CPU implementation).

Modes (--mode):
  none — Train on a single walks file (default).
  p1, p2 — Two-stage RDF2Vec: pretrain on protograph P1 or P2 walks, then finetune on
           instance walks from a file you supply (MASCHInE-style init when an ontology is available).
           Class-vector aggregation is configurable (--maschine-strategy).

Input: one space-separated walk per line.

During ``build_vocab``, Gensim INFO lines (sentence/word counts) go to stderr by default;
use ``--no-vocab-progress`` or ``--vocab-progress-per`` to tune.

Single-corpus mode logs per-epoch loss, CSV, PNG, optional per-step loss (--loss-every-steps).
p1/p2 mode writes pretrain_per_epoch.png and protograph_per_epoch.png; per-step PNGs use
--loss-every-steps-pretrain / --loss-every-steps-finetune (or --loss-every-steps N for both).
Without MASCHInE, optional predicate embedding transfer from stage 1 uses --transfer-predicate-embeddings (default on).
Writes instance_to_class.json (default): ``{ instance_iri_inner: [ class_iri_inner, ... ] }`` from the ontology file (.nt or .ttl).
Saves a .pt checkpoint compatible with evaluate_embeddings.py (embeddings + word2idx).

Reproducibility: pass ``--seed`` to fix Python/NumPy/torch RNGs. Gensim multi-threaded SGD remains
nondeterministic across runs when ``workers`` > 1.

Parallelism: ``workers`` are **in-process threads** (shared memory).  ``workers=1`` is never used
(minimum 2 when the machine has at least 2 logical CPUs).  Hyperparameter sweeps should
raise ``--jobs`` (concurrent training **processes**) before raising ``workers``; each subprocess
duplicates model RSS, threads do not.
``PYTHONHASHSEED`` is fixed only if set in the environment before starting the interpreter.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from contextlib import nullcontext
import os
import random
import sys
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from tqdm.auto import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _maschine_init import (
    INIT_STRATEGIES,
    apply_maschine_initialization,
    compute_finetune_init_stats,
    compute_instance_init_coverage,
    compute_relation_init_coverage,
    instance_class_mapping_from_ontology,
    normalize_init_strategy,
    snapshot_class_initialized_entity_vectors,
    transfer_predicate_embeddings_from_pretrained,
    user_facing_init_strategy,
    vocab_inner_iris,
    write_finetune_init_stats,
    write_finetune_token_init,
)
from _kg_io import RuntimeMetrics, write_runtime_metrics


def _workers_effective(workers: int, *, cpu_budget: int | None = None) -> int:
    """Resolve Gensim worker-thread count (0 = half of logical CPUs). Never returns 1."""
    cpus = os.cpu_count() or 1
    if cpu_budget is not None:
        cpus = max(1, min(cpus, cpu_budget))
    default_workers = max(2, cpus // 2) if cpus >= 2 else cpus
    if workers > 0:
        if workers == 1:
            raise SystemExit("--workers must be >= 2 (workers=1 is not allowed)")
        return workers
    return default_workers


def resolve_workers_for_sweep(
    workers: int,
    *,
    parallel_jobs: int,
    cpu_budget: int | None = None,
) -> int:
    """Pick Gensim ``workers`` for one training subprocess given sweep-level job count.

    Gensim ``workers`` are **threads inside one process** (shared RSS).  Sweep ``parallel_jobs``
    are **separate subprocesses** (RSS scales ~linearly with job count).  When only one job runs
    at a time, use all cores (``workers=0``).  With multiple concurrent jobs, split the CPU
    budget so ``parallel_jobs * threads ≈ cpu_count`` without oversubscribing.
    """
    cpus = os.cpu_count() or 1
    if cpu_budget is not None:
        cpus = max(1, min(cpus, cpu_budget))
    if workers > 0:
        return workers
    if parallel_jobs <= 1:
        return 0
    per_job = cpus // parallel_jobs
    return max(2, per_job) if cpus >= 2 else per_job


def _set_training_rng(seed: int) -> None:
    """Seed stdlib, NumPy, and PyTorch RNGs (Gensim Word2Vec uses NumPy for sampling)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Word2VecWithStepLoss(Word2Vec):
    """Hooks Gensim's per-job training to log loss on a step grid.

    A "step" is one :meth:`~gensim.models.word2vec.Word2Vec._do_train_job` call (a chunk
    of sentences up to ``batch_words``). Accurate per-step loss needs sequential jobs
    (single-threaded training); with ``workers`` > 1 the curve is approximate.

    Samples are recorded at the **first** job, every ``step_loss_interval`` jobs thereafter,
    plus one :class:`StepLossEndFlush` callback appends the **final** job if it was not
    already sampled (so short runs still get a usable curve).
    """

    def __init__(self, *args, step_loss_interval: int = 0, step_loss_quiet: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_loss_interval = step_loss_interval
        self.step_loss_quiet = step_loss_quiet
        self.train_epoch_idx: int = 0
        self.step_loss_rows: list[tuple[int, int, float, float]] = []
        self._job_step: int = 0
        self._cum_at_last_interval: float = 0.0

    def _do_train_job(self, sentences, alpha, inits):  # type: ignore[no-untyped-def]
        tally, raw_tally = super()._do_train_job(sentences, alpha, inits)
        if self.step_loss_interval <= 0:
            return tally, raw_tally
        self._job_step += 1
        cum = float(self.running_training_loss)
        n = self.step_loss_interval
        log_this = self._job_step == 1 or (self._job_step % n == 0)
        if log_this:
            interval = cum - self._cum_at_last_interval
            self.step_loss_rows.append(
                (
                    self._job_step,
                    self.train_epoch_idx,
                    cum,
                    interval,
                )
            )
            if not self.step_loss_quiet:
                tqdm.write(
                    f"[step loss] step={self._job_step} epoch={self.train_epoch_idx} "
                    f"cumulative={cum:.6f} interval={interval:.6f}",
                    file=sys.stdout,
                )
            self._cum_at_last_interval = cum
        return tally, raw_tally


class StepLossEndFlush(CallbackAny2Vec):
    """Append one step-loss sample at the last training job if it was not already logged."""

    def on_train_end(self, model: Word2Vec) -> None:
        if not isinstance(model, Word2VecWithStepLoss):
            return
        m = model
        if m.step_loss_interval <= 0 or m._job_step == 0:
            return
        last_logged = m.step_loss_rows[-1][0] if m.step_loss_rows else 0
        if last_logged == m._job_step:
            return
        cum = float(model.get_latest_training_loss())
        interval = cum - m._cum_at_last_interval
        m.step_loss_rows.append((m._job_step, m.train_epoch_idx, cum, interval))
        if not m.step_loss_quiet:
            tqdm.write(
                f"[step loss] step={m._job_step} epoch={m.train_epoch_idx} "
                f"cumulative={cum:.6f} interval={interval:.6f} (end flush)",
                file=sys.stdout,
            )
        m._cum_at_last_interval = cum


class EpochTagCallback(CallbackAny2Vec):
    """Sets ``model.train_epoch_idx`` at each epoch start (for step-loss CSV)."""

    def __init__(self, model: Word2VecWithStepLoss) -> None:
        self._model = model
        self._epoch = 0

    def on_epoch_begin(self, model: Word2Vec) -> None:
        self._epoch += 1
        self._model.train_epoch_idx = self._epoch


class EpochLossProgress(CallbackAny2Vec):
    """Per-epoch loss (from cumulative training loss) and tqdm over epochs."""

    def __init__(self, total_epochs: int, desc: str, *, echo: bool = False) -> None:
        self.total_epochs = total_epochs
        self.desc = desc
        self.echo = echo
        self.prev_cumulative = 0.0
        self.epoch_losses: list[float] = []
        self._pbar: tqdm | None = None

    def on_train_begin(self, model: Word2Vec) -> None:
        # Stage-2 finetune reuses the same model after pretrain; baseline cumulative
        # loss so per-epoch deltas exclude prior training stages.
        self.prev_cumulative = float(model.get_latest_training_loss())
        self._pbar = tqdm(
            total=self.total_epochs,
            desc=self.desc,
            unit="epoch",
            dynamic_ncols=True,
            colour="green",
            file=sys.stdout,
        )

    def on_epoch_end(self, model: Word2Vec) -> None:
        cumulative = float(model.get_latest_training_loss())
        epoch_loss = cumulative - self.prev_cumulative
        self.prev_cumulative = cumulative
        self.epoch_losses.append(epoch_loss)
        if self._pbar is not None:
            self._pbar.set_postfix(loss=f"{epoch_loss:.4f}", refresh=False)
            self._pbar.update(1)
        if self.echo:
            tqdm.write(
                f"[{self.desc}] epoch {len(self.epoch_losses)}/{self.total_epochs} "
                f"loss={epoch_loss:.6f}",
                file=sys.stdout,
            )

    def on_train_end(self, model: Word2Vec) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def finetune_epoch_checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    """Path for finetune-stage checkpoint (epoch 0 = before finetune training)."""
    return checkpoint_dir / f"finetune_epoch_{epoch:02d}.pt"


def save_word2vec_checkpoint(
    model: Word2Vec,
    path: Path,
    *,
    training_seed: int | None,
    finetune_epoch: int,
) -> None:
    ckpt = build_gensim_to_pt(model, training_seed=training_seed)
    ckpt["finetune_epoch"] = int(finetune_epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


class FinetuneEpochCheckpointCallback(CallbackAny2Vec):
    """Save a .pt checkpoint after each finetune epoch (1 .. N)."""

    def __init__(self, checkpoint_dir: Path, *, training_seed: int | None) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.training_seed = training_seed
        self._finetune_epoch = 0

    def on_epoch_end(self, model: Word2Vec) -> None:
        self._finetune_epoch += 1
        path = finetune_epoch_checkpoint_path(self.checkpoint_dir, self._finetune_epoch)
        save_word2vec_checkpoint(
            model,
            path,
            training_seed=self.training_seed,
            finetune_epoch=self._finetune_epoch,
        )
        print(f"Wrote finetune epoch checkpoint: {path}", flush=True)


class InstanceAnchoringCallback(CallbackAny2Vec):
    """After each finetune epoch, pull class-init entity vectors toward MASCHInE init.

    Anchoring is skipped after the final epoch so the saved model keeps the last
    epoch's learned vectors without a post-hoc blend toward initialization.
    """

    def __init__(
        self,
        init_vectors: dict[str, np.ndarray],
        alpha: float,
        total_epochs: int,
    ) -> None:
        self.init_vectors = init_vectors
        self.alpha = alpha
        self.total_epochs = total_epochs
        self._epoch = 0

    def on_epoch_end(self, model: Word2Vec) -> None:
        self._epoch += 1
        if self.alpha <= 0.0 or self._epoch >= self.total_epochs:
            return
        wv = model.wv
        b, a = 1.0 - self.alpha, self.alpha
        for token, v_init in self.init_vectors.items():
            idx = wv.key_to_index[token]
            wv.vectors[idx] = b * wv.vectors[idx] + a * v_init


def resolve_finetune_epoch_checkpoint_dir(args: argparse.Namespace) -> Path | None:
    if not args.save_finetune_epoch_checkpoints:
        return None
    if args.finetune_epoch_checkpoint_dir is not None:
        return args.finetune_epoch_checkpoint_dir
    if args.mode == "none":
        return args.output.parent / "ckpt"
    return args.out_dir / "ckpt"


def save_step_loss_csv(path: Path, rows: list[tuple[int, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "cumulative_loss", "interval_loss"])
        for step, epoch, cum, interval in rows:
            w.writerow([step, epoch, f"{cum:.8f}", f"{interval:.8f}"])


def save_loss_csv(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        for i, loss in enumerate(losses, start=1):
            w.writerow([i, f"{loss:.8f}"])


def _plot_loss_per_epoch(
    path: Path,
    losses: list[float],
) -> None:
    if not losses:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        list(epochs),
        losses,
        marker="o",
        markersize=4.5,
        markeredgewidth=1.2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_per_step(
    path: Path,
    rows: list[tuple[int, int, float, float]],
    *,
    title: str,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [r[0] for r in rows]
    cumulative = [r[2] for r in rows]
    interval = [r[3] for r in rows]
    smax = max(steps)
    x_right = max(smax * 1.02, smax + 1.0)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax0.plot(steps, cumulative, marker="o", markersize=2)
    ax0.set_ylabel("Cumulative loss")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(0.0, x_right)
    ax1.plot(steps, interval, marker="o", markersize=2, color="C1")
    ax1.set_xlabel("Step (batch index)")
    ax1.set_ylabel("Interval loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0, x_right)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_instance_to_class_json(
    path: Path,
    ontology_path: Path | None,
    *,
    instance_types_path: Path | None = None,
    subject_filter: set[str] | None = None,
) -> None:
    """Write ``{ instance_iri_inner: [ class_iri_inner, ... ] }`` from ontology (MASCHInE most-specific types)."""
    if ontology_path is None or not ontology_path.is_file():
        payload: dict[str, list[str]] = {}
    else:
        full = instance_class_mapping_from_ontology(
            ontology_path,
            instance_types_path=instance_types_path,
            subject_filter=subject_filter,
        )
        payload = {
            ent: d["most_specific_class_iris"]
            for ent, d in full.items()
            if d["most_specific_class_iris"]
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_gensim_to_pt(model: Word2Vec, *, training_seed: int | None = None) -> dict:
    """Checkpoint dict compatible with evaluate_embeddings.py / train_word2vec.py."""
    wv = model.wv
    emb = torch.from_numpy(wv.vectors.copy())
    idx2word = list(wv.index_to_key)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    out: dict = {
        "embeddings": emb,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "dim": int(model.vector_size),
        "architecture": "skipgram" if model.sg else "cbow",
        "window": int(model.window),
        "epochs": int(model.epochs),
        "trainer": "gensim",
        "gensim_version": __import__("gensim").__version__,
    }
    if training_seed is not None:
        out["training_seed"] = int(training_seed)
    return out


def train_word2vec_stage(
    walks_path: Path,
    *,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    sg: int,
    negative: int,
    epochs: int,
    seed: int | None,
    alpha: float,
    min_alpha: float,
    desc: str,
    sample: float = 0.0,
    hs: int = 0,
    step_loss_interval: int = 0,
    step_loss_quiet: bool = False,
) -> tuple[Word2Vec, list[float]]:
    """Train Word2Vec with per-epoch loss; optional per-batch step loss (``workers`` must be 1)."""
    sentences = LineSentence(str(walks_path))
    kw = dict(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        hs=hs,
        negative=negative,
        sample=sample,
        alpha=alpha,
        min_alpha=min_alpha,
        seed=seed,
        epochs=epochs,
    )
    model = Word2VecWithStepLoss(
        step_loss_interval=step_loss_interval,
        step_loss_quiet=step_loss_quiet,
        **kw,
    )
    model.build_vocab(sentences, progress_per=10000)
    cb = EpochLossProgress(total_epochs=epochs, desc=desc)
    callbacks: list[CallbackAny2Vec] = [cb]
    if step_loss_interval > 0:
        assert isinstance(model, Word2VecWithStepLoss)
        callbacks.insert(0, EpochTagCallback(model))
        callbacks.append(StepLossEndFlush())
    model.train(
        corpus_iterable=sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
        start_alpha=alpha,
        end_alpha=min_alpha,
        compute_loss=True,
        callbacks=callbacks,
    )
    return model, cb.epoch_losses


def _make_finetune_word2vec_model(
    pre_model: Word2Vec,
    *,
    finetune_epochs: int,
    finetune_loss_every: int,
    seed: int | None,
) -> Word2Vec:
    """Return a fresh Word2Vec model for stage-2 training on instance walks only."""
    kw = dict(
        vector_size=pre_model.vector_size,
        window=pre_model.window,
        min_count=pre_model.min_count,
        workers=pre_model.workers,
        sg=pre_model.sg,
        hs=pre_model.hs,
        negative=pre_model.negative,
        sample=pre_model.sample,
        alpha=pre_model.alpha,
        min_alpha=pre_model.min_alpha,
        seed=seed if seed is not None else pre_model.seed,
        epochs=finetune_epochs,
    )
    if finetune_loss_every > 0:
        return Word2VecWithStepLoss(
            step_loss_interval=finetune_loss_every,
            step_loss_quiet=False,
            **kw,
        )
    return Word2Vec(**kw)


def run_rdf2vec_two_stage(args: argparse.Namespace) -> None:
    """Pretrain on P1 or P2 protograph walks, then finetune on instance walks (``--mode p1`` / ``p2``)."""
    mode = args.mode
    assert mode in ("p1", "p2")

    if args.seed is not None:
        _set_training_rng(int(args.seed))

    if normalize_init_strategy(args.init_strategy) == "ancestor_weighted" and not (
        0.0 < args.maschine_ancestor_decay <= 1.0
    ):
        raise SystemExit(
            "--maschine-ancestor-decay must satisfy 0 < R <= 1 when using "
            "init_strategy weight_hierarchy (or legacy ancestor_weighted)."
        )

    if args.pretrain_walks is None:
        args.pretrain_walks = Path("walks_p1.txt") if mode == "p1" else Path("walks_p2.txt")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.loss_every_steps > 0:
        pretrain_loss_every = args.loss_every_steps
        finetune_loss_every = args.loss_every_steps
    else:
        pretrain_loss_every = args.loss_every_steps_pretrain
        finetune_loss_every = args.loss_every_steps_finetune

    any_step_loss = pretrain_loss_every > 0 or finetune_loss_every > 0
    if not any_step_loss and not args.no_loss_plots:
        print(
            "Note: per-step loss PNGs are disabled (use --loss-every-steps-pretrain / "
            "--loss-every-steps-finetune, or --loss-every-steps N for both).",
            file=sys.stderr,
            flush=True,
        )

    pretrained_path = out_dir / "rdf2vec_pretrained.model"
    final_pt_path = args.output
    if not final_pt_path.is_absolute():
        final_pt_path = out_dir / final_pt_path

    sg = 1 if args.architecture == "skipgram" else 0
    pretrain_losses: list[float] | None = None
    pretrain_step_rows: list[tuple[int, int, float, float]] = []
    runtime = RuntimeMetrics()

    if not args.skip_pretrain:
        if not args.pretrain_walks.is_file():
            raise SystemExit(f"Pretrain walks not found: {args.pretrain_walks}")
        print(f"Stage 1: training Word2Vec on {args.pretrain_walks} ...")
        st_interval = 0
        st_quiet = False
        if pretrain_loss_every > 0:
            st_interval = pretrain_loss_every
            st_quiet = True
            pre_workers = _workers_effective(args.workers)
            if pre_workers > 1:
                print(
                    "Note: per-step loss during stage 1 is approximate with workers>1.",
                    flush=True,
                )
        else:
            pre_workers = _workers_effective(args.workers)
        pre_lr = args.pretrain_lr if args.pretrain_lr is not None else args.lr
        pre_min_alpha = args.pretrain_min_alpha if args.pretrain_min_alpha is not None else args.min_alpha
        with runtime.stage("word2vec_pretrain"):
            pre_model, pretrain_losses = train_word2vec_stage(
                args.pretrain_walks,
                vector_size=args.dim,
                window=args.window,
                min_count=args.min_count,
                workers=pre_workers,
                sg=sg,
                negative=args.negative,
                epochs=args.pretrain_epochs,
                seed=args.seed,
                alpha=pre_lr,
                min_alpha=pre_min_alpha,
                sample=args.sample,
                hs=args.hs,
                desc="Stage 1 pretrain",
                step_loss_interval=st_interval,
                step_loss_quiet=st_quiet,
            )
            if isinstance(pre_model, Word2VecWithStepLoss) and pretrain_loss_every > 0:
                pretrain_step_rows = list(pre_model.step_loss_rows)
            pre_model.save(str(pretrained_path))
        print(f"Wrote {pretrained_path} (vocab size {len(pre_model.wv)})")
    else:
        if not pretrained_path.is_file():
            raise SystemExit(f"--skip-pretrain but missing {pretrained_path}")
        with runtime.stage("word2vec_pretrain_load"):
            pre_model = Word2Vec.load(str(pretrained_path))
        print(f"Loaded {pretrained_path} (vocab size {len(pre_model.wv)})")

    instance_path = args.instance_walks
    if instance_path is None:
        raise SystemExit(
            "Instance walks required for --mode p1/p2: pass positional <walks> or --instance-walks."
        )
    if not instance_path.is_file():
        raise SystemExit(f"Instance walks not found: {instance_path}")

    print(f"Stage 2: building vocabulary from instance walks {instance_path} ...", flush=True)

    ontology_path = args.ontology
    if ontology_path is None:
        for name in ("ontology.nt", "ontology.ttl", "ontology.turtle"):
            cand = instance_path.parent / name
            if cand.is_file():
                ontology_path = cand
                break

    stage1_vectors = {
        w: np.asarray(pre_model.wv[w], dtype=np.float32).copy() for w in pre_model.wv.index_to_key
    }
    stage1_vocab_size = len(stage1_vectors)
    pre_model = _make_finetune_word2vec_model(
        pre_model,
        finetune_epochs=args.finetune_epochs,
        finetune_loss_every=finetune_loss_every,
        seed=args.seed,
    )
    with runtime.stage("word2vec_finetune_vocab_build"):
        pre_model.build_vocab(LineSentence(str(instance_path)), progress_per=10000)
    finetune_vocab_size = len(pre_model.wv)
    n_inst = int(pre_model.corpus_count)

    from _maschine_init import restore_pretrained_vocab_rows  # noqa: PLC0415

    new_token_init: dict[str, str] = {}
    n_pretrained_restored = restore_pretrained_vocab_rows(
        pre_model,
        stage1_vectors,
        new_token_init=new_token_init,
    )
    n_pretrain_only_dropped = stage1_vocab_size - n_pretrained_restored
    print(
        f"Stage 2: finetune vocabulary from instance walks only "
        f"(size={finetune_vocab_size}, dropped {n_pretrain_only_dropped} pretrain-only tokens, "
        f"restored {n_pretrained_restored} overlapping pretrain rows) ...",
        flush=True,
    )
    print(f"Stage 2: continuing training on {instance_path} ({n_inst} sentences) ...", flush=True)
    maschine_applied = False
    n_ent = 0
    n_rel = 0
    n_rand = 0
    all_init_fallback = None
    init_strategy_user = user_facing_init_strategy(args.init_strategy)
    instance_types_path = getattr(args, "instance_types", None)
    vocab_subjects: set[str] | None = None
    if instance_types_path is not None:
        vocab_subjects = vocab_inner_iris(pre_model)
        if not vocab_subjects:
            vocab_subjects = None

    with runtime.stage("maschine_initialization"):
        if args.maschine_init and ontology_path is not None and ontology_path.is_file():
            maschine_applied = True
            n_ent, n_rel, n_rand, all_init_fallback = apply_maschine_initialization(
                pre_model,
                stage1_vectors,
                ontology_path,
                strategy=args.init_strategy,
                ancestor_decay=args.maschine_ancestor_decay,
                init_relations=args.init_relations,
                initialization_noise=args.initialization_noise,
                new_token_init=new_token_init,
                instance_types_path=instance_types_path,
                subject_filter=vocab_subjects,
            )
            rel_note = "on" if args.init_relations else "off"
            noise_note = (
                f", initialization_noise={args.initialization_noise:g}"
                if args.initialization_noise > 0
                else ""
            )
            print(
                f"MASCHInE init (init_strategy={init_strategy_user}, init_relations={rel_note}{noise_note}): "
                f"entity rows from class vectors={n_ent}, "
                f"relation rows copied from pretrain={n_rel}, left random={n_rand}"
            )
            if all_init_fallback is not None:
                fb = all_init_fallback.to_dict()
                print(
                    f"all_init fallback: instances_with_fallback={fb['instances_with_fallback']}, "
                    f"max_hops={fb['max_hops']}, avg_hops={fb['avg_hops']:.4f}",
                    flush=True,
                )
        elif args.maschine_init and (ontology_path is None or not ontology_path.is_file()):
            print(
                "MASCHInE init skipped (no ontology file); pass --ontology or place "
                "ontology.nt / ontology.ttl beside instance walks",
            )

        n_pred_transfer = 0
        if not maschine_applied and args.init_relations:
            n_pred_transfer = transfer_predicate_embeddings_from_pretrained(
                pre_model,
                stage1_vectors,
                new_token_init=new_token_init,
            )
            if n_pred_transfer:
                print(
                    f"Transferred predicate embeddings from pretrain (rows updated={n_pred_transfer}); "
                    "use --no-init-relations to disable.",
                    flush=True,
                )

        if not maschine_applied:
            for w in pre_model.wv.index_to_key:
                if w not in stage1_vectors and w not in new_token_init:
                    new_token_init[w] = "gensim_default"

        instance_coverage = None
        relation_coverage = None
        if ontology_path is not None and ontology_path.is_file():
            instance_coverage = compute_instance_init_coverage(
                ontology_path,
                pre_model,
                stage1_vectors,
                new_token_init,
                strategy=args.init_strategy,
                ancestor_decay=args.maschine_ancestor_decay,
                instance_types_path=instance_types_path,
                subject_filter=vocab_subjects,
            )
            relation_coverage = compute_relation_init_coverage(
                ontology_path,
                pre_model,
                stage1_vectors,
                new_token_init,
            )

        init_stats = compute_finetune_init_stats(pre_model, stage1_vectors, new_token_init)
        json_path, txt_path = write_finetune_init_stats(
            out_dir,
            init_stats,
            instance_coverage=instance_coverage,
            relation_coverage=relation_coverage,
            init_strategy=args.init_strategy if maschine_applied else None,
            all_init_fallback=all_init_fallback,
        )
        if instance_coverage is not None:
            cov = instance_coverage
            print(
                f"Instance init (ontology): from_protograph={cov.initialized_from_class} "
                f"random={cov.random_no_pretrained_class + cov.random_no_class_mapping} "
                f"not_in_walk_corpus={cov.not_in_finetune_vocab}",
                flush=True,
            )
        if relation_coverage is not None:
            rel = relation_coverage
            print(
                f"Relation init (ontology): total={rel.total} "
                f"from_protograph={rel.from_protograph} not_initialized={rel.not_initialized}",
                flush=True,
            )
        print(
            f"Finetune init stats: vocabulary_size={init_stats.instances.total} "
            f"(instances protograph={init_stats.instances.from_protograph} "
            f"random={init_stats.instances.random}; "
            f"relations protograph={init_stats.relations.from_protograph} "
            f"random={init_stats.relations.random})",
            flush=True,
        )
        print(f"Wrote {txt_path} and {json_path}", flush=True)
        token_init_path = write_finetune_token_init(
            out_dir, pre_model, stage1_vectors, new_token_init
        )
        print(f"Wrote {token_init_path}", flush=True)

    anchor_init_vectors = snapshot_class_initialized_entity_vectors(pre_model, new_token_init)

    if not args.no_pretrain_init_mapping:
        map_path = args.pretrain_init_mapping
        if map_path is None:
            map_path = out_dir / "instance_to_class.json"
        elif not map_path.is_absolute():
            map_path = out_dir / map_path
        with runtime.stage("instance_to_class_mapping"):
            save_instance_to_class_json(
                map_path,
                ontology_path if ontology_path is not None and ontology_path.is_file() else None,
                instance_types_path=instance_types_path,
                subject_filter=vocab_subjects,
            )
        print(f"Wrote instance→class mapping: {map_path}")

    ft_workers = _workers_effective(args.workers)
    if finetune_loss_every > 0 and ft_workers > 1:
        print(
            "Note: per-step loss during stage 2 is approximate with workers>1.",
            flush=True,
        )
    pre_model.workers = ft_workers

    ft_ckpt_dir = resolve_finetune_epoch_checkpoint_dir(args)
    if ft_ckpt_dir is not None:
        ft_ckpt_dir.mkdir(parents=True, exist_ok=True)
        init_path = finetune_epoch_checkpoint_path(ft_ckpt_dir, 0)
        save_word2vec_checkpoint(
            pre_model,
            init_path,
            training_seed=args.seed,
            finetune_epoch=0,
        )
        print(f"Wrote finetune epoch checkpoint (init): {init_path}", flush=True)

    ft_loss_cb = EpochLossProgress(
        total_epochs=args.finetune_epochs,
        desc="Stage 2 finetune",
    )
    ft_callbacks: list[CallbackAny2Vec] = [ft_loss_cb]
    if args.anchor_regularization > 0 and anchor_init_vectors:
        ft_callbacks.append(
            InstanceAnchoringCallback(
                anchor_init_vectors,
                args.anchor_regularization,
                total_epochs=args.finetune_epochs,
            )
        )
        print(
            f"Anchoring: alpha={args.anchor_regularization:g}, "
            f"entities={len(anchor_init_vectors)}, "
            f"epochs=1-{max(args.finetune_epochs - 1, 0)} "
            f"(skipped on last epoch)",
            flush=True,
        )
    if ft_ckpt_dir is not None:
        ft_callbacks.append(
            FinetuneEpochCheckpointCallback(ft_ckpt_dir, training_seed=args.seed)
        )
    step_model_ok = isinstance(pre_model, Word2VecWithStepLoss)
    if finetune_loss_every > 0 and not step_model_ok:
        print(
            "Warning: step-loss plots need a model saved from this script's pretrain "
            "(Word2VecWithStepLoss). Skipping per-step plots; epoch plot still saved.",
            file=sys.stderr,
        )
    if finetune_loss_every > 0 and step_model_ok:
        assert isinstance(pre_model, Word2VecWithStepLoss)
        pre_model.step_loss_interval = finetune_loss_every
        pre_model.step_loss_quiet = False
        pre_model.step_loss_rows.clear()
        pre_model._job_step = 0
        pre_model._cum_at_last_interval = 0.0
        pre_model.train_epoch_idx = 0
        ft_callbacks.insert(0, EpochTagCallback(pre_model))
        ft_callbacks.append(StepLossEndFlush())
    elif isinstance(pre_model, Word2VecWithStepLoss):
        pre_model.step_loss_interval = 0

    # Finetune LR: --lr / --min-alpha (independent of stage-1 pretrain-lr / pretrain-min-alpha).
    with runtime.stage("word2vec_finetune"):
        pre_model.train(
            corpus_iterable=LineSentence(str(instance_path)),
            total_examples=n_inst,
            epochs=args.finetune_epochs,
            start_alpha=args.lr,
            end_alpha=args.min_alpha,
            compute_loss=True,
            callbacks=ft_callbacks,
        )
    finetune_losses = ft_loss_cb.epoch_losses

    with runtime.stage("loss_artifacts"):
        if not args.skip_pretrain and pretrain_losses is not None and len(pretrain_losses) > 0:
            pre_csv = out_dir / "pretrain_loss.csv"
            save_loss_csv(pre_csv, pretrain_losses)
            print(f"Wrote pretrain loss log: {pre_csv}")
        if finetune_losses:
            ft_csv = out_dir / "finetune_loss.csv"
            save_loss_csv(ft_csv, finetune_losses)
            print(f"Wrote finetune loss log: {ft_csv}")
        if not args.skip_pretrain and pretrain_loss_every > 0 and pretrain_step_rows:
            pre_steps = out_dir / "pretrain_loss_steps.csv"
            save_step_loss_csv(pre_steps, pretrain_step_rows)
            print(f"Wrote pretrain step loss log: {pre_steps}")
        if finetune_loss_every > 0 and step_model_ok and isinstance(pre_model, Word2VecWithStepLoss):
            ft_rows = pre_model.step_loss_rows
            if ft_rows:
                ft_steps = out_dir / "finetune_loss_steps.csv"
                save_step_loss_csv(ft_steps, ft_rows)
                print(f"Wrote finetune step loss log: {ft_steps}")

        if not args.no_loss_plots:
            pre_ep = (
                args.loss_pretrain_epoch_plot
                if args.loss_pretrain_epoch_plot is not None
                else out_dir / "pretrain_per_epoch.png"
            )
            proto_ep = (
                args.loss_protograph_epoch_plot
                if args.loss_protograph_epoch_plot is not None
                else out_dir / "finetune_per_epoch.png"
            )
            pre_st = (
                args.loss_pretrain_steps_plot
                if args.loss_pretrain_steps_plot is not None
                else out_dir / "pretrain_per_step.png"
            )
            proto_st = (
                args.loss_steps_plot
                if args.loss_steps_plot is not None
                else out_dir / "finetune_per_step.png"
            )

            if not args.skip_pretrain and pretrain_losses and len(pretrain_losses) > 0:
                _plot_loss_per_epoch(pre_ep, pretrain_losses)
                print(f"Wrote pretrain per-epoch loss plot: {pre_ep}")
            elif not args.skip_pretrain:
                print("No pretrain epoch losses to plot.", file=sys.stderr)

            if finetune_losses:
                _plot_loss_per_epoch(proto_ep, finetune_losses)
                print(f"Wrote protograph per-epoch loss plot: {proto_ep}")
            else:
                print("No finetune epoch losses to plot.", file=sys.stderr)

            if not args.skip_pretrain and pretrain_loss_every > 0:
                if pretrain_step_rows:
                    _plot_loss_per_step(
                        pre_st,
                        pretrain_step_rows,
                        title="Stage 1 — pretrain (protograph walks) — loss vs batch step",
                    )
                    print(
                        f"Wrote pretrain per-step loss plot (every {pretrain_loss_every} batches): {pre_st}",
                    )
                else:
                    print(
                        "No stage-1 step-loss points to plot "
                        "(try a smaller --loss-every-steps-pretrain).",
                        file=sys.stderr,
                    )

            if finetune_loss_every > 0 and step_model_ok:
                assert isinstance(pre_model, Word2VecWithStepLoss)
                if pre_model.step_loss_rows:
                    _plot_loss_per_step(
                        proto_st,
                        pre_model.step_loss_rows,
                        title="Stage 2 — finetune (instance walks) — loss vs batch step",
                    )
                    print(
                        f"Wrote protograph per-step loss plot (every {finetune_loss_every} batches): {proto_st}",
                    )
                else:
                    print(
                        "No stage-2 step-loss points to plot (try a smaller --loss-every-steps-finetune).",
                        file=sys.stderr,
                    )

    with runtime.stage("checkpoint_save"):
        ckpt = build_gensim_to_pt(pre_model, training_seed=args.seed)
        ckpt["trainer"] = "rdf2vec_train"
        final_pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, final_pt_path)
    print(f"Wrote {final_pt_path} (vocab size {len(pre_model.wv)})")

    rt_json, rt_txt = write_runtime_metrics(out_dir, runtime)
    print(f"Wrote runtime metrics: {rt_txt}", flush=True)


def run_single_corpus_mode(args: argparse.Namespace) -> None:
    """Train on ``args.walks`` (``--mode none``)."""
    if args.walks is None:
        raise SystemExit("Walks file required when --mode is none.")
    if not args.walks.is_file():
        raise SystemExit(f"Walks file not found: {args.walks}")

    runtime = RuntimeMetrics()
    metrics_dir = args.output.resolve().parent

    if args.seed is not None:
        _set_training_rng(int(args.seed))

    if args.gensim_log_progress:
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s",
            level=logging.INFO,
        )

    workers = _workers_effective(args.workers)
    if args.loss_every_steps > 0 and workers > 1:
        tqdm.write(
            "Note: --loss-every-steps loss curve is approximate with workers>1.",
            file=sys.stderr,
        )

    sentences = LineSentence(str(args.walks))

    w2v_kw = dict(
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        workers=workers,
        sg=1 if args.architecture == "skipgram" else 0,
        hs=args.hs,
        negative=args.negative,
        sample=args.sample,
        alpha=args.lr,
        min_alpha=args.min_alpha,
        seed=args.seed,
        epochs=args.epochs,
    )
    if args.loss_every_steps > 0:
        model = Word2VecWithStepLoss(step_loss_interval=args.loss_every_steps, **w2v_kw)
    else:
        model = Word2Vec(**w2v_kw)

    with runtime.stage("word2vec_vocab_build"):
        model.build_vocab(sentences, progress_per=10000)

    if len(model.wv) == 0:
        raise SystemExit("Empty vocabulary after min_count filtering.")

    ft_ckpt_dir = resolve_finetune_epoch_checkpoint_dir(args)
    if ft_ckpt_dir is not None:
        ft_ckpt_dir.mkdir(parents=True, exist_ok=True)
        init_path = finetune_epoch_checkpoint_path(ft_ckpt_dir, 0)
        save_word2vec_checkpoint(
            model,
            init_path,
            training_seed=args.seed,
            finetune_epoch=0,
        )
        print(f"Wrote finetune epoch checkpoint (init): {init_path}", flush=True)

    cb = EpochLossProgress(
        total_epochs=args.epochs,
        desc=f"Gensim {args.architecture.upper()} [loss]",
    )
    callbacks: list[CallbackAny2Vec] = [cb]
    if ft_ckpt_dir is not None:
        callbacks.append(
            FinetuneEpochCheckpointCallback(ft_ckpt_dir, training_seed=args.seed)
        )
    if args.loss_every_steps > 0:
        assert isinstance(model, Word2VecWithStepLoss)
        callbacks.insert(0, EpochTagCallback(model))
        callbacks.append(StepLossEndFlush())

    train_kw: dict = {
        "corpus_iterable": sentences,
        "total_examples": model.corpus_count,
        "epochs": args.epochs,
        "start_alpha": args.lr,
        "end_alpha": args.min_alpha,
        "compute_loss": True,
        "callbacks": callbacks,
    }
    if args.gensim_log_progress:
        train_kw["report_delay"] = args.report_delay

    with runtime.stage("word2vec_train"):
        model.train(**train_kw)

    losses = cb.epoch_losses
    if len(losses) != args.epochs:
        tqdm.write(
            f"Warning: expected {args.epochs} epoch losses, got {len(losses)}.",
            file=sys.stderr,
        )

    stem = args.output.with_suffix("")
    loss_log = args.loss_log if args.loss_log is not None else Path(f"{stem}_loss.csv")
    loss_plot = args.loss_plot if args.loss_plot is not None else Path(f"{stem}_loss.png")

    with runtime.stage("loss_artifacts"):
        save_loss_csv(loss_log, losses)
        tqdm.write(f"Wrote loss log: {loss_log}")

        if args.loss_every_steps > 0 and isinstance(model, Word2VecWithStepLoss):
            steps_path = (
                args.loss_steps_log
                if args.loss_steps_log is not None
                else Path(f"{stem}_loss_steps.csv")
            )
            save_step_loss_csv(steps_path, model.step_loss_rows)
            tqdm.write(
                f"Wrote step loss log ({args.loss_every_steps} batch interval): {steps_path}",
            )
            if not args.no_plot and model.step_loss_rows:
                steps_plot = (
                    args.loss_steps_plot
                    if args.loss_steps_plot is not None
                    else Path(f"{stem}_loss_steps.png")
                )
                _plot_loss_per_step(
                    steps_plot,
                    model.step_loss_rows,
                    title="Gensim Word2Vec — loss vs training batch step",
                )
                tqdm.write(f"Wrote step loss plot: {steps_plot}")
            elif not args.no_plot and not model.step_loss_rows:
                tqdm.write(
                    "No step loss rows to plot (try a smaller --loss-every-steps or more data).",
                    file=sys.stderr,
                )

        if not args.no_plot:
            _plot_loss_per_epoch(loss_plot, losses)
            tqdm.write(f"Wrote loss plot: {loss_plot}")

    with runtime.stage("checkpoint_save"):
        ckpt = build_gensim_to_pt(model, training_seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.output)
    tqdm.write(
        f"Saved checkpoint to {args.output}  (vocab={len(ckpt['word2idx'])}, dim={args.dim})",
    )

    rt_json, rt_txt = write_runtime_metrics(metrics_dir, runtime)
    tqdm.write(f"Wrote runtime metrics: {rt_txt}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train Word2Vec with Gensim: single corpus (--mode none) or RDF2Vec P1/P2 two-stage.",
    )
    p.add_argument(
        "--mode",
        choices=("none", "p1", "p2"),
        default="none",
        help="none: train on one walks file. p1/p2: pretrain on protograph walks then finetune on instance walks.",
    )
    p.add_argument(
        "walks",
        type=Path,
        nargs="?",
        default=None,
        help="Walks file: required for --mode none; for p1/p2, instance walks unless --instance-walks is set.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .pt checkpoint (default: word2vec_gensim.pt or rdf2vec_final.pt in --out-dir)",
    )
    p.add_argument(
        "--architecture",
        "--arch",
        dest="architecture",
        choices=("skipgram", "cbow"),
        default="skipgram",
        help="sg=1 skip-gram, sg=0 CBOW",
    )
    p.add_argument("--dim", type=int, default=128, help="Vector size (vector_size)")
    p.add_argument("--window", type=int, default=5, help="Context window")
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (only --mode none; for p1/p2 use --pretrain-epochs / --finetune-epochs)",
    )
    p.add_argument("--negative", type=int, default=5, help="Negative samples")
    p.add_argument(
        "--sample",
        type=float,
        default=0.0,
        help="Subsampling threshold (0.0 = disable, matches jRDF2Vec default)",
    )
    p.add_argument(
        "--hs",
        type=int,
        choices=(0, 1),
        default=0,
        help="Hierarchical softmax (0 = negative sampling only)",
    )
    p.add_argument("--min-count", type=int, default=1, help="Ignore tokens below this count")
    p.add_argument("--lr", type=float, default=0.025, help="Initial learning rate (alpha)")
    p.add_argument(
        "--min-alpha",
        type=float,
        default=0.0001,
        help="Minimum learning rate after linear decay (min_alpha)",
    )
    p.add_argument(
        "--pretrain-lr",
        type=float,
        default=None,
        help="Stage-1 initial alpha (--mode p1/p2 only; default: same as --lr)",
    )
    p.add_argument(
        "--pretrain-min-alpha",
        type=float,
        default=None,
        help="Stage-1 min_alpha (--mode p1/p2 only; default: same as --min-alpha)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Gensim worker threads (0 = half of logical CPUs, minimum 2). For --mode p1/p2, stage 1/2 training.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (sets Python/NumPy/torch RNGs). Multi-threaded Word2Vec SGD remains nondeterministic.",
    )
    p.add_argument(
        "--loss-log",
        type=Path,
        default=None,
        help="CSV path for per-epoch losses (default: <output stem>_loss.csv)",
    )
    p.add_argument(
        "--loss-plot",
        type=Path,
        default=None,
        help="PNG path for loss curve (default: <output stem>_loss.png)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write loss PNGs (epoch curve or step curve)",
    )
    p.add_argument(
        "--gensim-log-progress",
        action="store_true",
        help="Enable Gensim INFO logging (within-epoch progress lines; can be noisy)",
    )
    p.add_argument(
        "--report-delay",
        type=float,
        default=1.0,
        help="Seconds between Gensim progress log lines (only with --gensim-log-progress)",
    )
    p.add_argument(
        "--loss-every-steps",
        type=int,
        default=0,
        metavar="N",
        help="Per-batch step loss: CSV/PNG in --mode none. For p1/p2, if N>0, sets both "
        "--loss-every-steps-pretrain and --loss-every-steps-finetune to N; if 0, use those flags separately.",
    )
    p.add_argument(
        "--loss-every-steps-pretrain",
        type=int,
        default=0,
        metavar="N",
        help="p1/p2 stage 1: log step loss every N batch jobs (default 0 = epoch bar only; N>0 slows "
        "pretrain and per-step loss is approximate with workers>1). Ignored when --loss-every-steps > 0.",
    )
    p.add_argument(
        "--loss-every-steps-finetune",
        type=int,
        default=0,
        metavar="N",
        help="p1/p2 stage 2: log step loss every N batch jobs (default 0 = epoch plots only). "
        "Ignored when --loss-every-steps > 0.",
    )
    p.add_argument(
        "--loss-steps-log",
        type=Path,
        default=None,
        help="CSV for step losses (default: <output stem>_loss_steps.csv when --loss-every-steps > 0)",
    )
    p.add_argument(
        "--loss-steps-plot",
        type=Path,
        default=None,
        help="PNG for step loss curves (default: <output stem>_loss_steps.png when --loss-every-steps > 0)",
    )

    # RDF2Vec (--mode p1 / p2)
    p.add_argument(
        "--pretrain-walks",
        type=Path,
        default=None,
        help="Stage-1 walks (default: walks_p1.txt for p1, walks_p2.txt for p2).",
    )
    p.add_argument(
        "--instance-walks",
        type=Path,
        default=None,
        help="Stage-2 instance walks file (required for p1/p2 unless set via positional walks).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for models and walk files for --mode p1/p2 (default: current directory).",
    )
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Load existing pretrained model from --out-dir instead of training stage 1.",
    )
    p.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Ontology with rdf:type and rdfs:subClassOf (.nt or .ttl). "
        "If omitted, uses <instance-walks-dir>/ontology.nt, ontology.ttl, or ontology.turtle when present.",
    )
    p.add_argument(
        "--instance-types",
        type=Path,
        default=None,
        help="Optional rdf:type source for DBpedia resources (.nt, .ttl, or .bz2). "
        "Merged with --ontology for MASCHInE init (filtered to finetune vocabulary).",
    )
    p.add_argument(
        "--maschine-init",
        dest="maschine_init",
        action="store_true",
        default=True,
        help="MASCHInE initialization for new instance tokens (default: on when ontology is available).",
    )
    p.add_argument(
        "--no-maschine-init",
        dest="maschine_init",
        action="store_false",
        help="Skip MASCHInE init; keep gensim random init for new instance-walk vocabulary tokens.",
    )
    p.add_argument(
        "--init-strategy",
        dest="init_strategy",
        type=str,
        default="most_specific",
        choices=sorted(INIT_STRATEGIES),
        help="Finetune embedding init for new instance rows from protograph class vectors: "
        "most_specific (default), all_init (most_specific with ancestor fallback), "
        "average_hier (unweighted mean over class hierarchy), "
        "or weight_hierarchy (hop-decayed mean; see --maschine-ancestor-decay).",
    )
    p.add_argument(
        "--init-relations",
        dest="init_relations",
        action="store_true",
        default=True,
        help="(--mode p1/p2) Initialize new P_* relation rows from stage-1 pretrain (default: on).",
    )
    p.add_argument(
        "--no-init-relations",
        dest="init_relations",
        action="store_false",
        help="(--mode p1/p2) Skip relation/P_* init; only initialize class and instance rows.",
    )
    p.add_argument(
        "--initialization-noise",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian noise std dev added to instance/relation rows at finetune init (default: 0).",
    )
    p.add_argument(
        "--anchor-regularization",
        type=float,
        default=0.0,
        metavar="ALPHA",
        help="After each finetune epoch, pull class-init entity vectors toward MASCHInE init "
        "(default: 0 = off).",
    )
    p.add_argument(
        "--transfer-predicate-embeddings",
        dest="init_relations",
        action="store_true",
        help="[deprecated] Alias for --init-relations.",
    )
    p.add_argument(
        "--no-transfer-predicate-embeddings",
        dest="init_relations",
        action="store_false",
        help="[deprecated] Alias for --no-init-relations.",
    )
    p.add_argument(
        "--maschine-strategy",
        dest="init_strategy",
        type=str,
        choices=sorted(INIT_STRATEGIES | {"ancestor_mean", "ancestor_weighted"}),
        help="[deprecated] Alias for --init-strategy; legacy names ancestor_mean / ancestor_weighted accepted.",
    )
    p.add_argument(
        "--maschine-ancestor-decay",
        type=float,
        default=0.5,
        metavar="R",
        help="For init_strategy weight_hierarchy (or legacy ancestor_weighted): per-hop multiplier "
        "so weight ∝ R**distance (smaller R emphasizes types closer to the most-specific class). "
        "Ignored otherwise. Required: 0 < R ≤ 1.",
    )
    p.add_argument(
        "--loss-pretrain-epoch-plot",
        type=Path,
        default=None,
        help="PNG for stage-1 pretrain per-epoch loss (default: <out-dir>/pretrain_per_epoch.png).",
    )
    p.add_argument(
        "--loss-protograph-epoch-plot",
        type=Path,
        default=None,
        help="PNG for stage-2 instance finetune per-epoch loss (default: <out-dir>/finetune_per_epoch.png).",
    )
    p.add_argument(
        "--loss-pretrain-steps-plot",
        type=Path,
        default=None,
        help="PNG for stage-1 pretrain per-step loss (default: <out-dir>/pretrain_per_step.png).",
    )
    p.add_argument(
        "--no-loss-plots",
        action="store_true",
        help="Do not write RDF2Vec loss PNGs.",
    )
    p.add_argument(
        "--pretrain-init-mapping",
        type=Path,
        default=None,
        help="JSON path for instance→class mapping (default: <out-dir>/instance_to_class.json).",
    )
    p.add_argument(
        "--no-pretrain-init-mapping",
        action="store_true",
        help="Do not write instance→class JSON.",
    )
    p.add_argument(
        "--save-finetune-epoch-checkpoints",
        action="store_true",
        help=(
            "Save .pt checkpoints for the finetune stage: finetune_epoch_00.pt before training "
            "(post-pretrain init for p1/p2; post-vocab for --mode none), finetune_epoch_k.pt "
            "after k finetune epochs."
        ),
    )
    p.add_argument(
        "--finetune-epoch-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-epoch checkpoints (default: <out-dir>/ckpt).",
    )

    args = p.parse_args()

    if args.mode == "none":
        args.output = args.output or Path("word2vec_gensim.pt")
        run_single_corpus_mode(args)
        return

    # p1 / p2
    if args.walks is not None:
        args.instance_walks = args.instance_walks or args.walks
    if args.instance_walks is None:
        raise SystemExit(
            "Instance walks required for --mode p1/p2: pass positional <walks> or --instance-walks."
        )
    args.output = args.output or Path("rdf2vec_final.pt")
    run_rdf2vec_two_stage(args)


if __name__ == "__main__":
    main()
