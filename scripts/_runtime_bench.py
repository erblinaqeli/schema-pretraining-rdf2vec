"""
Stage-level runtime benchmarks for protograph pipelines.

Measures SGNS pretrain / init / finetune only. Walk generation and protograph
construction are excluded — callers must pass ready-made walk corpora.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models.word2vec import LineSentence

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _synthetic_compare import (  # noqa: E402
    concept_bound_vectors,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
)

STAGE_KEYS = (
    "pretrain_sgns",
    "normalize_codes",
    "concept_bound",
    "build_vocab",
    "init_vectors",
    "finetune_sgns",
)


def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 3)


def pretrain_stage(
    proto_walks: Path,
    *,
    dim: int,
    pretrain_epochs: int,
    seed: int,
    workers: int,
    target_norm: float,
) -> tuple[dict[str, np.ndarray], float, dict[str, float]]:
    """Protograph SGNS pretrain + code normalization."""
    stages: dict[str, float] = {}

    t0 = time.perf_counter()
    pre = pretrain_protograph(
        proto_walks,
        dim=dim,
        epochs=pretrain_epochs,
        seed=seed,
        workers=workers,
    )
    stages["pretrain_sgns"] = _elapsed(t0)

    t0 = time.perf_counter()
    vecs, used_norm = normalized_stage1_vectors(pre.wv, target_norm=target_norm)
    stages["normalize_codes"] = _elapsed(t0)

    return vecs, used_norm, stages


def finetune_variant(
    name: str,
    *,
    vecs: dict[str, np.ndarray],
    used_norm: float,
    proto_kind: str,
    instance_walks: Path,
    ontology_nt: Path,
    graph_nt: Path | None,
    dim: int,
    epochs: int,
    finetune_alpha: float,
    min_alpha: float,
    seed: int,
    workers: int,
    pretrain_stages: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Init + finetune one classic/bound variant (pretrain excluded unless passed in)."""
    bound = name.endswith("_bound")
    stages: dict[str, float] = {k: 0.0 for k in STAGE_KEYS}
    if pretrain_stages:
        stages.update(pretrain_stages)

    bound_vectors = None
    if bound:
        if graph_nt is None:
            raise ValueError(f"{name} requires graph_nt")
        t0 = time.perf_counter()
        bound_vectors = concept_bound_vectors(
            graph_nt, ontology_nt, vecs, target_norm=used_norm,
        )
        stages["concept_bound"] = _elapsed(t0)

    t0 = time.perf_counter()
    model = new_skipgram_model(
        dim=dim,
        alpha=finetune_alpha,
        min_alpha=min_alpha,
        seed=seed,
        workers=workers,
    )
    model.build_vocab(LineSentence(str(instance_walks)))
    stages["build_vocab"] = _elapsed(t0)

    t0 = time.perf_counter()
    protograph_init(
        model,
        vecs,
        ontology_nt,
        strategy="all_init",
        target_norm=used_norm,
        bound_vectors=bound_vectors,
    )
    stages["init_vectors"] = _elapsed(t0)

    t0 = time.perf_counter()
    model.train(
        corpus_iterable=LineSentence(str(instance_walks)),
        total_examples=model.corpus_count,
        epochs=epochs,
        start_alpha=finetune_alpha,
        end_alpha=min_alpha,
    )
    stages["finetune_sgns"] = _elapsed(t0)

    stages["total"] = round(sum(stages[k] for k in STAGE_KEYS), 3)
    return {
        "variant": name,
        "proto_kind": proto_kind,
        "bound": bound,
        "stages": stages,
    }


def run_tc_timings(
    tc: str,
    paths: dict[str, Path],
    proto_walks: dict[str, Path],
    inst_walks: Path,
    *,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Benchmark all six protograph variants for one test case."""
    pretrain_cache: dict[str, tuple[dict[str, np.ndarray], float, dict[str, float]]] = {}
    pretrain_by_kind: dict[str, dict[str, float]] = {}
    variants: list[dict[str, Any]] = []

    for kind in ("p1", "p2", "p3"):
        vecs, used_norm, pre_stages = pretrain_stage(
            proto_walks[kind],
            dim=cfg["dim"],
            pretrain_epochs=cfg["pretrain_epochs"],
            seed=cfg["seed"],
            workers=cfg["workers"],
            target_norm=cfg["target_norm"],
        )
        pretrain_cache[kind] = (vecs, used_norm, pre_stages)
        pretrain_by_kind[kind] = {
            **pre_stages,
            "total_pretrain": round(pre_stages["pretrain_sgns"] + pre_stages["normalize_codes"], 3),
        }

    variant_specs = [
        ("p1_classic", "p1"),
        ("p2_classic", "p2"),
        ("p3_classic", "p3"),
        ("p1_bound", "p1"),
        ("p2_bound", "p2"),
        ("p3_bound", "p3"),
    ]
    for name, kind in variant_specs:
        vecs, used_norm, pre_stages = pretrain_cache[kind]
        row = finetune_variant(
            name,
            vecs=vecs,
            used_norm=used_norm,
            proto_kind=kind,
            instance_walks=inst_walks,
            ontology_nt=paths["ontology"],
            graph_nt=paths["graph"] if name.endswith("_bound") else None,
            dim=cfg["dim"],
            epochs=cfg["epochs"],
            finetune_alpha=cfg["finetune_alpha"],
            min_alpha=cfg["min_alpha"],
            seed=cfg["seed"],
            workers=cfg["workers"],
            pretrain_stages=pre_stages,
        )
        row["tc"] = tc
        variants.append(row)

    return {
        "tc": tc,
        "pretrain_by_kind": pretrain_by_kind,
        "variants": variants,
    }
