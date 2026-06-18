#!/usr/bin/env python3
"""
Legacy training on imported bina instance walks (old_protographs + old_train logic).

Uses exact protograph NT generation from old_protographs.py and exact graph Word2Vec
training from old_train.py (init from protograph model.kv + entity2classes.json).

Examples:
  uv run python scripts/train_old.py --tc tc01 tc02 tc03 --train-mode p1 p2 vanilla
  uv run python scripts/train_old.py --timestamp 20260607_bina_old_tc123 --no-plot
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shlex import join as shlex_join
from typing import Any, Literal

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
DEFAULT_WALKS_ROOT = REPO_ROOT / "all_walks"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_bina_old"
DEFAULT_GRAPH_EPOCHS = 5

# old_embed_protograph.py defaults
DEFAULT_NUM_WALKS = 200
DEFAULT_DEPTH = 3
DEFAULT_DIMENSIONS = 200

EPOCH_EVAL_CSV_NAME = "epoch_eval_accuracy.csv"
InitFallback = Literal["none", "ancestor", "p2kv"]


@dataclass
class OldTrainOptions:
    vector_dim: int = DEFAULT_DIMENSIONS
    jar_dim: int = DEFAULT_DIMENSIONS
    init_fallback: InitFallback = "none"
    init_relations: bool = False
    force_rebuild_protographs: bool = False
    audit: bool = True
    run_eval: bool = True


def resolve_jar_path() -> Path:
    candidates = [
        REPO_ROOT / "jrdf2vec-1.3-SNAPSHOT_seed.jar",
        REPO_ROOT / "jars" / "jrdf2vec-1.3-SNAPSHOT_seed.jar",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "jRDF2Vec jar not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def ensure_jrdf2vec_python_compat() -> None:
    """Prepare python-server for jRDF2Vec (Werkzeug 3.x + venv python)."""
    import site

    patch_src = REPO_ROOT / "python-server" / "sitecustomize.py"
    if patch_src.is_file():
        for sp in site.getsitepackages():
            dest = Path(sp) / "sitecustomize.py"
            try:
                if not dest.is_file() or dest.read_text(encoding="utf-8") != patch_src.read_text(
                    encoding="utf-8"
                ):
                    dest.write_text(patch_src.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                continue

    cmd_file = REPO_ROOT / "python-server" / "python_command.txt"
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        # Keep the venv shim path (do not .resolve()); the base interpreter
        # alone does not see packages installed in .venv/lib/.../site-packages.
        desired = f"{venv_python}\n"
        if not cmd_file.is_file() or cmd_file.read_text(encoding="utf-8") != desired:
            cmd_file.write_text(desired, encoding="utf-8")

random.seed(42)
np.random.seed(42)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_bina_walks import materialize_imported_walks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_old")

CLI_MODES = ("p1", "p2", "vanilla")

# ---------------------------------------------------------------------------
# old_protographs.py (verbatim logic; ontology path adapted in callers)
# ---------------------------------------------------------------------------

RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"


def parse_ontology_nt(path: Path):
    """Simple parser: assumes clean N-Triples with <s> <p> <o> . lines"""
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw == ".":
                continue

            if raw.endswith("."):
                raw = raw[:-1].strip()

            parts = raw.split()
            if len(parts) < 3:
                continue

            s, p, o = parts[0], parts[1], parts[2]

            def strip_b(x):
                return x[1:-1] if x.startswith("<") and x.endswith(">") else x

            triples.append((strip_b(s), strip_b(p), strip_b(o)))

    return triples


def build_indices(triples):
    rel2domain = {}
    rel2range = {}
    class2subclasses = {}  # it stores children grouped by parent

    for s, p, o in triples:
        if p == RDFS_DOMAIN:
            rel2domain[s] = o
        elif p == RDFS_RANGE:
            rel2range[s] = o
        elif p == RDFS_SUBCLASS:
            # s subClassOf o
            parent = o
            child = s
            class2subclasses.setdefault(parent, set()).add(child)

    return rel2domain, rel2range, class2subclasses


def write_nt(triples, out_path: Path):
    """Write triples in N-Triples format, wrapping with <...>."""

    def wrap(x):
        return x if (x.startswith("<") and x.endswith(">")) else f"<{x}>"

    with out_path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{wrap(s)} {wrap(p)} {wrap(o)} .\n")


def build_prototype_v1(rel2domain, rel2range, out_path: Path):
    """P1: DomainClass --rel--> RangeClass."""
    out_triples = []
    for rel, dom in rel2domain.items():
        if rel in rel2range:
            ran = rel2range[rel]
            out_triples.append((dom, rel, ran))

    write_nt(out_triples, out_path)
    logger.info("[P1] wrote %s (%d triples)", out_path, len(out_triples))


def build_prototype_v2(rel2domain, rel2range, class2subclasses, out_path: Path):
    """P2: P1 + direct subclasses expansion."""
    # Implements MASCHInE P2 protograph exactly:
    # (Ci, r, Cj), (C'i, r, Cj), (Ci, r, C'j)
    # No (C'i, r, C'j), direct subclasses only

    out_triples = []

    for rel, dom in rel2domain.items():
        if rel not in rel2range:
            continue
        ran = rel2range[rel]

        out_triples.append((dom, rel, ran))  # (pCi, r, pCj)

        # domain subclasses
        for dsub in class2subclasses.get(dom, []):
            out_triples.append((dsub, rel, ran))  # (pC′i, r, pCj)

        # range subclasses
        for rsub in class2subclasses.get(ran, []):
            out_triples.append((dom, rel, rsub))  # (pCi, r, pC′j)

    # dedupe
    seen = set()
    deduped = []
    for tri in out_triples:
        if tri not in seen:
            seen.add(tri)
            deduped.append(tri)

    write_nt(deduped, out_path)
    logger.info("[P2] wrote %s (%d triples)", out_path, len(deduped))


def ontology_path_for_tc(tc: str) -> Path:
    return REPO_ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology" / "ontology.nt"


def test_path_for_tc(tc: str) -> Path:
    return (
        REPO_ROOT
        / "v1"
        / "synthetic_ontology"
        / tc
        / "synthetic_ontology"
        / "1000"
        / "train_test"
        / "test.txt"
    )


def audit_dir(output_root: Path, tc: str) -> Path:
    return output_root / "_cache" / "audit" / tc


def build_class_parents(triples: list[tuple[str, str, str]]) -> dict[str, set[str]]:
    """Map child class -> parent classes via rdfs:subClassOf."""
    parents: dict[str, set[str]] = {}
    for s, p, o in triples:
        if p == RDFS_SUBCLASS:
            parents.setdefault(s, set()).add(o)
    return parents


def count_nt_triples(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip() and not line.startswith("#"))


def classes_in_protograph_nt(path: Path) -> set[str]:
    classes: set[str] = set()
    if not path.is_file():
        return classes
    for line in path.open(encoding="utf-8"):
        raw = line.strip()
        if not raw or raw == ".":
            continue
        parts = raw.split()
        if len(parts) < 3:
            continue
        for token in (parts[0], parts[2]):
            inner = strip_angle_brackets(token)
            if inner.startswith("C_"):
                classes.add(inner)
    return classes


def resolve_class_vectors(
    cls_ids: list[str],
    proto_kv: KeyedVectors,
    class_parents: dict[str, set[str]] | None,
    *,
    init_fallback: InitFallback,
    fallback_kv: KeyedVectors | None = None,
) -> list[np.ndarray]:
    cls_vecs = [proto_kv[c] for c in cls_ids if c in proto_kv]
    if cls_vecs:
        return cls_vecs

    if init_fallback == "p2kv" and fallback_kv is not None:
        cls_vecs = [fallback_kv[c] for c in cls_ids if c in fallback_kv]
        if cls_vecs:
            return cls_vecs

    if init_fallback != "ancestor" or not class_parents:
        return cls_vecs

    found: list[np.ndarray] = []
    for start in cls_ids:
        q: deque[str] = deque([start])
        seen = {start}
        while q:
            c = q.popleft()
            if c in proto_kv:
                found.append(proto_kv[c])
                break
            if init_fallback == "p2kv" and fallback_kv is not None and c in fallback_kv:
                found.append(fallback_kv[c])
                break
            for par in class_parents.get(c, ()):
                if par not in seen:
                    seen.add(par)
                    q.append(par)
    return found


def audit_record_path(output_root: Path, tc: str, timestamp: str) -> Path:
    return audit_dir(output_root, tc) / f"{timestamp}.json"


def write_audit_record(
    *,
    output_root: Path,
    tc: str,
    timestamp: str,
    stage: str,
    data: dict[str, Any],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    path = audit_record_path(output_root, tc, timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {}
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
    record.setdefault("tc", tc)
    record.setdefault("timestamp", timestamp)
    record[stage] = data
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Audit [%s] %s -> %s", tc, stage, path)


def audit_protograph_stage(
    *,
    tc: str,
    output_root: Path,
    timestamp: str,
    p1_path: Path,
    p2_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    data = {
        "p1_nt": str(p1_path),
        "p2_nt": str(p2_path),
        "p1_triple_count": count_nt_triples(p1_path),
        "p2_triple_count": count_nt_triples(p2_path),
        "p1_class_count": len(classes_in_protograph_nt(p1_path)),
        "p2_class_count": len(classes_in_protograph_nt(p2_path)),
    }
    write_audit_record(
        output_root=output_root,
        tc=tc,
        timestamp=timestamp,
        stage="protograph_nt",
        data=data,
        dry_run=dry_run,
    )
    return data


def audit_jar_embed_stage(
    *,
    tc: str,
    proto: str,
    output_root: Path,
    timestamp: str,
    embed_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    data: dict[str, Any] = {"proto": proto, "embed_dir": str(embed_dir)}
    if embed_dir.is_dir():
        meta_path = embed_dir / "meta.json"
        data["meta_present"] = meta_path.is_file()
        if meta_path.is_file():
            try:
                data["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data["meta"] = None
        try:
            kv_path = resolve_jar_kv(embed_dir)
            kv = KeyedVectors.load(str(kv_path), mmap="r")
            class_tokens = [k for k in kv.key_to_index if k.startswith("C_")]
            norms = [float(np.linalg.norm(kv[c])) for c in class_tokens[:100]]
            data.update(
                {
                    "kv_path": str(kv_path),
                    "kv_dim": kv.vector_size,
                    "kv_token_count": len(kv),
                    "class_token_count": len(class_tokens),
                    "sample_class_norm_mean": float(np.mean(norms)) if norms else None,
                }
            )
        except (OSError, FileNotFoundError) as exc:
            data["kv_error"] = str(exc)
    write_audit_record(
        output_root=output_root,
        tc=tc,
        timestamp=timestamp,
        stage=f"jar_embed_{proto}",
        data=data,
        dry_run=dry_run,
    )
    return data


def audit_entity2classes_stage(
    *,
    tc: str,
    output_root: Path,
    timestamp: str,
    mapping_path: Path,
    p1_path: Path,
    p2_path: Path,
    proto_kvs: dict[str, KeyedVectors],
    dry_run: bool,
) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    all_classes: set[str] = set()
    multi_class = 0
    for cls_list in mapping.values():
        if isinstance(cls_list, list) and len(cls_list) > 1:
            multi_class += 1
        if isinstance(cls_list, list):
            all_classes.update(cls_list)
        elif isinstance(cls_list, str):
            all_classes.add(cls_list)

    p1_classes = classes_in_protograph_nt(p1_path)
    p2_classes = classes_in_protograph_nt(p2_path)
    coverage: dict[str, Any] = {}
    for proto, kv in proto_kvs.items():
        inst_missing = sum(
            1
            for cls_list in mapping.values()
            if not resolve_class_vectors(
                cls_list if isinstance(cls_list, list) else [cls_list],
                kv,
                None,
                init_fallback="none",
            )
        )
        coverage[proto] = {
            "classes_in_mapping": len(all_classes),
            "classes_in_kv": len({k for k in kv.key_to_index if k.startswith("C_")}),
            "classes_in_mapping_not_in_kv": len(all_classes - {k for k in kv.key_to_index}),
            "instances_without_direct_class_vec": inst_missing,
        }

    data = {
        "mapping_path": str(mapping_path),
        "instance_count": len(mapping),
        "unique_classes": len(all_classes),
        "multi_class_instances": multi_class,
        "classes_in_p1_nt": len(p1_classes),
        "classes_in_p2_nt": len(p2_classes),
        "classes_in_mapping_not_in_p1_nt": len(all_classes - p1_classes),
        "classes_in_mapping_not_in_p2_nt": len(all_classes - p2_classes),
        "proto_coverage": coverage,
    }
    write_audit_record(
        output_root=output_root,
        tc=tc,
        timestamp=timestamp,
        stage="entity2classes",
        data=data,
        dry_run=dry_run,
    )
    return data


def audit_init_stage(
    *,
    tc: str,
    output_root: Path,
    timestamp: str,
    mode: str,
    model: Word2Vec,
    init_stats: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    wv = model.wv
    inst_norms = [
        float(np.linalg.norm(wv[t]))
        for t in wv.key_to_index
        if str(t).startswith("I_")
    ]
    data = {
        **init_stats,
        "mode": mode,
        "vector_dim": wv.vector_size,
        "instance_count_in_vocab": len(inst_norms),
        "instance_norm_mean": float(np.mean(inst_norms)) if inst_norms else None,
        "instance_norm_std": float(np.std(inst_norms)) if inst_norms else None,
    }
    write_audit_record(
        output_root=output_root,
        tc=tc,
        timestamp=timestamp,
        stage=f"init_{mode}",
        data=data,
        dry_run=dry_run,
    )
    return data


def protograph_cache_dir(output_root: Path, tc: str) -> Path:
    return output_root / "_cache" / "protograph" / tc


def build_protographs_for_tc(
    tc: str,
    *,
    output_root: Path,
    dry_run: bool = False,
    force_rebuild: bool = False,
) -> tuple[Path, Path]:
    """Generate protograph_p1.nt and protograph_p2.nt using old_protographs logic."""
    ont_path = ontology_path_for_tc(tc)
    if not ont_path.is_file():
        raise FileNotFoundError(f"ontology.nt not found: {ont_path}")

    out_dir = protograph_cache_dir(output_root, tc)
    p1_path = out_dir / "protograph_p1.nt"
    p2_path = out_dir / "protograph_p2.nt"

    if force_rebuild and not dry_run:
        for path in (p1_path, p2_path, out_dir / "entity2classes.json"):
            if path.is_file():
                logger.info("force-rebuild-protographs: removing %s", path)
                path.unlink()

    if p1_path.is_file() and p2_path.is_file():
        logger.info("reuse protographs for %s: %s", tc, out_dir)
        return p1_path, p2_path

    if dry_run:
        logger.info("would build protographs for %s -> %s", tc, out_dir)
        return p1_path, p2_path

    logger.info("Parsing ontology for %s: %s", tc, ont_path)
    triples = parse_ontology_nt(ont_path)
    rel2domain, rel2range, class2subclasses = build_indices(triples)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_prototype_v1(rel2domain, rel2range, p1_path)
    build_prototype_v2(rel2domain, rel2range, class2subclasses, p2_path)
    return p1_path, p2_path


def build_entity2classes_json(tc: str, *, output_root: Path, dry_run: bool = False) -> Path:
    """Build entity2classes.json from ontology rdf:type (old NT parser)."""
    ont_path = ontology_path_for_tc(tc)
    out_path = protograph_cache_dir(output_root, tc) / "entity2classes.json"

    if out_path.is_file():
        logger.info("reuse entity2classes for %s: %s", tc, out_path)
        return out_path

    if dry_run:
        logger.info("would build entity2classes for %s -> %s", tc, out_path)
        return out_path

    triples = parse_ontology_nt(ont_path)
    mapping: dict[str, list[str]] = {}
    for s, p, o in triples:
        if p != RDF_TYPE:
            continue
        if o == OWL_CLASS:
            continue
        if not s.startswith("I_") or not o.startswith("C_"):
            continue
        mapping.setdefault(s, []).append(o)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote entity2classes for %s: %d instances -> %s", tc, len(mapping), out_path)
    return out_path


# ---------------------------------------------------------------------------
# old_train.py (verbatim training logic)
# ---------------------------------------------------------------------------


def strip_angle_brackets(x: str) -> str:
    if isinstance(x, str) and x.startswith("<") and x.endswith(">"):
        return x[1:-1]
    return x


class MySentences:
    """Yield one token list per line from a file or directory of .gz walk files. Same tokenization as server."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __iter__(self):
        if self.path.is_dir():
            names = sorted(n for n in os.listdir(self.path) if n.endswith(".gz"))
            for name in names:
                p = self.path / name
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        yield line.split(" ")
        else:
            if str(self.path).endswith(".gz"):
                with gzip.open(self.path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        yield line.split(" ")
            else:
                with self.path.open("rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        yield line.split(" ")


class ReiterableGraphWalks:
    """Re-iterable corpus: each __iter__ yields a fresh pass over graph walks (so train(..., epochs=N) works)."""

    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        return iter(MySentences(self.path))


class CheckpointSaver(CallbackAny2Vec):
    """Saves model + KV after each epoch (epoch_1, epoch_2, ...). Appends to metadata_rows; records loss and runtime per epoch."""

    def __init__(self, out_dir: Path, save_each_epoch: bool, metadata_rows: list, init_stats: dict):
        self.out_dir = out_dir
        self.save_each_epoch = save_each_epoch
        self.metadata_rows = metadata_rows
        self.init_stats = init_stats
        self.epoch = 0
        self.cumulative_loss_before = 0.0
        self._epoch_start: float | None = None

    def on_epoch_begin(self, model):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, model):
        self.epoch += 1
        total = model.get_latest_training_loss()
        epoch_loss = total - self.cumulative_loss_before
        self.cumulative_loss_before = total
        runtime_sec = time.perf_counter() - self._epoch_start if self._epoch_start is not None else None
        if self.save_each_epoch:
            epoch_dir = self.out_dir / f"epoch_{self.epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir / "model.model"))
            model.wv.save(str(epoch_dir / "model.kv"))
            logger.info(
                "Saved checkpoint epoch %d -> %s (epoch_loss: %.4f, cumulative_loss: %.4f, runtime_sec: %.2f)",
                self.epoch,
                epoch_dir,
                epoch_loss,
                total,
                runtime_sec or 0,
            )
        else:
            logger.info(
                "Epoch %d done (epoch_loss: %.4f, cumulative_loss: %.4f, runtime_sec: %.2f)",
                self.epoch,
                epoch_loss,
                total,
                runtime_sec or 0,
            )
        self.metadata_rows.append(
            {
                "graph_epoch": self.epoch,
                "loss": epoch_loss,
                "runtime_sec": runtime_sec,
                **self.init_stats,
            }
        )


def init_instance_vectors_from_classes_instance_only(
    model: Word2Vec,
    proto_kv: KeyedVectors,
    inst2classes_raw: dict,
    *,
    class_parents: dict[str, set[str]] | None = None,
    init_fallback: InitFallback = "none",
    init_relations: bool = False,
    fallback_kv: KeyedVectors | None = None,
) -> tuple[int, int, int, int, int]:
    """
    Initialize only tokens that exist in the Word2Vec vocab using class vectors from proto_kv.
    Writes into model.wv.vectors.

    Returns: (initialized, skipped_not_in_vocab, skipped_no_classvec, skipped_bad, relations_initialized)
    """
    wv = model.wv
    vectors = wv.vectors

    initialized = 0
    skipped_not_in_vocab = 0
    skipped_no_classvec = 0
    skipped_bad = 0
    relations_initialized = 0

    for inst_uri, cls_list in inst2classes_raw.items():
        inst_id = strip_angle_brackets(inst_uri)

        if inst_id not in wv.key_to_index:
            skipped_not_in_vocab += 1
            continue

        if isinstance(cls_list, str):
            cls_ids = [strip_angle_brackets(cls_list)]
        elif isinstance(cls_list, list):
            cls_ids = [strip_angle_brackets(x) for x in cls_list]
        else:
            skipped_bad += 1
            continue

        cls_vecs = resolve_class_vectors(
            cls_ids,
            proto_kv,
            class_parents,
            init_fallback=init_fallback,
            fallback_kv=fallback_kv,
        )
        if not cls_vecs:
            skipped_no_classvec += 1
            continue

        idx = wv.key_to_index[inst_id]
        vectors[idx] = np.mean(np.vstack(cls_vecs), axis=0).astype(np.float32)
        initialized += 1

    if init_relations:
        for token in wv.key_to_index:
            if not str(token).startswith("P_"):
                continue
            if token not in proto_kv:
                continue
            idx = wv.key_to_index[token]
            vectors[idx] = np.asarray(proto_kv[token], dtype=np.float32)
            relations_initialized += 1

    return initialized, skipped_not_in_vocab, skipped_no_classvec, skipped_bad, relations_initialized


def resolve_workers(number_of_threads: int | None) -> int:
    """Return Gensim workers; never 1."""
    if number_of_threads is not None:
        if number_of_threads < 2:
            raise SystemExit("--number-of-threads must be >= 2 (workers=1 is not allowed)")
        return number_of_threads
    return os.cpu_count() or 4


def make_word2vec(*, vector_size: int, workers: int) -> Word2Vec:
    return Word2Vec(
        vector_size=vector_size,
        window=5,
        sg=1,
        hs=0,
        negative=5,
        min_count=1,
        sample=0.0,
        workers=workers,
        compute_loss=True,
        seed=42,
    )


def write_epochs_metadata(
    *,
    out_dir: Path,
    tc_str: str,
    mode: str,
    graph_walks_path: Path,
    graph_count: int,
    graph_epochs: int,
    workers: int,
    build_vocab_time_sec: float,
    init_vectors_time_sec: float,
    epoch0_save_time_sec: float,
    total_train_time_sec: float,
    total_pipeline_time_sec: float,
    init_stats: dict,
    metadata_rows: list[dict],
    initialized: int,
    skipped_not_in_vocab: int,
    skipped_no_classvec: int,
    skipped_bad: int,
    vocab_size: int,
) -> None:
    tsv_path = out_dir / "epochs_metadata.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "graph_epoch\tloss\truntime_sec\tcumulative_runtime_sec\tvocab_size\tinitialized\t"
            "skipped_not_in_vocab\tskipped_no_classvec\tskipped_bad\n"
        )
        f.write(
            f"0\t\t\t\t{vocab_size}\t{initialized}\t{skipped_not_in_vocab}\t"
            f"{skipped_no_classvec}\t{skipped_bad}\n"
        )
        cum = 0.0
        for r in metadata_rows:
            rt = r.get("runtime_sec")
            if rt is not None:
                cum += rt
            rt_str = f"{rt:.4f}" if rt is not None else ""
            cum_str = f"{cum:.4f}" if rt is not None else ""
            loss_str = "" if r.get("loss") is None else str(r["loss"])
            f.write(
                f"{r['graph_epoch']}\t{loss_str}\t{rt_str}\t{cum_str}\t{r['vocab_size']}\t{r['initialized']}\t"
                f"{r['skipped_not_in_vocab']}\t{r['skipped_no_classvec']}\t{r['skipped_bad']}\n"
            )

    epochs_json = [
        {
            "graph_epoch": 0,
            "loss": None,
            "runtime_sec": None,
            "cumulative_runtime_sec": None,
            **init_stats,
        }
    ]
    cum = 0.0
    for r in metadata_rows:
        rt = r.get("runtime_sec")
        if rt is not None:
            cum += rt
        epochs_json.append(
            {
                "graph_epoch": r["graph_epoch"],
                "loss": r.get("loss"),
                "runtime_sec": rt,
                "cumulative_runtime_sec": cum if rt is not None else None,
                **{k: r[k] for k in init_stats},
            }
        )

    json_path = out_dir / "epochs_metadata.json"
    json_path.write_text(
        json.dumps(
            {
                "tc": tc_str,
                "mode": mode,
                "graph_walks_path": str(graph_walks_path),
                "graph_corpus_count": graph_count,
                "graph_epochs": graph_epochs,
                "workers": workers,
                "build_vocab_time_sec": build_vocab_time_sec,
                "init_vectors_time_sec": init_vectors_time_sec,
                "epoch0_save_time_sec": epoch0_save_time_sec,
                "total_train_time_sec": total_train_time_sec,
                "total_pipeline_time_sec": total_pipeline_time_sec,
                "vocab_size": vocab_size,
                "initialized": initialized,
                "skipped_not_in_vocab": skipped_not_in_vocab,
                "skipped_no_classvec": skipped_no_classvec,
                "skipped_bad": skipped_bad,
                "epochs": epochs_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_loss_csv(metadata_rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for row in metadata_rows:
            writer.writerow([row["graph_epoch"], f"{row['loss']:.8f}"])


def write_manifest(out_dir: Path, manifest: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Prerequisites: protograph embedding via jRDF2Vec JAR (old_embed_protograph.py)
# ---------------------------------------------------------------------------


def protograph_embed_dir(output_root: Path, tc: str, proto: str) -> Path:
    return protograph_cache_dir(output_root, tc) / proto


def protograph_kv_path(output_root: Path, tc: str, proto: str) -> Path:
    return protograph_embed_dir(output_root, tc, proto) / "model.kv"


def resolve_jar_kv(out_dir: Path) -> Path:
    """Return the gensim .kv file produced by jRDF2Vec in *out_dir*."""
    preferred = out_dir / "model.kv"
    if preferred.is_file():
        return preferred
    candidates = sorted(out_dir.glob("*.kv"))
    if not candidates:
        raise FileNotFoundError(f"No .kv file found under jRDF2Vec output: {out_dir}")
    return candidates[0]


def _jar_embed_meta(proto_nt: Path, *, jar_dim: int) -> dict[str, Any]:
    return {
        "method": "jrdf2vec-jar",
        "jar": str(resolve_jar_path()),
        "numberOfWalks": DEFAULT_NUM_WALKS,
        "depth": DEFAULT_DEPTH,
        "dimension": jar_dim,
        "proto_nt": str(proto_nt.resolve()),
        "proto_nt_mtime": proto_nt.stat().st_mtime if proto_nt.is_file() else None,
    }


def _jar_embed_cache_valid(out_dir: Path, proto_nt: Path, *, jar_dim: int) -> bool:
    meta_path = out_dir / "meta.json"
    kv_path = out_dir / "model.kv"
    if not meta_path.is_file() or not kv_path.is_file():
        # Also accept any .kv if model.kv missing but meta matches
        if not meta_path.is_file():
            return False
        try:
            resolve_jar_kv(out_dir)
        except FileNotFoundError:
            return False
    try:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    desired = _jar_embed_meta(proto_nt, jar_dim=jar_dim)
    return cached == desired


def embed_protograph_via_jar(
    *,
    tc: str,
    proto: str,
    proto_nt: Path,
    output_root: Path,
    dry_run: bool,
    force: bool = False,
    jar_dim: int = DEFAULT_DIMENSIONS,
    timestamp: str | None = None,
    audit: bool = True,
) -> Path:
    """Run jRDF2Vec JAR on a protograph NT file (old_embed_protograph.py)."""
    out_dir = protograph_embed_dir(output_root, tc, proto)

    if force and out_dir.exists() and not dry_run:
        logger.info("force-reembed: removing %s", out_dir)
        shutil.rmtree(out_dir)

    if _jar_embed_cache_valid(out_dir, proto_nt, jar_dim=jar_dim):
        kv_path = resolve_jar_kv(out_dir)
        logger.info("reuse protograph JAR embed for %s %s: %s", tc, proto, kv_path)
        if audit and timestamp:
            audit_jar_embed_stage(
                tc=tc,
                proto=proto,
                output_root=output_root,
                timestamp=timestamp,
                embed_dir=out_dir,
                dry_run=dry_run,
            )
        return kv_path

    if dry_run:
        logger.info("would run jRDF2Vec JAR for %s %s -> %s", tc, proto, out_dir)
        return protograph_kv_path(output_root, tc, proto)

    if not proto_nt.is_file():
        raise FileNotFoundError(f"Protograph NT not found: {proto_nt}")

    jar_path = resolve_jar_path()
    ensure_jrdf2vec_python_compat()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-jar",
        str(jar_path),
        "-graph",
        str(proto_nt),
        "-walkDirectory",
        str(out_dir),
        "-numberOfWalks",
        str(DEFAULT_NUM_WALKS),
        "-depth",
        str(DEFAULT_DEPTH),
        "-dimension",
        str(jar_dim),
    ]
    logger.info("Running jRDF2Vec for %s %s: %s", tc, proto, shlex_join(cmd))
    env = os.environ.copy()
    # jRDF2Vec passes walk/model paths as HTTP headers with underscores; Werkzeug 2.3+
    # drops them unless this is set (inherited by the JAR-spawned python_server).
    env["WERKZEUG_HEADERS_WITH_UNDERSCORES"] = "1"
    venv_bin = REPO_ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )

    try:
        kv_path = resolve_jar_kv(out_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"jRDF2Vec finished but no .kv under {out_dir}. "
            "Ensure python-server deps are installed (flask, gensim) and "
            f"the gensim server can start. Original: {exc}"
        ) from exc
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(_jar_embed_meta(proto_nt, jar_dim=jar_dim), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Finished jRDF2Vec %s %s -> %s (kv dim check pending load)", tc, proto, kv_path)
    if audit and timestamp:
        audit_jar_embed_stage(
            tc=tc,
            proto=proto,
            output_root=output_root,
            timestamp=timestamp,
            embed_dir=out_dir,
            dry_run=dry_run,
        )
    return kv_path


def ensure_protograph_prereqs(
    *,
    tc: str,
    protos: tuple[str, ...],
    output_root: Path,
    dry_run: bool,
    force_reembed: bool = False,
    train_options: OldTrainOptions,
    timestamp: str,
) -> None:
    p1_nt, p2_nt = build_protographs_for_tc(
        tc,
        output_root=output_root,
        dry_run=dry_run,
        force_rebuild=train_options.force_rebuild_protographs,
    )
    if train_options.audit:
        audit_protograph_stage(
            tc=tc,
            output_root=output_root,
            timestamp=timestamp,
            p1_path=p1_nt,
            p2_path=p2_nt,
            dry_run=dry_run,
        )

    mapping_path = build_entity2classes_json(tc, output_root=output_root, dry_run=dry_run)

    proto_nt = {"p1": p1_nt, "p2": p2_nt}
    proto_kvs: dict[str, KeyedVectors] = {}
    for proto in protos:
        kv_path = embed_protograph_via_jar(
            tc=tc,
            proto=proto,
            proto_nt=proto_nt[proto],
            output_root=output_root,
            dry_run=dry_run,
            force=force_reembed,
            jar_dim=train_options.jar_dim,
            timestamp=timestamp if train_options.audit else None,
            audit=train_options.audit,
        )
        if not dry_run and kv_path.is_file():
            proto_kvs[proto] = KeyedVectors.load(str(kv_path), mmap="r")

    if train_options.audit and not dry_run and mapping_path.is_file() and proto_kvs:
        audit_entity2classes_stage(
            tc=tc,
            output_root=output_root,
            timestamp=timestamp,
            mapping_path=mapping_path,
            p1_path=p1_nt,
            p2_path=p2_nt,
            proto_kvs=proto_kvs,
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Graph training (P1 / P2 / vanilla)
# ---------------------------------------------------------------------------


def run_dir_for_mode(
    *,
    output_root: Path,
    tc: str,
    mode: str,
    timestamp: str,
) -> Path:
    if mode == "vanilla":
        return output_root / "synthetic" / "vanilla" / "vanilla_default" / tc / timestamp
    return output_root / "synthetic" / "protograph" / "default" / tc / timestamp / mode


def write_epoch_eval_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "finetune_epoch",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_epoch_checkpoints(out_dir: Path, graph_epochs: int) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    epoch0 = out_dir / "epoch_0" / "model.kv"
    if epoch0.is_file():
        checkpoints.append((0, epoch0))
    for epoch in range(1, graph_epochs + 1):
        ckpt = out_dir / f"epoch_{epoch}" / "model.kv"
        if ckpt.is_file():
            checkpoints.append((epoch, ckpt))
    return checkpoints


def run_epoch_eval(
    *,
    tc: str,
    mode: str,
    out_dir: Path,
    graph_epochs: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Evaluate saved epoch checkpoints via scripts/_evaluate.py."""
    test_path = test_path_for_tc(tc)
    if dry_run:
        logger.info("would evaluate %s %s checkpoints under %s", tc, mode, out_dir)
        return []

    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    rows: list[dict[str, Any]] = []
    checkpoints = discover_epoch_checkpoints(out_dir, graph_epochs)
    if not checkpoints:
        logger.warning("No epoch checkpoints found under %s", out_dir)
        return rows

    evaluate_script = SCRIPTS_DIR / "_evaluate.py"
    for epoch, ckpt_path in checkpoints:
        logger.info("Evaluating %s %s epoch %d: %s", tc, mode, epoch, ckpt_path)
        cmd = [
            sys.executable,
            str(evaluate_script),
            str(test_path),
            "-c",
            str(ckpt_path),
            "--label",
            f"{tc} {mode} epoch {epoch}",
        ]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "Eval failed for %s %s epoch %d:\n%s",
                tc,
                mode,
                epoch,
                result.stderr or result.stdout,
            )
            continue
        accuracy = precision = recall = f1 = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            key, val = parts
            if key == "accuracy":
                accuracy = float(val)
            elif key == "precision":
                precision = float(val)
            elif key == "recall":
                recall = float(val)
            elif key == "f1":
                f1 = float(val)
        if accuracy is None:
            logger.warning("Could not parse eval output for %s %s epoch %d", tc, mode, epoch)
            continue
        rows.append(
            {
                "finetune_epoch": epoch,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "checkpoint_path": str(ckpt_path.resolve()),
            }
        )
        logger.info(
            "%s %s epoch %d accuracy=%.4f f1=%.4f",
            tc,
            mode,
            epoch,
            accuracy,
            f1 or 0.0,
        )

    if rows:
        write_epoch_eval_csv(out_dir / EPOCH_EVAL_CSV_NAME, rows)
    return rows


def load_class_parents_for_tc(tc: str) -> dict[str, set[str]]:
    triples = parse_ontology_nt(ontology_path_for_tc(tc))
    return build_class_parents(triples)


def run_graph_train_p1_p2(
    *,
    tc: str,
    proto: str,
    graph_walks_path: Path,
    proto_kv_path: Path,
    mapping_path: Path,
    out_dir: Path,
    graph_epochs: int,
    save_each_epoch: bool,
    workers: int,
    timestamp: str,
    materialized_walks: Path,
    output_root: Path,
    train_options: OldTrainOptions,
    dry_run: bool,
) -> None:
    tc_str = tc
    if dry_run:
        logger.info("would run graph training %s %s -> %s", tc_str, proto, out_dir)
        return

    if not proto_kv_path.exists():
        raise FileNotFoundError(f"Protograph KV not found: {proto_kv_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")
    if not graph_walks_path.exists():
        raise FileNotFoundError(f"Graph walks path not found: {graph_walks_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "synthetic",
        "train_mode": proto,
        "tc": tc_str,
        "timestamp": timestamp,
        "run_dir": str(out_dir),
        "command": shlex_join([sys.executable, *sys.argv]),
        "workers": workers,
        "graph_epochs": graph_epochs,
        "materialized_walks": str(materialized_walks.resolve()),
        "proto_kv_path": str(proto_kv_path.resolve()),
        "entity2classes_path": str(mapping_path.resolve()),
        "training_logic": "old_train.py",
        "protograph_embed_method": "jrdf2vec-jar",
        "protograph_embed_params": {
            "numberOfWalks": DEFAULT_NUM_WALKS,
            "depth": DEFAULT_DEPTH,
            "dimension": train_options.jar_dim,
        },
        "init_fallback": train_options.init_fallback,
        "init_relations": train_options.init_relations,
    }
    write_manifest(out_dir, manifest, dry_run=False)

    logger.info("=== Graph-only training with init from protograph: %s %s (workers=%d) ===", tc_str, proto, workers)
    logger.info("Protograph KV:  %s", proto_kv_path)
    logger.info("entity2classes: %s", mapping_path)
    logger.info("Graph walks:    %s", graph_walks_path)
    logger.info("Output dir:     %s", out_dir)

    pipeline_start = time.perf_counter()

    proto_kv: KeyedVectors = KeyedVectors.load(str(proto_kv_path), mmap=None)
    logger.info("Loaded protograph KV: %d tokens, dim=%d", len(proto_kv), proto_kv.vector_size)

    sentences_for_vocab = MySentences(graph_walks_path)
    model = make_word2vec(vector_size=proto_kv.vector_size, workers=workers)
    t0 = time.perf_counter()
    model.build_vocab(sentences_for_vocab)
    build_vocab_time_sec = time.perf_counter() - t0
    vocab_size = len(model.wv)
    graph_count = model.corpus_count
    logger.info(
        "Vocabulary built. Vocab size: %d, corpus count: %d (build_vocab: %.2fs)",
        vocab_size,
        graph_count,
        build_vocab_time_sec,
    )

    inst2classes_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_parents = (
        load_class_parents_for_tc(tc)
        if train_options.init_fallback in ("ancestor", "p2kv")
        else None
    )
    fallback_kv: KeyedVectors | None = None
    if train_options.init_fallback == "p2kv" and proto != "p2":
        p2_kv_path = protograph_kv_path(output_root, tc, "p2")
        if p2_kv_path.is_file():
            fallback_kv = KeyedVectors.load(str(p2_kv_path), mmap="r")
            logger.info("P2 KV fallback loaded for %s: %d tokens", tc, len(fallback_kv))
    t0 = time.perf_counter()
    (
        initialized,
        skipped_not_in_vocab,
        skipped_no_classvec,
        skipped_bad,
        relations_initialized,
    ) = init_instance_vectors_from_classes_instance_only(
        model,
        proto_kv,
        inst2classes_raw,
        class_parents=class_parents,
        init_fallback=train_options.init_fallback,
        init_relations=train_options.init_relations,
        fallback_kv=fallback_kv,
    )
    init_vectors_time_sec = time.perf_counter() - t0
    logger.info(
        "Init from classes: initialized=%d | relations_initialized=%d | "
        "skipped_not_in_vocab=%d | skipped_no_classvec=%d | skipped_bad=%d (%.2fs)",
        initialized,
        relations_initialized,
        skipped_not_in_vocab,
        skipped_no_classvec,
        skipped_bad,
        init_vectors_time_sec,
    )

    init_stats = {
        "vocab_size": vocab_size,
        "initialized": initialized,
        "relations_initialized": relations_initialized,
        "skipped_not_in_vocab": skipped_not_in_vocab,
        "skipped_no_classvec": skipped_no_classvec,
        "skipped_bad": skipped_bad,
        "init_fallback": train_options.init_fallback,
        "init_relations": train_options.init_relations,
    }

    if train_options.audit:
        audit_init_stage(
            tc=tc,
            output_root=output_root,
            timestamp=timestamp,
            mode=proto,
            model=model,
            init_stats=init_stats,
            dry_run=False,
        )

    t0 = time.perf_counter()
    init_dir = out_dir / "epoch_0"
    init_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(init_dir / "model.model"))
    model.wv.save(str(init_dir / "model.kv"))
    epoch0_save_time_sec = time.perf_counter() - t0
    logger.info("Saved init-only checkpoint (epoch_0): %s (%.2fs)", init_dir, epoch0_save_time_sec)

    metadata_rows: list[dict] = []
    corpus = ReiterableGraphWalks(graph_walks_path)
    checkpoint_cb = CheckpointSaver(out_dir, save_each_epoch, metadata_rows, init_stats)
    logger.info("Training for %d graph epochs (single train() for correct alpha decay) ...", graph_epochs)
    t_train_start = time.perf_counter()
    model.train(
        corpus,
        total_examples=graph_count,
        epochs=graph_epochs,
        callbacks=[checkpoint_cb],
        compute_loss=True,
    )
    total_train_time_sec = time.perf_counter() - t_train_start
    logger.info("Model trained. total_train_time_sec=%.2f", total_train_time_sec)

    total_pipeline_time_sec = time.perf_counter() - pipeline_start

    write_epochs_metadata(
        out_dir=out_dir,
        tc_str=tc_str,
        mode=proto,
        graph_walks_path=graph_walks_path,
        graph_count=graph_count,
        graph_epochs=graph_epochs,
        workers=workers,
        build_vocab_time_sec=build_vocab_time_sec,
        init_vectors_time_sec=init_vectors_time_sec,
        epoch0_save_time_sec=epoch0_save_time_sec,
        total_train_time_sec=total_train_time_sec,
        total_pipeline_time_sec=total_pipeline_time_sec,
        init_stats=init_stats,
        metadata_rows=metadata_rows,
        initialized=initialized,
        skipped_not_in_vocab=skipped_not_in_vocab,
        skipped_no_classvec=skipped_no_classvec,
        skipped_bad=skipped_bad,
        vocab_size=vocab_size,
    )
    write_loss_csv(metadata_rows, out_dir / "finetune_loss.csv")
    if train_options.run_eval:
        run_epoch_eval(
            tc=tc,
            mode=proto,
            out_dir=out_dir,
            graph_epochs=graph_epochs,
            dry_run=False,
        )
    logger.info("Done %s %s.", tc_str, proto)


def run_graph_train_vanilla(
    *,
    tc: str,
    graph_walks_path: Path,
    out_dir: Path,
    graph_epochs: int,
    save_each_epoch: bool,
    workers: int,
    timestamp: str,
    materialized_walks: Path,
    output_root: Path,
    train_options: OldTrainOptions,
    dry_run: bool,
) -> None:
    tc_str = tc
    if dry_run:
        logger.info("would run vanilla graph training %s -> %s", tc_str, out_dir)
        return

    if not graph_walks_path.exists():
        raise FileNotFoundError(f"Graph walks path not found: {graph_walks_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "synthetic",
        "train_mode": "vanilla",
        "tc": tc_str,
        "timestamp": timestamp,
        "run_dir": str(out_dir),
        "command": shlex_join([sys.executable, *sys.argv]),
        "workers": workers,
        "graph_epochs": graph_epochs,
        "materialized_walks": str(materialized_walks.resolve()),
        "training_logic": "old_train.py (vanilla, no proto init)",
        "vector_dim": train_options.vector_dim,
    }
    write_manifest(out_dir, manifest, dry_run=False)

    logger.info("=== Vanilla graph-only training: %s (workers=%d) ===", tc_str, workers)
    logger.info("Graph walks: %s", graph_walks_path)
    logger.info("Output dir:  %s", out_dir)

    pipeline_start = time.perf_counter()

    sentences_for_vocab = MySentences(graph_walks_path)
    model = make_word2vec(vector_size=train_options.vector_dim, workers=workers)
    t0 = time.perf_counter()
    model.build_vocab(sentences_for_vocab)
    build_vocab_time_sec = time.perf_counter() - t0
    vocab_size = len(model.wv)
    graph_count = model.corpus_count
    logger.info(
        "Vocabulary built. Vocab size: %d, corpus count: %d (build_vocab: %.2fs)",
        vocab_size,
        graph_count,
        build_vocab_time_sec,
    )

    init_stats = {
        "vocab_size": vocab_size,
        "initialized": 0,
        "relations_initialized": 0,
        "skipped_not_in_vocab": 0,
        "skipped_no_classvec": 0,
        "skipped_bad": 0,
        "vector_dim": train_options.vector_dim,
    }

    if train_options.audit:
        audit_init_stage(
            tc=tc,
            output_root=output_root,
            timestamp=timestamp,
            mode="vanilla",
            model=model,
            init_stats=init_stats,
            dry_run=False,
        )

    t0 = time.perf_counter()
    init_dir = out_dir / "epoch_0"
    init_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(init_dir / "model.model"))
    model.wv.save(str(init_dir / "model.kv"))
    epoch0_save_time_sec = time.perf_counter() - t0
    logger.info("Saved init checkpoint (epoch_0): %s (%.2fs)", init_dir, epoch0_save_time_sec)

    metadata_rows: list[dict] = []
    corpus = ReiterableGraphWalks(graph_walks_path)
    checkpoint_cb = CheckpointSaver(out_dir, save_each_epoch, metadata_rows, init_stats)
    logger.info("Training for %d graph epochs ...", graph_epochs)
    t_train_start = time.perf_counter()
    model.train(
        corpus,
        total_examples=graph_count,
        epochs=graph_epochs,
        callbacks=[checkpoint_cb],
        compute_loss=True,
    )
    total_train_time_sec = time.perf_counter() - t_train_start
    logger.info("Model trained. total_train_time_sec=%.2f", total_train_time_sec)

    total_pipeline_time_sec = time.perf_counter() - pipeline_start

    write_epochs_metadata(
        out_dir=out_dir,
        tc_str=tc_str,
        mode="vanilla",
        graph_walks_path=graph_walks_path,
        graph_count=graph_count,
        graph_epochs=graph_epochs,
        workers=workers,
        build_vocab_time_sec=build_vocab_time_sec,
        init_vectors_time_sec=0.0,
        epoch0_save_time_sec=epoch0_save_time_sec,
        total_train_time_sec=total_train_time_sec,
        total_pipeline_time_sec=total_pipeline_time_sec,
        init_stats=init_stats,
        metadata_rows=metadata_rows,
        initialized=0,
        skipped_not_in_vocab=0,
        skipped_no_classvec=0,
        skipped_bad=0,
        vocab_size=vocab_size,
    )
    write_loss_csv(metadata_rows, out_dir / "rdf2vec_word2vec_loss.csv")
    if train_options.run_eval:
        run_epoch_eval(
            tc=tc,
            mode="vanilla",
            out_dir=out_dir,
            graph_epochs=graph_epochs,
            dry_run=False,
        )
    logger.info("Done %s vanilla.", tc_str)


def load_init_accuracy_csv(path: Path):
    """Load epoch-0 accuracy rows for init-quality plots."""
    import pandas as pd

    df = pd.read_csv(path)
    for col in ("finetune_epoch", "accuracy"):
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}")
    return df.loc[df["finetune_epoch"] == 0, ["finetune_epoch", "accuracy"]].sort_values(
        "finetune_epoch"
    )


def plot_init_quality(
    *,
    output_root: Path,
    timestamp: str,
    tcs: list[str],
    plot_out_dir: Path,
    slug: str = "default",
) -> None:
    plot_scripts = SCRIPTS_DIR / "plot"
    if str(plot_scripts) not in sys.path:
        sys.path.insert(0, str(plot_scripts))

    from _common import ExperimentRun, apply_plot_style, resolve_run_dirs  # noqa: E402
    from experiment import load_epoch_accuracy_csv, plot_single_tc, plot_tc_grid  # noqa: E402

    runs: list[ExperimentRun] = []
    for tc in tcs:
        run_dirs = resolve_run_dirs(
            output_root=output_root,
            dataset="synthetic",
            slug=slug,
            tc=tc,
            timestamp=timestamp,
        )
        missing_modes: list[str] = []
        for mode in CLI_MODES:
            if mode not in run_dirs:
                missing_modes.append(mode)
                continue
            csv_path = run_dirs[mode] / EPOCH_EVAL_CSV_NAME
            if not csv_path.is_file():
                missing_modes.append(mode)
        if missing_modes:
            logger.warning(
                "Skipping init plot for %s — missing: %s",
                tc,
                ", ".join(sorted(set(missing_modes))),
            )
            continue
        runs.append(
            ExperimentRun(
                tc=tc,
                timestamps=(timestamp,),
                run_dirs_by_timestamp={timestamp: run_dirs},
            )
        )

    if not runs:
        logger.warning("No runs available for init-quality plotting.")
        return

    apply_plot_style()
    slug_dir = plot_out_dir / timestamp
    slug_dir.mkdir(parents=True, exist_ok=True)

    acc_ylabel = "Test accuracy"
    init_path = slug_dir / "init_accuracy_p1_p2_vanilla.png"
    plot_tc_grid(
        runs=runs,
        ylabel=acc_ylabel,
        out_path=init_path,
        load_series=load_init_accuracy_csv,
        x_col="finetune_epoch",
        y_col="accuracy",
        sharey=True,
    )

    for run in runs:
        tc_acc_path = slug_dir / f"{run.tc}_init_accuracy.png"
        plot_single_tc(
            run=run,
            ylabel=acc_ylabel,
            out_path=tc_acc_path,
            load_series=load_init_accuracy_csv,
            x_col="finetune_epoch",
            y_col="accuracy",
        )

    conv_path = slug_dir / "eval_accuracy_p1_p2_vanilla.png"
    plot_tc_grid(
        runs=runs,
        ylabel=acc_ylabel,
        out_path=conv_path,
        load_series=load_epoch_accuracy_csv,
        x_col="finetune_epoch",
        y_col="accuracy",
        sharey=True,
    )

    logger.info("Wrote %s", init_path)
    logger.info("Wrote %s", conv_path)
    logger.info("Wrote %d per-TC init plots under %s", len(runs), slug_dir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_losses(
    *,
    output_root: Path,
    timestamp: str,
    tcs: list[str],
    plot_out_dir: Path,
    slug: str = "default",
) -> None:
    plot_scripts = SCRIPTS_DIR / "plot"
    if str(plot_scripts) not in sys.path:
        sys.path.insert(0, str(plot_scripts))

    from _common import ExperimentRun, apply_plot_style, resolve_run_dirs  # noqa: E402
    from experiment import (  # noqa: E402
        load_epoch_loss_csv,
        plot_single_tc,
        plot_tc_grid,
    )

    runs: list[ExperimentRun] = []
    for tc in tcs:
        run_dirs = resolve_run_dirs(
            output_root=output_root,
            dataset="synthetic",
            slug=slug,
            tc=tc,
            timestamp=timestamp,
        )
        missing_modes = [m for m in CLI_MODES if m not in run_dirs]
        for mode in CLI_MODES:
            if mode in missing_modes:
                continue
            csv_name = "finetune_loss.csv" if mode in ("p1", "p2") else "rdf2vec_word2vec_loss.csv"
            csv_path = run_dirs[mode] / csv_name
            if not csv_path.is_file():
                missing_modes.append(mode)
        if missing_modes:
            logger.warning("Skipping plot for %s — missing: %s", tc, ", ".join(sorted(set(missing_modes))))
            continue
        runs.append(
            ExperimentRun(
                tc=tc,
                timestamps=(timestamp,),
                run_dirs_by_timestamp={timestamp: run_dirs},
            )
        )

    if not runs:
        logger.warning("No runs available for plotting.")
        return

    apply_plot_style()
    slug_dir = plot_out_dir / timestamp
    slug_dir.mkdir(parents=True, exist_ok=True)

    loss_path = slug_dir / "training_loss_p1_p2_vanilla.png"
    loss_ylabel = "Training loss"
    plot_tc_grid(
        runs=runs,
        ylabel=loss_ylabel,
        out_path=loss_path,
        load_series=load_epoch_loss_csv,
        x_col="epoch",
        y_col="loss",
        sharey=False,
    )

    for run in runs:
        tc_loss_path = slug_dir / f"{run.tc}_training_loss.png"
        plot_single_tc(
            run=run,
            ylabel=loss_ylabel,
            out_path=tc_loss_path,
            load_series=load_epoch_loss_csv,
            x_col="epoch",
            y_col="loss",
        )

    logger.info("Wrote %s", loss_path)
    logger.info("Wrote %d per-TC loss plots under %s", len(runs), slug_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def normalize_tc(tc: str) -> str:
    tc = tc.strip().lower()
    if tc.startswith("tc"):
        return tc
    if tc.isdigit():
        return f"tc{int(tc):02d}"
    return tc


def parse_train_modes(arg: str) -> list[str]:
    s = arg.strip().lower()
    if s == "all":
        return list(CLI_MODES)
    modes = [m.strip().lower() for m in s.split(",") if m.strip()]
    bad = [m for m in modes if m not in CLI_MODES]
    if bad:
        raise SystemExit(f"Unknown train mode(s): {', '.join(bad)} (expected p1, p2, vanilla, all)")
    return modes


def list_available_tcs(walks_root: Path) -> list[str]:
    if not walks_root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(walks_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("tc"):
            continue
        if not any(p.glob("*.txt.gz")):
            continue
        graph_nt = REPO_ROOT / "v1" / "synthetic_ontology" / p.name / "synthetic_ontology" / "graph.nt"
        if graph_nt.is_file():
            out.append(p.name)
    return out


def resolve_tcs(requested: list[str] | None, walks_root: Path) -> list[str]:
    available = list_available_tcs(walks_root)
    if not available:
        raise SystemExit(
            f"No TCs with walks under {walks_root} and graphs under v1/synthetic_ontology/"
        )
    if requested is None:
        return available
    normalized = [normalize_tc(tc) for tc in requested]
    missing = [tc for tc in normalized if tc not in available]
    if missing:
        raise SystemExit(
            f"Requested TC(s) not available: {', '.join(missing)}. Available: {', '.join(available)}"
        )
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Legacy training on imported bina walks (old_protographs + old_train).",
    )
    ap.add_argument(
        "--walks-root",
        type=Path,
        default=DEFAULT_WALKS_ROOT,
        help="Root with tcXX/*.txt.gz walk shards (default: all_walks/)",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Experiment output root (default: output_bina_old/)",
    )
    ap.add_argument(
        "--tc",
        nargs="*",
        default=["tc01", "tc02", "tc03"],
        help="Test case ids (default: tc01 tc02 tc03)",
    )
    ap.add_argument(
        "--train-mode",
        dest="train_modes",
        nargs="+",
        default=list(CLI_MODES),
        help="Training modes: p1 p2 vanilla (default: all three)",
    )
    ap.add_argument("--graph-epochs", type=int, default=DEFAULT_GRAPH_EPOCHS)
    ap.add_argument("--timestamp", default=None)
    ap.add_argument(
        "--number-of-threads",
        type=int,
        default=None,
        dest="number_of_threads",
        help="Gensim workers (default: os.cpu_count(); must be >= 2 if set)",
    )
    ap.add_argument(
        "--save-each-epoch",
        action="store_true",
        default=True,
        help="Save model + KV after each graph epoch (default: True)",
    )
    ap.add_argument(
        "--no-save-each-epoch",
        action="store_false",
        dest="save_each_epoch",
        help="Do not save per-epoch checkpoints",
    )
    ap.add_argument("--no-plot", action="store_true", help="Skip loss and init-quality plotting")
    ap.add_argument(
        "--force-reembed",
        action="store_true",
        help="Delete cached protograph JAR outputs and re-embed before training",
    )
    ap.add_argument(
        "--force-rebuild-protographs",
        action="store_true",
        help="Delete cached protograph NT / entity2classes and rebuild",
    )
    ap.add_argument(
        "--vector-dim",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"Vanilla Word2Vec dimension (default: {DEFAULT_DIMENSIONS})",
    )
    ap.add_argument(
        "--jar-dim",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"jRDF2Vec protograph embedding dimension (default: {DEFAULT_DIMENSIONS})",
    )
    ap.add_argument(
        "--init-fallback",
        choices=("none", "ancestor", "p2kv"),
        default="none",
        help="When direct class vector missing: none, walk subClassOf (ancestor), or use P2 KV (p2kv)",
    )
    ap.add_argument(
        "--init-relations",
        action="store_true",
        help="Also initialize P_* relation tokens from protograph KV",
    )
    ap.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip per-stage audit JSON under output_bina_old/_cache/audit/",
    )
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip per-epoch downstream evaluation",
    )
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="Build/cache protographs and write audit records without training",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def build_train_options(args: argparse.Namespace) -> OldTrainOptions:
    return OldTrainOptions(
        vector_dim=args.vector_dim,
        jar_dim=args.jar_dim,
        init_fallback=args.init_fallback,
        init_relations=args.init_relations,
        force_rebuild_protographs=args.force_rebuild_protographs,
        audit=not args.no_audit,
        run_eval=not args.no_eval,
    )


def print_epoch0_summary(*, output_root: Path, timestamp: str, tcs: list[str]) -> None:
    print("\nEpoch-0 accuracy summary:")
    print(f"{'TC':<8} {'P1':>8} {'P2':>8} {'Vanilla':>8} {'P1>Van':>8} {'P2>Van':>8}")
    print("-" * 56)
    all_pass = True
    for tc in tcs:
        acc: dict[str, float | None] = {"p1": None, "p2": None, "vanilla": None}
        for mode in CLI_MODES:
            run_dir = run_dir_for_mode(
                output_root=output_root,
                tc=tc,
                mode=mode,
                timestamp=timestamp,
            )
            csv_path = run_dir / EPOCH_EVAL_CSV_NAME
            if not csv_path.is_file():
                continue
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["finetune_epoch"]) == 0:
                        acc[mode] = float(row["accuracy"])
                        break
        p1_ok = acc["p1"] is not None and acc["vanilla"] is not None and acc["p1"] > acc["vanilla"]
        p2_ok = acc["p2"] is not None and acc["vanilla"] is not None and acc["p2"] > acc["vanilla"]
        if acc["p1"] is None or acc["p2"] is None or acc["vanilla"] is None:
            all_pass = False
        elif not (p1_ok and p2_ok):
            all_pass = False
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        print(
            f"{tc:<8} {fmt(acc['p1']):>8} {fmt(acc['p2']):>8} {fmt(acc['vanilla']):>8} "
            f"{'yes' if p1_ok else 'NO':>8} {'yes' if p2_ok else 'NO':>8}"
        )
    if all_pass:
        print("\nAll TCs pass: P1 and P2 epoch-0 accuracy > vanilla.")
    else:
        print("\nSome TCs fail init-quality criterion (P1/P2 epoch-0 > vanilla).")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    walks_root = args.walks_root.resolve()
    output_root = args.output_root.resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    workers = resolve_workers(args.number_of_threads)
    train_options = build_train_options(args)
    if train_options.vector_dim != train_options.jar_dim:
        logger.warning(
            "vector_dim=%d differs from jar_dim=%d; vanilla and P1/P2 may not be comparable",
            train_options.vector_dim,
            train_options.jar_dim,
        )

    train_modes: list[str] = []
    for mode_arg in args.train_modes:
        train_modes.extend(parse_train_modes(mode_arg))
    deduped: list[str] = []
    seen_modes: set[str] = set()
    for mode in train_modes:
        if mode not in seen_modes:
            seen_modes.add(mode)
            deduped.append(mode)
    train_modes = deduped

    tcs = resolve_tcs(args.tc, walks_root)

    print(f"Walks root:   {walks_root}")
    print(f"Output root:  {output_root}")
    print(f"TCs:          {', '.join(tcs)}")
    print(f"Modes:        {', '.join(train_modes)}")
    print(f"Workers:      {workers}")
    print(f"Graph epochs: {args.graph_epochs}")
    print(f"Vector dim:   {train_options.vector_dim}")
    print(f"JAR dim:      {train_options.jar_dim}")
    print(f"Init fallback:{train_options.init_fallback}")
    print(f"Init relations:{train_options.init_relations}")
    print(f"Timestamp:    {timestamp}")

    protos_needed = tuple(m for m in train_modes if m in ("p1", "p2"))

    materialized_by_tc: dict[str, Path] = {}
    for tc in tcs:
        materialized_by_tc[tc] = materialize_imported_walks(
            tc,
            walks_root=walks_root,
            cache_root=output_root,
            dry_run=args.dry_run,
        )
        if protos_needed:
            ensure_protograph_prereqs(
                tc=tc,
                protos=protos_needed,
                output_root=output_root,
                dry_run=args.dry_run,
                force_reembed=args.force_reembed,
                train_options=train_options,
                timestamp=timestamp,
            )

    if args.audit_only:
        print("\nAudit-only run complete.")
        for tc in tcs:
            audit_path = audit_record_path(output_root, tc, timestamp)
            if audit_path.is_file():
                print(f"  {tc}: {audit_path}")
        return

    total = len(train_modes) * len(tcs)
    run_idx = 0

    for mode in train_modes:
        for tc in tcs:
            run_idx += 1
            out_dir = run_dir_for_mode(
                output_root=output_root,
                tc=tc,
                mode=mode,
                timestamp=timestamp,
            )
            walks_path = materialized_by_tc[tc]
            header = f"Run {run_idx}/{total} | {tc} | {mode} | {out_dir}"
            print(f"\n{'=' * 72}\n{header}\n{'=' * 72}")

            if mode in ("p1", "p2"):
                embed_dir = protograph_embed_dir(output_root, tc, mode)
                proto_kv = (
                    resolve_jar_kv(embed_dir)
                    if embed_dir.is_dir()
                    else protograph_kv_path(output_root, tc, mode)
                )
                run_graph_train_p1_p2(
                    tc=tc,
                    proto=mode,
                    graph_walks_path=walks_path,
                    proto_kv_path=proto_kv,
                    mapping_path=protograph_cache_dir(output_root, tc) / "entity2classes.json",
                    out_dir=out_dir,
                    graph_epochs=args.graph_epochs,
                    save_each_epoch=args.save_each_epoch,
                    workers=workers,
                    timestamp=timestamp,
                    materialized_walks=walks_path,
                    output_root=output_root,
                    train_options=train_options,
                    dry_run=args.dry_run,
                )
            else:
                run_graph_train_vanilla(
                    tc=tc,
                    graph_walks_path=walks_path,
                    out_dir=out_dir,
                    graph_epochs=args.graph_epochs,
                    save_each_epoch=args.save_each_epoch,
                    workers=workers,
                    timestamp=timestamp,
                    materialized_walks=walks_path,
                    output_root=output_root,
                    train_options=train_options,
                    dry_run=args.dry_run,
                )

    if not args.dry_run and train_options.run_eval:
        print_epoch0_summary(output_root=output_root, timestamp=timestamp, tcs=tcs)

    if not args.no_plot and not args.dry_run:
        plot_out = REPO_ROOT / "plots" / "bina_old"
        plot_training_losses(
            output_root=output_root,
            timestamp=timestamp,
            tcs=tcs,
            plot_out_dir=plot_out,
        )
        if train_options.run_eval:
            plot_init_quality(
                output_root=output_root,
                timestamp=timestamp,
                tcs=tcs,
                plot_out_dir=plot_out,
            )

    print(f"\nAll runs finished ({total} total).")


if __name__ == "__main__":
    main()
