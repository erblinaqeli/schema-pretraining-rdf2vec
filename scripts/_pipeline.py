#!/usr/bin/env python3
"""
Shared synthetic training pipeline: config flattening, walks, Word2Vec subprocess, eval.

Used by scripts/train.py, scripts/run_finetune_epoch_eval.py, and grid search.
Hyperparameter sweeps stay in scripts/grid_search/run_grid_search.py.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import re
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

try:
    import yaml
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyYAML is required. Install project deps: uv sync"
    ) from e

WalkRole = Literal["pretrain_proto", "instance", "no_pretrain"]

# scripts/ — anchors the sibling helper scripts (_walks.py, _word2vec.py,
# _evaluate.py, _protograph_gen.py) that _pipeline launches as subprocesses.
SCRIPTS_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    # Single source of truth: pyproject-anchored discovery in _kg_io.
    from _kg_io import repo_root as _repo_root

    return _repo_root()



def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _sample_from_lists(lists: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    """One independent random draw per key (uniform over each list)."""
    out: dict[str, Any] = {}
    for k, vals in lists.items():
        choices = as_list(vals)
        if not choices:
            raise SystemExit(f"Random search: empty list for key {k!r}")
        out[k] = rng.choice(choices)
    return out


def sample_random_run(
    cfg: dict[str, Any], rng: random.Random, tm: str
) -> tuple[str, dict[str, Any]]:
    """Sample one configuration: one draw per hyperparameter list for the given training mode."""
    tm = str(tm).strip()

    pre = cfg.get("pretrain") or {}
    fin = cfg.get("finetune") or {}

    pre_w = normalize_walk_block(pre.get("walks"))
    pre_w2v = normalize_w2v_block(pre.get("word2vec"))
    fin_w = normalize_walk_block(fin.get("walks"))
    fin_w2v = normalize_w2v_block(fin.get("word2vec"))

    w2v_lists = {k: v for k, v in fin_w2v.items() if k != "finetune_epochs"}
    if tm == "no_pretrain":
        if not w2v_lists.get("epochs"):
            raise SystemExit(
                "finetune.word2vec.epochs is required when --training-mode is no_pretrain"
            )
        flat = {
            "training_mode": tm,
            "finetune_walks": _sample_from_lists(fin_w, rng),
            "finetune_word2vec": _sample_from_lists(w2v_lists, rng),
        }
        return tm, flat

    if tm in ("p1", "p2"):
        w2v_lists = {k: v for k, v in fin_w2v.items() if k != "epochs"}
        if not w2v_lists.get("finetune_epochs"):
            raise SystemExit(
                "finetune.word2vec.finetune_epochs is required when --training-mode is p1 or p2"
            )
        flat = {
            "training_mode": tm,
            "pretrain_walks": _sample_from_lists(pre_w, rng),
            "pretrain_word2vec": _sample_from_lists(pre_w2v, rng),
            "finetune_walks": _sample_from_lists(fin_w, rng),
            "finetune_word2vec": _sample_from_lists(w2v_lists, rng),
        }
        return tm, flat

    raise SystemExit(f"Unknown training_mode: {tm!r} (expected no_pretrain, p1, p2)")


def _suggest_block_from_lists(
    trial: Any,
    prefix: str,
    lists: dict[str, list[Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, vals in lists.items():
        choices = as_list(vals)
        if not choices:
            raise SystemExit(f"Optuna search: empty list for key {prefix}/{k!r}")
        out[k] = trial.suggest_categorical(f"{prefix}/{k}", choices)
    return out


def suggest_flat_from_optuna(cfg: dict[str, Any], trial: Any, tm: str) -> tuple[str, dict[str, Any]]:
    """Suggest one hyperparameter configuration via Optuna (TPE over YAML lists)."""
    tm = str(tm).strip()

    pre = cfg.get("pretrain") or {}
    fin = cfg.get("finetune") or {}

    pre_w = normalize_walk_block(pre.get("walks"))
    pre_w2v = normalize_w2v_block(pre.get("word2vec"))
    fin_w = normalize_walk_block(fin.get("walks"))
    fin_w2v = normalize_w2v_block(fin.get("word2vec"))

    if tm == "no_pretrain":
        w2v_lists = {k: v for k, v in fin_w2v.items() if k != "finetune_epochs"}
        if not w2v_lists.get("epochs"):
            raise SystemExit(
                "finetune.word2vec.epochs is required when --training-mode is no_pretrain"
            )
        flat = {
            "training_mode": tm,
            "finetune_walks": _suggest_block_from_lists(trial, "finetune_walks", fin_w),
            "finetune_word2vec": _suggest_block_from_lists(trial, "finetune_word2vec", w2v_lists),
        }
        return tm, flat

    if tm in ("p1", "p2"):
        w2v_lists = {k: v for k, v in fin_w2v.items() if k != "epochs"}
        if not w2v_lists.get("finetune_epochs"):
            raise SystemExit(
                "finetune.word2vec.finetune_epochs is required when --training-mode is p1 or p2"
            )
        flat = {
            "training_mode": tm,
            "pretrain_walks": _suggest_block_from_lists(trial, "pretrain_walks", pre_w),
            "pretrain_word2vec": _suggest_block_from_lists(trial, "pretrain_word2vec", pre_w2v),
            "finetune_walks": _suggest_block_from_lists(trial, "finetune_walks", fin_w),
            "finetune_word2vec": _suggest_block_from_lists(trial, "finetune_word2vec", w2v_lists),
        }
        return tm, flat

    raise SystemExit(f"Unknown training_mode: {tm!r} (expected no_pretrain, p1, p2)")


def dict_product(d: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Cartesian product over keys; each value must be a list (use as_list)."""
    keys = list(d.keys())
    if not keys:
        yield {}
        return
    lists = [as_list(d[k]) for k in keys]
    for combo in itertools.product(*lists):
        yield dict(zip(keys, combo))


def normalize_walk_block(block: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not block:
        return {}
    out: dict[str, list[Any]] = {}
    for k, v in block.items():
        if k.startswith("#") or v is None:
            continue
        out[k] = as_list(v)
    return out


def normalize_w2v_block(block: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not block:
        return {}
    out: dict[str, list[Any]] = {}
    for k, v in block.items():
        if k.startswith("#") or v is None:
            continue
        out[k] = as_list(v)
    return out


def walk_cli_args(
    w: dict[str, Any],
    *,
    tm: str,
    role: WalkRole,
) -> list[str]:
    m = w.get("mode")
    if m is None:
        raise ValueError("walk params missing 'mode'")
    args = [
        "--mode",
        str(m),
        "--depth",
        str(int(w["depth"])),
        "--walks-per-entity",
        str(int(w["walks_per_entity"])),
    ]
    if "token_format" in w and w["token_format"] is not None:
        args.extend(["--token-format", str(w["token_format"])])

    from _kg_io import DEFAULT_WALK_SEED

    seed: int | None = None
    if "random_seed" in w and w["random_seed"] is not None:
        seed = int(w["random_seed"])
    elif role == "pretrain_proto" and tm in ("p1", "p2"):
        seed = DEFAULT_WALK_SEED
    elif role in ("instance", "no_pretrain"):
        seed = DEFAULT_WALK_SEED

    if seed is not None:
        args.extend(["--seed", str(seed)])

    # Protograph pretrain walks must cover every schema triple so all P1/P2
    # relations get protograph-derived embeddings (mirrors walk_cache_key "cov").
    if role == "pretrain_proto" and str(m) == "jrdf2vec-duplicate-free":
        args.append("--ensure-triple-coverage")
    return args


def pretrain_walk_cache_key(pw: dict[str, Any]) -> str:
    """Backward-compatible alias; prefer ``_io.walk_cache_key``."""
    from _kg_io import walk_cache_key

    return walk_cache_key(pw, tm="p1", role="pretrain_proto")


def protograph_script_argv(*, ontology_nt: Path, out_dir: Path) -> list[str]:
    return [
        *uv_run_python(SCRIPTS_DIR / "_protograph_gen.py"),
        "--schema",
        str(ontology_nt),
        "--ontology-only",
        "--out-dir",
        str(out_dir),
    ]


def train_w2v_common_args(w: dict[str, Any]) -> list[str]:
    args = [
        "--architecture",
        str(w["architecture"]),
        "--dim",
        str(int(w["dim"])),
        "--window",
        str(int(w["window"])),
        "--negative",
        str(int(w["negative"])),
        "--lr",
        str(float(w["lr"])),
        "--min-alpha",
        str(float(w["min_alpha"])),
        "--min-count",
        str(int(w["min_count"])),
    ]
    if "sample" in w:
        args.extend(["--sample", str(float(w["sample"]))])
    if "hs" in w:
        args.extend(["--hs", str(int(w["hs"]))])
    return args


def finetune_init_cli_args(w: dict[str, Any]) -> list[str]:
    """CLI flags for finetune embedding initialization (P1/P2 MASCHInE path)."""
    args: list[str] = []
    init_strategy = w.get("init_strategy", "most_specific")
    args.extend(["--init-strategy", str(init_strategy)])
    init_relations = w.get("init_relations", True)
    args.append("--init-relations" if init_relations else "--no-init-relations")
    initialization_noise = float(w.get("initialization_noise", 0.0))
    args.extend(["--initialization-noise", str(initialization_noise)])
    anchor_regularization = float(w.get("anchor_regularization", 0.0))
    args.extend(["--anchor-regularization", str(anchor_regularization)])
    return args


def uv_bin() -> str:
    u = shutil.which("uv")
    if not u:
        raise SystemExit(
            "Could not find 'uv' on PATH. Install uv or run from the same environment as the e2e scripts."
        )
    return u


def uv_run_python(script: Path) -> list[str]:
    return [uv_bin(), "run", "python", str(script)]


def run_cmd(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool,
    log_path: Path | None = None,
    log_append: bool = False,
    log_section: str | None = None,
) -> None:
    if dry_run:
        cmd_line = " ".join(shlex_quote(a) for a in argv)
        print(cmd_line)
        if log_path is not None:
            print(f"  # log: {log_path}")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if log_append else "w"
            with log_path.open(mode, encoding="utf-8") as lf:
                if log_section:
                    lf.write(f"\n{'=' * 72}\n{log_section}\n{'=' * 72}\n")
                lf.write(f"$ {cmd_line}\n")
        return
    merged = os.environ.copy()
    if env:
        merged.update(env)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if log_append else "w"
        with log_path.open(mode, encoding="utf-8") as lf:
            if log_section:
                lf.write(f"\n{'=' * 72}\n{log_section}\n{'=' * 72}\n")
            lf.write(f"$ {' '.join(shlex_quote(a) for a in argv)}\n\n")
            lf.flush()
            subprocess.run(
                argv,
                cwd=cwd,
                check=True,
                env=merged,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lf.write("\n")
    else:
        subprocess.run(argv, cwd=cwd, check=True, env=merged)


def shlex_quote(s: str) -> str:
    if not s or any(c in s for c in " \t\n\"'"):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def parse_accuracy_from_eval_log(path: Path) -> float | None:
    """Read best-model test accuracy from eval_metrics.txt."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    in_best_section = False
    for line in text.splitlines():
        if line.startswith("Best model:"):
            in_best_section = True
            continue
        if not in_best_section:
            continue
        if line.startswith("─"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "accuracy":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def read_accuracy(run_dir: Path) -> float | None:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            acc = data.get("accuracy")
            if acc is not None:
                return float(acc)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return parse_accuracy_from_eval_log(run_dir / "eval_metrics.txt")


def tc_paths(tc: str, root: Path) -> dict[str, Path]:
    base = root / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return {
        "base": base,
        "graph_nt": base / "graph.nt",
        "ontology_nt": base / "ontology.nt",
        "test_txt": base / "1000" / "train_test" / "test.txt",
    }


def iter_runs(cfg: dict[str, Any], tm: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (training_mode, flat_param_dict) for each grid point for the given training mode."""
    tm = str(tm).strip()
    pre = cfg.get("pretrain") or {}
    fin = cfg.get("finetune") or {}

    pre_w = normalize_walk_block(pre.get("walks"))
    pre_w2v = normalize_w2v_block(pre.get("word2vec"))
    fin_w = normalize_walk_block(fin.get("walks"))
    fin_w2v = normalize_w2v_block(fin.get("word2vec"))

    if tm == "no_pretrain":
        w2v = {k: v for k, v in fin_w2v.items() if k != "finetune_epochs"}
        if not w2v.get("epochs"):
            raise SystemExit(
                "finetune.word2vec.epochs is required when --training-mode is no_pretrain"
            )
        for fw in dict_product(fin_w):
            for fv in dict_product(w2v):
                flat = {
                    "training_mode": tm,
                    "finetune_walks": fw,
                    "finetune_word2vec": fv,
                }
                yield tm, flat
    elif tm in ("p1", "p2"):
        w2v = {k: v for k, v in fin_w2v.items() if k != "epochs"}
        if not w2v.get("finetune_epochs"):
            raise SystemExit(
                "finetune.word2vec.finetune_epochs is required when --training-mode is p1 or p2"
            )
        for pw in dict_product(pre_w):
            for p2v in dict_product(pre_w2v):
                for fw in dict_product(fin_w):
                    for fv in dict_product(w2v):
                        flat = {
                            "training_mode": tm,
                            "pretrain_walks": pw,
                            "pretrain_word2vec": p2v,
                            "finetune_walks": fw,
                            "finetune_word2vec": fv,
                        }
                        yield tm, flat
    else:
        raise SystemExit(f"Unknown --training-mode: {tm!r} (expected no_pretrain, p1, p2)")


# Process-wide registry of cache locks. SharedArtifacts instances created in
# different threads (e.g. --run-5-times --rep-jobs > 1) must serialize on the
# same lock when they target the same cache scope, otherwise concurrent cold
# builds corrupt shared walks/protograph files.
_CACHE_LOCKS: dict[str, threading.RLock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _cache_lock_for(key: str) -> threading.RLock:
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _CACHE_LOCKS[key] = lock
        return lock


@dataclass
class SharedArtifacts:
    """Artifact cache: global ``output/_cache/`` or sweep-local ``{root}/shared/``."""

    paths: dict[str, Path]
    tm: str
    dry_run: bool
    dataset: str = "synthetic"
    tc: str | None = None
    root: Path | None = None

    @property
    def lock(self) -> threading.RLock:
        if self.root is not None:
            key = f"root:{self.root.resolve()}"
        else:
            key = f"global:{self.dataset}:{self.tc or ''}"
        return _cache_lock_for(key)

    @property
    def use_global_cache(self) -> bool:
        return self.root is None

    @property
    def shared_dir(self) -> Path:
        if self.root is not None:
            return self.root / "shared"
        from _kg_io import cache_root

        return cache_root()

    @property
    def build_log(self) -> Path:
        if self.root is not None:
            return self.shared_dir / "build.log"
        from _kg_io import cache_root

        d = cache_root() / "_build_logs"
        name = f"{self.dataset}_{self.tc or 'global'}.log"
        return d / name

    def _protograph_dir(self) -> Path:
        if self.root is not None:
            return self.shared_dir
        from _kg_io import protograph_cache_dir

        return protograph_cache_dir(dataset=self.dataset, tc=self.tc)

    def _protograph_paths(self) -> tuple[Path, Path]:
        d = self._protograph_dir()
        return d / "protograph_p1.nt", d / "protograph_p2.nt"

    def _entity_class_paths(self) -> tuple[Path, Path]:
        d = self._protograph_dir()
        return d / "entity2classes.json", d / "entity2classes_hier.json"

    def ensure_protograph(self) -> tuple[Path, Path]:
        from _kg_io import check_cache_stale, write_cache_meta

        prot_p1, prot_p2 = self._protograph_paths()
        entity2classes, entity2classes_hier = self._entity_class_paths()
        if self.tm not in ("p1", "p2"):
            raise ValueError("protograph only for p1/p2")
        meta_path = self._protograph_dir() / "meta.json"
        ontology_nt = self.paths["ontology_nt"]
        with self.lock:
            have_files = (
                prot_p1.is_file()
                and (self.tm == "p1" or prot_p2.is_file())
                and entity2classes.is_file()
                and entity2classes_hier.is_file()
            )
            if have_files and not check_cache_stale(meta_path, ontology_nt):
                return prot_p1, prot_p2
            if not self.dry_run:
                self._protograph_dir().mkdir(parents=True, exist_ok=True)
            run_cmd(
                protograph_script_argv(
                    ontology_nt=ontology_nt,
                    out_dir=self._protograph_dir(),
                ),
                cwd=repo_root(),
                dry_run=self.dry_run,
                log_path=self.build_log,
                log_append=self.build_log.is_file(),
                log_section="protograph (cache)",
            )
            if not self.dry_run:
                write_cache_meta(
                    meta_path,
                    {
                        "dataset": self.dataset,
                        "tc": self.tc,
                        "ontology": str(ontology_nt),
                        "ontology_only": True,
                        "ontology_mtime": ontology_nt.stat().st_mtime if ontology_nt.is_file() else None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
        return prot_p1, prot_p2

    def ensure_instance_walks(self, fw: dict[str, Any]) -> Path:
        if self.root is not None:
            dest = self.shared_dir / "walks_instance.txt"
            with self.lock:
                if dest.is_file():
                    return dest
                if not self.dry_run:
                    self.shared_dir.mkdir(parents=True, exist_ok=True)
                self._generate_walks(self.paths["graph_nt"], dest, fw, role="instance")
            return dest

        from _kg_io import (
            check_cache_stale,
            walk_cache_dir,
            walk_cache_file,
            walk_cache_key,
            write_cache_meta,
        )

        wkey = walk_cache_key(fw, tm=self.tm, role="instance")
        dest = walk_cache_file(
            dataset=self.dataset,
            tc=self.tc,
            role="instance",
            walk_key=wkey,
        )
        meta_path = walk_cache_dir(
            dataset=self.dataset,
            tc=self.tc,
            role="instance",
            walk_key=wkey,
        ) / "meta.json"
        graph_nt = self.paths["graph_nt"]
        with self.lock:
            if dest.is_file() and not check_cache_stale(meta_path, graph_nt):
                return dest
            if not self.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
            self._generate_walks(graph_nt, dest, fw, role="instance")
            if not self.dry_run:
                write_cache_meta(
                    meta_path,
                    {
                        "walk_key": wkey,
                        "role": "instance",
                        "dataset": self.dataset,
                        "tc": self.tc,
                        "params": fw,
                        "input_graph": str(graph_nt),
                        "input_mtime": graph_nt.stat().st_mtime,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
        return dest

    def ensure_pretrain_walks(self, pw: dict[str, Any]) -> Path:
        if self.root is not None:
            cache_dir = self.shared_dir / "walks_pretrain"
            dest = cache_dir / f"{pretrain_walk_cache_key(pw)}.txt"
            with self.lock:
                if dest.is_file():
                    return dest
                if not self.dry_run:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                prot_p1, prot_p2 = self.ensure_protograph()
                proto_nt = prot_p1 if self.tm == "p1" else prot_p2
                self._generate_walks(proto_nt, dest, pw, role="pretrain_proto")
            return dest

        from _kg_io import (
            check_cache_stale,
            walk_cache_dir,
            walk_cache_file,
            walk_cache_key,
            write_cache_meta,
        )

        wkey = walk_cache_key(pw, tm=self.tm, role="pretrain_proto")
        dest = walk_cache_file(
            dataset=self.dataset,
            tc=self.tc,
            role="pretrain",
            walk_key=wkey,
            tm=self.tm,
        )
        meta_path = walk_cache_dir(
            dataset=self.dataset,
            tc=self.tc,
            role="pretrain",
            walk_key=wkey,
            tm=self.tm,
        ) / "meta.json"
        with self.lock:
            prot_p1, prot_p2 = self.ensure_protograph()
            proto_nt = prot_p1 if self.tm == "p1" else prot_p2
            if dest.is_file() and not check_cache_stale(meta_path, proto_nt):
                return dest
            if not self.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
            self._generate_walks(proto_nt, dest, pw, role="pretrain_proto")
            if not self.dry_run:
                write_cache_meta(
                    meta_path,
                    {
                        "walk_key": wkey,
                        "role": "pretrain",
                        "dataset": self.dataset,
                        "tc": self.tc,
                        "params": pw,
                        "input_graph": str(proto_nt),
                        "input_mtime": proto_nt.stat().st_mtime,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
        return dest

    def artifact_refs(self, fw: dict[str, Any], pw: dict[str, Any] | None) -> dict[str, str]:
        if self.root is not None:
            refs: dict[str, str] = {"shared_dir": str(self.shared_dir)}
            if self.tm in ("p1", "p2"):
                refs["protograph"] = str(self.shared_dir)
            return refs

        from _kg_io import (
            protograph_cache_dir,
            walk_cache_dir,
            walk_cache_key,
        )

        refs: dict[str, str] = {}
        if self.tm in ("p1", "p2"):
            refs["protograph"] = str(protograph_cache_dir(dataset=self.dataset, tc=self.tc))
            ikey = walk_cache_key(fw, tm=self.tm, role="instance")
            refs["instance_walks"] = str(
                walk_cache_dir(
                    dataset=self.dataset,
                    tc=self.tc,
                    role="instance",
                    walk_key=ikey,
                )
            )
            if pw is not None:
                pkey = walk_cache_key(pw, tm=self.tm, role="pretrain_proto")
                refs["pretrain_walks"] = str(
                    walk_cache_dir(
                        dataset=self.dataset,
                        tc=self.tc,
                        role="pretrain",
                        walk_key=pkey,
                        tm=self.tm,
                    )
                )
        else:
            ikey = walk_cache_key(fw, tm=self.tm, role="instance")
            refs["instance_walks"] = str(
                walk_cache_dir(
                    dataset=self.dataset,
                    tc=self.tc,
                    role="instance",
                    walk_key=ikey,
                )
            )
        return refs

    def _generate_walks(
        self,
        input_nt: Path,
        output_walks: Path,
        wdict: dict[str, Any],
        *,
        role: WalkRole,
    ) -> None:
        mode = str(wdict.get("mode", "jrdf2vec-duplicate-free"))
        if mode == "jrdf2vec-jar":
            from _jrdf2vec_jar import generate_and_merge_walks

            if not self.dry_run:
                output_walks.parent.mkdir(parents=True, exist_ok=True)
            line = (
                f"jRDF2Vec JAR walks: graph={input_nt} -> {output_walks} "
                f"(depth={wdict['depth']}, walks_per_entity={wdict['walks_per_entity']})"
            )
            print(line, flush=True)
            if self.build_log.is_file() or not self.dry_run:
                self.build_log.parent.mkdir(parents=True, exist_ok=True)
                with self.build_log.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n== random_walks ({role}, cache) ==\n{line}\n")
            generate_and_merge_walks(
                graph_nt=input_nt,
                output_walks=output_walks,
                number_of_walks=int(wdict["walks_per_entity"]),
                depth=int(wdict["depth"]),
                dry_run=self.dry_run,
            )
            return

        argv = [
            *uv_run_python(SCRIPTS_DIR / "_walks.py"),
            str(input_nt),
            str(output_walks),
            *walk_cli_args(wdict, tm=self.tm, role=role),
        ]
        run_cmd(
            argv,
            cwd=repo_root(),
            dry_run=self.dry_run,
            log_path=self.build_log,
            log_append=self.build_log.is_file(),
            log_section=f"random_walks ({role}, cache)",
        )


def _resolve_run_dir(out_dir: Path, run_index: int, run_subdir: str | None) -> Path:
    """Resolve per-run output directory under *out_dir*."""
    if run_subdir is not None:
        if run_subdir in (".", ""):
            return out_dir
        return out_dir / run_subdir
    return out_dir / f"run_{run_index:04d}"


def run_one(
    *,
    run_index: int,
    tm: str,
    flat: dict[str, Any],
    paths: dict[str, Path],
    out_dir: Path,
    dry_run: bool,
    no_eval: bool,
    workers: int | None,
    seed: int | None,
    reuse_shared: bool,
    shared: SharedArtifacts | None,
    save_finetune_epoch_checkpoints: bool = False,
    run_subdir: str | None = None,
    precomputed_instance_walks: Path | None = None,
    log_preamble: str | None = None,
    dataset: str = "synthetic",
    tc: str | None = None,
    all_classifiers: bool = False,
) -> None:
    from _kg_io import RuntimeMetrics, merge_runtime_metrics, write_artifact_refs

    root = repo_root()
    graph_nt = paths["graph_nt"]
    ontology_nt = paths["ontology_nt"]
    test_txt = paths["test_txt"]
    pipeline_metrics = RuntimeMetrics()

    run_dir = _resolve_run_dir(out_dir, run_index, run_subdir)
    run_label = run_dir.name if run_subdir in (".", "") else f"run_{run_index:04d}"
    run_log = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    preamble = log_preamble or ""
    meta = (
        f"# {run_label}  training_mode={tm}\n"
        f"# started {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
    )
    if dry_run:
        if not run_log.is_file():
            run_log.write_text(preamble + meta, encoding="utf-8")
    else:
        run_log.write_text(preamble + meta, encoding="utf-8")
        (run_dir / "params.json").write_text(
            json.dumps(flat, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    checkpoint = (
        run_dir / "rdf2vec_word2vec.pt"
        if tm == "no_pretrain"
        else run_dir / "rdf2vec_final.pt"
    )

    def _walk_cmd(
        input_nt: Path,
        output_walks: Path,
        wdict: dict[str, Any],
        *,
        role: WalkRole,
        metric_key: str,
    ) -> None:
        mode = str(wdict.get("mode", "jrdf2vec-duplicate-free"))
        with pipeline_metrics.stage(metric_key):
            if mode == "jrdf2vec-jar":
                from _jrdf2vec_jar import generate_and_merge_walks

                if not dry_run:
                    output_walks.parent.mkdir(parents=True, exist_ok=True)
                line = (
                    f"jRDF2Vec JAR walks: graph={input_nt} -> {output_walks} "
                    f"(depth={wdict['depth']}, walks_per_entity={wdict['walks_per_entity']})"
                )
                print(line, flush=True)
                if dry_run:
                    with run_log.open("a", encoding="utf-8") as lf:
                        lf.write(f"\n== random_walks ({role}) ==\n{line}\n")
                else:
                    with run_log.open("a", encoding="utf-8") as lf:
                        lf.write(f"\n== random_walks ({role}) ==\n{line}\n")
                generate_and_merge_walks(
                    graph_nt=input_nt,
                    output_walks=output_walks,
                    number_of_walks=int(wdict["walks_per_entity"]),
                    depth=int(wdict["depth"]),
                    dry_run=dry_run,
                )
                return

            argv = [
                *uv_run_python(SCRIPTS_DIR / "_walks.py"),
                str(input_nt),
                str(output_walks),
                *walk_cli_args(wdict, tm=tm, role=role),
            ]
            run_cmd(
                argv,
                cwd=root,
                dry_run=dry_run,
                log_path=run_log,
                log_append=True,
                log_section=f"random_walks ({role})",
            )

    def _train_cmd(extra: list[str], walks_positional: Path | None = None) -> None:
        argv = [*uv_run_python(SCRIPTS_DIR / "_word2vec.py")]
        if walks_positional is not None:
            argv.append(str(walks_positional))
        argv.extend(extra)
        argv.extend(["-o", str(checkpoint)])
        if save_finetune_epoch_checkpoints:
            argv.extend(
                [
                    "--save-finetune-epoch-checkpoints",
                    "--finetune-epoch-checkpoint-dir",
                    str(run_dir / "ckpt"),
                ]
            )
        if workers is not None:
            argv.extend(["--workers", str(workers)])
        if seed is not None:
            argv.extend(["--seed", str(seed)])
        run_cmd(
            argv,
            cwd=root,
            dry_run=dry_run,
            log_path=run_log,
            log_append=True,
            log_section="train_word2vec",
        )

    def _run_p1_p2_pipeline(walks_proto: Path, walks_instance: Path) -> None:
        pw = flat["pretrain_walks"]
        p2v = flat["pretrain_word2vec"]
        fv = flat["finetune_word2vec"]

        train_extra = [
            "--mode",
            tm,
            "--pretrain-walks",
            str(walks_proto),
            "--instance-walks",
            str(walks_instance),
            "--ontology",
            str(ontology_nt),
            "--out-dir",
            str(run_dir),
            "--no-loss-plots",
            *train_w2v_common_args(fv),
            *finetune_init_cli_args(fv),
            "--pretrain-epochs",
            str(int(p2v["pretrain_epochs"])),
        ]
        if "pretrain_lr" in p2v:
            train_extra.extend(["--pretrain-lr", str(float(p2v["pretrain_lr"]))])
        if "pretrain_min_alpha" in p2v:
            train_extra.extend(["--pretrain-min-alpha", str(float(p2v["pretrain_min_alpha"]))])
        train_extra.extend(
            [
                "--finetune-epochs",
                str(int(fv["finetune_epochs"])),
            ]
        )
        _train_cmd(train_extra, walks_positional=None)

    def _run_ephemeral_walks(walk_dir: Path) -> None:
        walks_instance = walk_dir / "walks_instance.txt"
        if tm == "no_pretrain":
            fw = flat["finetune_walks"]
            fv = flat["finetune_word2vec"]
            if precomputed_instance_walks is not None:
                instance_path = precomputed_instance_walks
            elif reuse_shared and shared is not None:
                with pipeline_metrics.stage("random_walks_instance"):
                    instance_path = shared.ensure_instance_walks(fw)
                write_artifact_refs(
                    run_dir,
                    shared.artifact_refs(fw, None),
                    dry_run=dry_run,
                )
            else:
                _walk_cmd(
                    graph_nt,
                    walks_instance,
                    fw,
                    role="no_pretrain",
                    metric_key="random_walks_instance",
                )
                instance_path = walks_instance
            _train_cmd(
                [
                    "--mode",
                    "none",
                    "--no-plot",
                    *train_w2v_common_args(fv),
                    "--epochs",
                    str(int(fv["epochs"])),
                ],
                walks_positional=instance_path,
            )
            return

        fw = flat["finetune_walks"]
        pw = flat["pretrain_walks"]
        walks_proto = walk_dir / ("walks_p1.txt" if tm == "p1" else "walks_p2.txt")

        if reuse_shared and shared is not None:
            if precomputed_instance_walks is not None:
                walks_instance_path = precomputed_instance_walks
            else:
                with pipeline_metrics.stage("random_walks_instance"):
                    walks_instance_path = shared.ensure_instance_walks(fw)
            with pipeline_metrics.stage("protograph_generation"):
                shared.ensure_protograph()
            with pipeline_metrics.stage("random_walks_protograph"):
                walks_proto_path = shared.ensure_pretrain_walks(pw)
            pw_ref = pw
            refs = shared.artifact_refs(fw, pw_ref)
            write_artifact_refs(run_dir, refs, dry_run=dry_run)
            _run_p1_p2_pipeline(walks_proto_path, walks_instance_path)
            return

        prot_p1 = run_dir / "protograph_p1.nt"
        prot_p2 = run_dir / "protograph_p2.nt"

        prot_argv = protograph_script_argv(
            ontology_nt=ontology_nt,
            out_dir=run_dir,
        )
        with pipeline_metrics.stage("protograph_generation"):
            run_cmd(
                prot_argv,
                cwd=root,
                dry_run=dry_run,
                log_path=run_log,
                log_append=True,
                log_section="protograph",
            )

        proto_nt = prot_p1 if tm == "p1" else prot_p2
        _walk_cmd(
            proto_nt,
            walks_proto,
            pw,
            role="pretrain_proto",
            metric_key="random_walks_protograph",
        )
        if precomputed_instance_walks is not None:
            walks_instance_path = precomputed_instance_walks
        else:
            _walk_cmd(
                graph_nt,
                walks_instance,
                fw,
                role="instance",
                metric_key="random_walks_instance",
            )
            walks_instance_path = walks_instance
        _run_p1_p2_pipeline(walks_proto, walks_instance_path)

    if dry_run:
        _run_ephemeral_walks(Path(tempfile.gettempdir()) / f"kg_grid_dryrun_run{run_index}")
    else:
        if tm in ("p1", "p2") and reuse_shared and shared is not None:
            _run_ephemeral_walks(Path(tempfile.gettempdir()) / f"kg_grid_shared_run{run_index}")
        else:
            with tempfile.TemporaryDirectory(prefix="kg_grid_walks_") as td:
                _run_ephemeral_walks(Path(td))

    if save_finetune_epoch_checkpoints:
        with pipeline_metrics.stage("per_epoch_evaluation"):
            rows = run_per_epoch_eval(
                test_txt=test_txt,
                ckpt_dir=run_dir / "ckpt",
                run_dir=run_dir,
                flat=flat,
                tm=tm,
                run_log=run_log,
                dry_run=dry_run,
                all_classifiers=all_classifiers,
            )
        write_epoch_eval_csv(run_dir / "epoch_eval_accuracy.csv", rows)

    if not no_eval and not save_finetune_epoch_checkpoints:
        eval_log = run_dir / "eval_metrics.txt"
        if dry_run:
            eval_argv = [
                *uv_run_python(SCRIPTS_DIR / "_evaluate.py"),
                str(test_txt),
                "-c",
                str(checkpoint),
            ]
            if all_classifiers:
                eval_argv.append("--all-classifiers")
            with pipeline_metrics.stage("evaluation"):
                run_cmd(eval_argv, cwd=root, dry_run=True, log_path=eval_log)
        else:
            with pipeline_metrics.stage("evaluation"):
                run_eval_inprocess(
                    test_txt,
                    checkpoint,
                    run_dir,
                    eval_log,
                    run_log=run_log,
                    all_classifiers=all_classifiers,
                )

    if not dry_run:
        json_path, txt_path = merge_runtime_metrics(run_dir, pipeline_metrics)
        print(f"Wrote runtime metrics: {txt_path}", flush=True)


_FINETUNE_EPOCH_CKPT_RE = re.compile(r"^finetune_epoch_(\d+)\.pt$")


def discover_finetune_epoch_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return (epoch, path) pairs sorted ascending from ckpt_dir."""
    found: list[tuple[int, Path]] = []
    if not ckpt_dir.is_dir():
        return found
    for path in ckpt_dir.iterdir():
        if not path.is_file():
            continue
        match = _FINETUNE_EPOCH_CKPT_RE.match(path.name)
        if match is None:
            continue
        found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


def _format_eval_metrics_lines(res: dict[str, Any]) -> list[str]:
    from _evaluate import format_eval_metrics_lines

    return format_eval_metrics_lines(res)


def write_metrics_json(run_dir: Path, res: dict[str, Any], checkpoint: Path) -> None:
    metrics_out = {
        "accuracy": res["accuracy"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "best_model": res["best_model"],
        "models": res["models"],
        "checkpoint_path": str(checkpoint.resolve()),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2) + "\n",
        encoding="utf-8",
    )


EPOCH_EVAL_CSV_FIELDNAMES = [
    "finetune_epoch",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "n_test",
    "n_test_maschine_initialized",
    "n_test_oov",
    "checkpoint_path",
]


def _epoch_eval_csv_row(epoch: int, ckpt_path: Path, res: dict[str, Any]) -> dict[str, Any]:
    maschine_init = res.get("n_test_maschine_initialized")
    return {
        "finetune_epoch": epoch,
        "accuracy": res["accuracy"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "n_test": res["n_test"],
        "n_test_maschine_initialized": "" if maschine_init is None else maschine_init,
        "n_test_oov": res["oov_test"],
        "checkpoint_path": str(ckpt_path.resolve()),
        "best_model": res["best_model"],
        "models": res["models"],
    }


def write_epoch_eval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = EPOCH_EVAL_CSV_FIELDNAMES
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _expected_finetune_epochs(tm: str, flat: dict[str, Any]) -> int:
    fv = flat["finetune_word2vec"]
    if tm == "no_pretrain":
        return int(fv["epochs"])
    return int(fv["finetune_epochs"])


def run_per_epoch_eval(
    *,
    test_txt: Path,
    ckpt_dir: Path,
    run_dir: Path,
    flat: dict[str, Any],
    tm: str,
    run_log: Path | None = None,
    dry_run: bool = False,
    all_classifiers: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate each finetune checkpoint (epoch 0 .. N); write per-epoch eval logs."""
    rows: list[dict[str, Any]] = []
    checkpoints = discover_finetune_epoch_checkpoints(ckpt_dir)
    if dry_run:
        n_epochs = _expected_finetune_epochs(tm, flat)
        checkpoints = [
            (epoch, ckpt_dir / f"finetune_epoch_{epoch:02d}.pt")
            for epoch in range(n_epochs + 1)
        ]

    if not checkpoints:
        print(f"Warning: no finetune epoch checkpoints under {ckpt_dir}", flush=True)
        return rows

    last_res: dict[str, Any] | None = None
    for epoch, ckpt_path in checkpoints:
        eval_log = run_dir / "eval" / f"epoch_{epoch}" / "eval_metrics.txt"
        if dry_run:
            eval_argv = [
                *uv_run_python(SCRIPTS_DIR / "_evaluate.py"),
                str(test_txt),
                "-c",
                str(ckpt_path),
            ]
            if all_classifiers:
                eval_argv.append("--all-classifiers")
            run_cmd(
                eval_argv,
                cwd=repo_root(),
                dry_run=True,
                log_path=eval_log,
            )
            rows.append(
                {
                    "finetune_epoch": epoch,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "n_test": 0,
                    "n_test_maschine_initialized": "",
                    "n_test_oov": 0,
                    "checkpoint_path": str(ckpt_path),
                }
            )
            continue

        sys.path.insert(0, str(SCRIPTS_DIR))
        from _evaluate import run_evaluation, write_eval_coverage_json  # noqa: PLC0415

        print(f"== Per-epoch eval epoch {epoch}: {ckpt_path}", flush=True)
        res = run_evaluation(
            test_txt,
            ckpt_path,
            progress=False,
            verbose=False,
            run_dir=run_dir,
            all_classifiers=all_classifiers,
        )
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        eval_log.write_text("\n".join(_format_eval_metrics_lines(res)) + "\n", encoding="utf-8")
        write_eval_coverage_json(eval_log.parent / "eval_coverage.json", res)
        rows.append(_epoch_eval_csv_row(epoch, ckpt_path, res))
        last_res = res
        if run_log is not None:
            with run_log.open("a", encoding="utf-8") as lf:
                lf.write(
                    f"\n{'=' * 72}\nper-epoch eval epoch {epoch}\n{'=' * 72}\n"
                )
                lf.write(eval_log.read_text(encoding="utf-8"))

    if rows and not dry_run:
        last = rows[-1]
        write_metrics_json(
            run_dir,
            {
                "accuracy": last["accuracy"],
                "precision": last["precision"],
                "recall": last["recall"],
                "f1": last["f1"],
                "best_model": last["best_model"],
                "models": last["models"],
            },
            Path(last["checkpoint_path"]),
        )
        if last_res is not None:
            eval_log = run_dir / "eval_metrics.txt"
            eval_log.write_text(
                "\n".join(_format_eval_metrics_lines(last_res)) + "\n",
                encoding="utf-8",
            )
            write_eval_coverage_json(run_dir / "eval_coverage.json", last_res)
            print(f"Wrote {run_dir / 'eval_coverage.json'}", flush=True)
    return rows


def run_eval_inprocess(
    test_txt: Path,
    checkpoint: Path,
    run_dir: Path,
    eval_log: Path,
    *,
    run_log: Path | None = None,
    all_classifiers: bool = False,
) -> None:
    """Evaluate checkpoint; write eval_metrics.txt, metrics.json, and eval_coverage.json."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from _evaluate import run_evaluation, write_eval_coverage_json  # noqa: PLC0415

    res = run_evaluation(
        test_txt,
        checkpoint,
        progress=False,
        verbose=False,
        run_dir=run_dir,
        all_classifiers=all_classifiers,
    )
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    eval_log.write_text("\n".join(_format_eval_metrics_lines(res)) + "\n", encoding="utf-8")
    write_metrics_json(run_dir, res, checkpoint)
    write_eval_coverage_json(run_dir / "eval_coverage.json", res)
    if run_log is not None:
        with run_log.open("a", encoding="utf-8") as lf:
            lf.write(f"\n{'=' * 72}\nevaluate_embeddings (in-process)\n{'=' * 72}\n")
            lf.write(eval_log.read_text(encoding="utf-8"))




def merge_epoch_eval_sweep_csv(
    runs: list[tuple[str, str, Path]],
    out_path: Path,
) -> None:
    """Merge per-run epoch_eval_accuracy.csv into a sweep-level CSV for plotting."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tc", "training_mode", "finetune_epoch", "accuracy"]
    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for tc, training_mode, run_dir in runs:
            csv_path = run_dir / "epoch_eval_accuracy.csv"
            if not csv_path.is_file():
                continue
            with csv_path.open(newline="", encoding="utf-8") as in_f:
                for row in csv.DictReader(in_f):
                    writer.writerow(
                        {
                            "tc": tc,
                            "training_mode": training_mode,
                            "finetune_epoch": row["finetune_epoch"],
                            "accuracy": row["accuracy"],
                        }
                    )


def first_flat_config(cfg: dict[str, Any], tm: str) -> dict[str, Any]:
    """Return the single hyperparameter tuple from a scalar YAML config."""
    runs = list(iter_runs(cfg, tm))
    if len(runs) != 1:
        raise SystemExit(
            f"Config for {tm!r} must define exactly one grid point (scalar YAML values); "
            f"got {len(runs)} combinations. Use scalar hyperparameters in the YAML."
        )
    return runs[0][1]


def list_synthetic_tcs(root: Path) -> list[str]:
    base = root / "v1" / "synthetic_ontology"
    if not base.is_dir():
        return []
    out: list[str] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir() or not p.name.startswith("tc"):
            continue
        if (p / "synthetic_ontology" / "graph.nt").is_file():
            out.append(p.name)
    return out


# Backward-compatible alias
_run_eval_inprocess = run_eval_inprocess
