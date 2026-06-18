#!/usr/bin/env python3
"""
Run hyperparameter search from YAML for a synthetic-ontology test case (TC)
and a fixed training mode (no_pretrain | p1 | p2), both chosen on the CLI.

Default search is random sampling over YAML lists (--limit N, default 50 trials).
Use --grid-search for the full Cartesian product. Use --optuna for TPE search.

For P1/P2 pretrain tuning use conf/grid_search_p1p2.yaml (finetune block fixed to
e2e_synthetic.py defaults). Artifacts go under output/<tc>_grid_search_<mode>/<timestamp>/.

With --reuse-shared (default for p1/p2), protograph, instance walks, and pretrain walks
are built once under shared/ and reused across trials. Console shows only the tqdm
progress bar; per-trial subprocess output goes to run_XXXX/run.log and shared builds
to shared/build.log. Sweep metadata is written to sweep.log.

Example:
  uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p2
  uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p1 --limit 3 --dry-run
  uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p2 --limit 35 --halving

  uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p2 \\
    --grid-search

  uv run python scripts/search.py conf/grid_search_p1p2.yaml --tc tc12 --training-mode p2 \\
    --optuna --n-trials 30 --halving
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import itertools
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from tqdm.auto import tqdm

try:
    import yaml
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyYAML is required. Install project deps: uv sync"
    ) from e

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _pipeline import (  # noqa: E402
    SharedArtifacts,
    as_list,
    dict_product,
    iter_runs,
    load_config,
    normalize_walk_block,
    normalize_w2v_block,
    parse_accuracy_from_eval_log,
    pretrain_walk_cache_key,
    read_accuracy,
    repo_root,
    run_cmd,
    run_eval_inprocess,
    run_one,
    sample_random_run,
    suggest_flat_from_optuna,
    shlex_quote,
    tc_paths,
    train_w2v_common_args,
    uv_bin,
    uv_run_python,
    walk_cli_args,
)

_run_eval_inprocess = run_eval_inprocess


# Default trial count when --limit is omitted (random search is the default mode).
DEFAULT_RANDOM_TRIALS = 50

_sweep_log_path: Path | None = None
_sweep_log_lock = threading.Lock()


def set_sweep_log(path: Path | None) -> None:
    global _sweep_log_path
    _sweep_log_path = path


def sweep_log(msg: str, *, tqdm_also: bool = False) -> None:
    """Append sweep-level messages to sweep.log; optionally mirror via tqdm.write."""
    line = msg if msg.endswith("\n") else f"{msg}\n"
    if _sweep_log_path is not None:
        with _sweep_log_lock:
            _sweep_log_path.parent.mkdir(parents=True, exist_ok=True)
            with _sweep_log_path.open("a", encoding="utf-8") as f:
                f.write(line)
    if tqdm_also:
        tqdm.write(msg.rstrip("\n"), file=sys.stderr)




def sweep_dir_name(tc: str, training_mode: str) -> str:
    """Output folder segment: <tc>_grid_search_<training_mode> (e.g. tc12_grid_search_p2)."""
    return f"{tc}_grid_search_{training_mode}"



def _read_meminfo_kb() -> dict[str, int]:
    info: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return info
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        val_kb = parts[1].strip().split()[0]
        try:
            info[key] = int(val_kb)
        except ValueError:
            continue
    return info


def available_memory_gb() -> float:
    """Best-effort free memory in GiB (Linux /proc/meminfo, else fallback)."""
    info = _read_meminfo_kb()
    if "MemAvailable" in info:
        return info["MemAvailable"] / (1024**2)
    if "MemFree" in info:
        return info["MemFree"] / (1024**2)
    return 8.0


def total_memory_gb() -> float:
    """Best-effort installed RAM in GiB."""
    info = _read_meminfo_kb()
    if "MemTotal" in info:
        return info["MemTotal"] / (1024**2)
    return available_memory_gb()


def wait_for_memory(min_free_gb: float, timeout_sec: float) -> None:
    """Block until at least ``min_free_gb`` is available, or raise on timeout."""
    if min_free_gb <= 0:
        return
    free = available_memory_gb()
    if free >= min_free_gb:
        return
    total = total_memory_gb()
    # MemAvailable cannot exceed installed RAM; waiting is pointless when the gate
    # exceeds what the machine can ever report (common with conservative estimates).
    if min_free_gb >= total * 0.98:
        sweep_log(
            f"WARNING: memory gate {min_free_gb:.1f} GiB exceeds installed RAM "
            f"({total:.1f} GiB); estimate likely too high — starting trial anyway "
            f"({free:.1f} GiB currently available). "
            "Override with --mem-per-job-gb if trials OOM."
        )
        return
    sweep_log(
        f"Waiting for memory: need {min_free_gb:.1f} GiB free, have {free:.1f} GiB "
        f"(timeout {timeout_sec:.0f}s)"
    )
    deadline = time.monotonic() + timeout_sec
    next_log = time.monotonic() + 30.0
    while True:
        free = available_memory_gb()
        if free >= min_free_gb:
            return
        now = time.monotonic()
        if now >= deadline:
            raise SystemExit(
                f"Timed out after {timeout_sec:.0f}s waiting for {min_free_gb:.1f} GiB free "
                f"(currently {free:.1f} GiB). Lower --jobs, set --mem-per-job-gb lower if the "
                "estimate is too high, or close other memory-heavy processes."
            )
        if now >= next_log:
            sweep_log(
                f"Still waiting for memory: need {min_free_gb:.1f} GiB free, have {free:.1f} GiB"
            )
            next_log = now + 30.0
        time.sleep(5.0)


def swap_memory_gb() -> tuple[float, float]:
    """Return (swap_total_gb, swap_free_gb); (0, 0) if unknown."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 0.0, 0.0
    total_kb = free_kb = 0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("SwapTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("SwapFree:"):
            free_kb = int(line.split()[1])
    return total_kb / (1024**2), free_kb / (1024**2)


def print_memory_status(*, reserve_gb: float = 0.0) -> None:
    """Log RAM/swap snapshot at sweep start."""
    free = available_memory_gb()
    swap_total, swap_free = swap_memory_gb()
    lines = [f"Memory: {free:.1f} GiB available"]
    if swap_total > 0:
        lines.append(f"swap {swap_free:.1f}/{swap_total:.1f} GiB free")
    if reserve_gb > 0:
        lines.append(f"will reserve {reserve_gb:.1f} GiB between trials")
    sweep_log("  |  ".join(lines))


class MemoryReserve:
    """
    Hold a RAM buffer so the OS/desktop cannot consume all memory between trials.

    Released immediately before each training run, then re-acquired after.
    """

    _CHUNK_BYTES = 256 * 1024 * 1024

    def __init__(self, gb: float) -> None:
        self._target_gb = max(0.0, float(gb))
        self._chunks: list[bytearray] = []

    @property
    def active_gb(self) -> float:
        return sum(len(c) for c in self._chunks) / (1024**3)

    def acquire(self) -> None:
        if self._target_gb <= 0 or self._chunks:
            return
        need = int(self._target_gb * (1024**3))
        acquired = 0
        while need > 0:
            sz = min(self._CHUNK_BYTES, need)
            try:
                self._chunks.append(bytearray(sz))
            except MemoryError:
                if self._chunks:
                    sweep_log(
                        f"WARNING: reserved only {self.active_gb:.1f}/{self._target_gb:.1f} GiB "
                        "(MemoryError on further allocation)"
                    )
                else:
                    sweep_log(
                        f"WARNING: could not reserve {self._target_gb:.1f} GiB RAM; "
                        "continuing without reserve"
                    )
                break
            acquired += sz
            need -= sz
        if self._chunks:
            sweep_log(
                f"Reserved {self.active_gb:.1f} GiB RAM headroom "
                "(released before each training run)"
            )

    def release(self) -> None:
        if not self._chunks:
            return
        self._chunks.clear()
        gc.collect()


def is_likely_oom(exit_code: int | None) -> bool:
    """True for typical Linux OOM-killer / SIGKILL exits."""
    return exit_code in (-9, 137, 247)


def _sigterm_to_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    """SIGTERM does not raise KeyboardInterrupt by default; map it so try/finally can flush."""
    raise KeyboardInterrupt




@dataclass(frozen=True)
class SweepParallelPlan:
    """Resolved sweep concurrency: separate training processes vs in-process Gensim threads."""

    jobs: int
    workers: int  # forwarded to train_word2vec (--workers 0 = all cores in that subprocess)
    mem_per_job_gb: float
    cpus: int
    available_gb: float

    @property
    def worker_threads(self) -> int:
        """Effective Gensim threads per subprocess (0 → all *cpus* when jobs==1). Never 1."""
        if self.workers > 0:
            return self.workers
        if self.jobs <= 1:
            return self.cpus
        per_job = self.cpus // self.jobs
        return max(2, per_job) if self.cpus >= 2 else per_job

    @property
    def total_threads(self) -> int:
        return self.jobs * self.worker_threads


def _grid_max_int(block: dict[str, list[Any]], key: str, default: int) -> int:
    vals = as_list(block.get(key) or default)
    return max(int(v) for v in vals)


def _walk_file_mb_from_graph(graph_mb: float, depth: int, wpe: int, *, proto: bool) -> float:
    """Rough walk .txt size from graph.nt (calibrated on tc01 instance walks)."""
    base_mb = max(graph_mb / 4.0, 0.1) if proto else max(graph_mb, 0.1)
    return max(base_mb, base_mb * int(depth) * int(wpe) / 9.0)


def _workers_per_job(cpus: int, jobs: int) -> int:
    """Gensim threads per subprocess when splitting CPUs across concurrent jobs. Never 1."""
    per_job = cpus // jobs
    return max(2, per_job) if cpus >= 2 else per_job


def _effective_gensim_workers(workers: int, cpus: int) -> int:
    if workers > 0:
        if workers == 1:
            raise SystemExit("--workers must be >= 2 or 0 for auto (workers=1 is not allowed)")
        return workers
    return cpus if cpus >= 2 else cpus


def estimate_mem_per_job_gb(
    cfg: dict[str, Any],
    training_mode: str,
    paths: dict[str, Path],
    *,
    instance_walks_path: Path | None = None,
    pretrain_walks_path: Path | None = None,
    workers: int = 1,
    cpus: int | None = None,
    flat: dict[str, Any] | None = None,
) -> float:
    """Conservative peak-RSS estimate for one train_word2vec subprocess.

    RSS scales with concurrent training **processes** (``--jobs``).  Within a process, Gensim
    ``workers`` (threads) can multiply peak RSS during ``build_vocab``; ``workers=0`` uses all CPUs
    and is treated as especially heavy.

    Walk corpora are much larger than ``graph.nt`` (e.g. tc01: 2.4 MiB graph → ~106 MiB instance
    walks at depth 4 / wpe 100).  Pass ``flat`` for a per-trial estimate from sampled walk params.
    """
    pre = cfg.get("pretrain") or {}
    pre_w = normalize_walk_block(pre.get("walks"))
    fin = cfg.get("finetune") or {}
    fin_w = normalize_walk_block(fin.get("walks"))

    if flat:
        fw = flat.get("finetune_walks") or {}
        inst_depth = int(fw.get("depth", _grid_max_int(fin_w, "depth", 4)))
        inst_wpe = int(fw.get("walks_per_entity", _grid_max_int(fin_w, "walks_per_entity", 100)))
        if training_mode in ("p1", "p2"):
            pw = flat.get("pretrain_walks") or {}
            max_depth = int(pw.get("depth", _grid_max_int(pre_w, "depth", 4)))
            max_wpe = int(
                pw.get("walks_per_entity", _grid_max_int(pre_w, "walks_per_entity", 100))
            )
        else:
            max_depth = _grid_max_int(fin_w, "depth", 4)
            max_wpe = _grid_max_int(fin_w, "walks_per_entity", 100)
    elif training_mode in ("p1", "p2"):
        max_depth = _grid_max_int(pre_w, "depth", 4)
        max_wpe = _grid_max_int(pre_w, "walks_per_entity", 100)
        inst_depth = _grid_max_int(fin_w, "depth", 4)
        inst_wpe = _grid_max_int(fin_w, "walks_per_entity", 100)
    else:
        max_depth = _grid_max_int(fin_w, "depth", 4)
        max_wpe = _grid_max_int(fin_w, "walks_per_entity", 100)
        inst_depth = max_depth
        inst_wpe = max_wpe

    graph_mb = paths["graph_nt"].stat().st_size / (1024**2)
    inst_mb = _walk_file_mb_from_graph(graph_mb, inst_depth, inst_wpe, proto=False)
    if instance_walks_path is not None and instance_walks_path.is_file():
        inst_mb = max(inst_mb, instance_walks_path.stat().st_size / (1024**2))

    proto_mb = 0.0
    if training_mode in ("p1", "p2"):
        proto_mb = _walk_file_mb_from_graph(graph_mb, max_depth, max_wpe, proto=True)
        if pretrain_walks_path is not None and pretrain_walks_path.is_file():
            proto_mb = max(proto_mb, pretrain_walks_path.stat().st_size / (1024**2))

    cpu_n = cpus if cpus is not None else (os.cpu_count() or 1)
    gensim_workers = _effective_gensim_workers(workers, cpu_n)
    # build_vocab RSS grows with threads but sublinearly; cap for estimation.
    est_workers = min(gensim_workers, 8)
    worker_mult = 1.0 + 0.25 * max(0, est_workers - 1)

    inst_gb = (inst_mb / 32.0) * worker_mult
    proto_gb = proto_mb / 256.0
    base_gb = 1.5
    two_stage_mult = 1.4 if training_mode in ("p1", "p2") else 1.0
    estimate = (base_gb + inst_gb + proto_gb) * two_stage_mult
    return max(2.0, min(estimate, 64.0))


def trial_memory_gb(
    *,
    cfg: dict[str, Any],
    training_mode: str,
    paths: dict[str, Path],
    flat: dict[str, Any],
    workers: int,
    cpus: int,
    shared: SharedArtifacts | None,
    fallback_gb: float,
) -> float:
    """Per-run memory estimate using this trial's walk params and cached shared walks if present."""
    inst_path: Path | None = None
    pre_path: Path | None = None
    if shared is not None:
        inst_path = shared.shared_dir / "walks_instance.txt"
        if training_mode in ("p1", "p2") and "pretrain_walks" in flat:
            pw = flat["pretrain_walks"]
            pre_path = (
                shared.shared_dir / "walks_pretrain" / f"{pretrain_walk_cache_key(pw)}.txt"
            )
    try:
        return estimate_mem_per_job_gb(
            cfg,
            training_mode,
            paths,
            instance_walks_path=inst_path,
            pretrain_walks_path=pre_path,
            workers=workers,
            cpus=cpus,
            flat=flat,
        )
    except OSError:
        return fallback_gb


def plan_sweep_parallelism(
    *,
    jobs_arg: str,
    workers_arg: int,
    mem_per_job_gb: float | None,
    max_jobs: int | None,
    cfg: dict[str, Any],
    training_mode: str,
    paths: dict[str, Path],
    optuna: bool,
    instance_walks_path: Path | None = None,
) -> SweepParallelPlan:
    """Choose (--jobs, --workers) from RAM and CPU count.

    * ``--jobs`` — concurrent training **subprocesses** (each loads its own Gensim model).
    * ``--workers`` — Gensim **threads per subprocess** (share RSS; 0 = all cores in that process).

    Strategy: use ``jobs=1, workers=0`` when only one subprocess fits in RAM (fastest per trial,
    lowest memory).  When RAM allows more, raise ``jobs`` up to ``min(mem_budget, cpus)`` and set
    ``workers = max(2, cpus // jobs)`` so total threads stay near ``cpu_count`` (never 1).
    """
    cpus = os.cpu_count() or 1
    avail = available_memory_gb()
    if workers_arg < 0 or workers_arg == 0:
        planned_workers = 1 if optuna else 0
    else:
        planned_workers = workers_arg
    mem = (
        float(mem_per_job_gb)
        if mem_per_job_gb is not None
        else estimate_mem_per_job_gb(
            cfg,
            training_mode,
            paths,
            instance_walks_path=instance_walks_path,
            workers=planned_workers,
            cpus=cpus,
        )
    )
    max_jobs_mem = max(1, int(avail * 0.85 // max(mem, 0.5)))

    if optuna:
        if workers_arg == 1:
            raise SystemExit("--workers must be >= 2 or 0 for auto (workers=1 is not allowed)")
        workers = 0 if workers_arg < 0 or workers_arg == 0 else workers_arg
        plan = SweepParallelPlan(
            jobs=1, workers=workers, mem_per_job_gb=mem, cpus=cpus, available_gb=avail
        )
        sweep_log(
            f"Parallelism (optuna): 1 trial at a time, {plan.worker_threads} Gensim thread(s)/trial, "
            f"~{mem:.1f} GiB/trial est., {avail:.1f} GiB available, {cpus} CPUs"
        )
        return plan

    auto_jobs = jobs_arg.strip().lower() == "auto"
    auto_workers = workers_arg < 0

    if auto_jobs:
        best_jobs = 1
        best_workers = 0
        best_threads = cpus
        upper = min(max_jobs_mem, cpus)
        for candidate in range(1, upper + 1):
            w_c = 0 if candidate == 1 else _workers_per_job(cpus, candidate)
            threads = candidate * (cpus if w_c == 0 else w_c)
            if candidate * mem > avail * 0.85:
                continue
            if threads > cpus + max(1, cpus // 4):
                continue
            if threads > best_threads or (threads == best_threads and candidate > best_jobs):
                best_jobs = candidate
                best_workers = w_c
                best_threads = threads
        jobs = best_jobs
        if auto_workers:
            workers = best_workers
        if max_jobs is not None:
            jobs = min(jobs, max_jobs)
            if auto_workers:
                workers = 0 if jobs == 1 else _workers_per_job(cpus, jobs)
    else:
        try:
            jobs = int(jobs_arg)
        except ValueError as e:
            raise SystemExit("--jobs must be a positive integer or 'auto'") from e
        if jobs < 1:
            raise SystemExit("--jobs must be >= 1")

    if auto_workers and not auto_jobs:
        workers = 0 if jobs == 1 else _workers_per_job(cpus, jobs)
    elif not auto_workers:
        workers = workers_arg
        if workers == 1:
            raise SystemExit("--workers must be >= 2 or 0 for auto (workers=1 is not allowed)")
        if workers < 0:
            raise SystemExit("--workers must be >= 0, or -1 for auto")

    plan = SweepParallelPlan(
        jobs=jobs, workers=workers, mem_per_job_gb=mem, cpus=cpus, available_gb=avail
    )

    if jobs > max_jobs_mem:
        sweep_log(
            f"WARNING: --jobs {jobs} exceeds memory budget ({max_jobs_mem} concurrent trial(s) at "
            f"~{mem:.1f} GiB/trial, {avail:.1f} GiB available). Expect OOM or swap; "
            "use --jobs auto or raise --mem-per-job-gb if trials are smaller than estimated."
        )

    if plan.total_threads > cpus + max(1, cpus // 4):
        sweep_log(
            f"WARNING: {plan.jobs} job(s) × {plan.worker_threads} thread(s) = {plan.total_threads} "
            f"may oversubscribe {cpus} CPU(s)."
        )

    sweep_log(
        f"Parallelism: {plan.jobs} concurrent trial(s), {plan.worker_threads} Gensim thread(s)/trial "
        f"({plan.total_threads} total), ~{mem:.1f} GiB/trial est., "
        f"{avail:.1f} GiB available, {cpus} CPUs"
    )
    return plan


def warn_if_memory_tight(plan: SweepParallelPlan) -> None:
    needed = plan.jobs * plan.mem_per_job_gb
    if needed > plan.available_gb * 0.85:
        sweep_log(
            f"WARNING: {plan.jobs} trial(s) × {plan.mem_per_job_gb:.1f} GiB ≈ {needed:.1f} GiB "
            f"may exceed available memory ({plan.available_gb:.1f} GiB). "
            "Use --jobs auto, --jobs 1, or raise --mem-per-job-gb if estimates are too low."
        )








def flat_config_key(flat: dict[str, Any]) -> str:
    """Stable key for halving promotion (same pretrain+finetune params, ignoring halving metadata)."""
    payload = {k: v for k, v in flat.items() if not str(k).startswith("_halving")}
    return json.dumps(payload, sort_keys=True)


def apply_halving_screen_epochs(flat: dict[str, Any], screen_epochs: int) -> dict[str, Any]:
    """Return a copy with pretrain_epochs capped for halving phase 1."""
    out = copy.deepcopy(flat)
    tm = out.get("training_mode", "")
    if tm in ("p1", "p2"):
        p2v = out["pretrain_word2vec"]
        full = int(p2v["pretrain_epochs"])
        out["_halving_full_pretrain_epochs"] = full
        p2v["pretrain_epochs"] = min(full, screen_epochs)
        out["_halving_phase"] = "screen"
    return out


def clear_halving_metadata(flat: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with full pretrain_epochs restored for halving phase 2."""
    out = copy.deepcopy(flat)
    if "_halving_full_pretrain_epochs" in out:
        out["pretrain_word2vec"]["pretrain_epochs"] = int(out.pop("_halving_full_pretrain_epochs"))
    out.pop("_halving_phase", None)
    return out









@dataclass(frozen=True)
class GridRunOutcome:
    run_index: int
    training_mode: str
    flat: dict[str, Any]
    test_accuracy: float | None
    manifest_status: str
    sys_exit_code: int | None
    reraise: Exception | None
    halving_phase: str | None = None


@dataclass
class SweepProgress:
    """Mutable sweep state for tqdm, best_run.json, and signal-handler flush."""

    out_root: Path
    tc: str
    training_mode: str
    no_eval: bool
    dry_run: bool
    planned_runs: int
    runs_completed: int = 0
    best_acc: float | None = None
    best_run_idx: int | None = None
    best_outcome: GridRunOutcome | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _execute_grid_point(
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
    halving_phase: str | None = None,
    mem_min_free_gb: float = 0.0,
    mem_per_job_gb: float = 0.0,
    mem_wait_timeout: float = 3600.0,
    memory_reserve: MemoryReserve | None = None,
    cfg: dict[str, Any] | None = None,
    cpus: int = 1,
    save_finetune_epoch_checkpoints: bool = False,
) -> GridRunOutcome:
    """Run one grid point; mirrors main()'s per-run try/except for subprocess and other errors."""
    if not dry_run:
        if memory_reserve is not None:
            memory_reserve.release()
        trial_mem = mem_per_job_gb
        if cfg is not None:
            w = workers if workers is not None else 1
            trial_mem = trial_memory_gb(
                cfg=cfg,
                training_mode=tm,
                paths=paths,
                flat=flat,
                workers=w,
                cpus=cpus,
                shared=shared,
                fallback_gb=mem_per_job_gb,
            )
        wait_for_memory(mem_min_free_gb + trial_mem, mem_wait_timeout)
        gc.collect()
    try:
        run_one(
            run_index=run_index,
            tm=tm,
            flat=flat,
            paths=paths,
            out_dir=out_dir,
            dry_run=dry_run,
            no_eval=no_eval,
            workers=workers,
            seed=seed,
            reuse_shared=reuse_shared,
            shared=shared,
            save_finetune_epoch_checkpoints=save_finetune_epoch_checkpoints,
        )
        acc: float | None = None
        if not no_eval and not dry_run:
            acc = read_accuracy(out_dir / f"run_{run_index:04d}")
        status = "ok"
        if halving_phase:
            status = f"ok ({halving_phase})"
        return GridRunOutcome(
            run_index=run_index,
            training_mode=tm,
            flat=flat,
            test_accuracy=acc,
            manifest_status=status,
            sys_exit_code=None,
            reraise=None,
            halving_phase=halving_phase,
        )
    except subprocess.CalledProcessError as e:
        st = f"error: exit {e.returncode}"
        if is_likely_oom(e.returncode):
            st = f"error: oom (exit {e.returncode})"
        if halving_phase:
            st = f"{st} ({halving_phase})"
        return GridRunOutcome(
            run_index=run_index,
            training_mode=tm,
            flat=flat,
            test_accuracy=None,
            manifest_status=st,
            sys_exit_code=e.returncode,
            reraise=None,
            halving_phase=halving_phase,
        )
    except Exception as e:
        st = f"error: {e!s}"
        if halving_phase:
            st = f"{st} ({halving_phase})"
        return GridRunOutcome(
            run_index=run_index,
            training_mode=tm,
            flat=flat,
            test_accuracy=None,
            manifest_status=st,
            sys_exit_code=None,
            reraise=e,
            halving_phase=halving_phase,
        )
    finally:
        if not dry_run and memory_reserve is not None:
            memory_reserve.acquire()


def write_best_run_record(
    out_root: Path,
    *,
    tc: str,
    training_mode: str,
    no_eval: bool,
    dry_run: bool,
    best_outcome: GridRunOutcome | None,
    runs_completed: int,
    planned_runs: int,
    sweep_status: str,
) -> None:
    """Write best_run.json: best-so-far accuracy run and sweep progress (updated often during a sweep)."""
    if dry_run:
        return
    record: dict[str, Any] = {
        "tc": tc,
        "training_mode": training_mode,
        "sweep_output_dir": str(out_root.resolve()),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sweep_status": sweep_status,
        "runs_completed": runs_completed,
        "planned_runs": planned_runs,
    }
    if no_eval:
        record["best"] = None
        record["reason"] = "evaluation disabled (--no-eval)"
    elif best_outcome is None or best_outcome.test_accuracy is None:
        record["best"] = None
        if runs_completed == 0:
            record["reason"] = "no runs completed yet"
        else:
            record["reason"] = "no run completed evaluation with a parsed test accuracy yet"
    else:
        run_dir = out_root / f"run_{best_outcome.run_index:04d}"
        record["best"] = {
            "run_index": best_outcome.run_index,
            "test_accuracy": best_outcome.test_accuracy,
            "run_dir": str(run_dir.resolve()),
            "params": best_outcome.flat,
            "halving_phase": best_outcome.halving_phase,
        }
    (out_root / "best_run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flush_best_run(progress: SweepProgress, sweep_status: str) -> None:
    """Snapshot progress to disk (thread-safe read of progress fields)."""
    if progress.dry_run:
        return
    with progress.lock:
        bo = progress.best_outcome
        rc = progress.runs_completed
    write_best_run_record(
        progress.out_root,
        tc=progress.tc,
        training_mode=progress.training_mode,
        no_eval=progress.no_eval,
        dry_run=progress.dry_run,
        best_outcome=bo,
        runs_completed=rc,
        planned_runs=progress.planned_runs,
        sweep_status=sweep_status,
    )


def select_halving_promoted(
    outcomes: list[GridRunOutcome],
    keep_frac: float,
) -> list[dict[str, Any]]:
    """Pick top configs by screen-phase accuracy for full pretrain rerun."""
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for o in outcomes:
        if o.test_accuracy is None:
            continue
        ranked.append((o.test_accuracy, flat_config_key(o.flat), clear_halving_metadata(o.flat)))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if not ranked:
        return []
    k = max(1, math.ceil(len(ranked) * keep_frac))
    seen: set[str] = set()
    promoted: list[dict[str, Any]] = []
    for _, key, flat in ranked:
        if key in seen:
            continue
        seen.add(key)
        flat = copy.deepcopy(flat)
        flat["_halving_phase"] = "promoted"
        promoted.append(flat)
        if len(promoted) >= k:
            break
    return promoted


def build_run_schedule(
    base_runs: list[tuple[str, dict[str, Any]]],
    *,
    halving: bool,
    screen_epochs: int,
    training_mode: str,
) -> list[tuple[str, dict[str, Any], str | None]]:
    """
    Return list of (tm, flat, halving_phase_label).
    halving_phase_label: None | 'screen' | 'promoted'
    """
    if not halving:
        return [(tm, flat, None) for tm, flat in base_runs]
    if training_mode not in ("p1", "p2"):
        raise SystemExit("--halving is only supported for --training-mode p1 or p2")
    scheduled: list[tuple[str, dict[str, Any], str | None]] = []
    for tm, flat in base_runs:
        scheduled.append((tm, apply_halving_screen_epochs(flat, screen_epochs), "screen"))
    return scheduled


def expand_halving_promote_phase(
    screen_outcomes: list[GridRunOutcome],
    keep_frac: float,
) -> list[tuple[str, dict[str, Any], str | None]]:
    promoted_flats = select_halving_promoted(screen_outcomes, keep_frac)
    tm = screen_outcomes[0].training_mode if screen_outcomes else "p1"
    return [(tm, flat, "promoted") for flat in promoted_flats]


def run_optuna_sweep(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    cfg_path: Path,
    paths: dict[str, Path],
    out_root: Path,
    reuse_shared: bool,
    n_trials: int,
    memory_reserve: MemoryReserve | None = None,
) -> None:
    """Optuna TPE search with optional halving pruner and memory-aware sequential trials."""
    try:
        import optuna
        from optuna.pruners import HyperbandPruner
        from optuna.samplers import TPESampler
    except ModuleNotFoundError as e:
        raise SystemExit("Optuna is required for --optuna. Install project deps: uv sync") from e

    if args.jobs != 1:
        sweep_log("Note: --optuna runs trials sequentially; forcing --jobs 1 to avoid OOM.")

    search_label = "optuna"
    if args.halving:
        search_label = "optuna+halving"

    manifest = out_root / "manifest.csv"
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "config_copy.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        meta = {
            "tc": args.tc,
            "training_mode": args.training_mode,
            "sweep_dir": sweep_dir_name(args.tc, args.training_mode),
            "search": search_label,
            "random_seed": args.random_seed,
            "n_trials": n_trials,
            "halving": args.halving,
            "halving_screen_epochs": args.halving_screen_epochs if args.halving else None,
            "reuse_shared": reuse_shared,
            "parallel_jobs": 1,
            "train_workers": args.workers,
            "mem_per_job_gb": args.mem_per_job_gb,
            "mem_min_free_gb": args.mem_min_free_gb,
            "reserve_mem_gb": args.reserve_mem_gb,
        }
        (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "run_index",
        "training_mode",
        "params_json",
        "run_dir",
        "status",
        "search",
        "halving_phase",
        "test_accuracy",
        "optuna_trial",
    ]

    shared: SharedArtifacts | None = None
    if reuse_shared and args.training_mode in ("p1", "p2"):
        shared = SharedArtifacts(
            root=out_root,
            paths=paths,
            tm=args.training_mode,
            dry_run=args.dry_run,
        )

    common_kw = dict(
        paths=paths,
        out_dir=out_root,
        dry_run=args.dry_run,
        no_eval=args.no_eval,
        workers=args.workers,
        seed=args.seed,
        reuse_shared=reuse_shared,
        shared=shared,
        mem_min_free_gb=float(args.mem_min_free_gb),
        mem_per_job_gb=float(args.mem_per_job_gb),
        mem_wait_timeout=float(args.mem_wait_timeout),
        memory_reserve=memory_reserve,
        cfg=cfg,
        cpus=os.cpu_count() or 1,
    )

    progress = SweepProgress(
        out_root=out_root,
        tc=args.tc,
        training_mode=args.training_mode,
        no_eval=args.no_eval,
        dry_run=args.dry_run,
        planned_runs=n_trials * (2 if args.halving and args.training_mode in ("p1", "p2") else 1),
    )

    run_counter = 0
    manifest_lock = threading.Lock()

    def write_manifest_row(
        idx: int,
        tm: str,
        flat: dict[str, Any],
        status: str,
        *,
        test_accuracy: float | None = None,
        halving_phase: str | None = None,
        optuna_trial: int | None = None,
    ) -> None:
        if args.dry_run:
            return
        row = {
            "run_index": idx,
            "training_mode": tm,
            "params_json": json.dumps(flat, sort_keys=True),
            "run_dir": str(out_root / f"run_{idx:04d}"),
            "status": status,
            "search": search_label,
            "halving_phase": halving_phase or "",
            "test_accuracy": "" if test_accuracy is None else f"{test_accuracy:.6f}",
            "optuna_trial": "" if optuna_trial is None else str(optuna_trial),
        }
        write_header = not manifest.is_file()
        with manifest_lock:
            with manifest.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow(row)

    def handle_outcome(outcome: GridRunOutcome, *, optuna_trial: int | None = None) -> None:
        write_manifest_row(
            outcome.run_index,
            outcome.training_mode,
            outcome.flat,
            outcome.manifest_status,
            test_accuracy=outcome.test_accuracy,
            halving_phase=outcome.halving_phase,
            optuna_trial=optuna_trial,
        )
        with progress.lock:
            acc = outcome.test_accuracy
            if acc is not None and (progress.best_acc is None or acc > progress.best_acc):
                progress.best_acc = acc
                progress.best_run_idx = outcome.run_index
                progress.best_outcome = outcome
        progress.runs_completed += 1
        flush_best_run(progress, "running")

    def execute_with_memory_guard(
        *,
        run_index: int,
        tm: str,
        flat: dict[str, Any],
        halving_phase: str | None,
    ) -> GridRunOutcome:
        nonlocal run_counter
        max_attempts = max(1, int(args.mem_retry))
        last: GridRunOutcome | None = None
        for attempt in range(1, max_attempts + 1):
            trial_mem = trial_memory_gb(
                cfg=cfg,
                training_mode=tm,
                paths=paths,
                flat=flat,
                workers=args.workers if args.workers is not None else 1,
                cpus=os.cpu_count() or 1,
                shared=shared,
                fallback_gb=float(args.mem_per_job_gb),
            )
            wait_for_memory(args.mem_min_free_gb + trial_mem, args.mem_wait_timeout)
            gc.collect()
            outcome = _execute_grid_point(
                run_index=run_index,
                tm=tm,
                flat=flat,
                halving_phase=halving_phase,
                **common_kw,
            )
            last = outcome
            if outcome.reraise is not None:
                raise outcome.reraise
            if outcome.sys_exit_code is None or not is_likely_oom(outcome.sys_exit_code):
                return outcome
            if attempt < max_attempts:
                sweep_log(
                    f"OOM on run_{run_index:04d} (attempt {attempt}/{max_attempts}); "
                    "waiting for memory before retry…"
                )
                time.sleep(15.0)
        assert last is not None
        return last

    if args.dry_run:
        class _DryTrial:
            number = 0

            def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
                return choices[0]

        tm, flat = suggest_flat_from_optuna(cfg, _DryTrial(), args.training_mode)
        print(json.dumps(flat, indent=2, sort_keys=True))
        run_one(
            run_index=1,
            tm=tm,
            flat=flat,
            paths=paths,
            out_dir=out_root,
            dry_run=True,
            no_eval=args.no_eval,
            workers=args.workers,
            seed=args.seed,
            reuse_shared=reuse_shared,
            shared=shared,
        )
        return

    storage_path = out_root / "optuna.db"
    study = optuna.create_study(
        study_name=f"{args.tc}_{args.training_mode}",
        storage=f"sqlite:///{storage_path.resolve()}",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.random_seed),
        pruner=(
            HyperbandPruner(min_resource=1, max_resource=2, reduction_factor=2)
            if args.halving and args.training_mode in ("p1", "p2")
            else None
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal run_counter
        tm, flat = suggest_flat_from_optuna(cfg, trial, args.training_mode)

        if args.halving and args.training_mode in ("p1", "p2"):
            flat_screen = apply_halving_screen_epochs(flat, args.halving_screen_epochs)
            run_counter += 1
            screen_idx = run_counter
            screen_outcome = execute_with_memory_guard(
                run_index=screen_idx,
                tm=tm,
                flat=flat_screen,
                halving_phase="screen",
            )
            handle_outcome(screen_outcome, optuna_trial=trial.number)
            if screen_outcome.sys_exit_code is not None:
                raise optuna.TrialPruned(f"screen failed: {screen_outcome.manifest_status}")
            if screen_outcome.test_accuracy is not None:
                trial.report(screen_outcome.test_accuracy, step=1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            flat = clear_halving_metadata(flat_screen)
            halving_phase: str | None = "promoted"
        else:
            halving_phase = None

        run_counter += 1
        run_idx = run_counter
        outcome = execute_with_memory_guard(
            run_index=run_idx,
            tm=tm,
            flat=flat,
            halving_phase=halving_phase,
        )
        handle_outcome(outcome, optuna_trial=trial.number)
        if outcome.sys_exit_code is not None:
            if is_likely_oom(outcome.sys_exit_code):
                raise optuna.TrialPruned(f"oom: {outcome.manifest_status}")
            raise RuntimeError(outcome.manifest_status)
        if outcome.test_accuracy is None:
            raise optuna.TrialPruned("no accuracy")
        if args.halving and args.training_mode in ("p1", "p2"):
            trial.report(outcome.test_accuracy, step=2)
        return outcome.test_accuracy

    if not args.dry_run:
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
        flush_best_run(progress, "running")

    completed_normally = False
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        completed_normally = True
    finally:
        if not args.dry_run:
            flush_best_run(progress, "complete" if completed_normally else "interrupted")
            trials_df_path = out_root / "optuna_trials.csv"
            try:
                study.trials_dataframe().to_csv(trials_df_path, index=False)
                sweep_log(f"Wrote {trials_df_path}")
            except Exception:
                pass

    if (
        not args.no_eval
        and progress.best_outcome is not None
        and progress.best_outcome.test_accuracy is not None
    ):
        best_params = study.best_params if study.best_trial is not None else {}
        best_record = {
            "tc": args.tc,
            "training_mode": args.training_mode,
            "search": search_label,
            "optuna_best_value": study.best_value if study.best_trial is not None else None,
            "optuna_best_params": best_params,
            "best_run_index": progress.best_outcome.run_index,
            "best_test_accuracy": progress.best_outcome.test_accuracy,
            "best_params_flat": progress.best_outcome.flat,
            "study_db": str(storage_path.resolve()),
        }
        (out_root / "optuna_best.json").write_text(
            json.dumps(best_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sweep_log(
            f"Optuna best: {study.best_value:.6f}  "
            f"trial={study.best_trial.number if study.best_trial else '—'}  "
            f"run_{progress.best_outcome.run_index:04d}  →  {out_root / 'optuna_best.json'}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Hyperparameter search from YAML for a TC under v1/synthetic_ontology/<TC>/ "
            "(default: random search; use --grid-search for full Cartesian grid)."
        ),
    )
    ap.add_argument(
        "config",
        type=Path,
        help="Grid YAML (e.g. conf/grid_search_p1p2.yaml)",
    )
    ap.add_argument(
        "--tc",
        required=True,
        help="Test case id (folder v1/synthetic_ontology/<TC>/synthetic_ontology/)",
    )
    ap.add_argument(
        "--training-mode",
        required=True,
        metavar="MODE",
        choices=("no_pretrain", "p1", "p2"),
        help=(
            "Training recipe for this sweep: no_pretrain (--mode none on instance walks), "
            "p1, or p2 (protograph pretrain then finetune). Not part of the grid YAML."
        ),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for this sweep "
            "(default: output/<tc>_grid_search_<mode>/<timestamp>/)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of running",
    )
    ap.add_argument(
        "--grid-search",
        action="store_true",
        help="Full Cartesian product over YAML lists (default is random search)",
    )
    ap.add_argument(
        "--random-search",
        action="store_true",
        help="Explicit random search (default; kept for backward-compatible scripts)",
    )
    ap.add_argument(
        "--optuna",
        action="store_true",
        help=(
            "Optuna TPE search over YAML lists (--n-trials or --limit). "
            "Trials run sequentially (--jobs forced to 1). Study saved to optuna.db."
        ),
    )
    ap.add_argument(
        "--n-trials",
        type=int,
        default=None,
        metavar="N",
        help="Number of Optuna trials (alias for --limit when --optuna is set)",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        metavar="S",
        help="RNG seed for random search (default) / --optuna TPE sampler (not train_word2vec --seed)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Number of random trials (default {DEFAULT_RANDOM_TRIALS} when omitted). "
            "With --grid-search: cap to first N grid points after expansion."
        ),
    )
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluate_embeddings.py after each train",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=-1,
        help=(
            "Gensim worker threads per training subprocess (-1 = auto: all CPUs when --jobs 1, "
            "else cpus//jobs). Threads share RSS; raising this does not substitute for --jobs."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Forwarded to train_word2vec.py --seed",
    )
    ap.add_argument(
        "--jobs",
        "-j",
        default="auto",
        metavar="N",
        help=(
            "Concurrent training subprocesses (default: auto from RAM and CPU count). "
            "Each subprocess loads its own Gensim model (~mem-per-job-gb). "
            "Ignored with --optuna (always 1 trial at a time)."
        ),
    )
    ap.add_argument(
        "--mem-per-job-gb",
        type=float,
        default=None,
        metavar="G",
        help=(
            "Estimated GiB per training subprocess for --jobs auto (default: estimated from grid YAML "
            "and walk corpus size)"
        ),
    )
    ap.add_argument(
        "--mem-min-free-gb",
        type=float,
        default=4.0,
        metavar="G",
        help="Headroom kept free on top of each trial before starting it (default: 4)",
    )
    ap.add_argument(
        "--mem-wait-timeout",
        type=float,
        default=3600.0,
        metavar="SEC",
        help="Max seconds to wait for --mem-min-free-gb before aborting (default: 3600)",
    )
    ap.add_argument(
        "--mem-retry",
        type=int,
        default=2,
        metavar="N",
        help="Retry a trial up to N times after likely OOM before marking failed (default: 2)",
    )
    ap.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        metavar="N",
        help="Cap for --jobs auto (default: no cap beyond CPU count)",
    )
    ap.add_argument(
        "--reserve-mem-gb",
        type=float,
        default=4.0,
        metavar="G",
        help=(
            "Keep this much RAM allocated between trials (released before each training run; "
            "default: 4, 0=off). Only used with --jobs 1."
        ),
    )
    ap.add_argument(
        "--reuse-shared",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse protograph, instance walks, and cached pretrain walks under shared/ "
            "(default: on for p1/p2, off for no_pretrain)"
        ),
    )
    ap.add_argument(
        "--halving",
        action="store_true",
        help=(
            "Successive halving for p1/p2: screens with reduced pretrain_epochs before full runs. "
            "With --optuna, uses HyperbandPruner; otherwise promotes top --halving-keep-frac after screen batch."
        ),
    )
    ap.add_argument(
        "--halving-screen-epochs",
        type=int,
        default=2,
        metavar="E",
        help="Pretrain epochs cap during halving screen phase (default: 2)",
    )
    ap.add_argument(
        "--halving-keep-frac",
        type=float,
        default=0.25,
        metavar="F",
        help="Fraction of screen trials promoted to full pretrain (default: 0.25)",
    )
    args = ap.parse_args()

    root = repo_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root
    if out_root is None:
        out_root = root / "output" / sweep_dir_name(args.tc, args.training_mode) / stamp
    else:
        out_root = out_root if out_root.is_absolute() else root / out_root
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        set_sweep_log(out_root / "sweep.log")
        sweep_log(f"Sweep output: {out_root.resolve()}")
    else:
        set_sweep_log(None)

    if args.optuna and args.grid_search:
        raise SystemExit("Use only one of --optuna or --grid-search")
    if args.grid_search and args.random_search:
        raise SystemExit("Use only one of --grid-search or --random-search")

    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    paths = tc_paths(args.tc, root)
    if not paths["graph_nt"].is_file():
        raise SystemExit(f"Missing instance graph: {paths['graph_nt']}")
    if not paths["ontology_nt"].is_file():
        raise SystemExit(f"Missing ontology: {paths['ontology_nt']}")
    if not paths["test_txt"].is_file():
        raise SystemExit(f"Missing test split: {paths['test_txt']}")

    parallel_plan = plan_sweep_parallelism(
        jobs_arg=str(args.jobs),
        workers_arg=int(args.workers),
        mem_per_job_gb=args.mem_per_job_gb,
        max_jobs=args.max_jobs,
        cfg=cfg,
        training_mode=args.training_mode,
        paths=paths,
        optuna=bool(args.optuna),
    )
    args.jobs = parallel_plan.jobs
    args.workers = parallel_plan.workers
    args.mem_per_job_gb = parallel_plan.mem_per_job_gb
    warn_if_memory_tight(parallel_plan)

    reserve_gb = float(args.reserve_mem_gb)
    if args.jobs > 1 and reserve_gb > 0:
        sweep_log("WARNING: --reserve-mem-gb is only supported with --jobs 1; disabling reserve.")
        reserve_gb = 0.0
    memory_reserve: MemoryReserve | None = None
    if reserve_gb > 0 and not args.dry_run:
        memory_reserve = MemoryReserve(reserve_gb)
        memory_reserve.acquire()
    print_memory_status(reserve_gb=reserve_gb)

    reuse_shared = args.reuse_shared
    if reuse_shared is None:
        reuse_shared = args.training_mode in ("p1", "p2")

    if args.optuna:
        n_trials = args.n_trials if args.n_trials is not None else args.limit
        if n_trials is None or n_trials < 1:
            raise SystemExit("--optuna requires --n-trials N or --limit N (>= 1)")
        sweep_log(
            f"Optuna TPE: {n_trials} trial(s)  |  free {available_memory_gb():.1f} GiB  →  {out_root}"
        )
        run_optuna_sweep(
            args=args,
            cfg=cfg,
            cfg_path=cfg_path,
            paths=paths,
            out_root=out_root,
            reuse_shared=reuse_shared,
            n_trials=n_trials,
            memory_reserve=memory_reserve,
        )
        if memory_reserve is not None:
            memory_reserve.release()
        return

    use_grid = args.grid_search
    search_label = "grid" if use_grid else "random"
    if args.halving:
        search_label = f"{search_label}+halving"

    if use_grid:
        base_runs = list(iter_runs(cfg, args.training_mode))
        if args.limit is not None:
            base_runs = base_runs[: max(0, args.limit)]
    else:
        n_trials = args.limit if args.limit is not None else DEFAULT_RANDOM_TRIALS
        if n_trials < 1:
            raise SystemExit("--limit must be >= 1")
        rng = random.Random(args.random_seed)
        base_runs = [
            sample_random_run(cfg, rng, args.training_mode) for _ in range(n_trials)
        ]
        if args.limit is None and not args.dry_run:
            sweep_log(f"Random search: {n_trials} trial(s) (default --limit {DEFAULT_RANDOM_TRIALS})")

    if args.halving:
        screen_schedule = build_run_schedule(
            base_runs,
            halving=True,
            screen_epochs=args.halving_screen_epochs,
            training_mode=args.training_mode,
        )
    else:
        screen_schedule = [(tm, flat, None) for tm, flat in base_runs]

    manifest = out_root / "manifest.csv"
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "config_copy.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        meta = {
            "tc": args.tc,
            "training_mode": args.training_mode,
            "sweep_dir": sweep_dir_name(args.tc, args.training_mode),
            "search": search_label,
            "random_seed": args.random_seed if not use_grid else None,
            "n_random_trials": len(base_runs) if not use_grid else None,
            "n_runs_planned_screen": len(screen_schedule),
            "halving": args.halving,
            "halving_screen_epochs": args.halving_screen_epochs if args.halving else None,
            "halving_keep_frac": args.halving_keep_frac if args.halving else None,
            "reuse_shared": reuse_shared,
            "parallel_jobs": args.jobs,
            "train_workers": args.workers,
            "mem_per_job_gb": args.mem_per_job_gb,
            "parallel_plan": {
                "jobs": parallel_plan.jobs,
                "workers": parallel_plan.workers,
                "worker_threads": parallel_plan.worker_threads,
                "total_threads": parallel_plan.total_threads,
                "mem_per_job_gb": parallel_plan.mem_per_job_gb,
                "cpus": parallel_plan.cpus,
                "available_gb": parallel_plan.available_gb,
            },
        }
        (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "run_index",
        "training_mode",
        "params_json",
        "run_dir",
        "status",
        "search",
        "halving_phase",
        "test_accuracy",
    ]

    def write_manifest_row(
        idx: int,
        tm: str,
        flat: dict[str, Any],
        status: str,
        *,
        test_accuracy: float | None = None,
        halving_phase: str | None = None,
    ) -> None:
        if args.dry_run:
            return
        row = {
            "run_index": idx,
            "training_mode": tm,
            "params_json": json.dumps(flat, sort_keys=True),
            "run_dir": str(out_root / f"run_{idx:04d}"),
            "status": status,
            "search": search_label,
            "halving_phase": halving_phase or "",
            "test_accuracy": "" if test_accuracy is None else f"{test_accuracy:.6f}",
        }
        write_header = not manifest.is_file()
        with manifest.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

    shared: SharedArtifacts | None = None
    if reuse_shared and args.training_mode in ("p1", "p2"):
        shared = SharedArtifacts(
            root=out_root,
            paths=paths,
            tm=args.training_mode,
            dry_run=args.dry_run,
        )

    label = "Grid points" if use_grid else "Random trials"
    phase_note = " (+ halving screen)" if args.halving else ""
    sweep_log(f"{label}{phase_note}: {len(screen_schedule)}  →  {out_root}")

    # Planned runs includes promote phase when halving (estimated upper bound for progress bar).
    promote_estimate = (
        max(1, math.ceil(len(screen_schedule) * args.halving_keep_frac))
        if args.halving
        else 0
    )
    progress = SweepProgress(
        out_root=out_root,
        tc=args.tc,
        training_mode=args.training_mode,
        no_eval=args.no_eval,
        dry_run=args.dry_run,
        planned_runs=len(screen_schedule) + promote_estimate,
    )

    def update_postfix(pbar: tqdm, outcome: GridRunOutcome) -> None:
        acc = outcome.test_accuracy
        with progress.lock:
            if acc is not None and (progress.best_acc is None or acc > progress.best_acc):
                progress.best_acc = acc
                progress.best_run_idx = outcome.run_index
                progress.best_outcome = outcome
            ba, br = progress.best_acc, progress.best_run_idx
        postfix: dict[str, str] = {}
        if not args.no_eval:
            postfix["last"] = f"{acc:.4f}" if acc is not None else "—"
            postfix["best"] = f"{ba:.4f}" if ba is not None else "—"
            if br is not None:
                postfix["best@"] = f"run_{br:04d}"
        else:
            postfix["eval"] = "off"
        if outcome.halving_phase:
            postfix["phase"] = outcome.halving_phase
        pbar.set_postfix(postfix, refresh=True)

    if not args.dry_run:
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
        flush_best_run(progress, "running")

    def handle_outcome(pbar: tqdm, outcome: GridRunOutcome, *, manifest_lock: threading.Lock | None) -> None:
        """Write manifest row; update tqdm; persist best_run.json; re-raise or exit like the original loop."""
        def do_write() -> None:
            write_manifest_row(
                outcome.run_index,
                outcome.training_mode,
                outcome.flat,
                outcome.manifest_status,
                test_accuracy=outcome.test_accuracy,
                halving_phase=outcome.halving_phase,
            )

        if manifest_lock is not None:
            with manifest_lock:
                do_write()
        else:
            do_write()
        update_postfix(pbar, outcome)
        progress.runs_completed += 1
        flush_best_run(progress, "running")
        if outcome.reraise is not None:
            raise outcome.reraise
        if outcome.sys_exit_code is not None:
            raise SystemExit(outcome.sys_exit_code)

    common_kw = dict(
        paths=paths,
        out_dir=out_root,
        dry_run=args.dry_run,
        no_eval=args.no_eval,
        workers=args.workers,
        seed=args.seed,
        reuse_shared=reuse_shared,
        shared=shared,
        mem_min_free_gb=float(args.mem_min_free_gb),
        mem_per_job_gb=float(args.mem_per_job_gb),
        mem_wait_timeout=float(args.mem_wait_timeout),
        memory_reserve=memory_reserve,
        cfg=cfg,
        cpus=parallel_plan.cpus,
    )

    def run_schedule(
        schedule: list[tuple[str, dict[str, Any], str | None]],
        *,
        start_index: int,
        pbar: tqdm | None,
        manifest_lock: threading.Lock | None,
    ) -> tuple[list[GridRunOutcome], int]:
        outcomes: list[GridRunOutcome] = []
        next_idx = start_index
        if args.jobs == 1:
            it = enumerate(schedule, start=0)
            if pbar is None:
                for offset, (tm, flat, hphase) in it:
                    i = start_index + offset
                    outcome = _execute_grid_point(
                        run_index=i,
                        tm=tm,
                        flat=flat,
                        halving_phase=hphase,
                        **common_kw,
                    )
                    outcomes.append(outcome)
                    if pbar is not None:
                        handle_outcome(pbar, outcome, manifest_lock=manifest_lock)
                return outcomes, start_index + len(schedule)

            for offset, (tm, flat, hphase) in it:
                i = start_index + offset
                pbar.set_description_str(f"[{i}/{progress.planned_runs}] {tm}", refresh=False)
                outcome = _execute_grid_point(
                    run_index=i,
                    tm=tm,
                    flat=flat,
                    halving_phase=hphase,
                    **common_kw,
                )
                outcomes.append(outcome)
                handle_outcome(pbar, outcome, manifest_lock=manifest_lock)
                pbar.update(1)
            return outcomes, start_index + len(schedule)

        tasks = [
            (start_index + offset, tm, flat, hphase)
            for offset, (tm, flat, hphase) in enumerate(schedule)
        ]
        lock = manifest_lock or threading.Lock()
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_map = {
                executor.submit(
                    _execute_grid_point,
                    run_index=i,
                    tm=tm,
                    flat=flat,
                    halving_phase=hphase,
                    **common_kw,
                ): (i, tm)
                for i, tm, flat, hphase in tasks
            }
            for fut in as_completed(future_map):
                i, tm = future_map[fut]
                if pbar is not None:
                    pbar.set_description_str(f"[done {i}] {tm}", refresh=False)
                outcome = fut.result()
                outcomes.append(outcome)
                if pbar is not None:
                    handle_outcome(pbar, outcome, manifest_lock=lock)
                    pbar.update(1)
        return outcomes, start_index + len(schedule)

    completed_normally = False
    screen_outcomes: list[GridRunOutcome] = []
    try:
        if args.jobs == 1:
            pbar = tqdm(
                total=progress.planned_runs,
                desc="grid search",
                unit="run",
                dynamic_ncols=True,
                file=sys.stderr,
                leave=True,
            )
            screen_outcomes, next_idx = run_schedule(
                screen_schedule, start_index=1, pbar=pbar, manifest_lock=None
            )
            if args.halving:
                promote_schedule = expand_halving_promote_phase(
                    screen_outcomes, args.halving_keep_frac
                )
                if promote_schedule:
                    sweep_log(
                        f"Halving promote: {len(promote_schedule)} configs "
                        f"(top {args.halving_keep_frac:.0%} of screen phase)"
                    )
                    progress.planned_runs = next_idx - 1 + len(promote_schedule)
                    run_schedule(
                        promote_schedule,
                        start_index=next_idx,
                        pbar=pbar,
                        manifest_lock=None,
                    )
            pbar.close()
        else:
            pbar = tqdm(
                total=progress.planned_runs,
                desc="grid search",
                unit="run",
                dynamic_ncols=True,
                file=sys.stderr,
                leave=True,
            )
            manifest_lock = threading.Lock()
            screen_outcomes, next_idx = run_schedule(
                screen_schedule,
                start_index=1,
                pbar=pbar,
                manifest_lock=manifest_lock,
            )
            if args.halving:
                promote_schedule = expand_halving_promote_phase(
                    screen_outcomes, args.halving_keep_frac
                )
                if promote_schedule:
                    sweep_log(f"Halving promote: {len(promote_schedule)} configs")
                    progress.planned_runs = next_idx - 1 + len(promote_schedule)
                    run_schedule(
                        promote_schedule,
                        start_index=next_idx,
                        pbar=pbar,
                        manifest_lock=manifest_lock,
                    )
            pbar.close()
        completed_normally = True
    finally:
        if memory_reserve is not None:
            memory_reserve.release()
        if not args.dry_run:
            flush_best_run(progress, "complete" if completed_normally else "interrupted")

    if (
        not args.dry_run
        and not args.no_eval
        and progress.best_outcome is not None
        and progress.best_outcome.test_accuracy is not None
    ):
        sweep_log(
            f"Best test accuracy: {progress.best_outcome.test_accuracy:.6f}  "
            f"(run_{progress.best_outcome.run_index:04d})  →  {out_root / 'best_run.json'}"
        )


if __name__ == "__main__":
    main()
