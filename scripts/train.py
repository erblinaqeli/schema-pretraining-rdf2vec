#!/usr/bin/env python3
"""
Universal RDF2Vec training for synthetic ontology TCs and DBpedia.

Output layout:
  output/synthetic/protograph/{slug}/{tc}/{timestamp}/{mode}/  (P1/P2; slug ``default`` at defaults)
  output/synthetic/vanilla/vanilla_default/{tc}/{timestamp}/     (vanilla at defaults)
  output/synthetic/vanilla/{slug}/{tc}/{timestamp}/              (vanilla with param diffs)
  output/dbpedia/{slug}/{timestamp}/{mode}/

Modes: p1, p2, vanilla (no protograph pretrain).

Examples:
  uv run python scripts/train.py --dataset synthetic --tc all --train-mode p1 p2 vanilla
  uv run python scripts/train.py --dataset dbpedia --train-mode p1 --finetune-lr 0.0001
  uv run python scripts/train.py --dataset dbpedia --train-mode p2 \\
    --instance-types dbpedia_graph/instance-types_lang=en_specific.ttl.bz2
  uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2 \\
    --pretrain-epochs 10 --finetune-epochs 5
  uv run python scripts/train.py --dataset synthetic --tc tc01 --train-mode vanilla \\
    --config-name finetune_epochs_5 --epochs 5 --dry-run
  uv run python scripts/train.py --dataset synthetic --tc tc01 --train-mode vanilla --run-5-times
  uv run python scripts/train.py --dataset synthetic --tc tc01 tc07 --train-mode p1 --run-5-times --rep-jobs 2
  uv run python scripts/train.py --dataset synthetic --tc tc01 --train-mode p2 \\
    --finetune-walks-per-entity 300 --finetune-depth 4
  uv run python scripts/train.py --dataset synthetic --tc tc01 --train-mode p2 \\
    --init-strategy all_init
  uv run python scripts/plot/experiment.py --slug init_strategy_all_init
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from shlex import join as shlex_join
from typing import Any, Iterator, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS_DIR))
from _kg_io import (  # noqa: E402
    append_index_entry,
    build_run_dir as io_build_run_dir,
    load_runtime_summary,
)
from _maschine_init import INIT_STRATEGIES  # noqa: E402
from _pipeline import (  # noqa: E402
    SharedArtifacts,
    first_flat_config,
    list_synthetic_tcs,
    load_config,
    merge_epoch_eval_sweep_csv,
    read_accuracy,
    run_cmd,
    run_eval_inprocess,
    run_one,
    tc_paths,
    uv_bin,
    uv_run_python,
)

CLI_MODES = ("p1", "p2", "vanilla")
MODE_TO_INTERNAL = {"vanilla": "no_pretrain", "p1": "p1", "p2": "p2"}
INTERNAL_TO_CLI = {v: k for k, v in MODE_TO_INTERNAL.items()}

MODE_CONFIG: dict[str, Path] = {
    "p1": REPO_ROOT / "conf" / "p1_default.yaml",
    "p2": REPO_ROOT / "conf" / "p2_default.yaml",
    "no_pretrain": REPO_ROOT / "conf" / "no_pretrain_default.yaml",
}

DBPEDIA_DEFAULTS: dict[str, Any] = {
    "pretrain_epochs": 5,
    "finetune_epochs": 5,
    "epochs": 5,
    "finetune_lr": 0.025,
    "pretrain_lr": None,
    "dim": 200,
    "window": 5,
    "architecture": "skipgram",
    "negative": 5,
    "min_count": 1,
    "sample": 0.0,
    "hs": 0,
    "workers": 0,
    "init_strategy": "most_specific",
    "init_relations": True,
    "initialization_noise": 0.0,
    "anchor_regularization": 0.0,
}

DEFAULT_INSTANCE_WALKS = REPO_ROOT / "walks" / "all_walks.txt"
DEFAULT_PRETRAIN_WALKS_P1 = REPO_ROOT / "walks" / "p1" / "walks_p1.txt"
DEFAULT_PRETRAIN_WALKS_P2 = REPO_ROOT / "walks" / "p2" / "walks_p2.txt"
DEFAULT_ONTOLOGY = REPO_ROOT / "dbpedia_graph" / "ontology.nt"
DEFAULT_DBPEDIA_GRAPH_DIR = REPO_ROOT / "dbpedia_graph"
DEFAULT_DBPEDIA_INSTANCE_TYPES_CANDIDATES = (
    DEFAULT_DBPEDIA_GRAPH_DIR / "instance-types_lang=en_specific.ttl.bz2",
    DEFAULT_DBPEDIA_GRAPH_DIR / "graph.nt",
)


@dataclass
class TrainArgs:
    datasets: list[str]
    tcs: list[str] | None
    train_modes: list[str]
    config_name: str | None
    config_path: Path | None
    timestamp: str | None
    pretrain_epochs: int | None
    finetune_epochs: int | None
    epochs: int | None
    finetune_lr: float | None
    pretrain_lr: float | None
    dim: int | None
    window: int | None
    workers: int | None
    seed: int | None
    negative: int | None
    min_count: int | None
    initialization_noise: float | None
    anchor_regularization: float | None
    init_strategy: str | None
    pretrain_walks_per_entity: int | None
    pretrain_depth: int | None
    finetune_walks_per_entity: int | None
    finetune_depth: int | None
    instance_walks: Path | None
    pretrain_walks_p1: Path | None
    pretrain_walks_p2: Path | None
    ontology: Path | None
    instance_types: Path | None
    precomputed_instance_walks: Path | None
    no_eval: bool
    dry_run: bool
    save_epoch_checkpoints: bool = False
    re_eval: bool = False
    run_dir: Path | None = None
    eval_stage: str = "finetuned"
    prepare_dbpedia: str | None = None
    run_5_times: bool = False
    rep_jobs: int = 2
    all_classifiers: bool = False
    extra_overrides: dict[str, Any] = field(default_factory=dict)


_REP_PRINT_LOCK = threading.Lock()


def _lr_tag(lr: float) -> str:
    s = f"{lr:g}"
    if s.startswith("0."):
        s = s[2:]
    return s.replace(".", "")


def _slug_value(key: str, value: Any) -> str:
    if key.endswith("_lr") or key == "lr":
        return _lr_tag(float(value))
    if isinstance(value, float):
        return f"{value:g}".replace(".", "_")
    return str(value)


def extract_slug_params(flat: dict[str, Any], tm: str) -> dict[str, Any]:
    """Training knobs used for config folder naming."""
    fv = flat["finetune_word2vec"]
    out: dict[str, Any] = {}
    if tm == "no_pretrain":
        for k in ("epochs", "lr", "dim", "window", "negative", "min_count", "sample", "hs"):
            if k in fv:
                out[k if k == "epochs" else f"finetune_{k}"] = fv[k]
    else:
        for k in ("finetune_epochs", "lr", "dim", "window", "negative", "min_count", "sample", "hs"):
            if k in fv:
                out[k if k != "lr" else "finetune_lr"] = fv[k]
        p2v = flat.get("pretrain_word2vec") or {}
        if "pretrain_epochs" in p2v:
            out["pretrain_epochs"] = p2v["pretrain_epochs"]
        if "pretrain_lr" in p2v:
            out["pretrain_lr"] = p2v["pretrain_lr"]
    for k in ("init_strategy", "init_relations", "initialization_noise", "anchor_regularization"):
        if k in fv:
            out[k] = fv[k]
    fw = flat.get("finetune_walks") or {}
    if "depth" in fw:
        out["finetune_depth"] = fw["depth"]
    if "walks_per_entity" in fw:
        out["finetune_walks_per_entity"] = fw["walks_per_entity"]
    if tm in ("p1", "p2"):
        pw = flat.get("pretrain_walks") or {}
        if "depth" in pw:
            out["pretrain_depth"] = pw["depth"]
        if "walks_per_entity" in pw:
            out["pretrain_walks_per_entity"] = pw["walks_per_entity"]
    return out


def extract_dbpedia_slug_params(params: dict[str, Any], tm: str) -> dict[str, Any]:
    keys = (
        "epochs",
        "finetune_epochs",
        "pretrain_epochs",
        "finetune_lr",
        "pretrain_lr",
        "dim",
        "window",
        "negative",
        "min_count",
        "init_strategy",
        "init_relations",
        "initialization_noise",
        "anchor_regularization",
    )
    out = {k: params[k] for k in keys if k in params and params[k] is not None}
    if tm == "no_pretrain":
        out.pop("finetune_epochs", None)
        out.pop("pretrain_epochs", None)
        out.pop("pretrain_lr", None)
        out.pop("init_strategy", None)
        out.pop("init_relations", None)
    else:
        out.pop("epochs", None)
    return out


def config_slug(
    current: dict[str, Any],
    defaults: dict[str, Any],
    *,
    explicit_name: str | None,
) -> str:
    if explicit_name:
        return explicit_name
    diffs: list[str] = []
    for key in sorted(current.keys()):
        cur = current[key]
        default = defaults.get(key)
        if cur != default:
            diffs.append(f"{key}_{_slug_value(key, cur)}")
    return "default" if not diffs else "_".join(diffs)


VANILLA_DEFAULT_SLUG = "vanilla_default"


def vanilla_slug(
    current: dict[str, Any],
    defaults: dict[str, Any],
    *,
    explicit_name: str | None,
) -> str:
    """Slug for no_pretrain runs; default-params runs use ``vanilla_default``."""
    if explicit_name:
        return explicit_name
    base = config_slug(current, defaults, explicit_name=None)
    return VANILLA_DEFAULT_SLUG if base == "default" else base


def run_config_slug(
    tm: str,
    current: dict[str, Any],
    defaults: dict[str, Any],
    *,
    explicit_name: str | None,
) -> str:
    if tm == "no_pretrain":
        return vanilla_slug(current, defaults, explicit_name=explicit_name)
    return config_slug(current, defaults, explicit_name=explicit_name)


def resolve_config_path(tm: str, config_path: Path | None) -> Path:
    cfg_path = (config_path or MODE_CONFIG[tm]).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Missing config: {cfg_path}")
    return cfg_path


def load_default_flat(tm: str, config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_config_path(tm, config_path)
    cfg = load_config(cfg_path)
    cfg_tm = cfg.get("training_mode")
    if cfg_tm is not None:
        cfg_tm = "no_pretrain" if str(cfg_tm) == "vanilla" else str(cfg_tm)
        if cfg_tm != tm:
            raise SystemExit(
                f"Config {cfg_path} declares training_mode={cfg.get('training_mode')!r}, "
                f"which does not match the requested train mode {tm!r}. "
                "Use a matching --config / --train-mode combination."
            )
    return copy.deepcopy(first_flat_config(cfg, tm))


def apply_synthetic_overrides(flat: dict[str, Any], tm: str, args: TrainArgs) -> dict[str, Any]:
    out = copy.deepcopy(flat)
    fv = out["finetune_word2vec"]
    if args.finetune_lr is not None:
        fv["lr"] = args.finetune_lr
    if args.dim is not None:
        fv["dim"] = args.dim
    if args.window is not None:
        fv["window"] = args.window
    if args.negative is not None:
        fv["negative"] = args.negative
    if args.min_count is not None:
        fv["min_count"] = args.min_count
    if args.initialization_noise is not None:
        fv["initialization_noise"] = args.initialization_noise
    if args.anchor_regularization is not None:
        fv["anchor_regularization"] = args.anchor_regularization
    if args.init_strategy is not None:
        fv["init_strategy"] = args.init_strategy

    if tm == "no_pretrain":
        if args.epochs is not None:
            fv["epochs"] = args.epochs
        if args.finetune_epochs is not None:
            fv["epochs"] = args.finetune_epochs
    else:
        if args.finetune_epochs is not None:
            fv["finetune_epochs"] = args.finetune_epochs
        p2v = out.setdefault("pretrain_word2vec", {})
        if args.pretrain_epochs is not None:
            p2v["pretrain_epochs"] = args.pretrain_epochs
        if args.pretrain_lr is not None:
            p2v["pretrain_lr"] = args.pretrain_lr

    fw = out["finetune_walks"]
    if args.finetune_depth is not None:
        fw["depth"] = args.finetune_depth
    if args.finetune_walks_per_entity is not None:
        fw["walks_per_entity"] = args.finetune_walks_per_entity
    if tm in ("p1", "p2"):
        pw = out["pretrain_walks"]
        if args.pretrain_depth is not None:
            pw["depth"] = args.pretrain_depth
        if args.pretrain_walks_per_entity is not None:
            pw["walks_per_entity"] = args.pretrain_walks_per_entity
    return out


def resolve_dbpedia_instance_types(args: TrainArgs) -> Path | None:
    """DBpedia resource rdf:type source for MASCHInE init (schema-only ontology is not enough)."""
    if args.instance_types is not None:
        return args.instance_types.resolve()
    for candidate in DEFAULT_DBPEDIA_INSTANCE_TYPES_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_dbpedia_params(args: TrainArgs, tm: str) -> dict[str, Any]:
    params = copy.deepcopy(DBPEDIA_DEFAULTS)
    if args.pretrain_epochs is not None:
        params["pretrain_epochs"] = args.pretrain_epochs
    if args.finetune_epochs is not None:
        params["finetune_epochs"] = args.finetune_epochs
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.finetune_lr is not None:
        params["finetune_lr"] = args.finetune_lr
    if args.pretrain_lr is not None:
        params["pretrain_lr"] = args.pretrain_lr
    if args.dim is not None:
        params["dim"] = args.dim
    if args.window is not None:
        params["window"] = args.window
    if args.negative is not None:
        params["negative"] = args.negative
    if args.min_count is not None:
        params["min_count"] = args.min_count
    if args.workers is not None:
        params["workers"] = args.workers
    if args.initialization_noise is not None:
        params["initialization_noise"] = args.initialization_noise
    if args.anchor_regularization is not None:
        params["anchor_regularization"] = args.anchor_regularization
    if args.init_strategy is not None:
        params["init_strategy"] = args.init_strategy
    if tm == "no_pretrain" and args.finetune_epochs is not None and args.epochs is None:
        params["epochs"] = args.finetune_epochs
    return params


def dbpedia_default_slug_params(tm: str) -> dict[str, Any]:
    return extract_dbpedia_slug_params(DBPEDIA_DEFAULTS, tm)


def cli_mode_to_internal(mode: str) -> str:
    if mode not in MODE_TO_INTERNAL:
        raise SystemExit(f"Unknown train mode: {mode!r} (expected p1, p2, vanilla)")
    return MODE_TO_INTERNAL[mode]


def build_run_dir(
    *,
    dataset: str,
    cli_mode: str,
    config_slug_name: str,
    timestamp: str,
    tc: str | None,
) -> Path:
    tm = cli_mode_to_internal(cli_mode)
    mode_leaf = None if tm == "no_pretrain" else tm
    family: str | None = None
    if dataset == "synthetic":
        family = "vanilla" if tm == "no_pretrain" else "protograph"
    return io_build_run_dir(
        dataset=dataset,
        slug=config_slug_name,
        timestamp=timestamp,
        tc=tc,
        mode=mode_leaf,
        family=family,
    )


def append_run_index(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    dataset: str,
    tc: str | None,
    cli_mode: str,
    slug: str,
    timestamp: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    tm = manifest.get("training_mode", cli_mode_to_internal(cli_mode))
    acc = read_accuracy(run_dir)
    metrics: dict[str, Any] = {}
    if acc is not None:
        metrics["accuracy"] = acc
    runtime = load_runtime_summary(run_dir)
    artifact_refs_path = run_dir / "artifact_refs.json"
    artifact_refs: dict[str, Any] = {}
    if artifact_refs_path.is_file():
        try:
            artifact_refs = json.loads(artifact_refs_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    run_id_parts = [dataset]
    if tc:
        run_id_parts.append(tc)
    if tm not in ("no_pretrain", None):
        run_id_parts.extend([str(tm), slug, timestamp])
    else:
        run_id_parts.extend([slug, timestamp])
    entry: dict[str, Any] = {
        "run_id": "/".join(run_id_parts),
        "dataset": dataset,
        "tc": tc,
        "slug": slug,
        "timestamp": timestamp,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "params": manifest.get("params", {}),
        "metrics": metrics,
        "runtime_seconds": runtime,
        "artifact_refs": artifact_refs,
        "command": manifest.get("command", ""),
    }
    if tm not in ("no_pretrain", None):
        entry["mode"] = tm
    append_index_entry(entry)


def format_run_log_header(
    *,
    header: str,
    manifest: dict[str, Any],
    config_source: str | None = None,
    extra_lines: list[str] | None = None,
) -> str:
    """Human-readable preamble for run.log (command + resolved hyperparameters)."""
    lines = [
        "=" * 72,
        header,
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "train.py command:",
        manifest["command"],
        "",
    ]
    if config_source:
        lines.extend(["Config source:", config_source, ""])
    lines.extend(
        [
            "Resolved parameters:",
            json.dumps(manifest.get("params"), indent=2, sort_keys=True),
            "",
        ]
    )
    if extra_lines:
        lines.extend(extra_lines)
        if extra_lines and extra_lines[-1] != "":
            lines.append("")
    lines.extend(["=" * 72, "", "Subcommands:", ""])
    return "\n".join(lines)


def write_manifest(run_dir: Path, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        print(f"  manifest → {run_dir / 'manifest.json'}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@contextmanager
def tee_run_log(log_path: Path, *, dry_run: bool) -> Iterator[TextIO | None]:
    if dry_run:
        yield None
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        yield log_file


def _log(log: TextIO | None, msg: str) -> None:
    print(msg, flush=True)
    if log is not None:
        log.write(msg + "\n")
        log.flush()


def _run_subprocess_logged(
    cmd: list[str],
    log: TextIO | None,
    *,
    cwd: Path,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:
    line = "+ " + shlex_join(cmd)
    _log(log, line)
    if dry_run:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(line + "\n")
        return
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log is not None:
                log.write(line)
                log.flush()
    finally:
        proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def checkpoint_for_mode(run_dir: Path, tm: str) -> Path:
    if tm == "no_pretrain":
        return run_dir / "rdf2vec_word2vec.pt"
    return run_dir / "rdf2vec_final.pt"


def train_synthetic(
    *,
    tc: str,
    cli_mode: str,
    tm: str,
    flat: dict[str, Any],
    config_slug_name: str,
    timestamp: str,
    args: TrainArgs,
    run_index: int,
    total_runs: int,
    training_seed: int | None = None,
) -> Path:
    run_dir = build_run_dir(
        dataset="synthetic",
        cli_mode=cli_mode,
        config_slug_name=config_slug_name,
        timestamp=timestamp,
        tc=tc,
    )
    paths = tc_paths(tc, REPO_ROOT)
    if not paths["graph_nt"].is_file():
        raise SystemExit(f"Missing graph for {tc}: {paths['graph_nt']}")

    manifest = {
        "dataset": "synthetic",
        "train_mode": cli_mode,
        "training_mode": tm,
        "config": config_slug_name,
        "tc": tc,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "command": shlex_join([sys.executable, *sys.argv]),
        "params": flat,
    }
    write_manifest(run_dir, manifest, dry_run=args.dry_run)

    header = (
        f"Run {run_index}/{total_runs} | synthetic | {cli_mode} | "
        f"{config_slug_name} | {tc} | {run_dir}"
    )
    print(f"\n{'=' * 72}\n{header}\n{'=' * 72}")

    precomputed = args.precomputed_instance_walks
    if precomputed is not None and not precomputed.is_file() and not args.dry_run:
        raise SystemExit(f"Precomputed instance walks not found: {precomputed}")

    config_source = str(resolve_config_path(tm, args.config_path))
    extra: list[str] = [
        f"Output directory: {run_dir}",
        f"Manifest: {run_dir / 'manifest.json'}",
    ]
    if precomputed is not None:
        extra.extend(["Precomputed instance walks:", str(precomputed.resolve()), ""])
    log_preamble = format_run_log_header(
        header=header,
        manifest=manifest,
        config_source=config_source,
        extra_lines=extra,
    )
    print(f"Command: {manifest['command']}")
    shared = SharedArtifacts(
        dataset="synthetic",
        tc=tc,
        paths=paths,
        tm=tm,
        dry_run=args.dry_run,
    )
    run_one(
        run_index=0,
        tm=tm,
        flat=flat,
        paths=paths,
        out_dir=run_dir,
        dry_run=args.dry_run,
        no_eval=True,
        workers=args.workers,
        seed=training_seed,
        reuse_shared=True,
        shared=shared,
        run_subdir=".",
        precomputed_instance_walks=precomputed,
        log_preamble=log_preamble,
        save_finetune_epoch_checkpoints=args.save_epoch_checkpoints,
        dataset="synthetic",
        tc=tc,
        all_classifiers=args.all_classifiers,
    )

    if not args.save_epoch_checkpoints:
        if not args.no_eval and not args.dry_run:
            from _kg_io import RuntimeMetrics, merge_runtime_metrics

            ckpt = checkpoint_for_mode(run_dir, tm)
            eval_log = run_dir / "eval_metrics.txt"
            print(f"== Evaluate {ckpt}")
            eval_runtime = RuntimeMetrics()
            with eval_runtime.stage("evaluation"):
                run_eval_inprocess(
                    paths["test_txt"],
                    ckpt,
                    run_dir,
                    eval_log,
                    run_log=run_dir / "run.log",
                    all_classifiers=args.all_classifiers,
                )
            merge_runtime_metrics(run_dir, eval_runtime)
        elif args.dry_run and not args.no_eval:
            eval_argv = [
                *uv_run_python(SCRIPTS_DIR / "_evaluate.py"),
                str(paths["test_txt"]),
                "-c",
                str(checkpoint_for_mode(run_dir, tm)),
            ]
            if args.all_classifiers:
                eval_argv.append("--all-classifiers")
            run_cmd(eval_argv, cwd=REPO_ROOT, dry_run=True, log_path=run_dir / "eval_metrics.txt")

    append_run_index(
        run_dir=run_dir,
        manifest=manifest,
        dataset="synthetic",
        tc=tc,
        cli_mode=cli_mode,
        slug=config_slug_name,
        timestamp=timestamp,
        dry_run=args.dry_run,
    )
    return run_dir


def train_dbpedia(
    *,
    cli_mode: str,
    tm: str,
    params: dict[str, Any],
    config_slug_name: str,
    timestamp: str,
    args: TrainArgs,
    run_index: int,
    total_runs: int,
    training_seed: int | None = None,
) -> None:
    run_dir = build_run_dir(
        dataset="dbpedia",
        cli_mode=cli_mode,
        config_slug_name=config_slug_name,
        timestamp=timestamp,
        tc=None,
    )
    instance_walks = (args.instance_walks or DEFAULT_INSTANCE_WALKS).resolve()
    pretrain_p1 = (args.pretrain_walks_p1 or DEFAULT_PRETRAIN_WALKS_P1).resolve()
    pretrain_p2 = (args.pretrain_walks_p2 or DEFAULT_PRETRAIN_WALKS_P2).resolve()
    ontology = (args.ontology or DEFAULT_ONTOLOGY).resolve()
    instance_types = resolve_dbpedia_instance_types(args)

    finetune_lr = float(params["finetune_lr"])
    pretrain_lr = params["pretrain_lr"]
    if pretrain_lr is None:
        pretrain_lr = finetune_lr
    else:
        pretrain_lr = float(pretrain_lr)

    manifest = {
        "dataset": "dbpedia",
        "train_mode": cli_mode,
        "training_mode": tm,
        "config": config_slug_name,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "command": shlex_join([sys.executable, *sys.argv]),
        "params": params,
        "instance_walks": str(instance_walks),
        "ontology": str(ontology),
        "instance_types": str(instance_types) if instance_types is not None else None,
    }
    write_manifest(run_dir, manifest, dry_run=args.dry_run)

    header = (
        f"Run {run_index}/{total_runs} | dbpedia | {cli_mode} | "
        f"{config_slug_name} | {run_dir}"
    )
    print(f"\n{'=' * 72}\n{header}\n{'=' * 72}")

    if not args.dry_run:
        for label, path in (
            ("instance walks", instance_walks),
            ("ontology", ontology),
        ):
            if not path.is_file():
                raise SystemExit(f"Missing {label}: {path}")
        if tm == "p1" and not pretrain_p1.is_file():
            raise SystemExit(f"Missing P1 pretrain walks: {pretrain_p1}")
        if tm == "p2" and not pretrain_p2.is_file():
            raise SystemExit(f"Missing P2 pretrain walks: {pretrain_p2}")

    if tm in ("p1", "p2") and instance_types is None:
        print(
            "Warning: no DBpedia instance-types file found; MASCHInE init will only see "
            "rdf:type rows in the schema ontology (typically 0 resource entities). "
            "Pass --instance-types or materialize dbpedia_graph with instance-types.",
            file=sys.stderr,
            flush=True,
        )

    loss_args = ["--loss-every-steps-pretrain", "0", "--loss-every-steps-finetune", "0"]
    w2v_common = [
        "--architecture",
        str(params["architecture"]),
        "--dim",
        str(int(params["dim"])),
        "--window",
        str(int(params["window"])),
        "--negative",
        str(int(params["negative"])),
        "--min-count",
        str(int(params["min_count"])),
        "--sample",
        str(float(params.get("sample", 0.0))),
        "--hs",
        str(int(params.get("hs", 0))),
        "--lr",
        str(finetune_lr),
    ]
    workers = int(params["workers"]) if params["workers"] is not None else 0

    if tm in ("p1", "p2"):
        pretrain_walks = pretrain_p1 if tm == "p1" else pretrain_p2
        cmd = [
            uv_bin(),
            "run",
            "python",
            str(SCRIPTS_DIR / "_word2vec.py"),
            "--mode",
            tm,
            "--pretrain-walks",
            str(pretrain_walks),
            "--instance-walks",
            str(instance_walks),
            "--ontology",
            str(ontology),
            "--out-dir",
            str(run_dir),
            *w2v_common,
            "--pretrain-lr",
            str(pretrain_lr),
            "--workers",
            str(workers),
            "--pretrain-epochs",
            str(int(params["pretrain_epochs"])),
            "--finetune-epochs",
            str(int(params["finetune_epochs"])),
            "--init-strategy",
            str(params["init_strategy"]),
            "--init-relations" if params["init_relations"] else "--no-init-relations",
            "--initialization-noise",
            str(float(params["initialization_noise"])),
            "--anchor-regularization",
            str(float(params["anchor_regularization"])),
            *loss_args,
        ]
        if instance_types is not None:
            cmd.extend(["--instance-types", str(instance_types)])
    else:
        cmd = [
            uv_bin(),
            "run",
            "python",
            str(SCRIPTS_DIR / "_word2vec.py"),
            str(instance_walks),
            *w2v_common,
            "--epochs",
            str(int(params["epochs"])),
            "--workers",
            str(workers),
            "-o",
            str(run_dir / "rdf2vec_final.pt"),
            *loss_args,
        ]
    if training_seed is not None:
        cmd.extend(["--seed", str(training_seed)])

    eval_out = run_dir / "eval"
    eval_checkpoint = run_dir / "rdf2vec_final.pt"
    do_eval = not args.no_eval

    log_header = format_run_log_header(
        header=header,
        manifest=manifest,
        config_source="scripts/train.py DBPEDIA_DEFAULTS",
        extra_lines=[
            f"Output directory: {run_dir}",
            f"Manifest: {run_dir / 'manifest.json'}",
            f"Instance walks: {instance_walks}",
            f"Ontology: {ontology}",
            f"Instance types: {instance_types or '(none)'}",
            "",
            "Training command:",
            shlex_join(cmd),
        ]
        + (
            ["", "Evaluation: in-process DLCC eval ->", str(eval_out)]
            if do_eval
            else []
        ),
    )

    from _dbpedia.eval import eval_all_splits
    from _kg_io import RuntimeMetrics, merge_runtime_metrics, write_runtime_metrics

    pipeline_runtime = RuntimeMetrics()

    with tee_run_log(run_dir / "run.log", dry_run=args.dry_run) as log:
        if log is not None:
            log.write(log_header)
            log.flush()
        elif args.dry_run:
            print(log_header)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.log").write_text(log_header, encoding="utf-8")

        if not args.dry_run:
            (run_dir / "params.json").write_text(
                json.dumps(params, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        run_log_path = run_dir / "run.log"
        _log(log, "Subcommands:")
        _run_subprocess_logged(
            cmd, log, cwd=REPO_ROOT, dry_run=args.dry_run, log_path=run_log_path
        )

        if do_eval:
            _log(log, "== Evaluate all DBpedia splits (in-process)")
            if args.dry_run:
                eval_all_splits(
                    eval_checkpoint,
                    eval_out,
                    dbpedia_root=REPO_ROOT / "v1" / "dbpedia",
                    dry_run=True,
                )
            else:
                with pipeline_runtime.stage("evaluation"):
                    eval_all_splits(
                        eval_checkpoint,
                        eval_out,
                        dbpedia_root=REPO_ROOT / "v1" / "dbpedia",
                        seed=int(training_seed) if training_seed is not None else 42,
                        log_path=eval_out / "eval.log",
                        all_classifiers=args.all_classifiers,
                    )

    if not args.dry_run and pipeline_runtime.stages:
        json_path, txt_path = merge_runtime_metrics(run_dir, pipeline_runtime)
        print(f"Wrote runtime metrics: {txt_path}", flush=True)

    append_run_index(
        run_dir=run_dir,
        manifest=manifest,
        dataset="dbpedia",
        tc=None,
        cli_mode=cli_mode,
        slug=config_slug_name,
        timestamp=timestamp,
        dry_run=args.dry_run,
    )


def resolve_tcs(args: TrainArgs, *, has_synthetic: bool) -> list[str]:
    if not has_synthetic:
        return []
    if args.tcs is None or args.tcs == ["all"]:
        tcs = list_synthetic_tcs(REPO_ROOT)
        if not tcs:
            raise SystemExit("No test cases found under v1/synthetic_ontology/")
        return tcs
    return list(args.tcs)


def count_runs(datasets: list[str], modes: list[str], tcs: list[str]) -> int:
    n = 0
    for dataset in datasets:
        if dataset == "synthetic":
            n += len(modes) * len(tcs)
        elif dataset == "dbpedia":
            n += len(modes)
    return n


def re_eval_run(args: TrainArgs) -> None:
    if args.run_dir is None:
        raise SystemExit("--re-eval requires --run-dir")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir() and not args.dry_run:
        raise SystemExit(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset = manifest.get("dataset", "synthetic")
    tm = manifest.get("training_mode", "p2")

    if dataset == "dbpedia" or "dbpedia" in str(run_dir):
        ckpt = run_dir / "rdf2vec_final.pt"
        if args.eval_stage == "protograph":
            ckpt = run_dir / "rdf2vec_pretrained.model"
        eval_out = run_dir / "eval"
        from _dbpedia.eval import eval_all_splits

        print(f"Re-eval DBpedia: {ckpt} -> {eval_out}")
        if args.dry_run:
            eval_all_splits(
                ckpt,
                eval_out,
                dbpedia_root=REPO_ROOT / "v1" / "dbpedia",
                dry_run=True,
            )
            return
        eval_all_splits(
            ckpt,
            eval_out,
            dbpedia_root=REPO_ROOT / "v1" / "dbpedia",
            seed=int(args.seed) if args.seed is not None else 42,
            log_path=eval_out / "eval.log",
            all_classifiers=args.all_classifiers,
        )
        return

    tc = manifest.get("tc")
    if tc:
        paths = tc_paths(str(tc), REPO_ROOT)
        test_txt = paths["test_txt"]
    else:
        raise SystemExit("Cannot determine test split for re-eval (missing tc in manifest)")

    ckpt = checkpoint_for_mode(run_dir, tm)
    eval_log = run_dir / "eval_metrics.txt"
    print(f"Re-eval synthetic: {ckpt}")
    if args.dry_run:
        print(f"  would evaluate {test_txt}")
        return
    run_eval_inprocess(
        test_txt,
        ckpt,
        run_dir,
        eval_log,
        run_log=run_dir / "run.log",
        all_classifiers=args.all_classifiers,
    )


def run_prepare_dbpedia(subcommand: str, argv: list[str]) -> None:
    from _dbpedia import cli as dbp_cli

    mapping = {
        "export-schema": "export-schema",
        "filter-instance": "filter-instance",
        "materialize": "materialize-dbpedia-graph-dir",
    }
    if subcommand == "all":
        dbp_cli.main(["materialize-dbpedia-graph-dir", "--dir", "dbpedia_graph"])
        return
    if subcommand not in mapping:
        raise SystemExit(
            f"Unknown --prepare-dbpedia subcommand: {subcommand!r} "
            f"(expected export-schema, filter-instance, materialize, all)"
        )
    dbp_cli.main([mapping[subcommand], *argv])


def parse_args(argv: list[str] | None = None) -> tuple[TrainArgs, list[str]]:
    ap = argparse.ArgumentParser(
        description="Universal RDF2Vec training for synthetic and DBpedia datasets.",
    )
    ap.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        default=None,
        choices=("synthetic", "dbpedia"),
        help="Dataset(s) to train on",
    )
    ap.add_argument(
        "--tc",
        nargs="*",
        default=None,
        help="Synthetic test case ids, or 'all' (default: all when synthetic is selected)",
    )
    ap.add_argument(
        "--train-mode",
        dest="train_modes",
        nargs="+",
        choices=CLI_MODES,
        default=list(CLI_MODES),
        help="Training modes (default: all)",
    )
    ap.add_argument(
        "--config-name",
        default=None,
        help="Force output config folder name (default: auto from hyperparameters)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: conf/p1_default.yaml / p2_default / no_pretrain_default)",
    )
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Vanilla / no-pretrain epochs")
    ap.add_argument("--finetune-lr", "--lr", dest="finetune_lr", type=float, default=None)
    ap.add_argument("--pretrain-lr", type=float, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Word2Vec training RNG seed (does not affect cached random walks)",
    )
    ap.add_argument("--negative", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=None)
    ap.add_argument(
        "--initialization-noise",
        type=float,
        default=None,
        help="Gaussian noise std dev for P1/P2 finetune init (overrides config)",
    )
    ap.add_argument(
        "--anchor-regularization",
        type=float,
        default=None,
        help="After each finetune epoch, pull class-init entity vectors toward MASCHInE init "
        "(overrides config; default 0 = off)",
    )
    ap.add_argument(
        "--init-strategy",
        choices=sorted(INIT_STRATEGIES),
        default=None,
        help="P1/P2 finetune instance init from protograph class vectors (overrides config). "
        "Runs are stored under slug init_strategy_<value> when not default (most_specific).",
    )
    ap.add_argument("--pretrain-walks-per-entity", type=int, default=None)
    ap.add_argument("--pretrain-depth", type=int, default=None)
    ap.add_argument("--finetune-walks-per-entity", type=int, default=None)
    ap.add_argument("--finetune-depth", type=int, default=None)
    ap.add_argument(
        "--instance-walks",
        type=Path,
        default=None,
        help="DBpedia instance walks (default: walks/all_walks.txt)",
    )
    ap.add_argument("--pretrain-walks-p1", type=Path, default=None)
    ap.add_argument("--pretrain-walks-p2", type=Path, default=None)
    ap.add_argument("--ontology", type=Path, default=None, help="DBpedia MASCHInE ontology .nt")
    ap.add_argument(
        "--instance-types",
        type=Path,
        default=None,
        help="DBpedia resource rdf:type source (.nt/.ttl/.bz2) for MASCHInE init. "
        "Default: dbpedia_graph/instance-types_lang=en_specific.ttl.bz2 or graph.nt when present.",
    )
    ap.add_argument(
        "--precomputed-instance-walks",
        type=Path,
        default=None,
        help="Synthetic: reuse instance walks instead of generating",
    )
    ap.add_argument("--timestamp", default=None, help="Shared run timestamp (default: now)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--run-5-times",
        action="store_true",
        help="Repeat the full training run 5 times with a fresh timestamp. "
        "When --seed is set, increment Word2Vec seed per repetition (base + rep index); "
        "otherwise training runs without a fixed seed. Cached walks are reused. "
        "Runs up to --rep-jobs repetitions in parallel (default 2).",
    )
    ap.add_argument(
        "--rep-jobs",
        type=int,
        default=2,
        metavar="N",
        help="Concurrent repetitions when using --run-5-times (default: 2). "
        "Ignored otherwise.",
    )
    ap.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Save finetune epoch checkpoints and eval after each epoch",
    )
    ap.add_argument(
        "--re-eval",
        action="store_true",
        help="Re-run evaluation on an existing run (--run-dir required)",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory for --re-eval",
    )
    ap.add_argument(
        "--eval-stage",
        choices=("finetuned", "protograph"),
        default="finetuned",
        help="DBpedia eval stage (default: finetuned)",
    )
    ap.add_argument(
        "--prepare-dbpedia",
        metavar="SUBCMD",
        default=None,
        help="DBpedia graph prep: export-schema, filter-instance, materialize, all (+ subcommand args)",
    )
    ap.add_argument(
        "--all-classifiers",
        action="store_true",
        help="Evaluate with NB, SVM, MLP, and LogReg and pick the best by accuracy "
        "(default: LogReg only)",
    )
    ns, remainder = ap.parse_known_args(argv)

    if ns.workers == 1:
        ap.error("--workers must be >= 2 (workers=1 is rejected by _word2vec.py; use 0 for auto)")

    prepare = ns.prepare_dbpedia

    tcs = ns.tc
    if tcs is not None and len(tcs) == 1 and tcs[0].lower() == "all":
        tcs = ["all"]

    if ns.datasets is None and not ns.re_eval and ns.prepare_dbpedia is None:
        ap.error("--dataset is required unless using --re-eval or --prepare-dbpedia")

    return TrainArgs(
        datasets=list(ns.datasets or ["synthetic"]),
        tcs=tcs,
        train_modes=list(ns.train_modes),
        config_name=ns.config_name,
        config_path=ns.config,
        timestamp=ns.timestamp,
        pretrain_epochs=ns.pretrain_epochs,
        finetune_epochs=ns.finetune_epochs,
        epochs=ns.epochs,
        finetune_lr=ns.finetune_lr,
        pretrain_lr=ns.pretrain_lr,
        dim=ns.dim,
        window=ns.window,
        workers=ns.workers,
        seed=ns.seed,
        negative=ns.negative,
        min_count=ns.min_count,
        initialization_noise=ns.initialization_noise,
        anchor_regularization=ns.anchor_regularization,
        init_strategy=ns.init_strategy,
        pretrain_walks_per_entity=ns.pretrain_walks_per_entity,
        pretrain_depth=ns.pretrain_depth,
        finetune_walks_per_entity=ns.finetune_walks_per_entity,
        finetune_depth=ns.finetune_depth,
        instance_walks=ns.instance_walks,
        pretrain_walks_p1=ns.pretrain_walks_p1,
        pretrain_walks_p2=ns.pretrain_walks_p2,
        ontology=ns.ontology,
        instance_types=ns.instance_types,
        precomputed_instance_walks=ns.precomputed_instance_walks,
        no_eval=ns.no_eval,
        dry_run=ns.dry_run,
        save_epoch_checkpoints=ns.save_epoch_checkpoints,
        re_eval=ns.re_eval,
        run_dir=ns.run_dir,
        eval_stage=ns.eval_stage,
        prepare_dbpedia=prepare,
        run_5_times=ns.run_5_times,
        rep_jobs=ns.rep_jobs,
        all_classifiers=ns.all_classifiers,
    ), remainder


def _execute_training(
    args: TrainArgs,
    *,
    timestamp: str,
    training_seed: int | None = None,
) -> None:
    has_synthetic = "synthetic" in args.datasets
    has_dbpedia = "dbpedia" in args.datasets
    tcs = resolve_tcs(args, has_synthetic=has_synthetic)

    total = count_runs(args.datasets, args.train_modes, tcs)
    run_idx = 0
    epoch_eval_runs: list[tuple[str, str, Path]] = []

    for cli_mode in args.train_modes:
        tm = cli_mode_to_internal(cli_mode)

        if has_synthetic:
            default_flat = load_default_flat(tm, args.config_path)
            flat = apply_synthetic_overrides(default_flat, tm, args)
            slug_defaults = extract_slug_params(
                load_default_flat(tm),
                tm,
            )
            slug_current = extract_slug_params(flat, tm)
            syn_config = run_config_slug(
                tm,
                slug_current,
                slug_defaults,
                explicit_name=args.config_name,
            )

            for tc in tcs:
                run_idx += 1
                run_dir = train_synthetic(
                    tc=tc,
                    cli_mode=cli_mode,
                    tm=tm,
                    flat=flat,
                    config_slug_name=syn_config,
                    timestamp=timestamp,
                    args=args,
                    run_index=run_idx,
                    total_runs=total,
                    training_seed=training_seed,
                )
                if args.save_epoch_checkpoints:
                    epoch_eval_runs.append((tc, tm, run_dir))

        if has_dbpedia:
            params = resolve_dbpedia_params(args, tm)
            slug_defaults = dbpedia_default_slug_params(tm)
            slug_current = extract_dbpedia_slug_params(params, tm)
            dbp_config = run_config_slug(
                tm,
                slug_current,
                slug_defaults,
                explicit_name=args.config_name,
            )
            run_idx += 1
            train_dbpedia(
                cli_mode=cli_mode,
                tm=tm,
                params=params,
                config_slug_name=dbp_config,
                timestamp=timestamp,
                args=args,
                run_index=run_idx,
                total_runs=total,
                training_seed=training_seed,
            )

    if args.save_epoch_checkpoints and epoch_eval_runs and has_synthetic and not args.dry_run:
        sweep_csv = (
            REPO_ROOT
            / "output"
            / "finetune_epoch_eval"
            / timestamp
            / "finetune_epoch_accuracy_all.csv"
        )
        merge_epoch_eval_sweep_csv(epoch_eval_runs, sweep_csv)
        print(f"Wrote sweep epoch accuracy CSV: {sweep_csv}", flush=True)

    print(f"\nAll runs finished ({total} total).")


def _repetition_timestamp(rep: int) -> str:
    # Always suffix with the repetition index: sequential (or dry-run) repetitions
    # can start within the same second and would otherwise share a run directory.
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{rep + 1:02d}"


def _run_repetition(
    args: TrainArgs,
    *,
    rep: int,
    repetitions: int,
    base_seed: int | None,
) -> None:
    training_seed = base_seed + rep if base_seed is not None else None
    timestamp = _repetition_timestamp(rep)
    seed_note = (
        f"training seed {training_seed}, "
        if training_seed is not None
        else "no training seed, "
    )
    with _REP_PRINT_LOCK:
        print(
            f"\n===== Repetition {rep + 1}/{repetitions} "
            f"({seed_note}timestamp {timestamp}) =====\n",
            flush=True,
        )
    _execute_training(args, timestamp=timestamp, training_seed=training_seed)


def _run_repetitions(
    args: TrainArgs, *, repetitions: int, base_seed: int | None
) -> None:
    rep_jobs = max(1, args.rep_jobs)
    parallel = rep_jobs > 1
    if parallel:
        print(
            f"Running {repetitions} repetition(s) with up to {rep_jobs} in parallel.",
            flush=True,
        )
    if rep_jobs == 1 or args.dry_run:
        for rep in range(repetitions):
            _run_repetition(
                args,
                rep=rep,
                repetitions=repetitions,
                base_seed=base_seed,
            )
        return

    with ThreadPoolExecutor(max_workers=rep_jobs) as executor:
        futures = {
            executor.submit(
                _run_repetition,
                args,
                rep=rep,
                repetitions=repetitions,
                base_seed=base_seed,
            ): rep
            for rep in range(repetitions)
        }
        for fut in as_completed(futures):
            rep = futures[fut]
            try:
                fut.result()
            except Exception:
                with _REP_PRINT_LOCK:
                    print(
                        f"\nRepetition {rep + 1}/{repetitions} failed.",
                        flush=True,
                    )
                raise


def main(argv: list[str] | None = None) -> None:
    args, prepare_remainder = parse_args(argv)

    if args.prepare_dbpedia:
        run_prepare_dbpedia(args.prepare_dbpedia, prepare_remainder)
        return

    if args.re_eval:
        re_eval_run(args)
        return

    has_dbpedia = "dbpedia" in args.datasets
    if has_dbpedia and args.tcs is not None:
        print(
            "Note: --tc is ignored for DBpedia (global training + eval on all TC splits).",
            file=sys.stderr,
        )

    if args.run_5_times:
        if args.rep_jobs < 1:
            raise SystemExit("--rep-jobs must be >= 1")
        _run_repetitions(args, repetitions=5, base_seed=args.seed)
        return

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    _execute_training(args, timestamp=timestamp, training_seed=args.seed)


if __name__ == "__main__":
    main()
