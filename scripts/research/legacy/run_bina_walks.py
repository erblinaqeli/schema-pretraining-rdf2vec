#!/usr/bin/env python3
"""
Train on imported instance walks from all_walks/ (.txt.gz shards).

Materializes gzipped walk shards into plain text, then runs the synthetic
training pipeline (p1, p2, vanilla) with outputs under output_bina/.

Examples:
  uv run python scripts/run_bina_walks.py
  uv run python scripts/run_bina_walks.py --tc tc04 tc12 --train-mode vanilla
  uv run python scripts/run_bina_walks.py --finetune-epochs 10 --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from shlex import join as shlex_join
from typing import Any

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

from _kg_io import RuntimeMetrics, load_runtime_summary, merge_runtime_metrics  # noqa: E402
from _pipeline import (  # noqa: E402
    SharedArtifacts,
    read_accuracy,
    run_cmd,
    run_eval_inprocess,
    run_one,
    tc_paths,
    uv_run_python,
)
from train import (  # noqa: E402
    MODE_CONFIG,
    TrainArgs,
    apply_synthetic_overrides,
    checkpoint_for_mode,
    cli_mode_to_internal,
    extract_slug_params,
    format_run_log_header,
    load_default_flat,
    run_config_slug,
    write_manifest,
)

CLI_MODES = ("p1", "p2", "vanilla")
DEFAULT_WALKS_ROOT = REPO_ROOT / "all_walks"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_bina"


def _sources_manifest(sources: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(p.resolve()), "mtime": p.stat().st_mtime}
        for p in sources
    ]


def materialize_imported_walks(
    tc: str,
    *,
    walks_root: Path,
    cache_root: Path,
    dry_run: bool = False,
) -> Path:
    """Decompress and concatenate all *.txt.gz shards for a TC into a cached plain-text file."""
    src_dir = walks_root / tc
    out_dir = cache_root / "_cache" / "walks" / "imported" / tc
    out_path = out_dir / "instance_walks.txt"
    manifest_path = out_dir / "sources.json"

    if not src_dir.is_dir():
        raise SystemExit(f"Missing walks directory: {src_dir}")

    sources = sorted(src_dir.glob("*.txt.gz"))
    if not sources:
        raise SystemExit(f"No .txt.gz walk files found under {src_dir}")

    desired = _sources_manifest(sources)
    if out_path.is_file() and manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached == desired:
                print(f"  reuse materialized walks: {out_path}")
                return out_path
        except (OSError, json.JSONDecodeError):
            pass

    if dry_run:
        print(f"  would materialize {len(sources)} shard(s) → {out_path}")
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for gz_path in sources:
            with gzip.open(gz_path, "rt", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line if line.endswith("\n") else line + "\n")

    manifest_path.write_text(
        json.dumps(desired, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  materialized {len(sources)} shard(s) → {out_path}")
    return out_path


def build_run_dir_bina(
    *,
    output_root: Path,
    cli_mode: str,
    config_slug_name: str,
    timestamp: str,
    tc: str,
) -> Path:
    """Mirror scripts/_kg_io.build_run_dir under output_bina/."""
    tm = cli_mode_to_internal(cli_mode)
    mode_leaf = None if tm == "no_pretrain" else tm
    family = "vanilla" if tm == "no_pretrain" else "protograph"

    parts: list[str] = [output_root.name, "synthetic", family, config_slug_name, tc, timestamp]
    if mode_leaf is not None:
        parts.append(mode_leaf)
    return REPO_ROOT.joinpath(*parts)


def bina_index_path(output_root: Path) -> Path:
    return output_root / "index" / "experiments.jsonl"


def append_bina_index(
    *,
    output_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    tc: str,
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

    run_id_parts = ["synthetic", tc]
    if tm not in ("no_pretrain", None):
        run_id_parts.extend([str(tm), slug, timestamp])
    else:
        run_id_parts.extend([slug, timestamp])

    entry: dict[str, Any] = {
        "run_id": "/".join(run_id_parts),
        "dataset": "synthetic",
        "tc": tc,
        "slug": slug,
        "timestamp": timestamp,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "params": manifest.get("params", {}),
        "metrics": metrics,
        "runtime_seconds": runtime,
        "artifact_refs": artifact_refs,
        "command": manifest.get("command", ""),
        "imported_walks_source": manifest.get("imported_walks_source"),
        "materialized_walks": manifest.get("materialized_walks"),
    }
    if tm not in ("no_pretrain", None):
        entry["mode"] = tm

    path = bina_index_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def list_imported_tcs(walks_root: Path) -> list[str]:
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
    available = list_imported_tcs(walks_root)
    if not available:
        raise SystemExit(
            f"No TCs with both walks under {walks_root} and graphs under v1/synthetic_ontology/"
        )
    if requested is None or requested == ["all"]:
        return available
    missing = [tc for tc in requested if tc not in available]
    if missing:
        raise SystemExit(
            f"Requested TC(s) not available (need walks + graph): {', '.join(missing)}. "
            f"Available: {', '.join(available)}"
        )
    return list(requested)


def train_bina_tc(
    *,
    tc: str,
    cli_mode: str,
    tm: str,
    flat: dict[str, Any],
    config_slug_name: str,
    timestamp: str,
    args: TrainArgs,
    walks_root: Path,
    output_root: Path,
    materialized_walks: Path,
    run_index: int,
    total_runs: int,
    training_seed: int | None = None,
) -> Path:
    run_dir = build_run_dir_bina(
        output_root=output_root,
        cli_mode=cli_mode,
        config_slug_name=config_slug_name,
        timestamp=timestamp,
        tc=tc,
    )
    paths = tc_paths(tc, REPO_ROOT)
    if not paths["graph_nt"].is_file():
        raise SystemExit(f"Missing graph for {tc}: {paths['graph_nt']}")

    manifest: dict[str, Any] = {
        "dataset": "synthetic",
        "train_mode": cli_mode,
        "training_mode": tm,
        "config": config_slug_name,
        "tc": tc,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "command": shlex_join([sys.executable, *sys.argv]),
        "params": flat,
        "imported_walks_source": str((walks_root / tc).resolve()),
        "materialized_walks": str(materialized_walks.resolve()),
    }
    write_manifest(run_dir, manifest, dry_run=args.dry_run)

    header = (
        f"Run {run_index}/{total_runs} | synthetic | {cli_mode} | "
        f"{config_slug_name} | {tc} | {run_dir}"
    )
    print(f"\n{'=' * 72}\n{header}\n{'=' * 72}")

    if not materialized_walks.is_file() and not args.dry_run:
        raise SystemExit(f"Materialized instance walks not found: {materialized_walks}")

    config_source = str(MODE_CONFIG[tm].resolve())
    extra: list[str] = [
        f"Output directory: {run_dir}",
        f"Manifest: {run_dir / 'manifest.json'}",
        f"Imported walks source: {walks_root / tc}",
        "Materialized instance walks:",
        str(materialized_walks.resolve()),
        "",
    ]
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
        precomputed_instance_walks=materialized_walks,
        log_preamble=log_preamble,
        save_finetune_epoch_checkpoints=args.save_epoch_checkpoints,
        dataset="synthetic",
        tc=tc,
    )

    if not args.save_epoch_checkpoints:
        if not args.no_eval and not args.dry_run:
            ckpt = checkpoint_for_mode(run_dir, tm)
            eval_log = run_dir / "eval_metrics.txt"
            print(f"== Evaluate {ckpt}")
            eval_runtime = RuntimeMetrics()
            with eval_runtime.stage("evaluation"):
                run_eval_inprocess(
                    paths["test_txt"], ckpt, run_dir, eval_log, run_log=run_dir / "run.log"
                )
            merge_runtime_metrics(run_dir, eval_runtime)
        elif args.dry_run and not args.no_eval:
            eval_argv = [
                *uv_run_python(SCRIPTS_DIR / "_evaluate.py"),
                str(paths["test_txt"]),
                "-c",
                str(checkpoint_for_mode(run_dir, tm)),
            ]
            run_cmd(eval_argv, cwd=REPO_ROOT, dry_run=True, log_path=run_dir / "eval_metrics.txt")

    append_bina_index(
        output_root=output_root,
        run_dir=run_dir,
        manifest=manifest,
        tc=tc,
        cli_mode=cli_mode,
        slug=config_slug_name,
        timestamp=timestamp,
        dry_run=args.dry_run,
    )
    return run_dir


def parse_args(argv: list[str] | None = None) -> tuple[TrainArgs, Path, Path]:
    ap = argparse.ArgumentParser(
        description="Train on imported all_walks/ instance walks with output under output_bina/.",
    )
    ap.add_argument(
        "--walks-root",
        type=Path,
        default=DEFAULT_WALKS_ROOT,
        help="Root directory with tcXX/*.txt.gz walk shards (default: all_walks/)",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Experiment output root (default: output_bina/)",
    )
    ap.add_argument(
        "--tc",
        nargs="*",
        default=None,
        help="Test case ids with imported walks, or 'all' (default: all available)",
    )
    ap.add_argument(
        "--train-mode",
        dest="train_modes",
        nargs="+",
        choices=CLI_MODES,
        default=list(CLI_MODES),
        help="Training modes (default: p1 p2 vanilla)",
    )
    ap.add_argument("--config-name", default=None)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--finetune-lr", "--lr", dest="finetune_lr", type=float, default=None)
    ap.add_argument("--pretrain-lr", type=float, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--negative", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=None)
    ap.add_argument("--initialization-noise", type=float, default=None)
    ap.add_argument("--anchor-regularization", type=float, default=None)
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save-epoch-checkpoints", action="store_true")

    ns = ap.parse_args(argv)
    tcs = ns.tc
    if tcs is not None and len(tcs) == 1 and tcs[0].lower() == "all":
        tcs = ["all"]

    return TrainArgs(
        datasets=["synthetic"],
        tcs=tcs,
        train_modes=list(ns.train_modes),
        config_name=ns.config_name,
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
        pretrain_walks_per_entity=None,
        pretrain_depth=None,
        finetune_walks_per_entity=None,
        finetune_depth=None,
        instance_walks=None,
        pretrain_walks_p1=None,
        pretrain_walks_p2=None,
        ontology=None,
        precomputed_instance_walks=None,
        no_eval=ns.no_eval,
        dry_run=ns.dry_run,
        save_epoch_checkpoints=ns.save_epoch_checkpoints,
        re_eval=False,
        run_dir=None,
        eval_stage="finetuned",
        prepare_dbpedia=None,
        run_5_times=False,
    ), ns.walks_root.resolve(), ns.output_root.resolve()


def main(argv: list[str] | None = None) -> None:
    args, walks_root, output_root = parse_args(argv)
    tcs = resolve_tcs(args.tcs, walks_root)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    total = len(args.train_modes) * len(tcs)
    run_idx = 0

    print(f"Walks root: {walks_root}")
    print(f"Output root: {output_root}")
    print(f"TCs: {', '.join(tcs)}")
    print(f"Modes: {', '.join(args.train_modes)}")
    print(f"Timestamp: {timestamp}")

    materialized_by_tc: dict[str, Path] = {}
    for tc in tcs:
        materialized_by_tc[tc] = materialize_imported_walks(
            tc,
            walks_root=walks_root,
            cache_root=output_root,
            dry_run=args.dry_run,
        )

    for cli_mode in args.train_modes:
        tm = cli_mode_to_internal(cli_mode)
        default_flat = load_default_flat(tm)
        flat = apply_synthetic_overrides(default_flat, tm, args)
        slug_defaults = extract_slug_params(default_flat, tm)
        slug_current = extract_slug_params(flat, tm)
        config_slug_name = run_config_slug(
            tm,
            slug_current,
            slug_defaults,
            explicit_name=args.config_name,
        )

        for tc in tcs:
            run_idx += 1
            train_bina_tc(
                tc=tc,
                cli_mode=cli_mode,
                tm=tm,
                flat=flat,
                config_slug_name=config_slug_name,
                timestamp=timestamp,
                args=args,
                walks_root=walks_root,
                output_root=output_root,
                materialized_walks=materialized_by_tc[tc],
                run_index=run_idx,
                total_runs=total,
                training_seed=args.seed,
            )

    print(f"\nAll runs finished ({total} total).")


if __name__ == "__main__":
    main()
