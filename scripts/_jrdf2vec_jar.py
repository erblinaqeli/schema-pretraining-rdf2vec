#!/usr/bin/env python3
"""Generate RDF2Vec walks via the jRDF2Vec JAR (-onlyWalks) and merge to plain text."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from _kg_io import repo_root


def resolve_jar_path() -> Path:
    root = repo_root()
    candidates = [
        root / "jrdf2vec-1.3-SNAPSHOT_seed.jar",
        root / "jars" / "jrdf2vec-1.3-SNAPSHOT_seed.jar",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "jRDF2Vec jar not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def default_jar_threads() -> int:
    return max(1, (os.cpu_count() or 1) // 2)


def ensure_jrdf2vec_python_compat() -> None:
    """Prepare python-server for jRDF2Vec (Werkzeug 3.x + venv python)."""
    import site

    root = repo_root()
    patch_src = root / "python-server" / "sitecustomize.py"
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

    cmd_file = root / "python-server" / "python_command.txt"
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        desired = f"{venv_python}\n"
        if not cmd_file.is_file() or cmd_file.read_text(encoding="utf-8") != desired:
            cmd_file.write_text(desired, encoding="utf-8")


def _jar_env() -> dict[str, str]:
    env = os.environ.copy()
    env["WERKZEUG_HEADERS_WITH_UNDERSCORES"] = "1"
    venv_bin = repo_root() / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    return env


def _run_jar(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("+ " + " ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=cwd, env=_jar_env(), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )


def generate_and_merge_walks(
    *,
    graph_nt: Path,
    output_walks: Path,
    number_of_walks: int = 200,
    depth: int = 3,
    threads: int | None = None,
    dry_run: bool = False,
) -> None:
    """
    Run jRDF2Vec with -onlyWalks, then -mergeWalks into a single plain-text walks file.

    Uses jRDF2Vec defaults: RANDOM_WALKS_DUPLICATE_FREE walk mode, no training.
    """
    jar_path = resolve_jar_path()
    thread_count = threads if threads is not None else default_jar_threads()
    root = repo_root()

    if dry_run:
        walk_dir = output_walks.parent / "_jrdf2vec_walks_tmp"
        gen_cmd = [
            "java",
            "-jar",
            str(jar_path),
            "-graph",
            str(graph_nt),
            "-onlyWalks",
            "-walkDirectory",
            str(walk_dir),
            "-numberOfWalks",
            str(number_of_walks),
            "-depth",
            str(depth),
            "-threads",
            str(thread_count),
        ]
        merge_cmd = [
            "java",
            "-jar",
            str(jar_path),
            "-mergeWalks",
            "-walkDirectory",
            str(walk_dir),
            "-o",
            str(output_walks),
        ]
        _run_jar(gen_cmd, cwd=root, dry_run=True)
        _run_jar(merge_cmd, cwd=root, dry_run=True)
        return

    if not graph_nt.is_file():
        raise FileNotFoundError(f"Graph not found for walk generation: {graph_nt}")

    ensure_jrdf2vec_python_compat()
    output_walks.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jrdf2vec_walks_") as tmp:
        walk_dir = Path(tmp)
        gen_cmd = [
            "java",
            "-jar",
            str(jar_path),
            "-graph",
            str(graph_nt),
            "-onlyWalks",
            "-walkDirectory",
            str(walk_dir),
            "-numberOfWalks",
            str(number_of_walks),
            "-depth",
            str(depth),
            "-threads",
            str(thread_count),
        ]
        _run_jar(gen_cmd, cwd=root, dry_run=False)

        merge_cmd = [
            "java",
            "-jar",
            str(jar_path),
            "-mergeWalks",
            "-walkDirectory",
            str(walk_dir),
            "-o",
            str(output_walks),
        ]
        _run_jar(merge_cmd, cwd=root, dry_run=False)

    if not output_walks.is_file() or output_walks.stat().st_size == 0:
        raise RuntimeError(f"jRDF2Vec walk merge produced no output: {output_walks}")
