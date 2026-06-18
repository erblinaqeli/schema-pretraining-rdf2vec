#!/usr/bin/env python3
"""Seed output/_cache instance walks from pre-baked all_walks/ shards."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _kg_io import repo_root, walk_cache_dir, write_cache_meta  # noqa: E402
from _pipeline import tc_paths  # noqa: E402

DEFAULT_WALKS_ROOT = repo_root() / "all_walks"
# jar-mode keys carry no seed suffix: the JAR's walk RNG cannot be seeded.
WALK_KEY = "jrdf2vec-jar_d3_wpe200"
WALK_PARAMS = {
    "mode": "jrdf2vec-jar",
    "depth": 3,
    "walks_per_entity": 200,
}


def list_tcs(walks_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in walks_root.iterdir()
        if p.is_dir() and p.name.startswith("tc") and list(p.glob("*.txt.gz"))
    )


def materialize_shards(src_dir: Path, dest: Path) -> int:
    sources = sorted(src_dir.glob("*.txt.gz"))
    if not sources:
        raise SystemExit(f"No .txt.gz walk files found under {src_dir}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with dest.open("w", encoding="utf-8") as out_f:
        for gz_path in sources:
            with gzip.open(gz_path, "rt", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line if line.endswith("\n") else line + "\n")
                    lines += 1
    return lines


def seed_tc(tc: str, *, walks_root: Path, force: bool = False) -> str:
    cache_dir = walk_cache_dir(
        dataset="synthetic",
        tc=tc,
        role="instance",
        walk_key=WALK_KEY,
    )
    dest = cache_dir / "walks.txt"
    meta_path = cache_dir / "meta.json"

    paths = tc_paths(tc, repo_root())
    graph_nt = paths["graph_nt"]
    if not graph_nt.is_file():
        raise SystemExit(f"Missing graph for {tc}: {graph_nt}")

    if dest.is_file() and meta_path.is_file() and not force:
        return f"{tc}: reuse {dest} ({sum(1 for _ in dest.open())} lines)"

    src_dir = walks_root / tc
    lines = materialize_shards(src_dir, dest)
    write_cache_meta(
        meta_path,
        {
            "walk_key": WALK_KEY,
            "role": "instance",
            "dataset": "synthetic",
            "tc": tc,
            "params": WALK_PARAMS,
            "input_graph": str(graph_nt.resolve()),
            "input_mtime": graph_nt.stat().st_mtime,
            "source": str((src_dir / "*.txt.gz").as_posix()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return f"{tc}: wrote {dest} ({lines} lines)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy all_walks/ instance shards into output/_cache under a walk key.",
    )
    parser.add_argument(
        "--walks-root",
        type=Path,
        default=DEFAULT_WALKS_ROOT,
        help=f"Root with tcXX/*.txt.gz shards (default: {DEFAULT_WALKS_ROOT})",
    )
    parser.add_argument(
        "--tc",
        nargs="*",
        default=None,
        help="Test cases to seed (default: all available under walks-root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache entries",
    )
    args = parser.parse_args()

    tcs = args.tc if args.tc else list_tcs(args.walks_root)
    if not tcs:
        raise SystemExit(f"No tcXX walk directories found under {args.walks_root}")

    for tc in tcs:
        print(seed_tc(tc, walks_root=args.walks_root, force=args.force))


if __name__ == "__main__":
    main()
