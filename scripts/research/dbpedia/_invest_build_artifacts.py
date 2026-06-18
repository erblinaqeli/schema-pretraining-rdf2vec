#!/usr/bin/env python3
"""Build & cache schema, instance types, stage-1 codes, and the epoch-0
concept-bound init for P2 and P3, so the init/finetune experiments don't
re-stream the 4.2 GB graph.nt each time."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

from _dbpedia_investigate import build_artifacts  # noqa: E402

build_artifacts(
    ontology=ROOT / "dbpedia_graph" / "ontology.nt",
    graph=ROOT / "dbpedia_graph" / "graph.nt",
    out_dir=ROOT / "notebooks" / "dbpedia_investigate" / "artifacts",
    compare_dir=ROOT / "notebooks" / "dbpedia_compare",
    corpus_vocab_pkl=ROOT / "walks" / "all_walks_vocab.pkl",
    kinds=("p2", "p3"),
)
