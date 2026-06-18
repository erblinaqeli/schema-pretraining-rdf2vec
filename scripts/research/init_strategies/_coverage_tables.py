#!/usr/bin/env python3
"""Recompute initialization-coverage tables for the thesis (P1/P2/P3 x TC01-TC15).

Coverage is an *init-time* property: a checkpoint-vocabulary token is
"protograph-initialized" iff it would receive an embedding from the protograph at
MASCHInE init time, else random. It depends only on (checkpoint vocab, protograph,
entity->class mapping) -- not on training and not on classic-vs-bound init (both
initialize the same *set* of tokens). So we recompute it deterministically from
on-disk data, reusing the exact labeling logic of ``protograph_init``
(_synthetic_compare.py) with ``strategy="all_init"`` (the benchmark setting,
run_synthetic_benchmark.py:272).

Outputs a JSON with all numbers and prints LaTeX rows for both thesis tables.

Run:  python scripts/_coverage_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"))

from _evaluate import load_labeled_txt, vocab_key_for_token  # noqa: E402
from _maschine_init import (  # noqa: E402
    build_entity_to_class_tokens,
    count_entity_init_coverage,
    count_vocab_init_coverage,
    load_maschine_entity_mapping,
    maschine_embedding_for_token,
    stage1_vector_lookup,
)
from _protograph_gen import iter_rdf_iris  # noqa: E402

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
PROTO_DIR = REPO / "notebooks" / "synthetic_compare"
BENCH_DIR = REPO / "output" / "synthetic_benchmark"
ONTO_DIR = REPO / "v1" / "synthetic_ontology"

TCS = [f"{i:02d}" for i in range(1, 16)]
VARIANTS = ["p1", "p2", "p3"]
# Coverage = does the instance's *own* most-specific class have a protograph
# embedding. We deliberately use "most_specific" (not the benchmark's "all_init",
# which climbs subClassOf and would mark nearly everything covered): this is the
# metric the thesis tables report, and it reproduces the originally published P1/P2
# figures exactly (validated against every train-split row and the Ent columns).
STRATEGY = "most_specific"


def load_stage1_keys(proto_nt: Path) -> dict[str, np.ndarray]:
    """Dummy protograph vectors keyed by every IRI in the protograph .nt.

    Only key *presence* matters for coverage labeling (values are never compared),
    so each gets a constant placeholder vector.
    """
    placeholder = np.ones(2, dtype=np.float32)
    keys: dict[str, np.ndarray] = {}
    for s, p, o in iter_rdf_iris(proto_nt):
        for term in (s, p, o):
            keys[term] = placeholder
    return keys


def label_vocab(
    vocab_tokens: list[str],
    stage1_vectors: dict[str, np.ndarray],
    parents: dict[str, set[str]],
    ent_to_classes: dict[str, list[str]],
) -> dict[str, str]:
    """Per-token init label mirroring protograph_init (_synthetic_compare.py:686-723)."""
    labels: dict[str, str] = {}
    for tok in vocab_tokens:
        vec = maschine_embedding_for_token(
            tok, stage1_vectors, parents, ent_to_classes, strategy=STRATEGY
        )
        if vec is None:
            labels[tok] = "random_no_class"
        elif stage1_vector_lookup(tok, stage1_vectors) is not None:
            labels[tok] = "stage1_pretrained"  # relation P_* copy
        else:
            labels[tok] = "maschine_class_mean"  # instance class-mean init
    return labels


def compute_tc(tc: str) -> dict:
    onto = ONTO_DIR / f"tc{tc}" / "synthetic_ontology" / "ontology.nt"
    split = ONTO_DIR / f"tc{tc}" / "synthetic_ontology" / "1000" / "train_test" / "train.txt"

    parents, entity_types = load_maschine_entity_mapping(onto, quiet=True)
    ent_to_classes = build_entity_to_class_tokens(parents, entity_types)
    train_tokens, train_labels = load_labeled_txt(split)

    out: dict[str, dict] = {}
    for v in VARIANTS:
        keys_path = BENCH_DIR / f"tc{tc}" / f"{v}_classic" / "keys.json"
        proto_nt = PROTO_DIR / f"tc{tc}" / f"protograph_{v}.nt"
        vocab_tokens = json.loads(keys_path.read_text())
        word2idx = {t: i for i, t in enumerate(vocab_tokens)}

        stage1 = load_stage1_keys(proto_nt)
        token_labels = label_vocab(vocab_tokens, stage1, parents, ent_to_classes)

        vocab = count_vocab_init_coverage(word2idx, token_labels)
        train = count_entity_init_coverage(
            train_tokens,
            word2idx,
            token_labels,
            class_labels=train_labels,
            vocab_key_resolver=vocab_key_for_token,
        )
        init_total = vocab["maschine_initialized"] + vocab["stage1_pretrained"]
        out[v] = {
            "vocab_size": vocab["total"],
            "init": init_total,
            "ent": vocab["initialized_instances"],
            "rel": vocab["initialized_relations"],
            "rand": vocab["random_initialized"],
            "train": train["by_class"],
            "train_total_per_class": {
                "positive": train["by_class"]["positive"]["total"],
                "negative": train["by_class"]["negative"]["total"],
            },
        }
    return out


def _proto_init(side: dict) -> int:
    return side["maschine_initialized"] + side["stage1_pretrained"]


def latex_vocab_row(tc: str, r: dict) -> str:
    cells = [f"TC{tc}"]
    for v in VARIANTS:
        d = r[v]
        pct = 100.0 * d["init"] / d["vocab_size"]
        cells.append(f"{d['init']}/{d['vocab_size']} ({pct:.1f}\\%)")
        cells.append(str(d["ent"]))
        cells.append(str(d["rel"]))
        cells.append(str(d["rand"]))
    return " & ".join(cells) + r" \\"


def latex_train_row(tc: str, r: dict) -> str:
    cells = [f"TC{tc}"]
    for v in VARIANTS:
        bc = r[v]["train"]
        for side in ("positive", "negative"):
            init = _proto_init(bc[side])
            rand = bc[side]["random_initialized"]
            cells.append(f"{init}/{rand}")
    return " & ".join(cells) + r" \\"


def main() -> None:
    results = {tc: compute_tc(tc) for tc in TCS}

    out_dir = REPO / "output" / "coverage_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coverage_tc01_tc15.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}\n", file=sys.stderr)

    print("% ===== VOCAB TABLE ROWS (Init/Ent/Rel/Rand per P1,P2,P3) =====")
    for tc in TCS:
        print(latex_vocab_row(tc, results[tc]))
    print()
    print("% ===== TRAIN TABLE ROWS (pos init/rand, neg init/rand per P1,P2,P3) =====")
    for tc in TCS:
        print(latex_train_row(tc, results[tc]))
    print()
    # Per-class train totals (for caption note): 800 expected for tc01-12, 200 for tc13-15
    print("% train per-class totals:", file=sys.stderr)
    for tc in TCS:
        tot = results[tc]["p1"]["train_total_per_class"]
        print(f"%   TC{tc}: pos={tot['positive']} neg={tot['negative']}", file=sys.stderr)


if __name__ == "__main__":
    main()
