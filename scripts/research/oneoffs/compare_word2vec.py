#!/usr/bin/env python3
"""Check whether two gensim Word2Vec / KeyedVectors checkpoints are equivalent."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec


@dataclass(frozen=True)
class EmbeddingStore:
    """Minimal token -> vector view for comparing checkpoints."""

    vectors: np.ndarray
    key_to_index: dict[str, int]

    @property
    def vector_size(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.key_to_index)

    def __getitem__(self, token: str) -> np.ndarray:
        return self.vectors[self.key_to_index[token]]


def load_embeddings(path: Path) -> EmbeddingStore:
    """Load embeddings from a .kv, .model, or PyTorch .pt checkpoint."""
    suffix = path.suffix.lower()
    if suffix == ".kv":
        wv = KeyedVectors.load(str(path), mmap="r")
        return EmbeddingStore(
            vectors=np.asarray(wv.vectors, dtype=np.float32),
            key_to_index=dict(wv.key_to_index),
        )

    if suffix == ".model":
        from _word2vec import Word2VecWithStepLoss

        # Checkpoints from train_word2vec.py pickle the class as __main__.Word2VecWithStepLoss.
        sys.modules["__main__"].Word2VecWithStepLoss = Word2VecWithStepLoss  # type: ignore[attr-defined]
        wv = Word2Vec.load(str(path)).wv
        return EmbeddingStore(
            vectors=np.asarray(wv.vectors, dtype=np.float32),
            key_to_index=dict(wv.key_to_index),
        )

    if suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "embeddings" not in ckpt or "word2idx" not in ckpt:
            raise ValueError(
                f"{path}: checkpoint must contain 'embeddings' and 'word2idx'."
            )
        emb_t = ckpt["embeddings"]
        if emb_t.dim() != 2:
            raise ValueError(f"{path}: 'embeddings' must be 2D [vocab, dim].")
        return EmbeddingStore(
            vectors=emb_t.detach().cpu().numpy().astype(np.float32, copy=False),
            key_to_index=ckpt["word2idx"],
        )

    raise ValueError(
        f"Unsupported checkpoint type {path.suffix!r}; expected .kv, .model, or .pt"
    )


def compare_vocabularies(
    wv_a: EmbeddingStore,
    wv_b: EmbeddingStore,
) -> tuple[bool, set[str], set[str]]:
    """Return (vocab_equal, tokens_only_in_a, tokens_only_in_b)."""
    vocab_a = set(wv_a.key_to_index)
    vocab_b = set(wv_b.key_to_index)
    only_a = vocab_a - vocab_b
    only_b = vocab_b - vocab_a
    return vocab_a == vocab_b, only_a, only_b


def compare_embeddings(
    wv_a: EmbeddingStore,
    wv_b: EmbeddingStore,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, list[str], float]:
    """Compare shared-token embeddings with np.allclose.

    Returns (all_close, mismatched_tokens, max_abs_diff).
    """
    shared = sorted(set(wv_a.key_to_index) & set(wv_b.key_to_index))
    if not shared:
        return True, [], 0.0

    if wv_a.vector_size != wv_b.vector_size:
        return False, shared, float("inf")

    mismatched: list[str] = []
    max_abs_diff = 0.0
    for token in shared:
        vec_a = np.asarray(wv_a[token], dtype=np.float64)
        vec_b = np.asarray(wv_b[token], dtype=np.float64)
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(vec_a - vec_b))))
        if not np.allclose(vec_a, vec_b, rtol=rtol, atol=atol):
            mismatched.append(token)

    return not mismatched, mismatched, max_abs_diff


def compare_word2vec_models(
    path_a: Path,
    path_b: Path,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Load two checkpoints and report whether they are equivalent."""
    wv_a = load_embeddings(path_a)
    wv_b = load_embeddings(path_b)

    vocab_equal, only_a, only_b = compare_vocabularies(wv_a, wv_b)
    emb_equal, mismatched, max_abs_diff = compare_embeddings(
        wv_a, wv_b, rtol=rtol, atol=atol
    )

    print(f"A: {path_a}  (vocab={len(wv_a)}, dim={wv_a.vector_size})")
    print(f"B: {path_b}  (vocab={len(wv_b)}, dim={wv_b.vector_size})")
    print()

    print(f"Vocabularies equal: {vocab_equal}")
    if not vocab_equal:
        print(f"  only in A: {len(only_a)}")
        for token in sorted(only_a)[:10]:
            print(f"    {token}")
        if len(only_a) > 10:
            print(f"    ... and {len(only_a) - 10} more")

        print(f"  only in B: {len(only_b)}")
        for token in sorted(only_b)[:10]:
            print(f"    {token}")
        if len(only_b) > 10:
            print(f"    ... and {len(only_b) - 10} more")

    print()
    if wv_a.vector_size != wv_b.vector_size:
        print(
            f"Embedding dimensions differ: A={wv_a.vector_size}, B={wv_b.vector_size}"
        )
        emb_equal = False
    else:
        shared_count = len(set(wv_a.key_to_index) & set(wv_b.key_to_index))
        print(
            f"Embeddings allclose on shared tokens ({shared_count}): {emb_equal} "
            f"(rtol={rtol}, atol={atol})"
        )
        print(f"  max |A - B| over shared tokens: {max_abs_diff:.6g}")
        if not emb_equal:
            print(f"  mismatched tokens: {len(mismatched)}")
            for token in mismatched[:10]:
                print(f"    {token}")
            if len(mismatched) > 10:
                print(f"    ... and {len(mismatched) - 10} more")

    same = vocab_equal and emb_equal
    print()
    print("Models are the same." if same else "Models differ.")
    return same


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two embedding checkpoints (.kv, .model, or .pt) by vocabulary "
            "and embedding vectors."
        )
    )
    parser.add_argument("model_a", type=Path, help="First checkpoint path")
    parser.add_argument("model_b", type=Path, help="Second checkpoint path")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for np.allclose (default: 1e-5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for np.allclose (default: 1e-8)",
    )
    args = parser.parse_args()

    scripts_dir = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    for path in (args.model_a, args.model_b):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")

    same = compare_word2vec_models(
        args.model_a,
        args.model_b,
        rtol=args.rtol,
        atol=args.atol,
    )
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
