#!/usr/bin/env python3
"""
2D projection of DBpedia RDF2Vec embeddings colored by top-level schema type.

This is the RDF2Vec analog of the MASCHInE paper figure (TransE/DistMult V vs P2):
instead of two embedding models we have one model (RDF2Vec) trained four ways, so
the figure is a 2x2 grid of *variants* projected separately and colored by the same
schema-style top type:

    Vanilla            P1
    P2                 P2-bound

Each entity is bucketed into one top-level domain (Person, Place, Organization,
CreativeWork, Taxon, ...) by taking its most-specific ``rdf:type`` from the DBpedia
instance-types dump and climbing ``rdfs:subClassOf`` in the ontology until a chosen
top-level class is reached. The *same* sampled entity set is plotted in every panel,
so panels are directly comparable.

Inputs
------
* Embeddings: gensim ``.kv`` (default, mmapped so only sampled rows are read),
  full ``Word2Vec`` ``.model``, or a ``.pt`` checkpoint with ``embeddings`` + ``word2idx``.
  All four default variants live in ``notebooks/dbpedia_compare/`` and share one vocab.
* Types: ``instance-types_lang=en_specific.ttl.bz2`` (one most-specific type per entity).
* Hierarchy: ``ontology.nt`` (``rdfs:subClassOf`` edges).

Example
-------
  uv run scripts/plot_dbpedia_schema_projection.py
  uv run scripts/plot_dbpedia_schema_projection.py --method tsne --per-domain 800
"""

from __future__ import annotations

import argparse
import bz2
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO / "scripts"))  # for the lazy `from _word2vec import` below
DBO = "http://dbpedia.org/ontology/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

# Top-level buckets, in priority order (first match wins for multi-typed entities).
# An entity is bucketed into the first domain (in list order) whose class (or a
# subClassOf-ancestor of the entity's most-specific type) matches. Display names
# mirror the schema.org-style legend in the MASCHInE figure.
#
# Two selectable presets:
#   * "toplevel" - broad schema.org-style top types (the paper-style figure).
#   * "fine"     - specific DBpedia classes (movies, books, albums, cities, ...).
PRESETS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "toplevel": (
        ("Person", (f"{DBO}Person",)),
        ("Organization", (f"{DBO}Organisation",)),
        ("Place", (f"{DBO}Place",)),
        ("CreativeWork", (f"{DBO}Work",)),
        ("Taxon", (f"{DBO}Species",)),
        ("MedicalEntity", (f"{DBO}Disease", f"{DBO}Drug", f"{DBO}AnatomicalStructure")),
        ("ChemicalEntity", (f"{DBO}ChemicalSubstance",)),
        ("MeanOfTransport", (f"{DBO}MeanOfTransportation",)),
        ("Event", (f"{DBO}Event",)),
    ),
    # Species excluded on purpose: dbo:Person -> dbo:Animal -> dbo:Eukaryote ->
    # dbo:Species, so a Species bucket overlaps Person. We keep Person only.
    "fine": (
        ("Person", (f"{DBO}Person",)),
        ("Movies", (f"{DBO}Film",)),
        ("Books", (f"{DBO}Book",)),
        ("Albums", (f"{DBO}Album",)),
        ("Cities", (f"{DBO}City",)),
        ("Countries", (f"{DBO}Country",)),
    ),
}

# Distinct, high-contrast palette; assigned to domains in list order.
PALETTE = (
    "#E73F74",  # magenta/pink
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#CC79A7",  # purple/pink
    "#7F3C8D",  # violet
    "#80BA5A",  # light green
    "#000000",  # black
    "#D55E00",  # vermillion
    "#11A579",  # teal
    "#F2B701",  # gold
)

# Per-domain color overrides that take precedence over the positional PALETTE
# assignment (keeps the toplevel preset untouched while tweaking specific classes).
COLOR_OVERRIDE: dict[str, str] = {
    "Countries": "#8B5A2B",  # brown
}

# Populated by configure() once the preset is chosen.
DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = PRESETS["toplevel"]
CLASS_TO_DOMAIN: dict[str, str] = {}
DOMAIN_PRIORITY: dict[str, int] = {}
DOMAIN_COLOR: dict[str, str] = {}


def configure(preset: str) -> None:
    """Select the active domain preset and rebuild the lookup tables."""
    global DOMAINS, CLASS_TO_DOMAIN, DOMAIN_PRIORITY, DOMAIN_COLOR
    DOMAINS = PRESETS[preset]
    CLASS_TO_DOMAIN = {iri: name for name, iris in DOMAINS for iri in iris}
    DOMAIN_PRIORITY = {name: i for i, (name, _) in enumerate(DOMAINS)}
    DOMAIN_COLOR = {
        name: COLOR_OVERRIDE.get(name, PALETTE[i % len(PALETTE)])
        for i, (name, _) in enumerate(DOMAINS)
    }

DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("Vanilla", "notebooks/dbpedia_compare/vanilla.kv"),
    ("P1", "notebooks/dbpedia_compare/p1_classic.kv"),
    ("P1-bound", "notebooks/dbpedia_compare/p1_bound.kv"),
    ("P2", "notebooks/dbpedia_compare/p2_classic.kv"),
    ("P2-bound", "notebooks/dbpedia_compare/p2_bound.kv"),
    ("P2-bound cap16", "notebooks/dbpedia_investigate/p2_bound_cap16_lr025_FULL.kv"),
    ("P3", "notebooks/dbpedia_investigate/p3_classic_FULL.kv"),
    ("P3-bound", "notebooks/dbpedia_compare/p3_bound.kv"),
    ("P3-bound cap16", "notebooks/dbpedia_investigate/p3_bound_cap16_lr025_FULL.kv"),
)


def strip_iri(tok: str) -> str:
    """``<http://...>`` -> ``http://...`` (leaves bare IRIs untouched)."""
    tok = tok.strip()
    if tok.startswith("<") and tok.endswith(">"):
        return tok[1:-1]
    return tok


def load_subclass_parents(ontology_nt: Path) -> dict[str, list[str]]:
    """child class IRI -> direct parent class IRIs (``rdfs:subClassOf``), bare IRIs."""
    parents: dict[str, set[str]] = defaultdict(set)
    with ontology_nt.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "subClassOf" not in line:
                continue
            parts = line.rstrip(" .\n").split(" ", 2)
            if len(parts) != 3:
                continue
            s, p, o = (strip_iri(x) for x in parts)
            if p == RDFS_SUBCLASS:
                parents[s].add(o)
    return {c: sorted(ps) for c, ps in parents.items()}


def make_domain_resolver(parents: dict[str, list[str]]):
    """Return ``cls_iri -> domain name | None`` (climbs subClassOf, memoized)."""

    @lru_cache(maxsize=None)
    def domain_for_class(cls: str) -> str | None:
        seen = {cls}
        stack = [cls]
        hits: set[str] = set()
        while stack:
            c = stack.pop()
            if c in CLASS_TO_DOMAIN:
                hits.add(CLASS_TO_DOMAIN[c])
            for par in parents.get(c, ()):
                if par not in seen:
                    seen.add(par)
                    stack.append(par)
        if not hits:
            return None
        return min(hits, key=lambda d: DOMAIN_PRIORITY[d])

    return domain_for_class


def stream_instance_types(types_path: Path):
    """Yield ``(subject_iri, object_class_iri)`` (bare) from an N-Triples (.bz2) types dump."""
    opener = bz2.open if types_path.suffix == ".bz2" else open
    with opener(types_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip(" .\n").split(" ", 2)
            if len(parts) != 3:
                continue
            s, p, o = parts
            if strip_iri(p) != RDF_TYPE:
                continue
            yield strip_iri(s), strip_iri(o)


def build_entity_domains(
    types_path: Path,
    vocab: set[str],
    domain_for_class,
) -> dict[str, str]:
    """entity IRI (in vocab) -> highest-priority top-level domain name."""
    ent_domain: dict[str, str] = {}
    for subj, obj in stream_instance_types(types_path):
        if subj not in vocab:
            continue
        dom = domain_for_class(obj)
        if dom is None:
            continue
        prev = ent_domain.get(subj)
        if prev is None or DOMAIN_PRIORITY[dom] < DOMAIN_PRIORITY[prev]:
            ent_domain[subj] = dom
    return ent_domain


def sample_plot_set(
    ent_domain: dict[str, str],
    per_domain: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample up to ``per_domain`` entities per domain. Returns [(entity, domain)]."""
    by_domain: dict[str, list[str]] = defaultdict(list)
    for ent, dom in ent_domain.items():
        by_domain[dom].append(ent)

    rng = np.random.default_rng(seed)
    plot: list[tuple[str, str]] = []
    for name, _ in DOMAINS:
        ents = by_domain.get(name, [])
        ents.sort()  # deterministic before sampling
        if per_domain > 0 and len(ents) > per_domain:
            idx = rng.choice(len(ents), size=per_domain, replace=False)
            ents = [ents[i] for i in idx]
        plot.extend((e, name) for e in ents)
    return plot


# --------------------------------------------------------------------------- #
# Embedding access
# --------------------------------------------------------------------------- #


class Vectors:
    """Uniform read access over .kv (mmapped) / .model / .pt embedding stores."""

    def __init__(self, key_to_index: dict[str, int], getter):
        self.key_to_index = key_to_index
        self._get = getter

    def vector(self, key: str) -> np.ndarray:
        return np.asarray(self._get(key), dtype=np.float32)


def open_vectors(path: Path) -> Vectors:
    suffix = path.suffix.lower()
    if suffix == ".kv":
        from gensim.models import KeyedVectors

        wv = KeyedVectors.load(str(path), mmap="r")
        return Vectors(dict(wv.key_to_index), wv.get_vector)
    if suffix == ".model":
        from gensim.models import Word2Vec

        try:  # checkpoints may pickle a custom subclass as __main__.Word2VecWithStepLoss
            from _word2vec import Word2VecWithStepLoss

            sys.modules["__main__"].Word2VecWithStepLoss = Word2VecWithStepLoss
        except Exception:
            pass
        wv = Word2Vec.load(str(path)).wv
        return Vectors(dict(wv.key_to_index), wv.get_vector)

    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    emb = np.asarray(ckpt["embeddings"], dtype=np.float32)
    w2i = ckpt["word2idx"]
    return Vectors(w2i, lambda k: emb[w2i[k]])


# --------------------------------------------------------------------------- #
# Projection + plotting
# --------------------------------------------------------------------------- #


def project(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)

    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        init="pca",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(X)


def plot_grid(
    panels: list[tuple[str, np.ndarray]],
    domains: np.ndarray,
    method: str,
    out_path: Path,
    seed: int,
    dpi: int,
) -> None:
    present = [name for name, _ in DOMAINS if (domains == name).any()]
    n = len(panels)
    ncols = 1 if n == 1 else (2 if n <= 4 else 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 6.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, X) in zip(axes, panels):
        Z = project(X, method, seed)
        for name in present:
            mask = domains == name
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                s=10,
                alpha=0.7,
                linewidths=0.2,
                edgecolors="#333333",
                color=DOMAIN_COLOR[name],
            )
        ax.set_title(label, fontsize=15, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=14,
               markerfacecolor=DOMAIN_COLOR[name], markeredgecolor="#333333", label=name)
        for name in present
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(present), 5),
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.suptitle(
        f"DBpedia RDF2Vec entity embeddings ({method.upper()}), colored by entity type",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--embeddings",
        action="append",
        metavar="LABEL=PATH",
        help="Variant to plot as label=path (repeatable). Defaults to the 4-variant compare set.",
    )
    ap.add_argument("--instance-types", type=Path,
                    default=REPO / "dbpedia_graph/instance-types_lang=en_specific.ttl.bz2")
    ap.add_argument("--ontology", type=Path, default=REPO / "dbpedia_graph/ontology.nt")
    ap.add_argument("--preset", choices=tuple(PRESETS), default="toplevel",
                    help="Domain coloring scheme: broad top types or specific classes.")
    ap.add_argument("--method", choices=("tsne", "pca", "both"), default="both")
    ap.add_argument("--per-domain", type=int, default=600,
                    help="Max entities sampled per top-level domain (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=REPO / "output/dbpedia/schema_projection")
    ap.add_argument("--dpi", type=int, default=170)
    return ap.parse_args()


def resolve_variants(raw: list[str] | None) -> list[tuple[str, Path]]:
    if not raw:
        return [(label, REPO / path) for label, path in DEFAULT_VARIANTS]
    out: list[tuple[str, Path]] = []
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--embeddings expects label=path, got: {item}")
        label, path = item.split("=", 1)
        out.append((label, Path(path)))
    return out


def main() -> None:
    args = parse_args()
    configure(args.preset)
    variants = resolve_variants(args.embeddings)
    for label, path in variants:
        if not path.is_file():
            raise SystemExit(f"missing embeddings for {label!r}: {path}")

    print(f"opening {len(variants)} variants ...", flush=True)
    stores = [(label, open_vectors(path)) for label, path in variants]

    # Shared vocab = entities present in every variant (same family -> ~identical).
    vocab = set(stores[0][1].key_to_index)
    for _, store in stores[1:]:
        vocab &= set(store.key_to_index)
    vocab = {k for k in vocab if "/resource/" in k}
    print(f"shared resource vocab: {len(vocab):,}", flush=True)

    print("loading ontology subClassOf ...", flush=True)
    parents = load_subclass_parents(args.ontology)
    domain_for_class = make_domain_resolver(parents)

    print("streaming instance types ...", flush=True)
    ent_domain = build_entity_domains(args.instance_types, vocab, domain_for_class)
    counts = defaultdict(int)
    for dom in ent_domain.values():
        counts[dom] += 1
    print("typed entities per domain:", flush=True)
    for name, _ in DOMAINS:
        print(f"  {name:16s} {counts.get(name, 0):>8,}", flush=True)

    plot = sample_plot_set(ent_domain, args.per_domain, args.seed)
    if not plot:
        raise SystemExit("no entities matched any top-level domain; check inputs")
    entities = [e for e, _ in plot]
    domains = np.array([d for _, d in plot], dtype=object)
    print(f"plotting {len(entities):,} entities across {len(set(domains))} domains", flush=True)

    # Same entity set -> one vector matrix per variant.
    panels: list[tuple[str, np.ndarray]] = []
    for label, store in stores:
        X = np.stack([store.vector(e) for e in entities])
        panels.append((label, X))
        print(f"  gathered {label}: {X.shape}", flush=True)

    methods = ("tsne", "pca") if args.method == "both" else (args.method,)
    for method in methods:
        out_path = args.out_dir / f"dbpedia_schema_{args.preset}_{method}.png"
        plot_grid(panels, domains, method, out_path, args.seed, args.dpi)


if __name__ == "__main__":
    main()
