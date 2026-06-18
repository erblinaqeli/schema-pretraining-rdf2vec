"""
MASCHInE-style transfer of protograph (P2) embeddings to instance RDF2Vec vocabulary
(Hubert et al., arXiv:2306.03659, Section 3.2).

Builds entity → most-specific class mapping from ontology, then initializes new
Word2Vec rows from pretrained class vectors. Strategies: most-specific only (default),
unweighted mean over the full superclass closure, or distance-decayed mean toward roots.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from gensim.models import Word2Vec

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _protograph_gen import iter_rdf_iris
from _walks import nt_term

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_PROPERTY = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
OWL_NS = "http://www.w3.org/2002/07/owl#"
OWL_CLASS = f"{OWL_NS}Class"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"

# Subjects with any of these as asserted rdf:type are RDF properties, not individuals.
RDF_PROPERTY_TYPE_IRIS: frozenset[str] = frozenset(
    {
        RDF_PROPERTY,
        f"{OWL_NS}ObjectProperty",
        f"{OWL_NS}DatatypeProperty",
        f"{OWL_NS}AnnotationProperty",
        f"{OWL_NS}FunctionalProperty",
        f"{OWL_NS}InverseFunctionalProperty",
        f"{OWL_NS}SymmetricProperty",
        f"{OWL_NS}AsymmetricProperty",
        f"{OWL_NS}ReflexiveProperty",
        f"{OWL_NS}IrreflexiveProperty",
        f"{OWL_NS}TransitiveProperty",
    }
)

# Synthetic DLCC-style extra individuals (any test case):
# EXTRA_I_FOR_CLASS_C_tc07_152_640 -> C_tc07_152
EXTRA_CLASS_RE = re.compile(r"^EXTRA_I_FOR_CLASS_(C_tc\d+_\d+)_")

INIT_STRATEGIES = frozenset({"most_specific", "all_init", "average_hier", "weight_hierarchy"})

MASCHINE_CLASS_INIT_STRATEGIES = frozenset(
    {"most_specific", "all_init", "ancestor_mean", "ancestor_weighted"},
)

_INIT_STRATEGY_TO_INTERNAL: dict[str, str] = {
    "most_specific": "most_specific",
    "all_init": "all_init",
    "average_hier": "ancestor_mean",
    "weight_hierarchy": "ancestor_weighted",
    # Legacy CLI / internal names (backward compatibility).
    "ancestor_mean": "ancestor_mean",
    "ancestor_weighted": "ancestor_weighted",
}

_INTERNAL_TO_USER_INIT_STRATEGY: dict[str, str] = {
    "most_specific": "most_specific",
    "all_init": "all_init",
    "ancestor_mean": "average_hier",
    "ancestor_weighted": "weight_hierarchy",
}


def normalize_init_strategy(value: str) -> str:
    """Map user-facing or legacy strategy names to internal MASCHInE strategy keys."""
    internal = _INIT_STRATEGY_TO_INTERNAL.get(value)
    if internal is None:
        accepted = sorted(INIT_STRATEGIES | {"all_init", "ancestor_mean", "ancestor_weighted"})
        raise ValueError(f"unknown init_strategy {value!r}; expected one of {accepted}")
    return internal


def user_facing_init_strategy(value: str) -> str:
    """Return canonical user-facing init_strategy name for logging and stats."""
    return _INTERNAL_TO_USER_INIT_STRATEGY[normalize_init_strategy(value)]

PROTOGRAPH_INIT_LABELS = frozenset(
    {
        "maschine_class_mean",
        "maschine_ancestor_mean",
        "maschine_ancestor_weighted",
        "maschine_relation_copy",
        "predicate_transfer",
        "stage1_pretrained",
    }
)

TokenCategory = Literal["class", "relation", "instance", "other"]

INSTANCE_CLASS_INIT_LABELS = frozenset(
    {
        "maschine_class_mean",
        "maschine_ancestor_mean",
        "maschine_ancestor_weighted",
    }
)


@dataclass
class CategoryInitCounts:
    """Per-category init counts for new stage-2 vocabulary tokens."""

    from_protograph: int = 0
    random: int = 0

    @property
    def total(self) -> int:
        return self.from_protograph + self.random

    def to_dict(self) -> dict[str, int]:
        return {
            "from_protograph": self.from_protograph,
            "random": self.random,
            "total": self.total,
        }


@dataclass
class FinetuneInitStats:
    """Counts for new finetune-vocabulary tokens (not in stage-1 protograph model)."""

    instances: CategoryInitCounts | None = None
    classes: CategoryInitCounts | None = None
    relations: CategoryInitCounts | None = None
    other: CategoryInitCounts | None = None

    def __post_init__(self) -> None:
        if self.instances is None:
            self.instances = CategoryInitCounts()
        if self.classes is None:
            self.classes = CategoryInitCounts()
        if self.relations is None:
            self.relations = CategoryInitCounts()
        if self.other is None:
            self.other = CategoryInitCounts()

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "instances": self.instances.to_dict(),
            "classes": self.classes.to_dict(),
            "relations": self.relations.to_dict(),
            "other": self.other.to_dict(),
        }


@dataclass
class InstanceInitCoverageStats:
    """Per-instance init outcomes for ontology individuals (``I_*`` / ``EXTRA_I_*``)."""

    total_instances: int = 0
    total_classes: int = 0
    with_class_mapping: int = 0
    without_class_mapping: int = 0
    initialized_from_class: int = 0
    random_no_pretrained_class: int = 0
    random_no_class_mapping: int = 0
    not_in_finetune_vocab: int = 0

    @property
    def in_finetune_vocab(self) -> int:
        return (
            self.initialized_from_class
            + self.random_no_pretrained_class
            + self.random_no_class_mapping
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "total_instances": self.total_instances,
            "total_classes": self.total_classes,
            "with_class_mapping": self.with_class_mapping,
            "without_class_mapping": self.without_class_mapping,
            "in_finetune_vocab": self.in_finetune_vocab,
            "initialized_from_class": self.initialized_from_class,
            "random_no_pretrained_class": self.random_no_pretrained_class,
            "random_no_class_mapping": self.random_no_class_mapping,
            "not_in_finetune_vocab": self.not_in_finetune_vocab,
        }


@dataclass
class RelationInitCoverageStats:
    """Per-relation init outcomes for ontology ``P_*`` properties."""

    total: int = 0
    from_protograph: int = 0
    not_initialized: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_graph_relations": self.total,
            "init_from_protograph": self.from_protograph,
            "not_initialized": self.not_initialized,
        }


@dataclass
class AllInitFallbackStats:
    """Hop statistics for ``all_init`` strategy (ancestor fallback from most-specific class)."""

    instances_with_fallback: int = 0
    max_hops: int = 0
    _hop_sum: float = 0.0
    _initialized_count: int = 0

    def record(self, hops: int) -> None:
        """Record hop count for one MASCHInE-initialized instance."""
        self._initialized_count += 1
        self._hop_sum += hops
        if hops > self.max_hops:
            self.max_hops = hops
        if hops > 0:
            self.instances_with_fallback += 1

    @property
    def avg_hops(self) -> float:
        if self._initialized_count == 0:
            return 0.0
        return self._hop_sum / self._initialized_count

    def to_dict(self) -> dict[str, float | int]:
        return {
            "instances_with_fallback": self.instances_with_fallback,
            "max_hops": self.max_hops,
            "avg_hops": self.avg_hops,
        }


FINETUNE_INIT_STATS_FILENAME = "finetune_init_stats.json"


def is_ontology_instance(ent: str) -> bool:
    """True for dataset individuals used in MASCHInE instance init (``I_*`` or ``EXTRA_I_*``)."""
    if ent.startswith("I_"):
        return True
    return _class_from_extra_name(ent) is not None


def is_maschine_instance_entity(ent: str) -> bool:
    """True for individuals tracked in MASCHInE init coverage (synthetic + DBpedia resources)."""
    if is_ontology_instance(ent):
        return True
    return ent.startswith("http://dbpedia.org/resource/")


def vocab_inner_iris(model: Word2Vec) -> set[str]:
    """Bare IRI strings for finetune-vocabulary tokens (angle-bracket or bare IRIs)."""
    return {token_inner_iri(token) for token in model.wv.index_to_key}


def iter_instance_type_triples(path: Path, subjects: set[str] | None = None):
    """Yield ``(subject, predicate, object)`` for ``rdf:type`` rows with IRI objects."""
    from _dbpedia.iri_nt_materialize import _guess_format, _open_binary_maybe_compressed
    from pyoxigraph import NamedNode, parse

    with _open_binary_maybe_compressed(path) as bio:
        for t in parse(input=bio, format=_guess_format(path), lenient=True):
            if not isinstance(t.subject, NamedNode):
                continue
            if not isinstance(t.predicate, NamedNode):
                continue
            if not isinstance(t.object, NamedNode):
                continue
            s, p, o = t.subject.value, t.predicate.value, t.object.value
            if p != RDF_TYPE:
                continue
            if subjects is not None and s not in subjects:
                continue
            yield s, p, o


def load_maschine_entity_mapping(
    ontology_nt: Path,
    *,
    instance_types_path: Path | None = None,
    subject_filter: set[str] | None = None,
    quiet: bool = False,
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """
    Schema hierarchy and entity typing for MASCHInE init.

    ``ontology_nt`` supplies ``rdfs:subClassOf`` and any ``rdf:type`` rows present there.
    When ``instance_types_path`` is set, additional resource ``rdf:type`` rows are merged in
    (optionally limited to ``subject_filter`` for large DBpedia dumps).
    """
    parents, entity_types = load_ontology_mapping(ontology_nt)
    if instance_types_path is None:
        return parents, entity_types
    if not instance_types_path.is_file():
        print(
            f"Warning: instance-types file not found: {instance_types_path}",
            file=sys.stderr,
            flush=True,
        )
        return parents, entity_types
    triple_iter = None
    suffix = instance_types_path.suffix.lower()
    if suffix in (".bz2", ".gz", ".ttl", ".turtle") or instance_types_path.name.endswith(".ttl.bz2"):
        triple_iter = iter_instance_type_triples(instance_types_path, subject_filter)
    n = merge_instance_rdf_types(
        parents,
        entity_types,
        instance_types_path,
        subject_filter=subject_filter,
        triple_iter=triple_iter,
    )
    if not quiet:
        print(
            f"Merged rdf:type for {n} instance subjects from {instance_types_path}",
            flush=True,
        )
    return parents, entity_types


def collect_ontology_relations(ontology_nt: Path) -> set[str]:
    """Collect ``P_*`` property IRIs declared via domain/range or ``rdf:type`` in the ontology."""
    relations: set[str] = set()
    for s, p, _o in iter_rdf_iris(ontology_nt):
        if not s.startswith("P_"):
            continue
        if p in (RDFS_DOMAIN, RDFS_RANGE, RDF_TYPE):
            relations.add(s)
    return relations


def count_ontology_classes(
    parents: dict[str, set[str]],
    entity_types: dict[str, list[str]],
) -> int:
    """Count unique ``C_*`` class IRIs declared or referenced in the ontology."""
    classes: set[str] = set()
    for c in parents:
        if c.startswith("C_"):
            classes.add(c)
    for pars in parents.values():
        classes.update(p for p in pars if p.startswith("C_"))
    for types in entity_types.values():
        classes.update(t for t in types if t.startswith("C_"))
    return len(classes)


def token_inner_iri(token: str) -> str:
    """Normalize a walk/checkpoint token to the bare IRI string."""
    token = token.strip()
    if len(token) >= 2 and token[0] == "<" and token[-1] == ">":
        return token[1:-1]
    return token


def resolve_vocab_token(inner: str, wv) -> str | None:
    """Return the finetune-vocab token string for a bare ontology IRI, if present."""
    inner = token_inner_iri(inner)
    for cand in (inner, nt_term(inner)):
        if cand in wv.key_to_index:
            return cand
    return None


def init_label_for_token(token: str, new_token_init: dict[str, str]) -> str | None:
    """Look up a finetune init label across bare and ``<IRI>`` token spellings."""
    inner = token_inner_iri(token)
    for cand in (token, inner, nt_term(inner)):
        label = new_token_init.get(cand)
        if label is not None:
            return label
    return None


def stage1_vector_lookup(
    token: str,
    stage1_vectors: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Return a protograph-pretrained row when ``token`` is already in stage-1 vocabulary."""
    token = token.strip()
    if not token:
        return None
    inner = token_inner_iri(token)
    for cand in (token, inner, nt_term(inner)):
        vec = stage1_vectors.get(cand)
        if vec is not None:
            return vec
    return None


def _lookup_class_vector(
    class_token: str,
    stage1_vectors: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Resolve a class IRI or ``<IRI>`` walk token to a protograph-pretrained vector."""
    return stage1_vector_lookup(class_token, stage1_vectors)


def _lookup_class_vector_with_fallback(
    class_inner: str,
    parents: dict[str, set[str]],
    stage1_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, int]:
    """BFS upward from *class_inner*; return nearest pretrain vector and hop count (0 = direct)."""
    q: deque[tuple[str, int]] = deque([(class_inner, 0)])
    seen = {class_inner}
    while q:
        node, hops = q.popleft()
        vec = _lookup_class_vector(node, stage1_vectors)
        if vec is not None:
            return vec, hops
        for par in parents.get(node, ()):
            if par not in seen:
                seen.add(par)
                q.append((par, hops + 1))
    return None, 0


def maschine_embedding_for_token(
    token: str,
    stage1_vectors: dict[str, np.ndarray],
    parents: dict[str, set[str]],
    ent_to_classes: dict[str, list[str]],
    *,
    strategy: str = "most_specific",
    ancestor_decay: float = 0.5,
    normalize_class_codes: bool = False,
) -> np.ndarray | None:
    """
    Initialize one instance/predicate token from protograph class vectors.

    Returns a pretrained row unchanged when ``token`` is already in ``stage1_vectors``.
    Otherwise applies the same rules as :func:`apply_maschine_initialization`.

    ``normalize_class_codes`` — unit-normalize each class code before averaging.
    """
    direct = stage1_vector_lookup(token, stage1_vectors)
    if direct is not None:
        return direct

    inner = token_inner_iri(token)
    class_tokens: list[str] | None = ent_to_classes.get(inner)

    if class_tokens is None and token_is_instance_entity(nt_term(inner)):
        c = _class_from_extra_name(inner)
        if c is not None:
            mst = _most_specific_types({c}, parents)
            class_tokens = [nt_term(x) for x in mst]

    roots_inner: list[str] = []
    if class_tokens:
        roots_inner = [
            t[1:-1]
            for t in class_tokens
            if len(t) >= 2 and t[0] == "<" and t[-1] == ">"
        ]

    if roots_inner:
        return init_vector_from_class_roots(
            roots_inner,
            parents,
            stage1_vectors,
            strategy=strategy,
            ancestor_decay=ancestor_decay,
            normalize_class_codes=normalize_class_codes,
        )

    if inner.startswith("P_"):
        return _lookup_class_vector(inner, stage1_vectors)

    return None


def min_hops_upward_from_roots(
    roots: list[str],
    parents: dict[str, set[str]],
) -> dict[str, int]:
    """For each class reachable via ``rdfs:subClassOf`` upward from ``roots``, shortest hop count from the nearest root."""
    if not roots:
        return {}
    dist: dict[str, int] = {}
    q: deque[str] = deque()
    for r in roots:
        if r not in dist:
            dist[r] = 0
            q.append(r)
    while q:
        u = q.popleft()
        du = dist[u]
        for par in parents.get(u, ()):
            nd = du + 1
            if par not in dist or nd < dist[par]:
                dist[par] = nd
                q.append(par)
    return dist


def init_vector_from_class_roots(
    roots_inner: list[str],
    parents: dict[str, set[str]],
    stage1_vectors: dict[str, np.ndarray],
    *,
    strategy: str,
    ancestor_decay: float = 0.5,
    normalize_class_codes: bool = False,
) -> np.ndarray | None:
    """
    Combine pretrained class vectors for initializing an instance row.

    * ``most_specific`` — mean of vectors for ``roots_inner`` (after MASCHInE filtering).
    * ``all_init`` — like ``most_specific``, but climb ``subClassOf`` when a root class is OOV in pretrain.
    * ``ancestor_mean`` — mean over ``roots_inner`` and all transitive superclasses.
    * ``ancestor_weighted`` — weighted mean over that same closure with weight ``ancestor_decay**d``,
      where ``d`` is hop distance upward from the nearest root.

    ``normalize_class_codes`` — unit-normalize each class code before averaging.
    """
    vec, _hops = init_vector_from_class_roots_with_hops(
        roots_inner,
        parents,
        stage1_vectors,
        strategy=strategy,
        ancestor_decay=ancestor_decay,
        normalize_class_codes=normalize_class_codes,
    )
    return vec


def init_vector_from_class_roots_with_hops(
    roots_inner: list[str],
    parents: dict[str, set[str]],
    stage1_vectors: dict[str, np.ndarray],
    *,
    strategy: str,
    ancestor_decay: float = 0.5,
    normalize_class_codes: bool = False,
) -> tuple[np.ndarray | None, int]:
    """Like :func:`init_vector_from_class_roots`, also returning max hops per root for ``all_init``.

    ``normalize_class_codes`` (ablation): when ``True``, L2-normalize each class
    code to unit length BEFORE averaging, so every class contributes equally
    regardless of its raw pretrain norm. Default ``False`` keeps the raw norms.
    """
    if strategy not in MASCHINE_CLASS_INIT_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(MASCHINE_CLASS_INIT_STRATEGIES)}")
    if not roots_inner:
        return None, 0

    def _u(v: np.ndarray) -> np.ndarray:
        if not normalize_class_codes:
            return v
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    if strategy == "most_specific":
        parts = [_lookup_class_vector(r, stage1_vectors) for r in roots_inner]
        parts = [_u(p) for p in parts if p is not None]
        if not parts:
            return None, 0
        return np.mean(parts, axis=0).astype(np.float32, copy=False), 0

    if strategy == "all_init":
        parts: list[np.ndarray] = []
        hops_per_root: list[int] = []
        for r in roots_inner:
            vec, hops = _lookup_class_vector_with_fallback(r, parents, stage1_vectors)
            if vec is not None:
                parts.append(_u(vec))
                hops_per_root.append(hops)
        if not parts:
            return None, 0
        return np.mean(parts, axis=0).astype(np.float32, copy=False), max(hops_per_root)

    closure = min_hops_upward_from_roots(roots_inner, parents)
    if not closure:
        return None, 0
    if strategy == "ancestor_mean":
        parts = [
            _u(v)
            for c_inner in closure
            if (v := _lookup_class_vector(c_inner, stage1_vectors)) is not None
        ]
        if not parts:
            return None, 0
        return np.mean(parts, axis=0).astype(np.float32, copy=False), 0

    if strategy == "ancestor_weighted":
        if not (0.0 < ancestor_decay <= 1.0):
            raise ValueError("ancestor_weighted requires 0 < --maschine-ancestor-decay <= 1")
        w_accum = 0.0
        vec_accum: np.ndarray | None = None
        for c_inner, d in closure.items():
            v = _lookup_class_vector(c_inner, stage1_vectors)
            if v is None:
                continue
            v = _u(v)
            w = ancestor_decay**d
            if vec_accum is None:
                vec_accum = (w * v).astype(np.float32, copy=False)
            else:
                vec_accum = vec_accum + w * v
            w_accum += w
        if vec_accum is None or w_accum <= 0.0:
            return None, 0
        return (vec_accum / w_accum).astype(np.float32, copy=False), 0
    return None, 0


def _ancestors_generalizations(node: str, parents: dict[str, set[str]]) -> set[str]:
    """All classes reachable from ``node`` by walking up ``subClassOf`` (including ``node``)."""
    out: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(parents.get(cur, ()))
    return out


def _most_specific_types(asserted: set[str], parents: dict[str, set[str]]) -> list[str]:
    """
    Among asserted class IRIs (inner strings), keep those not strictly generalized
    by another asserted type (MASCHInE: most specific class(es)).
    """
    if not asserted:
        return []
    anc = {t: _ancestors_generalizations(t, parents) for t in asserted}
    minimal: list[str] = []
    for t in asserted:
        more_specific_exists = False
        for t_other in asserted:
            if t_other == t:
                continue
            if t in anc[t_other]:
                more_specific_exists = True
                break
        if not more_specific_exists:
            minimal.append(t)
    return minimal


def _subject_is_rdf_property(raw_types: dict[str, set[str]], ent: str) -> bool:
    """True if ``ent`` is declared in the ontology as an RDF/OWL property (not an individual)."""
    ts = raw_types.get(ent)
    if not ts:
        return False
    return bool(ts & RDF_PROPERTY_TYPE_IRIS)


def merge_instance_rdf_types(
    parents: dict[str, set[str]],
    entity_types: dict[str, list[str]],
    types_path: Path,
    *,
    subject_filter: set[str] | None = None,
    triple_iter=None,
) -> int:
    """
    Add or update ``entity_types`` from ``rdf:type`` rows in ``types_path``.

    ``triple_iter``, when given, must yield ``(subject, predicate, object)`` IRI strings
    (e.g. :func:`src.dbpedia.nt_stream.iter_nt_iri_triples` for ``.nt`` / ``.bz2`` dumps).
    Otherwise :func:`iter_rdf_iris` is used (``.nt`` / ``.ttl``).

    Returns the number of subjects that received at least one type.
    """
    if triple_iter is None:
        triple_iter = iter_rdf_iris(types_path)
    asserted: dict[str, set[str]] = defaultdict(set)
    for s, p, o in triple_iter:
        if p != RDF_TYPE:
            continue
        if subject_filter is not None and s not in subject_filter:
            continue
        if o == OWL_CLASS:
            continue
        asserted[s].add(o)
    n = 0
    for ent, types in asserted.items():
        mst = _most_specific_types(types, parents)
        if mst:
            entity_types[ent] = mst
            n += 1
    return n


def load_ontology_mapping(ontology_nt: Path) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """
    Returns:
        parents: class IRI inner -> direct parent class IRI inners (subClassOf edges child -> parents)
        entity_types: entity IRI inner -> asserted rdf:type class inners (owl:Class declarations skipped)
    """
    parents: dict[str, set[str]] = defaultdict(set)
    raw_types: dict[str, set[str]] = defaultdict(set)

    for s, p, o in iter_rdf_iris(ontology_nt):
        if p == RDFS_SUBCLASS:
            parents[s].add(o)
            continue
        if p != RDF_TYPE:
            continue
        if o == OWL_CLASS:
            continue
        raw_types[s].add(o)

    raw_dict = dict(raw_types)
    entity_types: dict[str, list[str]] = {}
    for ent, types in raw_dict.items():
        mst = _most_specific_types(types, parents)
        entity_types[ent] = mst

    for ent in list(entity_types.keys()):
        if _subject_is_rdf_property(raw_dict, ent):
            del entity_types[ent]

    return dict(parents), entity_types


def _class_from_extra_name(entity_inner: str) -> str | None:
    m = EXTRA_CLASS_RE.match(entity_inner)
    if m:
        return m.group(1)
    return None


def instance_class_mapping_from_ontology(
    ontology_nt: Path,
    *,
    instance_types_path: Path | None = None,
    subject_filter: set[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    For each individual in ``ontology_nt`` with ``rdf:type`` (or derivable EXTRA_I_* typing),
    return most-specific class IRIs and the corresponding walk tokens used for MASCHInE init.

    Keys are **entity IRI strings** (same as N-Triples subject inner, without angle brackets).
    """
    parents, entity_types = load_maschine_entity_mapping(
        ontology_nt,
        instance_types_path=instance_types_path,
        subject_filter=subject_filter,
        quiet=True,
    )
    ent_to_tokens = build_entity_to_class_tokens(parents, entity_types)
    return {
        ent: {
            "most_specific_class_iris": list(entity_types[ent]),
            "class_tokens": list(ent_to_tokens.get(ent, [])),
        }
        for ent in entity_types
    }


def build_entity_to_class_tokens(
    parents: dict[str, set[str]],
    entity_types: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Map entity IRI inner string -> list of class tokens ``<C_...>`` for averaging.
    Uses rdf:type when present; otherwise EXTRA_I_* name pattern.
    """
    out: dict[str, list[str]] = {}
    for ent, cls_list in entity_types.items():
        if cls_list:
            out[ent] = [nt_term(c) for c in cls_list]
    for ent in entity_types:
        if ent in out:
            continue
        c = _class_from_extra_name(ent)
        if c is not None:
            mst = _most_specific_types({c}, parents)
            out[ent] = [nt_term(x) for x in mst]
    return out


def token_vocab_category(token: str) -> TokenCategory:
    """Bucket a walk token for init statistics (class / relation / instance / other)."""
    inner = token_inner_iri(token)
    if inner.startswith("P_"):
        return "relation"
    if inner.startswith("C_"):
        return "class"
    if inner.startswith("I_") or inner.startswith("EXTRA_I_FOR_CLASS_"):
        return "instance"
    if inner.startswith("http://dbpedia.org/resource/"):
        return "instance"
    return "other"


def token_is_instance_entity(token: str) -> bool:
    if len(token) < 2 or token[0] != "<" or token[-1] != ">":
        return False
    inner = token[1:-1]
    if inner.startswith("I_"):
        return True
    if inner.startswith("EXTRA_I_FOR_CLASS_"):
        return True
    return False


def restore_pretrained_vocab_rows(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    *,
    new_token_init: dict[str, str] | None = None,
) -> int:
    """
    After rebuilding finetune vocabulary from instance walks, copy stage-1 rows back for
    tokens that appear in both the new vocabulary and ``stage1_vectors``.

    Pretrain-only protograph tokens that never appear in instance walks are dropped.
    """
    wv = model.wv
    n = 0
    for token in wv.index_to_key:
        vec = stage1_vectors.get(token)
        if vec is None:
            continue
        idx = wv.key_to_index[token]
        wv.vectors[idx] = vec.copy()
        if model.syn1neg is not None:
            model.syn1neg[idx] = vec.copy()
        n += 1
        if new_token_init is not None and token not in new_token_init:
            new_token_init[token] = "stage1_pretrained"
    return n


def transfer_predicate_embeddings_from_pretrained(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    *,
    new_token_init: dict[str, str] | None = None,
) -> int:
    """
    For each vocabulary token not present in ``stage1_vectors`` whose inner IRI starts with
    ``P_`` (DLCC relation/predicate walks), copy the pretrained row if that token existed in stage 1.

    Use when MASCHInE class init is disabled but RDF2Vec should still reuse relation vectors
    from protograph pretraining. Idempotent for rows already matching ``stage1_vectors``.

    Returns the number of predicate rows overwritten.
    """
    wv = model.wv
    n = 0
    for token in wv.index_to_key:
        if token in stage1_vectors:
            continue
        inner = token[1:-1] if len(token) >= 2 and token[0] == "<" else token
        if not inner.startswith("P_"):
            continue
        vec = _lookup_class_vector(inner, stage1_vectors)
        if vec is None:
            continue
        idx = wv.key_to_index[token]
        wv.vectors[idx] = vec.copy()
        if model.syn1neg is not None:
            model.syn1neg[idx] = vec.copy()
        n += 1
        if new_token_init is not None:
            new_token_init[token] = "predicate_transfer"
    return n


def compute_finetune_init_stats(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
) -> FinetuneInitStats:
    """
    Summarize how new finetune-vocabulary rows were initialized, by token category.

    * **instances** — ``I_*`` / ``EXTRA_I_*`` instance entities.
    * **classes** — ``C_*`` class tokens.
    * **relations** — ``P_*`` predicate/relation tokens.
    """
    stats = FinetuneInitStats()
    wv = model.wv
    buckets = {
        "instance": stats.instances,
        "class": stats.classes,
        "relation": stats.relations,
        "other": stats.other,
    }
    for token in wv.index_to_key:
        if token in stage1_vectors:
            continue
        category = token_vocab_category(token)
        label = new_token_init.get(token, "random")
        from_protograph = label in PROTOGRAPH_INIT_LABELS
        bucket = buckets[category]
        if from_protograph:
            bucket.from_protograph += 1
        else:
            bucket.random += 1
    return stats


def _classify_instance_init_outcome(
    ent: str,
    token: str,
    *,
    in_vocab: bool,
    has_class_mapping: bool,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
    parents: dict[str, set[str]],
    ent_to_classes: dict[str, list[str]],
    strategy: str,
    ancestor_decay: float,
) -> Literal[
    "initialized_from_class",
    "random_no_pretrained_class",
    "random_no_class_mapping",
    "not_in_finetune_vocab",
]:
    if not in_vocab:
        return "not_in_finetune_vocab"

    if stage1_vector_lookup(token, stage1_vectors) is not None:
        return "initialized_from_class"

    label = init_label_for_token(token, new_token_init)
    if label in INSTANCE_CLASS_INIT_LABELS:
        return "initialized_from_class"

    if label == "random_no_pretrained_class":
        return "random_no_pretrained_class"
    if label == "random_no_class_mapping":
        return "random_no_class_mapping"

    if label in PROTOGRAPH_INIT_LABELS:
        return "initialized_from_class"

    if maschine_embedding_for_token(
        token,
        stage1_vectors,
        parents,
        ent_to_classes,
        strategy=strategy,
        ancestor_decay=ancestor_decay,
    ) is not None:
        return "initialized_from_class"
    if has_class_mapping:
        return "random_no_pretrained_class"
    return "random_no_class_mapping"


def compute_instance_init_coverage(
    ontology_nt: Path,
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
    *,
    strategy: str = "most_specific",
    ancestor_decay: float = 0.5,
    instance_types_path: Path | None = None,
    subject_filter: set[str] | None = None,
) -> InstanceInitCoverageStats:
    """
    Classify each ontology instance by finetune-vocab presence and init outcome.

    Instances are ``I_*`` / ``EXTRA_I_*`` (synthetic) and DBpedia resource IRIs when typed.
    """
    strategy = normalize_init_strategy(strategy)
    parents, entity_types = load_maschine_entity_mapping(
        ontology_nt,
        instance_types_path=instance_types_path,
        subject_filter=subject_filter,
        quiet=True,
    )
    ent_to_classes = build_entity_to_class_tokens(parents, entity_types)
    wv = model.wv

    stats = InstanceInitCoverageStats()
    stats.total_classes = count_ontology_classes(parents, entity_types)

    for ent in sorted(entity_types):
        if not is_maschine_instance_entity(ent):
            continue
        stats.total_instances += 1
        has_class_mapping = ent in ent_to_classes
        if has_class_mapping:
            stats.with_class_mapping += 1
        else:
            stats.without_class_mapping += 1

        vocab_token = resolve_vocab_token(ent, wv)
        in_vocab = vocab_token is not None
        outcome = _classify_instance_init_outcome(
            ent,
            vocab_token if vocab_token is not None else nt_term(ent),
            in_vocab=in_vocab,
            has_class_mapping=has_class_mapping,
            stage1_vectors=stage1_vectors,
            new_token_init=new_token_init,
            parents=parents,
            ent_to_classes=ent_to_classes,
            strategy=strategy,
            ancestor_decay=ancestor_decay,
        )
        if outcome == "initialized_from_class":
            stats.initialized_from_class += 1
        elif outcome == "random_no_pretrained_class":
            stats.random_no_pretrained_class += 1
        elif outcome == "random_no_class_mapping":
            stats.random_no_class_mapping += 1
        else:
            stats.not_in_finetune_vocab += 1

    return stats


def _classify_relation_init_outcome(
    token: str,
    *,
    in_vocab: bool,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
) -> Literal["from_protograph", "not_initialized"]:
    if not in_vocab:
        return "not_initialized"

    if stage1_vector_lookup(token, stage1_vectors) is not None:
        return "from_protograph"

    label = init_label_for_token(token, new_token_init)
    if label in PROTOGRAPH_INIT_LABELS:
        return "from_protograph"

    inner = token_inner_iri(token)
    if inner.startswith("P_") and stage1_vector_lookup(inner, stage1_vectors) is not None:
        return "from_protograph"

    return "not_initialized"


def compute_relation_init_coverage(
    ontology_nt: Path,
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
) -> RelationInitCoverageStats:
    """Classify each ontology ``P_*`` relation by finetune-vocab presence and init outcome."""
    relations = collect_ontology_relations(ontology_nt)
    wv = model.wv
    stats = RelationInitCoverageStats()
    stats.total = len(relations)

    for rel in sorted(relations):
        vocab_token = resolve_vocab_token(rel, wv)
        in_vocab = vocab_token is not None
        outcome = _classify_relation_init_outcome(
            vocab_token if vocab_token is not None else nt_term(rel),
            in_vocab=in_vocab,
            stage1_vectors=stage1_vectors,
            new_token_init=new_token_init,
        )
        if outcome == "from_protograph":
            stats.from_protograph += 1
        else:
            stats.not_initialized += 1

    return stats


def build_finetune_init_stats_payload(
    stats: FinetuneInitStats,
    *,
    instance_coverage: InstanceInitCoverageStats | None = None,
    relation_coverage: RelationInitCoverageStats | None = None,
    init_strategy: str | None = None,
    all_init_fallback: AllInitFallbackStats | None = None,
) -> dict[str, object]:
    """Build the simplified finetune init statistics JSON payload."""
    if instance_coverage is not None and instance_coverage.total_instances > 0:
        instances_from_protograph = instance_coverage.initialized_from_class
        instances_random = (
            instance_coverage.random_no_pretrained_class + instance_coverage.random_no_class_mapping
        )
        not_in_walk_corpus = instance_coverage.not_in_finetune_vocab
        vocabulary_size = instance_coverage.in_finetune_vocab
        total_graph_instances = instance_coverage.total_instances
    else:
        instances_from_protograph = stats.instances.from_protograph
        instances_random = stats.instances.random
        not_in_walk_corpus = 0
        vocabulary_size = stats.instances.total
        total_graph_instances = stats.instances.total

    if relation_coverage is not None:
        relations = relation_coverage.to_dict()
    else:
        relations = {
            "total_graph_relations": stats.relations.total,
            "init_from_protograph": stats.relations.from_protograph,
            "not_initialized": stats.relations.random,
        }

    payload: dict[str, object] = {
        "vocabulary_size": vocabulary_size,
        "instances": {
            "total_graph_instances": total_graph_instances,
            "init_from_protograph": instances_from_protograph,
            "init_random": instances_random,
            "not_in_walk_corpus": not_in_walk_corpus,
        },
        "relations": relations,
    }
    if init_strategy is not None:
        payload["init_strategy"] = user_facing_init_strategy(init_strategy)
    if all_init_fallback is not None:
        payload["all_init_fallback"] = all_init_fallback.to_dict()
    return payload


def format_finetune_init_stats_text(
    stats: FinetuneInitStats,
    *,
    instance_coverage: InstanceInitCoverageStats | None = None,
    relation_coverage: RelationInitCoverageStats | None = None,
    init_strategy: str | None = None,
    all_init_fallback: AllInitFallbackStats | None = None,
) -> str:
    payload = build_finetune_init_stats_payload(
        stats,
        instance_coverage=instance_coverage,
        relation_coverage=relation_coverage,
        init_strategy=init_strategy,
        all_init_fallback=all_init_fallback,
    )
    instances = payload["instances"]
    relations = payload["relations"]
    lines = [
        "Finetune vocabulary initialization summary",
        f"vocabulary_size: {payload['vocabulary_size']}",
        "",
        "Instances:",
        f"  total_graph_instances: {instances['total_graph_instances']}",
        f"  init_from_protograph:  {instances['init_from_protograph']}",
        f"  init_random:           {instances['init_random']}",
        f"  not_in_walk_corpus:    {instances['not_in_walk_corpus']}",
        "",
        "Relations:",
        f"  total_graph_relations: {relations['total_graph_relations']}",
        f"  init_from_protograph:  {relations['init_from_protograph']}",
        f"  not_initialized:       {relations['not_initialized']}",
    ]
    if init_strategy is not None:
        lines.extend(["", f"init_strategy: {user_facing_init_strategy(init_strategy)}"])
    if all_init_fallback is not None:
        fb = all_init_fallback.to_dict()
        lines.extend(
            [
                "",
                "all_init_fallback:",
                f"  instances_with_fallback: {fb['instances_with_fallback']}",
                f"  max_hops:                {fb['max_hops']}",
                f"  avg_hops:                {fb['avg_hops']:.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_finetune_init_stats(
    out_dir: Path,
    stats: FinetuneInitStats,
    *,
    instance_coverage: InstanceInitCoverageStats | None = None,
    relation_coverage: RelationInitCoverageStats | None = None,
    init_strategy: str | None = None,
    all_init_fallback: AllInitFallbackStats | None = None,
) -> tuple[Path, Path]:
    """Write JSON + plain-text finetune init statistics under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_finetune_init_stats_payload(
        stats,
        instance_coverage=instance_coverage,
        relation_coverage=relation_coverage,
        init_strategy=init_strategy,
        all_init_fallback=all_init_fallback,
    )
    json_path = out_dir / FINETUNE_INIT_STATS_FILENAME
    txt_path = out_dir / "finetune_init_stats.txt"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(
        format_finetune_init_stats_text(
            stats,
            instance_coverage=instance_coverage,
            relation_coverage=relation_coverage,
            init_strategy=init_strategy,
            all_init_fallback=all_init_fallback,
        ),
        encoding="utf-8",
    )
    return json_path, txt_path


def load_finetune_init_stats(path: Path) -> dict[str, object]:
    """Load finetune init statistics JSON written by :func:`write_finetune_init_stats`."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_finetune_init_stats_path(
    *,
    checkpoint_path: Path,
    run_dir: Path | None = None,
) -> Path | None:
    """Locate finetune_init_stats.json for an evaluation checkpoint."""
    if run_dir is not None:
        candidate = run_dir.resolve() / FINETUNE_INIT_STATS_FILENAME
        if candidate.is_file():
            return candidate
    ckpt = checkpoint_path.resolve()
    for base in (ckpt.parent, ckpt.parent.parent):
        candidate = base / FINETUNE_INIT_STATS_FILENAME
        if candidate.is_file():
            return candidate
    return None


FINETUNE_TOKEN_INIT_VERSION = 1
FINETUNE_TOKEN_INIT_FILENAME = "finetune_token_init.json"


def build_vocab_init_label_map(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
) -> dict[str, str]:
    """Map each finetune-vocab token to its initialization label."""
    labels: dict[str, str] = {}
    for token in model.wv.index_to_key:
        if token in stage1_vectors:
            labels[token] = "stage1_pretrained"
        else:
            labels[token] = new_token_init.get(token, "unknown")
    return labels


def write_finetune_token_init(
    out_dir: Path,
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    new_token_init: dict[str, str],
) -> Path:
    """Persist per-token finetune init labels for later eval coverage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FINETUNE_TOKEN_INIT_FILENAME
    payload = {
        "version": FINETUNE_TOKEN_INIT_VERSION,
        "tokens": build_vocab_init_label_map(model, stage1_vectors, new_token_init),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_finetune_token_init(path: Path) -> dict[str, str]:
    """Load token -> init label map from *finetune_token_init.json*."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError(f"Invalid finetune token init file: {path}")
    return {str(k): str(v) for k, v in tokens.items()}


def resolve_finetune_token_init_path(
    *,
    checkpoint_path: Path,
    run_dir: Path | None = None,
    init_labels_path: Path | None = None,
) -> Path | None:
    """Locate finetune_token_init.json for an evaluation checkpoint."""
    if init_labels_path is not None:
        resolved = init_labels_path.resolve()
        if resolved.is_file():
            return resolved
    if run_dir is not None:
        candidate = run_dir.resolve() / FINETUNE_TOKEN_INIT_FILENAME
        if candidate.is_file():
            return candidate
    ckpt = checkpoint_path.resolve()
    for base in (ckpt.parent, ckpt.parent.parent):
        candidate = base / FINETUNE_TOKEN_INIT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _token_init_bucket(label: str) -> str:
    if label in INSTANCE_CLASS_INIT_LABELS:
        return "maschine_initialized"
    if label == "stage1_pretrained":
        return "stage1_pretrained"
    if label.startswith("random") or label == "gensim_default":
        return "random_initialized"
    return "unknown"


def _empty_init_coverage_counts() -> dict[str, int]:
    return {
        "total": 0,
        "oov": 0,
        "in_vocab": 0,
        "maschine_initialized": 0,
        "random_initialized": 0,
        "stage1_pretrained": 0,
        "unknown": 0,
    }


def _class_side_from_label(class_label: int) -> Literal["positive", "negative"]:
    return "positive" if int(class_label) == 1 else "negative"


def count_entity_init_coverage(
    tokens: list[str],
    word2idx: dict[str, int],
    token_init_labels: dict[str, str],
    *,
    class_labels: np.ndarray | None = None,
    vocab_key_resolver: Callable[[str, dict[str, int]], str | None] | None = None,
) -> dict[str, int | dict[str, dict[str, int]]]:
    """
    Classify labeled entities by checkpoint vocab presence and finetune init label.

    When *class_labels* is provided (binary 0/1), also count per-class breakdown
    for positive (1) and negative (0) entities.

    *vocab_key_resolver* maps a labeled entity token to the matching checkpoint vocab
    key; when omitted, only exact ``word2idx`` keys are matched.
    """
    resolve = vocab_key_resolver
    if resolve is None:

        def resolve(token: str, w2i: dict[str, int]) -> str | None:
            return token if token in w2i else None

    counts: dict[str, int | dict[str, dict[str, int]]] = _empty_init_coverage_counts()
    counts["total"] = len(tokens)
    by_class: dict[str, dict[str, int]] | None = None
    if class_labels is not None:
        if len(class_labels) != len(tokens):
            raise ValueError("class_labels length must match tokens length")
        by_class = {
            "positive": _empty_init_coverage_counts(),
            "negative": _empty_init_coverage_counts(),
        }

    for i, token in enumerate(tokens):
        class_side: Literal["positive", "negative"] | None = None
        if by_class is not None:
            class_side = _class_side_from_label(int(class_labels[i]))  # type: ignore[index]
            by_class[class_side]["total"] += 1

        vocab_key = resolve(token, word2idx)
        if vocab_key is None:
            counts["oov"] += 1
            if by_class is not None and class_side is not None:
                by_class[class_side]["oov"] += 1
            continue

        counts["in_vocab"] += 1
        init_label = token_init_labels.get(vocab_key, "unknown")
        bucket = _token_init_bucket(init_label)
        counts[bucket] += 1
        if by_class is not None and class_side is not None:
            by_class[class_side]["in_vocab"] += 1
            by_class[class_side][bucket] += 1

    if by_class is not None:
        counts["by_class"] = by_class
    return counts


def count_vocab_init_coverage(
    word2idx: dict[str, int],
    token_init_labels: dict[str, str],
) -> dict[str, int]:
    """Count checkpoint vocabulary tokens by finetune initialization label."""
    counts = _empty_init_coverage_counts()
    counts["total"] = len(word2idx)
    counts["initialized_instances"] = 0
    counts["initialized_relations"] = 0
    for token in word2idx:
        init_label = token_init_labels.get(token, "unknown")
        bucket = _token_init_bucket(init_label)
        counts[bucket] += 1
        if init_label not in PROTOGRAPH_INIT_LABELS:
            continue
        category = token_vocab_category(token)
        if category == "instance":
            counts["initialized_instances"] += 1
        elif category == "relation":
            counts["initialized_relations"] += 1
    return counts


def _add_initialization_noise(vec: np.ndarray, initialization_noise: float) -> np.ndarray:
    """Return *vec* plus N(0, initialization_noise) when noise std dev is positive."""
    if initialization_noise <= 0:
        return vec
    return vec + np.random.normal(0.0, initialization_noise, size=vec.shape).astype(np.float32)


def apply_maschine_initialization(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    ontology_nt: Path,
    *,
    strategy: str = "most_specific",
    ancestor_decay: float = 0.5,
    init_relations: bool = True,
    initialization_noise: float = 0.0,
    new_token_init: dict[str, str] | None = None,
    instance_types_path: Path | None = None,
    subject_filter: set[str] | None = None,
) -> tuple[int, int, int, AllInitFallbackStats | None]:
    """
    After ``build_vocab`` on instance walks (finetune vocabulary only), overwrite new entity
    rows using pretrained class vectors. Copies ``syn1neg`` rows to match ``wv.vectors``.

    ``strategy`` selects how class vectors are combined (see ``init_vector_from_class_roots``).
    Accepts user-facing names (``average_hier``, ``weight_hierarchy``, ``all_init``) or legacy internal names.

    When ``init_relations`` is False, ``P_*`` relation rows are left at Gensim default init.

    When ``initialization_noise`` > 0, Gaussian noise with that standard deviation is added
    to instance entity rows initialized from class vectors and to relation rows copied from
    stage-1 pretrain.

    If ``new_token_init`` is provided, it is filled for each vocabulary token *not* in
    ``stage1_vectors`` with one of: ``maschine_class_mean``, ``maschine_ancestor_mean``,
    ``maschine_ancestor_weighted``, ``maschine_relation_copy``, ``random``.

    Returns:
        (n_entity_initialized, n_relation_copied, n_left_random, all_init_fallback_stats)
    """
    strategy = normalize_init_strategy(strategy)

    parents, entity_types = load_maschine_entity_mapping(
        ontology_nt,
        instance_types_path=instance_types_path,
        subject_filter=subject_filter,
    )
    ent_to_classes = build_entity_to_class_tokens(parents, entity_types)

    init_labels = {
        "most_specific": "maschine_class_mean",
        "all_init": "maschine_class_mean",
        "ancestor_mean": "maschine_ancestor_mean",
        "ancestor_weighted": "maschine_ancestor_weighted",
    }
    init_label = init_labels[strategy]

    wv = model.wv
    n_entity = 0
    n_rel = 0
    n_random = 0
    all_init_stats = AllInitFallbackStats() if strategy == "all_init" else None

    for token in wv.index_to_key:
        if token in stage1_vectors:
            continue

        inner = token[1:-1] if len(token) >= 2 and token[0] == "<" else token
        class_tokens: list[str] | None = ent_to_classes.get(inner)

        if class_tokens is None and token_is_instance_entity(token):
            c = _class_from_extra_name(inner)
            if c is not None:
                mst = _most_specific_types({c}, parents)
                class_tokens = [nt_term(x) for x in mst]

        roots_inner: list[str] = []
        if class_tokens:
            roots_inner = [t[1:-1] for t in class_tokens if len(t) >= 2 and t[0] == "<" and t[-1] == ">"]

        init_vec: np.ndarray | None = None
        init_hops = 0
        if roots_inner:
            if strategy == "all_init":
                init_vec, init_hops = init_vector_from_class_roots_with_hops(
                    roots_inner,
                    parents,
                    stage1_vectors,
                    strategy=strategy,
                    ancestor_decay=ancestor_decay,
                )
            else:
                init_vec = init_vector_from_class_roots(
                    roots_inner,
                    parents,
                    stage1_vectors,
                    strategy=strategy,
                    ancestor_decay=ancestor_decay,
                )

        if init_vec is not None:
            init_vec = _add_initialization_noise(init_vec, initialization_noise)
            idx = wv.key_to_index[token]
            wv.vectors[idx] = init_vec
            if model.syn1neg is not None:
                model.syn1neg[idx] = init_vec.copy()
            n_entity += 1
            if all_init_stats is not None:
                all_init_stats.record(init_hops)
            if new_token_init is not None:
                new_token_init[token] = init_label
            continue

        if init_relations and inner.startswith("P_"):
            vec = _lookup_class_vector(inner, stage1_vectors)
            if vec is not None:
                vec = _add_initialization_noise(vec.copy(), initialization_noise)
                idx = wv.key_to_index[token]
                wv.vectors[idx] = vec
                if model.syn1neg is not None:
                    model.syn1neg[idx] = vec.copy()
                n_rel += 1
                if new_token_init is not None:
                    new_token_init[token] = "maschine_relation_copy"
                continue

        n_random += 1
        if new_token_init is not None:
            category = token_vocab_category(token)
            if category == "relation":
                new_token_init[token] = "random_relation"
            elif category == "instance":
                if roots_inner:
                    new_token_init[token] = "random_no_pretrained_class"
                else:
                    new_token_init[token] = "random_no_class_mapping"
            elif category == "class":
                new_token_init[token] = "random_class_or_instance"
            else:
                new_token_init[token] = "random"

    return n_entity, n_rel, n_random, all_init_stats


def snapshot_class_initialized_entity_vectors(
    model: Word2Vec,
    new_token_init: dict[str, str],
) -> dict[str, np.ndarray]:
    """Snapshot ``wv.vectors`` for instance entities initialized from class vectors."""
    wv = model.wv
    out: dict[str, np.ndarray] = {}
    for token, label in new_token_init.items():
        if label not in INSTANCE_CLASS_INIT_LABELS:
            continue
        idx = wv.key_to_index[token]
        out[token] = np.asarray(wv.vectors[idx], dtype=np.float32).copy()
    return out
