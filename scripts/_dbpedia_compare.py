"""
Shared pipeline for notebooks/dbpedia_compare.ipynb.

Ports the synthetic concept-bound-init comparison (scripts/_synthetic_compare.py,
notebooks/synthetic_compare.ipynb) to the real DBpedia graph. The DBpedia data is
much noisier than the DLCC synthetic graphs, so a preprocessing layer cleans the
schema and instance typing before anything else:

Schema (dbpedia_graph/ontology.nt):
  * A protograph node must be a *declared* class: ``<X> rdf:type owl:Class``.
    This one rule removes datatype ranges (xsd:string, dbpedia datatypes,
    rdf:langString), external classes never declared here (skos:Concept,
    schema.org classes, owl:Thing) and malformed mapping-wiki IRIs
    (``dbo:Organisation,_Person``).
  * rdfs:domain / rdfs:range axioms are kept only when they point to declared
    classes; rdfs:subClassOf edges only when both ends are declared.

Instance graph (dbpedia_graph/graph.nt = mapping-based objects + specific
instance types, 29M IRI triples):
  * entity typing uses only ``rdf:type`` rows whose object is a declared dbo
    class (drops owl:Thing, bibo:*, schema.org noise),
  * ``rdf:type`` is never treated as a relation in the concept-bound features
    (the alpha term already covers own-class membership),
  * class ancestors are materialized through the cleaned subClassOf hierarchy,
    so leaf-typed entities still project onto the P1 domain/range classes.

Protographs: P1 (domain x range per property), P2 (P1 + direct-subclass
expansion), P3 (P1 + the full subClassOf hierarchy as bidirectional edges, so
*every* declared class has a pretrained code).

Variants (same protocol as synthetic_compare):
  * ``vanilla``      — from-scratch skip-gram on instance walks, LR 0.025;
  * ``p{1,2,3}_classic`` — protograph pretrain -> own-class-mean init
    (``all_init`` ancestor fallback), normalized to TARGET_NORM, protected
    finetune LR;
  * ``p{1,2,3}_bound``   — protograph pretrain -> concept-bound init
    (Hadamard-bound relation x neighbour-class concepts, accumulated counts,
    one global rescale), same protected finetune recipe.

Evaluation: the DLCC DBpedia gold standard under ``v1/dbpedia`` (tc01-tc12 x
{books, cities, movies, people, species}), LogReg test accuracy after init and
after every finetune epoch.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _evaluate import load_labeled_txt  # noqa: E402
from _protograph_gen import build_p1, build_p2, iter_nt_iris, triples_to_nt_lines  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    HAS_SUBCLASS_TOKEN,
    SUBCLASS_TOKEN,
    build_p3,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
)
from _walks import run_jrdf2vec_duplicate_free  # noqa: E402

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
RDFS_DOMAIN = f"{RDFS}domain"
RDFS_RANGE = f"{RDFS}range"
RDFS_SUBCLASS = f"{RDFS}subClassOf"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
DBO = "http://dbpedia.org/ontology/"

PROTOGRAPH_KINDS = ("p1", "p2", "p3")
VARIANTS = (
    "vanilla",
    "p1_classic", "p2_classic", "p3_classic",
    "p1_bound", "p2_bound", "p3_bound",
)


# ---------------------------------------------------------------------------
# Schema preprocessing
# ---------------------------------------------------------------------------

@dataclass
class DBpediaSchema:
    """Cleaned DBpedia schema: declared classes only, filtered axioms."""

    declared: set[str]
    domains: dict[str, set[str]]      # property -> declared domain classes
    ranges: dict[str, set[str]]       # property -> declared range classes
    parents: dict[str, set[str]]      # child -> direct declared parents
    children: dict[str, set[str]]     # parent -> direct declared children
    stats: dict = field(default_factory=dict)


def load_schema(ontology_nt: Path) -> DBpediaSchema:
    """One pass over ontology.nt; keep only axioms between declared owl:Class IRIs."""
    declared: set[str] = set()
    raw_domains: dict[str, set[str]] = defaultdict(set)
    raw_ranges: dict[str, set[str]] = defaultdict(set)
    raw_sub: list[tuple[str, str]] = []
    n_triples = 0
    for s, p, o in iter_nt_iris(ontology_nt):
        n_triples += 1
        if p == RDF_TYPE and o == OWL_CLASS:
            declared.add(s)
        elif p == RDFS_DOMAIN:
            raw_domains[s].add(o)
        elif p == RDFS_RANGE:
            raw_ranges[s].add(o)
        elif p == RDFS_SUBCLASS:
            raw_sub.append((s, o))

    domains = {
        r: {c for c in cs if c in declared}
        for r, cs in raw_domains.items()
    }
    ranges = {
        r: {c for c in cs if c in declared}
        for r, cs in raw_ranges.items()
    }
    domains = {r: cs for r, cs in domains.items() if cs}
    ranges = {r: cs for r, cs in ranges.items() if cs}

    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    n_sub_kept = 0
    for c, par in raw_sub:
        if c in declared and par in declared:
            parents[c].add(par)
            children[par].add(c)
            n_sub_kept += 1

    rels_both = set(domains) & set(ranges)
    stats = {
        "ontology_iri_triples": n_triples,
        "declared_classes": len(declared),
        "subclass_edges_raw": len(raw_sub),
        "subclass_edges_kept": n_sub_kept,
        "props_with_domain_raw": len(raw_domains),
        "props_with_range_raw": len(raw_ranges),
        "props_with_domain_and_range_raw": len(set(raw_domains) & set(raw_ranges)),
        "props_with_declared_domain_and_range": len(rels_both),
    }
    return DBpediaSchema(
        declared=declared,
        domains=domains,
        ranges=ranges,
        parents=dict(parents),
        children=dict(children),
        stats=stats,
    )


def write_protographs(schema: DBpediaSchema, out_dir: Path) -> dict[str, Path]:
    """Build P1/P2/P3 from the cleaned schema, write as N-Triples; returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = build_p1(schema.domains, schema.ranges)
    graphs = {
        "p1": p1,
        "p2": build_p2(p1, schema.children),
        "p3": build_p3(p1, schema.children),
    }
    paths: dict[str, Path] = {}
    for kind, triples in graphs.items():
        path = out_dir / f"protograph_{kind}.nt"
        if not path.is_file():
            path.write_text("".join(triples_to_nt_lines(triples)), encoding="utf-8")
        paths[kind] = path
    return paths


def protograph_class_coverage(schema: DBpediaSchema) -> dict[str, int]:
    """How many declared classes appear as nodes in each protograph."""
    p1 = build_p1(schema.domains, schema.ranges)
    p2 = build_p2(p1, schema.children)
    p3 = build_p3(p1, schema.children)

    def classes_of(triples):
        out = set()
        for s, p, o in triples:
            if s in schema.declared:
                out.add(s)
            if o in schema.declared:
                out.add(o)
        return out

    return {
        "declared": len(schema.declared),
        "p1_triples": len(p1), "p1_classes": len(classes_of(p1)),
        "p2_triples": len(p2), "p2_classes": len(classes_of(p2)),
        "p3_triples": len(p3), "p3_classes": len(classes_of(p3)),
    }


# ---------------------------------------------------------------------------
# Instance typing (cleaned, ancestor-materialized)
# ---------------------------------------------------------------------------

@dataclass
class InstanceTypes:
    """entity -> asserted declared-class ids; class id helpers."""

    class_list: list[str]                 # id -> class IRI
    class_index: dict[str, int]           # class IRI -> id
    ent_types: dict[str, tuple[int, ...]]  # entity IRI -> asserted class ids
    ancestors: list[tuple[int, ...]]      # class id -> materialized ancestor ids (incl self)
    parent_ids: list[tuple[int, ...]] = field(default_factory=list)  # class id -> direct parent ids
    stats: dict = field(default_factory=dict)

    def materialized(self, cids: tuple[int, ...]) -> tuple[int, ...]:
        if len(cids) == 1:
            return self.ancestors[cids[0]]
        out: set[int] = set()
        for c in cids:
            out.update(self.ancestors[c])
        return tuple(sorted(out))

    def most_specific(self, cids: tuple[int, ...]) -> list[int]:
        """Asserted classes that are not a (transitive) superclass of another asserted class."""
        leaves = [
            c for c in cids
            if not any(d != c and c in self.ancestors[d] for d in cids)
        ]
        return leaves or list(cids)

    def class_weights(
        self, cids: tuple[int, ...], *, mode: str = "uniform", ancestor_decay: float = 0.5,
    ) -> tuple[tuple[int, float], ...]:
        """(class id, weight) pairs for the concept-bound init, mirroring
        scripts/_synthetic_compare.load_weighted_types:

          * ``uniform``       — every ancestor-materialized class, weight 1.0 (base).
          * ``most_specific`` — only the most-specific asserted class(es), weight 1.0.
          * ``decay``         — every materialized class weighted ``ancestor_decay ** hops``
            from the nearest most-specific class (leaf=1.0, parent=decay, ...).
        """
        if mode == "uniform":
            return tuple((c, 1.0) for c in self.materialized(cids))
        leaves = sorted(set(self.most_specific(cids)))
        if mode == "most_specific":
            return tuple((c, 1.0) for c in leaves)
        if mode == "decay":
            depth: dict[int, int] = {}
            queue = deque((c, 0) for c in leaves)
            while queue:
                c, d = queue.popleft()
                if c in depth:
                    continue
                depth[c] = d
                for par in self.parent_ids[c]:
                    if par not in depth:
                        queue.append((par, d + 1))
            return tuple((c, float(ancestor_decay) ** depth[c]) for c in sorted(depth))
        raise ValueError(f"unknown class weight mode {mode!r}")


def _ancestor_closure(schema: DBpediaSchema, class_list: list[str], class_index: dict[str, int]) -> list[tuple[int, ...]]:
    closure: list[tuple[int, ...]] = []
    for cid, c in enumerate(class_list):
        seen = {c}
        stack = [c]
        while stack:
            for par in schema.parents.get(stack.pop(), ()):
                if par not in seen:
                    seen.add(par)
                    stack.append(par)
        closure.append(tuple(sorted(class_index[x] for x in seen)))
    return closure


def load_instance_types(graph_nt: Path, schema: DBpediaSchema) -> InstanceTypes:
    """
    entity -> asserted declared classes from rdf:type rows of the instance graph.

    Noise filter: only objects that are *declared* dbo classes count (drops
    owl:Thing, external vocabularies, malformed IRIs).
    """
    class_list = sorted(schema.declared)
    class_index = {c: i for i, c in enumerate(class_list)}
    ent_types_raw: dict[str, int | tuple[int, ...]] = {}
    n_type_rows = 0
    n_type_kept = 0
    for s, p, o in iter_nt_iris(graph_nt):
        if p != RDF_TYPE:
            continue
        n_type_rows += 1
        cid = class_index.get(o)
        if cid is None:
            continue
        n_type_kept += 1
        prev = ent_types_raw.get(s)
        if prev is None:
            ent_types_raw[s] = cid
        elif isinstance(prev, int):
            if prev != cid:
                ent_types_raw[s] = (prev, cid)
        elif cid not in prev:
            ent_types_raw[s] = prev + (cid,)

    ent_types = {
        e: (v,) if isinstance(v, int) else v
        for e, v in ent_types_raw.items()
    }
    ancestors = _ancestor_closure(schema, class_list, class_index)
    parent_ids = [
        tuple(sorted(class_index[p] for p in schema.parents.get(c, ()) if p in class_index))
        for c in class_list
    ]
    stats = {
        "type_rows_total": n_type_rows,
        "type_rows_declared_class": n_type_kept,
        "typed_entities": len(ent_types),
        "classes_used": len({c for v in ent_types.values() for c in v}),
    }
    return InstanceTypes(
        class_list=class_list,
        class_index=class_index,
        ent_types=ent_types,
        ancestors=ancestors,
        parent_ids=parent_ids,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Protograph walks + pretraining
# ---------------------------------------------------------------------------

def ensure_walks(
    graph_path: Path,
    out_path: Path,
    *,
    walks_per_entity: int,
    depth: int,
    seed: int = 42,
    threads: int = 8,
) -> Path:
    """Duplicate-free walks with *bare* IRI tokens (matching walks/all_walks.txt)."""
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    run_jrdf2vec_duplicate_free(
        graph_path,
        tmp,
        walks_per_entity=walks_per_entity,
        depth=depth,
        threads=threads,
        seed=seed,
        token_format="bare",
        input_format="nt",
        ensure_triple_coverage=True,
    )
    tmp.rename(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Instance-corpus vocabulary cache (the corpus is ~50 GB; scan it once)
# ---------------------------------------------------------------------------

def corpus_vocab(walks_path: Path, cache_path: Path) -> tuple[dict[str, int], int, int]:
    """(token -> count, n_lines, n_tokens), cached as a pickle next to the corpus."""
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        return payload["freq"], payload["n_lines"], payload["n_tokens"]
    freq: dict[str, int] = {}
    get = freq.get
    n_lines = 0
    n_tokens = 0
    with walks_path.open(encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            n_lines += 1
            n_tokens += len(toks)
            for t in toks:
                freq[t] = get(t, 0) + 1
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"freq": freq, "n_lines": n_lines, "n_tokens": n_tokens}, f, protocol=4)
    tmp.rename(cache_path)
    return freq, n_lines, n_tokens


def build_model_with_vocab(
    freq: dict[str, int],
    n_lines: int,
    n_tokens: int,
    *,
    dim: int,
    alpha: float,
    min_alpha: float,
    seed: int,
    workers: int,
    window: int = 5,
) -> Word2Vec:
    model = new_skipgram_model(
        dim=dim, alpha=alpha, min_alpha=min_alpha, seed=seed, workers=workers, window=window
    )
    model.build_vocab_from_freq(freq)
    model.corpus_count = n_lines
    model.corpus_total_words = n_tokens
    return model


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _class_codes(
    itypes: InstanceTypes,
    stage1: dict[str, np.ndarray],
) -> list[np.ndarray | None]:
    """Unit code per class id (direct stage-1 lookup; None when OOV in pretrain)."""
    out: list[np.ndarray | None] = []
    for c in itypes.class_list:
        v = stage1.get(c)
        out.append(_unit(np.asarray(v, dtype=np.float32)) if v is not None else None)
    return out


def _resolved_class_codes(
    itypes: InstanceTypes,
    schema: DBpediaSchema,
    stage1: dict[str, np.ndarray],
) -> list[np.ndarray | None]:
    """Per class id: stage-1 vector, climbing subClassOf to the nearest coded ancestor
    when the class itself is OOV in pretrain (the ``all_init`` fallback)."""
    out: list[np.ndarray | None] = []
    for c in itypes.class_list:
        vec = None
        q: deque[str] = deque([c])
        seen = {c}
        while q:
            node = q.popleft()
            v = stage1.get(node)
            if v is not None:
                vec = np.asarray(v, dtype=np.float32)
                break
            for par in schema.parents.get(node, ()):
                if par not in seen:
                    seen.add(par)
                    q.append(par)
        out.append(vec)
    return out


def classic_init(
    model: Word2Vec,
    stage1: dict[str, np.ndarray],
    itypes: InstanceTypes,
    schema: DBpediaSchema,
    *,
    target_norm: float,
    bound_vectors: dict[str, np.ndarray] | None = None,
    normalize_class_codes: bool = False,
    class_init_strategy: str = "all_init",
) -> dict[str, int]:
    """
    MASCHInE-style transfer into a fresh instance model (rows go to wv AND syn1neg):

      1. tokens present in stage-1 pretrain (classes, relations) -> copy;
      2. ``bound_vectors`` (when given) for instance tokens -> copy as-is
         (already globally scaled; per-vector renorm would erase counts);
      3. typed instances -> mean of their class codes, renormalized to
         ``target_norm`` (see ``class_init_strategy``);
      4. everything else keeps gensim's random init.

    ``class_init_strategy`` selects which classes seed an instance and whether the
    subClassOf hierarchy is climbed for out-of-pretrain classes:

      * ``"all_init"`` (default) — mean over **all** asserted classes, each resolved
        to the nearest coded ancestor when OOV in pretrain (``_resolved_class_codes``).
      * ``"most_specific"`` — mean over only the **most-specific** asserted class(es),
        using their **direct** stage-1 code with **no** ancestor fallback; an instance
        whose most-specific class is OOV gets a random init. Same raw scale as
        ``all_init`` so ``normalize_class_codes`` / the final renorm behave identically.

    ``normalize_class_codes``: when ``True``, L2-normalize each class code to unit
    length BEFORE averaging (matching the concept-bound class mix, which uses
    ``_class_codes`` -> ``_unit``). Default ``False`` keeps the raw stage-1 norm,
    so classes with larger pretrain norms dominate the mean.
    """
    if class_init_strategy == "all_init":
        codes = _resolved_class_codes(itypes, schema, stage1)
    elif class_init_strategy == "most_specific":
        # leaf class direct code, no subClassOf fallback; raw scale matches all_init
        codes = [
            np.asarray(v, dtype=np.float32) if (v := stage1.get(c)) is not None else None
            for c in itypes.class_list
        ]
    else:
        raise ValueError(
            f"unknown class_init_strategy {class_init_strategy!r}; "
            "expected 'all_init' or 'most_specific'"
        )
    wv = model.wv
    vectors = wv.vectors
    syn1neg = model.syn1neg
    ent_types = itypes.ent_types

    ent_cache: dict[tuple[int, ...], np.ndarray | None] = {}

    def class_mean(cids: tuple[int, ...]) -> np.ndarray | None:
        cached = ent_cache.get(cids)
        if cached is not None or cids in ent_cache:
            return cached
        parts = [codes[c] for c in cids if codes[c] is not None]
        if not parts:
            ent_cache[cids] = None
            return None
        if normalize_class_codes:
            parts = [_unit(p) for p in parts]
        v = np.mean(parts, axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v *= target_norm / n
        ent_cache[cids] = v
        return v

    n_stage1 = n_class = n_bound = n_random = 0
    for token, idx in wv.key_to_index.items():
        if bound_vectors is not None:
            bv = bound_vectors.get(token)
            if bv is not None:
                vectors[idx] = bv
                syn1neg[idx] = bv
                n_bound += 1
                continue
        sv = stage1.get(token)
        if sv is not None:
            vectors[idx] = sv
            syn1neg[idx] = sv
            n_stage1 += 1
            continue
        cids = ent_types.get(token)
        if cids is not None:
            sel = cids if class_init_strategy == "all_init" else tuple(sorted(itypes.most_specific(cids)))
            v = class_mean(sel)
            if v is not None:
                vectors[idx] = v
                syn1neg[idx] = v
                n_class += 1
                continue
        n_random += 1

    return {
        "stage1_copied": n_stage1,
        "class_init": n_class,
        "bound_init": n_bound,
        "random": n_random,
    }


def mixed_init(
    model: Word2Vec,
    stage1: dict[str, np.ndarray],
    itypes: InstanceTypes,
    schema: DBpediaSchema,
    *,
    target_norm: float,
    bound_vectors: dict[str, np.ndarray] | None,
    mix_lambda: float,
) -> dict[str, int]:
    """Damped bound-offset init (class vector + damped bound offset).

    For an instance token that has *both* a class-mean code ``cv`` (renormed to
    ``target_norm``) and a bound vector ``bv``: scale ``bv`` per-row to
    ``target_norm`` (so norm can't dominate the blend) then set
    ``(1-λ)·cv + λ·bv_scaled``. Tokens with only one source use it as-is; schema
    tokens (classes/relations) copy their stage-1 code. λ=0 ⇒ class on the
    overlap, λ=1 ⇒ bound-direction on the overlap.
    """
    resolved = _resolved_class_codes(itypes, schema, stage1)
    wv = model.wv
    vectors = wv.vectors
    syn1neg = model.syn1neg
    ent_types = itypes.ent_types
    lam = float(mix_lambda)
    ent_cache: dict[tuple[int, ...], np.ndarray | None] = {}

    def class_mean(cids: tuple[int, ...]) -> np.ndarray | None:
        cached = ent_cache.get(cids)
        if cached is not None or cids in ent_cache:
            return cached
        parts = [resolved[c] for c in cids if resolved[c] is not None]
        if not parts:
            ent_cache[cids] = None
            return None
        v = np.mean(parts, axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v *= target_norm / n
        ent_cache[cids] = v
        return v

    n_stage1 = n_mix = n_class = n_bound = n_random = 0
    for token, idx in wv.key_to_index.items():
        sv = stage1.get(token)
        if sv is not None:
            vectors[idx] = sv
            syn1neg[idx] = sv
            n_stage1 += 1
            continue
        cids = ent_types.get(token)
        cv = class_mean(cids) if cids is not None else None
        bv = bound_vectors.get(token) if bound_vectors is not None else None
        if cv is not None and bv is not None:
            bn = float(np.linalg.norm(bv))
            bv_scaled = bv / bn * target_norm if bn > 0 else bv
            vec = ((1.0 - lam) * cv + lam * bv_scaled).astype(np.float32)
            vectors[idx] = vec
            syn1neg[idx] = vec
            n_mix += 1
            continue
        if cv is not None:
            vectors[idx] = cv
            syn1neg[idx] = cv
            n_class += 1
            continue
        if bv is not None:
            vectors[idx] = bv
            syn1neg[idx] = bv
            n_bound += 1
            continue
        n_random += 1

    return {
        "stage1_copied": n_stage1,
        "mixed_init": n_mix,
        "class_only": n_class,
        "bound_only": n_bound,
        "random": n_random,
        "mix_lambda": lam,
    }


DIRECTION_TAGS = ("shared", "rolled", "negated", "keyed")


def tag_code(rc: np.ndarray, tag: str) -> np.ndarray:
    """
    Code added for an INCOMING edge of relation r (outgoing always uses ``rc``).

    ``rolled`` is the production choice: a circular shift of a high-dimensional
    unit vector is nearly orthogonal to it, so incoming counts live in their own
    quasi-independent channel that a linear probe can read out separately. The
    ablation in notebooks/direction_aware_cardinality.ipynb shows this
    direction tag is worth ~7 accuracy points end-to-end on the synthetic
    cardinality TCs (up to ~14 on the inverse-direction ones) over a
    direction-blind ``shared`` count term, and that any fixed decorrelating
    norm-preserving transform (e.g. ``keyed``) works equally well — ``roll`` is
    simply the cheapest one.
    """
    if tag == "shared":
        return rc
    if tag == "rolled":
        return np.roll(rc, 1)
    if tag == "negated":
        return -rc
    if tag == "keyed":
        key = np.random.default_rng(7).choice([-1.0, 1.0], size=rc.shape[0]).astype(np.float32)
        return rc * key
    raise ValueError(f"unknown direction tag {tag!r}; expected one of {DIRECTION_TAGS}")


def concept_bound_vectors(
    graph_nt: Path,
    itypes: InstanceTypes,
    stage1: dict[str, np.ndarray],
    vocab: set[str] | dict[str, int],
    *,
    direction_tag: str = "rolled",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    target_norm: float = 1.0,
    progress_every: int = 5_000_000,
    alpha_weights: dict[int, float] | None = None,
    compress: str | None = None,
    do_rescale: bool = True,
    class_weight_mode: str = "uniform",
    ancestor_decay: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Concept-bound instance initialization on DBpedia with **direction-aware
    cardinality** (see _synthetic_compare for the rationale and
    notebooks/direction_aware_cardinality.ipynb for the direction-tag ablation):

        init(e) =  alpha * unit(mean code of cls*(e))
                 + beta  * sum over (e, r, o), c in cls*(o):  unit(code(r) o code(c))
                 + gamma * sum over (s, r, e), c in cls*(s):  unit(tag(code(r)) o code(c))
                 + delta * [ sum over (e, r, .): code(r)  +  sum over (., r, e): tag(code(r)) ]

    with cls* the ancestor-materialized declared-class set, ``o`` the Hadamard
    product, and ``tag`` the *direction tag* applied to incoming-edge relation
    codes in both the gamma binding term and the delta count term (default
    ``rolled`` = ``np.roll(code(r), 1)``, a cheap decorrelating tag that puts
    incoming counts into a nearly orthogonal channel). Repeats are accumulated
    (cardinality signal) and only one global rescale (mean norm -> target_norm)
    is applied at the end.

    DBpedia-specific preprocessing baked in: rdf:type edges are skipped (the
    alpha term covers class membership), relations without a pretrained code
    (no clean domain/range axioms) are skipped, and only entities present in
    the walk-corpus ``vocab`` are accumulated.

    Optional experiment hooks (all default to the production behaviour):
      * ``alpha_weights``: per-class-id weight for the alpha class-mean term
        (e.g. IDF specificity weights); ``None`` = uniform mean as before.
      * ``compress``: per-row monotone norm compression applied *before* the
        final global rescale — ``"sqrt"`` (norm -> sqrt(norm)) or ``"log1p"``
        (norm -> log1p(norm)). Tames the hub-norm tail at the source so the
        post-hoc per-row cap is no longer needed; ``None`` = no compression.
      * ``do_rescale``: when ``False``, skip the single global mean->target
        rescale and return the raw accumulation (for caching/recomposition).
      * ``class_weight_mode`` / ``ancestor_decay``: how the class hierarchy enters
        every class-bearing term (alpha mean + beta/gamma neighbour bindings), via
        :meth:`InstanceTypes.class_weights` — ``"uniform"`` (default, base behavior),
        ``"most_specific"`` (leaf only), or ``"decay"`` (ancestors weighted
        ``ancestor_decay ** hops`` from the leaf). Ported from the synthetic
        ``class_weight_mode`` ablation (notebooks/clean_bound.ipynb).
    """
    cls_codes = _class_codes(itypes, stage1)
    ent_types = itypes.ent_types

    wmat_cache: dict[tuple[int, ...], tuple[tuple[int, float], ...]] = {}

    def weighted_mat(cids: tuple[int, ...]) -> tuple[tuple[int, float], ...]:
        cached = wmat_cache.get(cids)
        if cached is None:
            cached = itypes.class_weights(
                cids, mode=class_weight_mode, ancestor_decay=ancestor_decay
            )
            wmat_cache[cids] = cached
        return cached

    rel_codes: dict[str, np.ndarray | None] = {}

    def rel_code(r: str) -> np.ndarray | None:
        if r in rel_codes:
            return rel_codes[r]
        v = stage1.get(r)
        u = _unit(np.asarray(v, dtype=np.float32)) if v is not None else None
        rel_codes[r] = u
        return u

    pair_cache: dict[tuple[str, int, bool], np.ndarray | None] = {}

    def pair_code(r: str, rc: np.ndarray, cid: int, inverse: bool) -> np.ndarray | None:
        key = (r, cid, inverse)
        cached = pair_cache.get(key)
        if cached is not None or key in pair_cache:
            return cached
        cc = cls_codes[cid]
        if cc is None:
            out = None
        else:
            base = tag_code(rc, direction_tag) if inverse else rc
            out = _unit(base * cc)
        pair_cache[key] = out
        return out

    acc: dict[str, np.ndarray] = {}
    dim = len(next(iter(stage1.values())))

    def bump(ent: str, vec: np.ndarray, w: float) -> None:
        row = acc.get(ent)
        if row is None:
            row = np.zeros(dim, dtype=np.float32)
            acc[ent] = row
        row += w * vec

    # alpha term: own (hierarchy-weighted) classes
    if alpha != 0.0:
        own_cache: dict[tuple[int, ...], np.ndarray | None] = {}
        for ent, cids in ent_types.items():
            if ent not in vocab:
                continue
            mix = own_cache.get(cids)
            if mix is None and cids not in own_cache:
                pairs = [(c, hw) for c, hw in weighted_mat(cids) if cls_codes[c] is not None]
                if alpha_weights is not None:
                    pairs = [(c, hw * float(alpha_weights.get(c, 1.0))) for c, hw in pairs]
                sw = float(sum(hw for _c, hw in pairs))
                if not pairs or sw <= 0:
                    mix = None
                else:
                    stacked = np.stack([cls_codes[c] for c, _hw in pairs]).astype(np.float32)
                    wv = np.asarray([hw for _c, hw in pairs], dtype=np.float32)
                    mix = _unit((stacked * wv[:, None]).sum(axis=0) / sw)
                own_cache[cids] = mix
            if mix is not None:
                bump(ent, mix, alpha)

    # edge terms
    n_edges = 0
    t0 = time.time()
    for s, r, o in iter_nt_iris(graph_nt):
        if r == RDF_TYPE:
            continue
        rc = rel_code(r)
        if rc is None:
            continue
        n_edges += 1
        if progress_every and n_edges % progress_every == 0:
            print(f"    bound: {n_edges/1e6:.0f}M edges in {time.time()-t0:.0f}s", flush=True)
        s_in = s in vocab
        o_in = o in vocab
        if not (s_in or o_in):
            continue
        rc_inv = tag_code(rc, direction_tag)
        if delta != 0.0:
            if s_in:
                bump(s, rc, delta)
            if o_in:
                bump(o, rc_inv, delta)
        if beta != 0.0 and s_in:
            cids = ent_types.get(o)
            if cids is not None:
                for c, hw in weighted_mat(cids):
                    pc = pair_code(r, rc, c, inverse=False)
                    if pc is not None:
                        bump(s, pc, beta * hw)
        if gamma != 0.0 and o_in:
            cids = ent_types.get(s)
            if cids is not None:
                for c, hw in weighted_mat(cids):
                    pc = pair_code(r, rc, c, inverse=True)
                    if pc is not None:
                        bump(o, pc, gamma * hw)

    if compress is not None:
        for vec in acc.values():
            n = float(np.linalg.norm(vec))
            if n > 0:
                if compress == "sqrt":
                    vec *= np.float32(1.0 / np.sqrt(n))
                elif compress == "log1p":
                    vec *= np.float32(np.log1p(n) / n)
                else:
                    raise ValueError(f"unknown compress {compress!r}; expected 'sqrt' or 'log1p'")

    norms = np.array([float(np.linalg.norm(v)) for v in acc.values()], dtype=np.float64)
    mean_norm = float(norms.mean()) if len(norms) else 1.0
    scale = np.float32(target_norm / mean_norm) if (do_rescale and mean_norm > 0) else np.float32(1.0)
    out = {ent: vec * scale for ent, vec in acc.items() if float(np.linalg.norm(vec)) > 0}
    print(
        f"    bound: {len(out)} entity vectors from {n_edges/1e6:.1f}M relation edges "
        f"({time.time()-t0:.0f}s, mean_norm={mean_norm:.3f}, compress={compress}, "
        f"rescale={do_rescale})",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# DLCC DBpedia evaluation (all tc x domain splits, per-epoch)
# ---------------------------------------------------------------------------

@dataclass
class EvalSplit:
    name: str
    tc: str
    domain: str
    train_tokens: list[str]
    y_train: np.ndarray
    test_tokens: list[str]
    y_test: np.ndarray


def load_eval_splits(dbpedia_root: Path, *, k: int = 5000, include_hard: bool = True) -> list[EvalSplit]:
    splits: list[EvalSplit] = []
    for tc_dir in sorted(dbpedia_root.glob("tc*")):
        if not tc_dir.is_dir():
            continue
        for domain_dir in sorted(p for p in tc_dir.iterdir() if p.is_dir()):
            k_dir = domain_dir / str(int(k))
            candidates = [("", k_dir / "train_test")]
            if include_hard:
                candidates.append(("_hard", k_dir / "train_test_hard"))
            for suffix, d in candidates:
                train, test = d / "train.txt", d / "test.txt"
                if not (train.is_file() and test.is_file()):
                    continue
                tr_tok, tr_y = load_labeled_txt(train)
                te_tok, te_y = load_labeled_txt(test)
                splits.append(
                    EvalSplit(
                        name=f"{tc_dir.name}/{domain_dir.name}{suffix}",
                        tc=tc_dir.name,
                        domain=f"{domain_dir.name}{suffix}",
                        train_tokens=tr_tok,
                        y_train=tr_y,
                        test_tokens=te_tok,
                        y_test=te_y,
                    )
                )
    if not splits:
        raise SystemExit(f"No eval splits found under {dbpedia_root} (k={k})")
    return splits


def _lookup(tokens: list[str], emb: np.ndarray, w2i: dict[str, int]) -> tuple[np.ndarray, int]:
    out = np.zeros((len(tokens), emb.shape[1]), dtype=np.float32)
    oov = 0
    for i, t in enumerate(tokens):
        j = w2i.get(t)
        if j is None:
            oov += 1
        else:
            out[i] = emb[j]
    return out, oov


def _fit_eval_one(x_tr, y_tr, x_te, y_te, seed: int, max_iter: int) -> float:
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(x_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(x_te)))


def make_eval_fn(
    splits: list[EvalSplit],
    *,
    seed: int = 42,
    max_iter: int = 1000,
    n_jobs: int = 8,
):
    """LogReg test accuracy per split on the current model state (parallel fits)."""
    from joblib import Parallel, delayed

    def evaluate(model: Word2Vec) -> dict[str, float]:
        emb = np.asarray(model.wv.vectors)
        w2i = model.wv.key_to_index
        tasks = []
        for sp in splits:
            x_tr, _ = _lookup(sp.train_tokens, emb, w2i)
            x_te, _ = _lookup(sp.test_tokens, emb, w2i)
            tasks.append((x_tr, sp.y_train, x_te, sp.y_test))
        accs = Parallel(n_jobs=n_jobs)(
            delayed(_fit_eval_one)(x_tr, y_tr, x_te, y_te, seed, max_iter)
            for x_tr, y_tr, x_te, y_te in tasks
        )
        return {sp.name: acc for sp, acc in zip(splits, accs)}

    return evaluate


def oov_report(splits: list[EvalSplit], model: Word2Vec) -> dict[str, dict[str, int]]:
    w2i = model.wv.key_to_index
    out: dict[str, dict[str, int]] = {}
    for sp in splits:
        out[sp.name] = {
            "train_oov": sum(1 for t in sp.train_tokens if t not in w2i),
            "test_oov": sum(1 for t in sp.test_tokens if t not in w2i),
            "n_train": len(sp.train_tokens),
            "n_test": len(sp.test_tokens),
        }
    return out


class _EpochEval(CallbackAny2Vec):
    def __init__(self, eval_fn, rows: list[dict[str, float]],
                 losses: list[float] | None = None, label: str = ""):
        self.eval_fn = eval_fn
        self.rows = rows
        self.losses = losses          # per-epoch SGNS loss (None = not tracked)
        self.label = label
        self.epoch = 0
        self._prev_cum = 0.0
        self._t0 = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        t = time.time()
        loss_str = ""
        if self.losses is not None:
            cum = float(model.get_latest_training_loss())  # cumulative across epochs
            self.losses.append(cum - self._prev_cum)
            self._prev_cum = cum
            loss_str = f" loss {self.losses[-1]:.0f}"
        accs = self.eval_fn(model)
        self.rows.append(accs)
        mean = float(np.mean(list(accs.values())))
        print(
            f"    [{time.time()-self._t0:7.0f}s] {self.label} epoch {self.epoch}: "
            f"mean acc {mean:.4f}{loss_str} (eval {time.time()-t:.0f}s)",
            flush=True,
        )


def train_with_eval(
    model: Word2Vec,
    walks_path: Path,
    *,
    epochs: int,
    alpha: float,
    min_alpha: float,
    total_examples: int,
    eval_fn,
    label: str = "",
    save_init: Path | None = None,
) -> tuple[list[dict[str, float]], list[float]]:
    """Train; return ([accs@init, accs@epoch1, ...], [loss@epoch1, ...]).

    The accuracy list has one dict per checkpoint (init + each epoch); the loss
    list has one per-epoch SGNS loss (cumulative deltas). When ``save_init`` is
    given, the post-init / pre-finetune embeddings are saved there (epoch 0).
    """
    rows: list[dict[str, float]] = []
    losses: list[float] = []
    t = time.time()
    accs0 = eval_fn(model)
    rows.append(accs0)
    print(
        f"    {label} epoch 0 (init): mean acc {float(np.mean(list(accs0.values()))):.4f} "
        f"(eval {time.time()-t:.0f}s)",
        flush=True,
    )
    if save_init is not None:
        model.wv.save(str(save_init))
    model.train(
        corpus_iterable=LineSentence(str(walks_path)),
        total_examples=total_examples,
        epochs=epochs,
        start_alpha=alpha,
        end_alpha=min_alpha,
        compute_loss=True,
        callbacks=[_EpochEval(eval_fn, rows, losses, label)],
    )
    return rows, losses


# ---------------------------------------------------------------------------
# One full variant run
# ---------------------------------------------------------------------------

def _result_row(name: str, rows: list[dict[str, float]], extra: dict) -> dict:
    split_names = list(rows[0].keys())
    split_accs = {s: [r[s] for r in rows] for s in split_names}
    mean_accs = [float(np.mean(list(r.values()))) for r in rows]
    return {
        "variant": name,
        "split_accs": split_accs,
        "mean_accs": mean_accs,
        "init_mean_acc": mean_accs[0],
        "final_mean_acc": mean_accs[-1],
        **extra,
    }


def run_vanilla(
    instance_walks: Path,
    freq: dict[str, int],
    n_lines: int,
    n_tokens: int,
    splits: list[EvalSplit],
    *,
    dim: int = 200,
    epochs: int = 5,
    alpha: float = 0.025,
    min_alpha: float = 0.0001,
    seed: int = 42,
    workers: int = 20,
    save_kv: Path | None = None,
) -> dict:
    t0 = time.time()
    model = build_model_with_vocab(
        freq, n_lines, n_tokens, dim=dim, alpha=alpha, min_alpha=min_alpha,
        seed=seed, workers=workers,
    )
    eval_fn = make_eval_fn(splits, seed=seed)
    init_kv = save_kv.with_name(save_kv.stem + ".init.kv") if save_kv is not None else None
    rows, losses = train_with_eval(
        model, instance_walks, epochs=epochs, alpha=alpha, min_alpha=min_alpha,
        total_examples=n_lines, eval_fn=eval_fn, label="vanilla", save_init=init_kv,
    )
    if save_kv is not None:
        model.wv.save(str(save_kv))
    res = _result_row(
        "vanilla", rows,
        dict(init_stats=None, losses=losses, target_norm=None, seconds=round(time.time() - t0, 1),
             vocab_size=len(model.wv), oov=oov_report(splits, model)),
    )
    del model
    return res


def run_protograph_variant(
    name: str,
    proto_walks: Path,
    instance_walks: Path,
    graph_nt: Path,
    freq: dict[str, int],
    n_lines: int,
    n_tokens: int,
    splits: list[EvalSplit],
    itypes: InstanceTypes,
    schema: DBpediaSchema,
    *,
    bound: bool,
    dim: int = 200,
    epochs: int = 5,
    pretrain_epochs: int = 5,
    finetune_alpha: float = 0.0025,
    min_alpha: float = 0.0001,
    target_norm: float = 8.0,
    direction_tag: str = "rolled",
    bound_alpha: float = 1.0,
    bound_beta: float = 1.0,
    bound_gamma: float = 1.0,
    bound_delta: float = 1.0,
    bound_class_weight_mode: str = "uniform",
    bound_ancestor_decay: float = 0.5,
    classic_strategy: str = "all_init",
    seed: int = 42,
    workers: int = 20,
    save_kv: Path | None = None,
) -> dict:
    """Pretrain on protograph walks -> normalized codes -> classic or bound init -> finetune.

    ``classic_strategy`` (``all_init`` | ``most_specific``) controls the class-mean
    instance init for classic variants; bound variants always use the ``all_init``
    fallback for tokens not covered by the concept-bound construction.
    """
    t0 = time.time()
    pre = pretrain_protograph(proto_walks, dim=dim, epochs=pretrain_epochs, seed=seed)
    stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=target_norm)
    del pre

    model = build_model_with_vocab(
        freq, n_lines, n_tokens, dim=dim, alpha=finetune_alpha, min_alpha=min_alpha,
        seed=seed, workers=workers,
    )

    bound_vectors = None
    if bound:
        bound_vectors = concept_bound_vectors(
            graph_nt, itypes, stage1, model.wv.key_to_index,
            direction_tag=direction_tag,
            alpha=bound_alpha, beta=bound_beta, gamma=bound_gamma, delta=bound_delta,
            target_norm=used_norm,
            class_weight_mode=bound_class_weight_mode,
            ancestor_decay=bound_ancestor_decay,
        )

    init_stats = classic_init(
        model, stage1, itypes, schema,
        target_norm=used_norm, bound_vectors=bound_vectors,
        class_init_strategy=("all_init" if bound else classic_strategy),
    )
    del bound_vectors
    print(f"    {name} init: {init_stats}", flush=True)

    eval_fn = make_eval_fn(splits, seed=seed)
    init_kv = save_kv.with_name(save_kv.stem + ".init.kv") if save_kv is not None else None
    rows, losses = train_with_eval(
        model, instance_walks, epochs=epochs, alpha=finetune_alpha, min_alpha=min_alpha,
        total_examples=n_lines, eval_fn=eval_fn, label=name, save_init=init_kv,
    )
    if save_kv is not None:
        model.wv.save(str(save_kv))
    res = _result_row(
        name, rows,
        dict(init_stats=init_stats, losses=losses, target_norm=used_norm,
             seconds=round(time.time() - t0, 1), vocab_size=len(model.wv),
             oov=oov_report(splits, model)),
    )
    del model
    return res


def run_protograph_mixed(
    name: str,
    proto_walks: Path,
    instance_walks: Path,
    graph_nt: Path,
    freq: dict[str, int],
    n_lines: int,
    n_tokens: int,
    splits: list[EvalSplit],
    itypes: InstanceTypes,
    schema: DBpediaSchema,
    *,
    mix_lambda: float,
    dim: int = 200,
    epochs: int = 5,
    pretrain_epochs: int = 5,
    finetune_alpha: float = 0.0025,
    min_alpha: float = 0.0001,
    target_norm: float = 8.0,
    direction_tag: str = "rolled",
    bound_alpha: float = 1.0,
    bound_beta: float = 1.0,
    bound_gamma: float = 1.0,
    bound_delta: float = 1.0,
    seed: int = 42,
    workers: int = 20,
    save_kv: Path | None = None,
) -> dict:
    """Protograph pretrain -> damped bound-offset init (lambda) -> finetune.

    Identical recipe to ``run_protograph_variant`` except the instance init is
    the class+bound blend from ``mixed_init`` instead of class-only / bound-only.
    """
    t0 = time.time()
    pre = pretrain_protograph(proto_walks, dim=dim, epochs=pretrain_epochs, seed=seed)
    stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=target_norm)
    del pre

    model = build_model_with_vocab(
        freq, n_lines, n_tokens, dim=dim, alpha=finetune_alpha, min_alpha=min_alpha,
        seed=seed, workers=workers,
    )

    bound_vectors = concept_bound_vectors(
        graph_nt, itypes, stage1, model.wv.key_to_index,
        direction_tag=direction_tag,
        alpha=bound_alpha, beta=bound_beta, gamma=bound_gamma, delta=bound_delta,
        target_norm=used_norm,
    )

    init_stats = mixed_init(
        model, stage1, itypes, schema,
        target_norm=used_norm, bound_vectors=bound_vectors, mix_lambda=mix_lambda,
    )
    del bound_vectors
    print(f"    {name} init: {init_stats}", flush=True)

    eval_fn = make_eval_fn(splits, seed=seed)
    init_kv = save_kv.with_name(save_kv.stem + ".init.kv") if save_kv is not None else None
    rows, losses = train_with_eval(
        model, instance_walks, epochs=epochs, alpha=finetune_alpha, min_alpha=min_alpha,
        total_examples=n_lines, eval_fn=eval_fn, label=name, save_init=init_kv,
    )
    if save_kv is not None:
        model.wv.save(str(save_kv))
    res = _result_row(
        name, rows,
        dict(init_stats=init_stats, losses=losses, target_norm=used_norm, mix_lambda=mix_lambda,
             seconds=round(time.time() - t0, 1), vocab_size=len(model.wv),
             oov=oov_report(splits, model)),
    )
    del model
    return res


# ---------------------------------------------------------------------------
# Results cache
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict[str, dict]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    tmp.rename(path)
