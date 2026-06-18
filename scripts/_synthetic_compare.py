"""
Shared pipeline for notebooks/synthetic_compare.ipynb.

Compares vanilla RDF2Vec against protograph-pretrained variants (P1 / P2 / P3)
on the DLCC synthetic test cases. The protograph recipe follows the lessons from
notebooks/synthetic.ipynb (tc14):

  * keep pretrained rows at a large, uniform scale (small norms make every SGNS
    dot product ~0 and the init is erased within one epoch),
  * finetune with a reduced LR (0.005) instead of the from-scratch LR (0.025),
  * mirror initialized rows into syn1neg.

On top of that this module adds:

  * **normalized initialization** — every transferred vector is L2-normalized to a
    single target norm (mean native pretrain norm by default), so rare leaf
    classes are not penalized by their low pretrain frequency;
  * **P3 protograph** — P1 (domain, r, range) triples plus the full rdfs:subClassOf
    hierarchy as explicit edges (both directions). Unlike P1 (leaf classes missing)
    and P2 (only direct subclasses of P1 classes), P3 puts *every* class in the
    pretrain vocabulary and gives subclasses contexts that overlap with their
    ancestors, so class-mean inits cluster by hierarchy.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from sklearn.metrics import accuracy_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _evaluate import load_labeled_txt, make_classifiers, tokens_to_embeddings  # noqa: E402
from _maschine_init import (  # noqa: E402
    build_entity_to_class_tokens,
    load_maschine_entity_mapping,
    maschine_embedding_for_token,
    normalize_init_strategy,
    stage1_vector_lookup,
)
from _protograph_gen import build_p1, build_p2, load_ontology, triples_to_nt_lines  # noqa: E402
from _walks import run_jrdf2vec_duplicate_free  # noqa: E402

SUBCLASS_TOKEN = "SCHEMA_subClassOf"
HAS_SUBCLASS_TOKEN = "SCHEMA_hasSubclass"

PROTOGRAPH_KINDS = ("p1", "p2", "p3")


# ---------------------------------------------------------------------------
# Protograph construction
# ---------------------------------------------------------------------------

def build_p3(
    p1: set[tuple[str, str, str]],
    children: dict[str, set[str]],
) -> set[tuple[str, str, str]]:
    """
    P3 = P1 + the full subClassOf hierarchy as bidirectional edges.

    P1 only contains classes used as a domain/range; P2 adds direct subclasses of
    those but still misses deeper leaves. The DLCC instances are typed with leaf
    classes, so under P1/P2 most class-mean inits fall back to random. P3 keeps
    every class reachable: walks through (child, subClassOf, parent) and
    (parent, hasSubclass, child) edges give each class a vector that shares
    context with its ancestors and siblings while remaining a distinct token.
    """
    out = set(p1)
    for parent, kids in children.items():
        for child in kids:
            out.add((child, SUBCLASS_TOKEN, parent))
            out.add((parent, HAS_SUBCLASS_TOKEN, child))
    return out


def write_protographs(ontology_nt: Path, out_dir: Path) -> dict[str, Path]:
    """Build P1/P2/P3 from the ontology and write them as N-Triples. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    domains, ranges, children, _instance_types, _class_parents = load_ontology(ontology_nt)
    p1 = build_p1(domains, ranges)
    graphs = {
        "p1": p1,
        "p2": build_p2(p1, children),
        "p3": build_p3(p1, children),
    }
    paths: dict[str, Path] = {}
    for kind, triples in graphs.items():
        path = out_dir / f"protograph_{kind}.nt"
        path.write_text("".join(triples_to_nt_lines(triples)), encoding="utf-8")
        paths[kind] = path
    return paths


TYPE_TOKEN = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def iter_instance_type_triples(ontology_nt: Path, *, materialize_ancestors: bool = False):
    """
    (instance, rdf:type, class) triples for synthetic individuals (I_* / EXTRA_I_*).

    With ``materialize_ancestors`` the RDFS entailment is materialized: each
    instance is additionally typed with every transitive superclass of its
    asserted classes.
    """
    from _protograph_gen import RDF_TYPE, RDFS_SUBCLASS, iter_rdf_iris

    parents: dict[str, set[str]] = {}
    asserted: dict[str, set[str]] = {}
    for s, p, o in iter_rdf_iris(ontology_nt):
        if p == RDFS_SUBCLASS and s.startswith("C_") and o.startswith("C_"):
            parents.setdefault(s, set()).add(o)
            continue
        if p != RDF_TYPE or not o.startswith("C_"):
            continue
        if s.startswith("C_") or s.startswith("P_"):
            continue
        asserted.setdefault(s, set()).add(o)

    def ancestors(c: str) -> set[str]:
        out: set[str] = set()
        stack = [c]
        while stack:
            for par in parents.get(stack.pop(), ()):
                if par not in out:
                    out.add(par)
                    stack.append(par)
        return out

    for s in sorted(asserted):
        classes = set(asserted[s])
        if materialize_ancestors:
            for c in list(classes):
                classes |= ancestors(c)
        for o in sorted(classes):
            yield s, TYPE_TOKEN, o


def write_augmented_graph(
    graph_nt: Path,
    ontology_nt: Path,
    protograph_nt: Path,
    out_path: Path,
    *,
    materialize_ancestors: bool = False,
) -> Path:
    """
    Instance graph + (instance, rdf:type, class) edges + protograph triples.

    This *attaches* the protograph to the instance graph: random walks can step
    from an instance into its class and onward through the protograph, so class
    tokens keep occurring in the finetune corpus instead of living only in the
    initialization (where instance-only walks gradually erase them). With
    ``materialize_ancestors``, discriminative superclasses co-occur with labeled
    entities directly inside the skip-gram window instead of requiring the walk
    to route through several subClassOf hops.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        out.write(graph_nt.read_text(encoding="utf-8"))
        for s, p, o in iter_instance_type_triples(
            ontology_nt, materialize_ancestors=materialize_ancestors
        ):
            out.write(f"<{s}> <{p}> <{o}> .\n")
        out.write(protograph_nt.read_text(encoding="utf-8"))
    return out_path


def load_materialized_types(ontology_nt: Path) -> dict[str, list[str]]:
    """instance -> sorted asserted + inherited (ancestor-materialized) classes."""
    out: dict[str, set[str]] = {}
    for s, _p, o in iter_instance_type_triples(ontology_nt, materialize_ancestors=True):
        out.setdefault(s, set()).add(o)
    return {s: sorted(cs) for s, cs in out.items()}


def _instance_type_structure(ontology_nt: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(subClassOf parent map, asserted instance->classes) parsed from the ontology."""
    from _protograph_gen import RDF_TYPE, RDFS_SUBCLASS, iter_rdf_iris

    parents: dict[str, set[str]] = {}
    asserted: dict[str, set[str]] = {}
    for s, p, o in iter_rdf_iris(ontology_nt):
        if p == RDFS_SUBCLASS and s.startswith("C_") and o.startswith("C_"):
            parents.setdefault(s, set()).add(o)
            continue
        if p != RDF_TYPE or not o.startswith("C_"):
            continue
        if s.startswith("C_") or s.startswith("P_"):
            continue
        asserted.setdefault(s, set()).add(o)
    return parents, asserted


def load_weighted_types(
    ontology_nt: Path,
    *,
    mode: str = "uniform",
    ancestor_decay: float = 0.5,
) -> dict[str, dict[str, float]]:
    """instance -> {class: weight} controlling how the class hierarchy enters the
    concept-bound init (own-class mean + the relation⊙class neighbour bindings).

    ``mode`` selects how ancestors are mixed in:

      * ``"uniform"`` — every asserted class **and** transitive superclass gets
        weight 1.0 (ancestor-materialized). This is the base bound behavior; the
        returned class set matches :func:`load_materialized_types`.
      * ``"most_specific"`` — only the instance's most-specific asserted class(es)
        get weight 1.0; all ancestors are dropped (ablation variant (b): no
        hierarchy noise, just the leaf type).
      * ``"decay"`` — every materialized class ``c`` gets weight
        ``ancestor_decay ** hops``, where ``hops`` is the shortest subClassOf
        distance from a most-specific asserted class (leaf → 1.0, parent →
        ``ancestor_decay``, grandparent → ``ancestor_decay**2`` …). Variant (a):
        keep ancestors but let distant ones fade so the leaf dominates.
    """
    if mode not in ("uniform", "most_specific", "decay"):
        raise ValueError(f"unknown class weight mode {mode!r}")
    parents, asserted = _instance_type_structure(ontology_nt)

    def most_specific(classes: set[str]) -> set[str]:
        # drop any asserted class that is a transitive superclass of another
        # asserted class of the same instance
        superclasses: set[str] = set()
        for c in classes:
            seen: set[str] = set()
            stack = list(parents.get(c, ()))
            while stack:
                par = stack.pop()
                if par in seen:
                    continue
                seen.add(par)
                stack.extend(parents.get(par, ()))
            superclasses |= seen & classes
        return classes - superclasses

    out: dict[str, dict[str, float]] = {}
    for inst in sorted(asserted):
        leaves = most_specific(set(asserted[inst]))
        if mode == "most_specific":
            out[inst] = {c: 1.0 for c in sorted(leaves)}
            continue
        # multi-source BFS from the leaves → shortest subClassOf hop count per class
        depth: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((c, 0) for c in sorted(leaves))
        while queue:
            c, d = queue.popleft()
            if c in depth:
                continue
            depth[c] = d
            for par in sorted(parents.get(c, ())):
                if par not in depth:
                    queue.append((par, d + 1))
        if mode == "uniform":
            out[inst] = {c: 1.0 for c in sorted(depth)}
        else:  # decay
            out[inst] = {c: float(ancestor_decay) ** d for c, d in sorted(depth.items())}
    return out


def concept_token(relation: str, cls: str, *, inverse: bool) -> str:
    """Token for the DL concept ∃r.C (or ∃r⁻.C): a relation-class *pair* node."""
    return f"EXINV_{relation}__{cls}" if inverse else f"EX_{relation}__{cls}"


def write_concept_attached_graph(
    graph_nt: Path,
    ontology_nt: Path,
    protograph_nt: Path,
    out_path: Path,
) -> Path:
    """
    P4 attachment: instance graph + DL *concept nodes* for existential restrictions.

    P1/P2 (and class-mean inits derived from them) inject class knowledge per
    *token*, but skip-gram representations are sums of context vectors, so the
    binding between a relation and the class of its object is lost: an entity
    with P19 -> (C281 thing) plus some-other-relation -> (C390 thing) looks like
    an entity with P19 -> (C390 thing). The DLCC tasks tc07-tc12 are defined by
    exactly such bindings (∃r.C / ∃r⁻.C).

    For every instance triple (s, r, o) and every (ancestor-materialized) class c:

        c ∈ classes(o):  add (s, r, EX_r__c)       — s satisfies ∃r.c
        c ∈ classes(s):  add (o, r, EXINV_r__c)    — o satisfies ∃r⁻.c

    plus plain (instance, rdf:type, class) edges with materialized ancestors
    (class-membership tasks, tc13-tc16) and the protograph triples themselves
    (class-class structure). Concept nodes are sinks: they only ever occur as
    walk *contexts*, turning each labeled entity's embedding into a bag of the
    schema concepts it satisfies.
    """
    types = load_materialized_types(ontology_nt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from _protograph_gen import iter_rdf_iris

    with out_path.open("w", encoding="utf-8") as out:
        out.write(graph_nt.read_text(encoding="utf-8"))
        for s, r, o in iter_rdf_iris(graph_nt):
            for c in types.get(o, ()):
                out.write(f"<{s}> <{r}> <{concept_token(r, c, inverse=False)}> .\n")
            for c in types.get(s, ()):
                out.write(f"<{o}> <{r}> <{concept_token(r, c, inverse=True)}> .\n")
        for s, cs in types.items():
            for c in cs:
                out.write(f"<{s}> <{TYPE_TOKEN}> <{c}> .\n")
        out.write(protograph_nt.read_text(encoding="utf-8"))
    return out_path


# ---------------------------------------------------------------------------
# Walks (cached on disk)
# ---------------------------------------------------------------------------

def ensure_walks(
    graph_path: Path,
    out_path: Path,
    *,
    walks_per_entity: int,
    depth: int,
    seed: int = 42,
    threads: int = 8,
    ensure_triple_coverage: bool = False,
) -> Path:
    """Generate duplicate-free random walks unless the corpus already exists."""
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
        token_format="angled",
        input_format="nt",
        ensure_triple_coverage=ensure_triple_coverage,
    )
    tmp.rename(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Word2Vec training with per-epoch evaluation
# ---------------------------------------------------------------------------

def make_eval_fn(train_txt: Path, test_txt: Path, *, seed: int = 42):
    """LogReg test accuracy on the current model state (same protocol as synthetic.ipynb)."""
    train_tokens, y_train = load_labeled_txt(train_txt)
    test_tokens, y_test = load_labeled_txt(test_txt)

    def evaluate(model: Word2Vec) -> float:
        emb = np.asarray(model.wv.vectors, dtype=np.float32)
        w2i = dict(model.wv.key_to_index)
        x_tr, _ = tokens_to_embeddings(train_tokens, emb, w2i, "", 4096, progress=False)
        x_te, _ = tokens_to_embeddings(test_tokens, emb, w2i, "", 4096, progress=False)
        clf = make_classifiers(max_iter=1000, seed=seed)["LogReg"]
        clf.fit(x_tr, y_train)
        return float(accuracy_score(y_test, clf.predict(x_te)))

    return evaluate


class _EpochEval(CallbackAny2Vec):
    def __init__(self, eval_fn, accs: list[float]):
        self.eval_fn = eval_fn
        self.accs = accs

    def on_epoch_end(self, model):
        self.accs.append(self.eval_fn(model))


def new_skipgram_model(
    *,
    dim: int,
    alpha: float,
    min_alpha: float,
    seed: int,
    workers: int,
    window: int = 5,
) -> Word2Vec:
    return Word2Vec(
        vector_size=dim,
        window=window,
        sg=1,
        hs=0,
        negative=5,
        min_count=1,
        sample=0.0,
        alpha=alpha,
        min_alpha=min_alpha,
        workers=workers,
        seed=seed,
    )


def train_with_eval(
    model: Word2Vec,
    walks_path: Path,
    *,
    epochs: int,
    alpha: float,
    min_alpha: float,
    eval_fn=None,
    extra_callbacks: list | None = None,
) -> list[float]:
    """Train and return [acc@init, acc@epoch1, ..., acc@epochN] (empty without eval_fn).

    ``extra_callbacks`` (e.g. an anchoring callback) run *before* the eval
    callback on every ``on_epoch_end``, so the reported accuracy reflects the
    state after the extra callbacks have modified the model.
    """
    accs: list[float] = []
    callbacks = list(extra_callbacks or [])
    if eval_fn is not None:
        accs.append(eval_fn(model))
        callbacks.append(_EpochEval(eval_fn, accs))
    model.train(
        corpus_iterable=LineSentence(str(walks_path)),
        total_examples=model.corpus_count,
        epochs=epochs,
        start_alpha=alpha,
        end_alpha=min_alpha,
        callbacks=callbacks,
    )
    return accs


def pretrain_protograph(
    walks_path: Path,
    *,
    dim: int,
    epochs: int = 5,
    seed: int = 42,
    workers: int = 8,
) -> Word2Vec:
    """Pretrain skip-gram on protograph walks (same hyperparameters as synthetic.ipynb)."""
    model = new_skipgram_model(dim=dim, alpha=0.025, min_alpha=0.0001, seed=seed, workers=workers)
    model.build_vocab(LineSentence(str(walks_path)))
    model.train(
        corpus_iterable=LineSentence(str(walks_path)),
        total_examples=model.corpus_count,
        epochs=epochs,
    )
    return model


# ---------------------------------------------------------------------------
# Normalized MASCHInE-style initialization
# ---------------------------------------------------------------------------

def normalized_stage1_vectors(
    wv,
    *,
    target_norm: float | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    """
    Protograph vectors L2-normalized to one shared target norm.

    By default the target is the mean native pretrain norm: large enough that
    consistent pairs saturate the SGNS sigmoid (protecting the init during
    finetuning, see synthetic.ipynb step 5), while removing per-token norm
    differences caused by pretrain token frequency.
    """
    vecs = np.asarray(wv.vectors, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    if target_norm is None:
        target_norm = float(norms.mean())
    out: dict[str, np.ndarray] = {}
    for tok, idx in wv.key_to_index.items():
        n = norms[idx]
        v = vecs[idx] * (target_norm / n) if n > 0 else vecs[idx]
        out[tok] = v.astype(np.float32, copy=False)
    return out, target_norm


def concept_bound_vectors(
    graph_nt: Path,
    ontology_nt: Path,
    stage1_vectors: dict[str, np.ndarray],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    target_norm: float = 1.0,
    normalize_components: bool = True,
    class_weight_mode: str = "uniform",
    ancestor_decay: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Concept-bound instance initialization (the P1/P2 fix found in this work).

    Skip-gram embeddings are *sums* of context vectors, so the binding between a
    relation and the class of its object is lost — yet the DLCC tasks tc07-tc12
    are defined by exactly such bindings (∃r.C / ∃r⁻.C, plus cardinalities over
    them). Classic MASCHInE init only injects an entity's *own* class, which
    carries no signal when positives and negatives share a class (tc07: all
    labeled entities are C_152).

    Here every instance e is initialized with a superposition of the schema
    concepts it satisfies, built from *pretrained protograph codes*:

        init(e) =  alpha * unit(mean of codes of e's materialized classes)
                 + beta  * sum over (e, r, o), c in classes(o) of unit(code(r) ⊙ code(c))
                 + gamma * sum over (s, r, e), c in classes(s) of unit(roll(code(r), 1) ⊙ code(c))
                 + delta * sum over (e, r, o) of code(r)  +  sum over (s, r, e) of roll(code(r), 1)

    The Hadamard product ⊙ keeps (r, c) pairs distinguishable from the sum of
    their parts (binding), while inheriting the protograph geometry: codes of
    sibling classes are close, so concept features generalize across the class
    hierarchy. ``roll`` distinguishes incoming from outgoing edges; the delta
    term adds plain per-relation counts (qualified cardinality tasks tc09-tc12).

    Repeated (r, class) pairs are accumulated, NOT deduplicated: the DLCC
    cardinality tasks are separable only through these magnitudes, which is also
    why the final vectors get one *global* rescale (mean norm -> ``target_norm``)
    instead of a per-vector normalization that would erase the count signal.

    Returns a dict keyed by ``<IRI>`` walk tokens (instances only).

    ``normalize_components`` (ablation): when ``False``, the per-component ``unit``
    L2-normalizations (codes, class mix, each relation⊙class pair) are skipped, so
    components are combined at their raw stage-1 scale; the single global
    mean-norm -> ``target_norm`` rescale still applies. Default ``True`` = current
    behavior.

    ``class_weight_mode`` controls how an instance's class hierarchy enters every
    class-bearing term (own-class mean and the relation⊙class neighbour bindings),
    see :func:`load_weighted_types`:

      * ``"uniform"`` (default) — ancestor-materialized, equal weight: the base
        bound init. Reproduces the prior behavior exactly.
      * ``"most_specific"`` — only the most-specific asserted class, no ancestors.
      * ``"decay"`` — ancestors kept but weighted ``ancestor_decay ** hops`` from
        the leaf, so distant superclasses fade.

    ``ancestor_decay`` is the per-hop decay used by ``class_weight_mode="decay"``.
    """
    weights = load_weighted_types(
        ontology_nt, mode=class_weight_mode, ancestor_decay=ancestor_decay
    )

    def unit(v: np.ndarray) -> np.ndarray:
        # ablation lever: when normalize_components is False, every component
        # (codes, class mix, relation⊙class pairs) keeps its raw scale; only the
        # final global mean->target_norm rescale is applied.
        if not normalize_components:
            return v
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    codes: dict[str, np.ndarray] = {}

    def code_of(inner: str) -> np.ndarray | None:
        cached = codes.get(inner)
        if cached is not None:
            return cached
        vec = stage1_vector_lookup(inner, stage1_vectors)
        if vec is None:
            return None
        u = unit(np.asarray(vec, dtype=np.float32))
        codes[inner] = u
        return u

    def class_mix(class_weights: dict[str, float]) -> np.ndarray | None:
        # weighted mean of the class codes; with uniform weights this is the plain
        # mean (unit() removes the shared scale), so the base path is unchanged.
        parts = []
        wsum = 0.0
        for c, w in class_weights.items():
            code = code_of(c)
            if code is not None:
                parts.append(w * code)
                wsum += w
        if not parts or wsum <= 0:
            return None
        return unit(np.sum(parts, axis=0) / wsum)

    from _protograph_gen import iter_rdf_iris

    acc: dict[str, np.ndarray] = {}

    def bump(ent: str, vec: np.ndarray, w: float) -> None:
        if ent not in acc:
            acc[ent] = np.zeros_like(vec)
        acc[ent] += w * vec

    for ent, cw in weights.items():
        mix = class_mix(cw)
        if mix is not None and alpha != 0.0:
            bump(ent, mix, alpha)

    for s, r, o in iter_rdf_iris(graph_nt):
        rc = code_of(r)
        if rc is None:
            continue
        rc_inv = np.roll(rc, 1)
        if delta != 0.0:
            bump(s, rc, delta)
            bump(o, rc_inv, delta)
        if beta != 0.0:
            for c, w in weights.get(o, {}).items():
                cc = code_of(c)
                if cc is not None:
                    bump(s, unit(rc * cc), beta * w)
        if gamma != 0.0:
            for c, w in weights.get(s, {}).items():
                cc = code_of(c)
                if cc is not None:
                    bump(o, unit(rc_inv * cc), gamma * w)

    norms = [float(np.linalg.norm(v)) for v in acc.values()]
    mean_norm = float(np.mean(norms)) if norms else 1.0
    scale = target_norm / mean_norm if mean_norm > 0 else 1.0
    return {
        f"<{ent}>": (vec * scale).astype(np.float32)
        for ent, vec in acc.items()
        if float(np.linalg.norm(vec)) > 0
    }


INIT_MODES = ("both", "context_only", "vectors_only")


def protograph_init(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    ontology_nt: Path,
    *,
    strategy: str = "most_specific",
    ancestor_decay: float = 0.5,
    target_norm: float | None = None,
    renormalize: bool = True,
    noise: float = 0.0,
    mode: str = "both",
    bound_vectors: dict[str, np.ndarray] | None = None,
    record_class_init: list[str] | None = None,
    normalize_class_codes: bool = False,
) -> dict[str, int]:
    """
    Initialize a fresh instance-walk model from (normalized) protograph vectors.

    When ``bound_vectors`` is given (see :func:`concept_bound_vectors`), instance
    tokens present in it use the concept-bound vector; all other tokens fall back
    to the standard MASCHInE rules.

      1. protograph rows (classes/relations) present in the new vocabulary are copied,
      2. typed instances are initialized from class vectors (`strategy` picks how:
         most_specific / all_init / average_hier / weight_hierarchy),
      3. initialized rows are renormalized to ``target_norm`` (means of unit
         vectors shrink; without this step instances of "mixed" classes start
         closer to the origin and lose the sigmoid-saturation protection).

    ``mode`` selects which matrices receive the init:

      * ``"both"`` — wv.vectors and syn1neg (MASCHInE-style, as in synthetic.ipynb);
      * ``"context_only"`` — only syn1neg; wv keeps gensim's tiny random init.
        Center rows then *accumulate* the class codes of their walk contexts
        (every SGNS update adds context rows to the center row), so an entity's
        final vector approximates a bag of its neighbours' classes — exactly the
        feature the relation-centric DLCC tasks (tc07-tc12) need;
      * ``"vectors_only"`` — only wv.vectors.
    """
    if mode not in INIT_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {INIT_MODES}")
    strategy_internal = normalize_init_strategy(strategy)
    parents, entity_types = load_maschine_entity_mapping(ontology_nt, quiet=True)
    ent_to_classes = build_entity_to_class_tokens(parents, entity_types)

    wv = model.wv
    rng = np.random.default_rng(12345)
    n_stage1 = n_class = n_bound = n_random = 0
    for token in wv.index_to_key:
        is_bound = False
        vec = bound_vectors.get(token) if bound_vectors is not None else None
        if vec is not None:
            is_bound = True
        else:
            vec = maschine_embedding_for_token(
                token,
                stage1_vectors,
                parents,
                ent_to_classes,
                strategy=strategy_internal,
                ancestor_decay=ancestor_decay,
                normalize_class_codes=normalize_class_codes,
            )
        if vec is None:
            n_random += 1
            continue
        v = np.asarray(vec, dtype=np.float32).copy()
        if renormalize and target_norm is not None and not is_bound:
            n = float(np.linalg.norm(v))
            if n > 0:
                v *= target_norm / n
        if noise > 0:
            v = v + rng.normal(0.0, noise, size=v.shape).astype(np.float32)
        idx = wv.key_to_index[token]
        if mode in ("both", "vectors_only"):
            wv.vectors[idx] = v
        if mode in ("both", "context_only") and model.syn1neg is not None:
            model.syn1neg[idx] = v.copy()
        if is_bound:
            n_bound += 1
        elif stage1_vector_lookup(token, stage1_vectors) is not None:
            n_stage1 += 1
        else:
            n_class += 1
            if record_class_init is not None:
                record_class_init.append(token)
    return {
        "stage1_copied": n_stage1,
        "class_init": n_class,
        "bound_init": n_bound,
        "random": n_random,
    }


# ---------------------------------------------------------------------------
# One full variant run
# ---------------------------------------------------------------------------

def run_vanilla(
    instance_walks: Path,
    train_txt: Path,
    test_txt: Path,
    *,
    dim: int = 200,
    epochs: int = 5,
    alpha: float = 0.025,
    min_alpha: float = 0.0001,
    seed: int = 42,
    workers: int = 16,
    window: int = 5,
) -> dict:
    """From-scratch RDF2Vec on instance walks (the baseline)."""
    t0 = time.time()
    eval_fn = make_eval_fn(train_txt, test_txt)
    model = new_skipgram_model(
        dim=dim, alpha=alpha, min_alpha=min_alpha, seed=seed, workers=workers, window=window
    )
    model.build_vocab(LineSentence(str(instance_walks)))
    accs = train_with_eval(
        model, instance_walks, epochs=epochs, alpha=alpha, min_alpha=min_alpha, eval_fn=eval_fn
    )
    return {
        "variant": "vanilla",
        "accs": accs,
        "final_acc": accs[-1],
        "init_stats": None,
        "seconds": round(time.time() - t0, 1),
        "model": model,
    }


def run_protograph_variant(
    name: str,
    proto_walks: Path,
    instance_walks: Path,
    ontology_nt: Path,
    train_txt: Path,
    test_txt: Path,
    *,
    strategy: str = "most_specific",
    ancestor_decay: float = 0.5,
    normalize: bool = True,
    target_norm: float | None = None,
    noise: float = 0.0,
    mode: str = "both",
    bound_graph: Path | None = None,
    bound_alpha: float = 1.0,
    bound_beta: float = 1.0,
    bound_gamma: float = 1.0,
    bound_delta: float = 1.0,
    bound_class_weight_mode: str = "uniform",
    bound_ancestor_decay: float = 0.5,
    dim: int = 200,
    pretrain_epochs: int = 5,
    epochs: int = 5,
    finetune_alpha: float = 0.005,
    min_alpha: float = 0.0001,
    seed: int = 42,
    workers: int = 16,
    window: int = 5,
) -> dict:
    """Pretrain on protograph walks, transfer with (normalized) init, finetune, evaluate."""
    t0 = time.time()
    pre = pretrain_protograph(
        proto_walks, dim=dim, epochs=pretrain_epochs, seed=seed, workers=workers
    )
    if normalize:
        stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=target_norm)
    else:
        stage1 = {tok: np.asarray(pre.wv[tok], dtype=np.float32) for tok in pre.wv.index_to_key}
        used_norm = None

    bound_vectors = None
    if bound_graph is not None:
        bound_vectors = concept_bound_vectors(
            bound_graph,
            ontology_nt,
            stage1,
            alpha=bound_alpha,
            beta=bound_beta,
            gamma=bound_gamma,
            delta=bound_delta,
            target_norm=used_norm if used_norm is not None else 1.0,
            class_weight_mode=bound_class_weight_mode,
            ancestor_decay=bound_ancestor_decay,
        )

    eval_fn = make_eval_fn(train_txt, test_txt)
    model = new_skipgram_model(
        dim=dim, alpha=finetune_alpha, min_alpha=min_alpha, seed=seed, workers=workers,
        window=window,
    )
    model.build_vocab(LineSentence(str(instance_walks)))
    init_stats = protograph_init(
        model,
        stage1,
        ontology_nt,
        strategy=strategy,
        ancestor_decay=ancestor_decay,
        target_norm=used_norm,
        renormalize=normalize,
        noise=noise,
        mode=mode,
        bound_vectors=bound_vectors,
    )
    accs = train_with_eval(
        model,
        instance_walks,
        epochs=epochs,
        alpha=finetune_alpha,
        min_alpha=min_alpha,
        eval_fn=eval_fn,
    )
    return {
        "variant": name,
        "accs": accs,
        "final_acc": accs[-1],
        "init_stats": init_stats,
        "target_norm": used_norm,
        "seconds": round(time.time() - t0, 1),
        "model": model,
    }
