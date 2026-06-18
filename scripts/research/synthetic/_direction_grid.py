"""Direction-tag x protograph (P1/P2/P3) grid for the thesis direction tables.

Extends ``notebooks/direction_aware_cardinality.ipynb``. The notebook ablates the
direction tag of the cardinality term on the P2 protograph only; for the thesis
tables we need the same pipeline run for every (test case, protograph kind,
method) triple so the appendix can average each method across P1/P2/P3.

Methods
-------
no_card
    Bound init WITHOUT the cardinality count term (delta=0) and with
    direction-blind binding (tag=shared) -- the count-free, direction-agnostic
    baseline the direction tags improve on.
shared / rolled / negated / keyed
    Full bound recipe (alpha=beta=gamma=delta=1) with that direction tag on the
    incoming-edge code (both the gamma binding term and the delta count term).

Recipe matches ``p2_bound`` in synthetic_compare: P2/P1/P3 pretrain -> bound init
mirrored into syn1neg -> 5 protected finetune epochs at LR 0.0025. Results are
cached to ``pipeline_grid_results.json`` keyed ``"{tc}|{kind}|{method}"``.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models.word2vec import LineSentence

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPTS = ROOT / "scripts"
COMPARE_ROOT = ROOT / "notebooks" / "synthetic_compare"
OUT_ROOT = ROOT / "notebooks" / "direction_aware_cardinality"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _maschine_init import stage1_vector_lookup  # noqa: E402
from _protograph_gen import iter_rdf_iris  # noqa: E402
from _synthetic_compare import (  # noqa: E402
    ensure_walks,
    load_materialized_types,
    make_eval_fn,
    new_skipgram_model,
    normalized_stage1_vectors,
    pretrain_protograph,
    protograph_init,
    train_with_eval,
    write_protographs,
)

# Directional DLCC test cases (existence controls tc01/tc02 + cardinality pairs).
TCS = ("tc01", "tc02", "tc09", "tc10", "tc11", "tc12")
PROTOS = ("p1", "p2", "p3")

# method -> directional_bound_vectors kwargs.
GRID_METHODS: dict[str, dict] = {
    "no_card": dict(direction_tag="shared", alpha=1.0, beta=1.0, gamma=1.0, delta=0.0),
    "shared": dict(direction_tag="shared", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0),
    "rolled": dict(direction_tag="rolled", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0),
    "negated": dict(direction_tag="negated", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0),
    "keyed": dict(direction_tag="keyed", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0),
}

CFG = dict(
    dim=200,
    walks_per_entity=100,
    proto_walks_per_entity=200,
    depth=3,
    epochs=5,
    pretrain_epochs=5,
    finetune_alpha=0.0025,
    min_alpha=0.0001,
    target_norm=8.0,
    seed=42,
)

GRID_JSON = OUT_ROOT / "pipeline_grid_results.json"


def tc_paths(tc: str) -> dict:
    tc_dir = ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return dict(
        ontology=tc_dir / "ontology.nt",
        graph=tc_dir / "graph.nt",
        config=tc_dir / "configuration.txt",
        train=tc_dir / "1000" / "train_test" / "train.txt",
        test=tc_dir / "1000" / "train_test" / "test.txt",
        cache=COMPARE_ROOT / tc,
        out=OUT_ROOT / tc,
    )


# ---------------------------------------------------------------------------
# Direction-tagged concept-bound init (mirrors the notebook's Part B definition)
# ---------------------------------------------------------------------------

def direction_key(dim: int) -> np.ndarray:
    return np.random.default_rng(7).choice([-1.0, 1.0], size=dim).astype(np.float32)


def tag_code(rc: np.ndarray, tag: str) -> np.ndarray:
    """Code added for an INCOMING edge of relation r (outgoing always uses rc)."""
    if tag == "shared":
        return rc
    if tag == "rolled":
        return np.roll(rc, 1)
    if tag == "negated":
        return -rc
    if tag == "keyed":
        return rc * direction_key(rc.shape[0])
    raise ValueError(f"unknown direction tag {tag!r}")


def directional_bound_vectors(
    graph_nt: Path,
    ontology_nt: Path,
    stage1_vectors: dict[str, np.ndarray],
    *,
    direction_tag: str = "rolled",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    target_norm: float = 1.0,
) -> dict[str, np.ndarray]:
    """concept_bound_vectors with the incoming-edge code parameterized by tag."""
    types = load_materialized_types(ontology_nt)

    def unit(v: np.ndarray) -> np.ndarray:
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

    def class_mix(inners) -> np.ndarray | None:
        parts = [code_of(c) for c in inners]
        parts = [p for p in parts if p is not None]
        return unit(np.mean(parts, axis=0)) if parts else None

    acc: dict[str, np.ndarray] = {}

    def bump(ent: str, vec: np.ndarray, w: float) -> None:
        if ent not in acc:
            acc[ent] = np.zeros_like(vec)
        acc[ent] += w * vec

    for ent, cs in types.items():
        mix = class_mix(cs)
        if mix is not None and alpha != 0.0:
            bump(ent, mix, alpha)

    for s, r, o in iter_rdf_iris(graph_nt):
        rc = code_of(r)
        if rc is None:
            continue
        rc_inv = tag_code(rc, direction_tag)
        if delta != 0.0:
            bump(s, rc, delta)
            bump(o, rc_inv, delta)
        if beta != 0.0:
            for c in types.get(o, ()):
                cc = code_of(c)
                if cc is not None:
                    bump(s, unit(rc * cc), beta)
        if gamma != 0.0:
            for c in types.get(s, ()):
                cc = code_of(c)
                if cc is not None:
                    bump(o, unit(rc_inv * cc), gamma)

    norms = [float(np.linalg.norm(v)) for v in acc.values()]
    mean_norm = float(np.mean(norms)) if norms else 1.0
    scale = target_norm / mean_norm if mean_norm > 0 else 1.0
    return {
        f"<{ent}>": (vec * scale).astype(np.float32)
        for ent, vec in acc.items()
        if float(np.linalg.norm(vec)) > 0
    }


def stage1_codes(tc: str, kind: str) -> tuple[dict[str, np.ndarray], float]:
    """Normalized pretrain codes for one protograph kind, cached on disk."""
    p = tc_paths(tc)
    p["out"].mkdir(parents=True, exist_ok=True)
    pkl = p["out"] / f"stage1_{kind}.pkl"
    if pkl.is_file():
        return pickle.loads(pkl.read_bytes())
    p["cache"].mkdir(parents=True, exist_ok=True)
    proto_paths = write_protographs(p["ontology"], p["cache"])
    walks = ensure_walks(
        proto_paths[kind], p["cache"] / f"walks_{kind}.txt",
        walks_per_entity=CFG["proto_walks_per_entity"], depth=CFG["depth"],
        seed=CFG["seed"], ensure_triple_coverage=True,
    )
    pre = pretrain_protograph(
        walks, dim=CFG["dim"], epochs=CFG["pretrain_epochs"], seed=CFG["seed"]
    )
    stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=CFG["target_norm"])
    pkl.write_bytes(pickle.dumps((stage1, used_norm)))
    return stage1, used_norm


def pipeline_run(tc: str, kind: str, method: str) -> dict:
    p = tc_paths(tc)
    stage1, used_norm = stage1_codes(tc, kind)
    inst_walks = ensure_walks(
        p["graph"],
        p["cache"] / f"walks_instance_w{CFG['walks_per_entity']}_d{CFG['depth']}.txt",
        walks_per_entity=CFG["walks_per_entity"], depth=CFG["depth"], seed=CFG["seed"],
    )
    bound = directional_bound_vectors(
        p["graph"], p["ontology"], stage1, target_norm=used_norm, **GRID_METHODS[method]
    )
    model = new_skipgram_model(
        dim=CFG["dim"], alpha=CFG["finetune_alpha"], min_alpha=CFG["min_alpha"],
        seed=CFG["seed"], workers=16,
    )
    model.build_vocab(LineSentence(str(inst_walks)))
    protograph_init(
        model, stage1, p["ontology"], strategy="all_init",
        target_norm=used_norm, bound_vectors=bound,
    )
    accs = train_with_eval(
        model, inst_walks, epochs=CFG["epochs"], alpha=CFG["finetune_alpha"],
        min_alpha=CFG["min_alpha"], eval_fn=make_eval_fn(p["train"], p["test"]),
    )
    return dict(tc=tc, kind=kind, method=method, accs=accs,
                init_acc=accs[0], final_acc=accs[-1])


def seed_from_notebook_cache(results: dict) -> int:
    """Reuse the notebook's P2 direction-tag pipeline runs (identical recipe)."""
    nb = OUT_ROOT / "pipeline_results.json"
    if not nb.is_file():
        return 0
    cached = json.loads(nb.read_text())
    added = 0
    for key, rec in cached.items():
        tc, tag = key.split("|")
        if tag not in GRID_METHODS:
            continue
        grid_key = f"{tc}|p2|{tag}"
        if grid_key in results:
            continue
        results[grid_key] = dict(
            tc=tc, kind="p2", method=tag, accs=rec["accs"],
            init_acc=rec["init_acc"], final_acc=rec["final_acc"],
        )
        added += 1
    return added


def compute_grid(verbose: bool = True) -> dict:
    results = json.loads(GRID_JSON.read_text()) if GRID_JSON.is_file() else {}
    added = seed_from_notebook_cache(results)
    if added and verbose:
        print(f"seeded {added} P2 tag runs from pipeline_results.json", flush=True)
    if added:
        GRID_JSON.write_text(json.dumps(results, indent=2) + "\n")

    t0 = time.time()
    for tc in TCS:
        for kind in PROTOS:
            for method in GRID_METHODS:
                key = f"{tc}|{kind}|{method}"
                if key in results:
                    continue
                results[key] = pipeline_run(tc, kind, method)
                GRID_JSON.write_text(json.dumps(results, indent=2) + "\n")
                if verbose:
                    r = results[key]
                    print(f"[{time.time()-t0:7.1f}s] {key:22s} final={r['final_acc']:.3f}",
                          flush=True)
    return results


if __name__ == "__main__":
    compute_grid()
    print(f"\nwrote {GRID_JSON}")
