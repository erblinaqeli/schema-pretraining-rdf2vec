"""
Investigation harness for notebooks/dbpedia_investigate.ipynb.

Goal: understand *why* the concept-bound / P3 init that wins big on the synthetic
DLCC suite does **not** win end-to-end on real DBpedia, and find a recipe that
does. The companion notebooks/dbpedia_compare.ipynb established the diagnosis:

  * the concept-bound init is the best *initializer* (epoch-0 mean acc 0.815 vs
    vanilla 0.501), but
  * vanilla has almost no headroom on the 1.16B-token corpus (0.50 -> 0.815 in
    one epoch), the benchmark mass is degree-shaped (a 4-number degree probe
    scores 0.79 normal / 0.74 hard), and
  * the *protected* finetune (LR 0.0025 + one global rescale that leaves hub
    rows with norms up to ~64k) freezes the bound model, blocking the recovery
    vanilla gets on the {e} family (tc04-tc06).

This module provides the pieces to test the fixes that were prescribed but never
run on DBpedia:

  1. evaluation lenses ......... raw vs per-dim standardize vs row-L2 (the cached
     bound vectors have a pathological norm tail; raw LogReg is ill-conditioned);
  2. feature fusion ........... evaluate [vanilla | bound] concatenations, i.e.
     use the bound init as an extra channel rather than a replacement;
  3. norm-cap init ............ per-row norm cap instead of one global rescale,
     so hub rows are no longer frozen during finetune;
  4. relaxed / per-row finetune  full-LR or norm-gated learning rates.

Most heavy artifacts (schema, instance types, pretrained stage-1 codes, the
concept-bound init vectors) are built once and cached as .npz / .pkl so the
notebook can iterate on eval + finetune policy without re-streaming graph.nt.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _dbpedia_compare import EvalSplit, load_eval_splits  # noqa: E402

ROOT = _SCRIPTS_DIR.parent

# DLCC DBpedia test-case families (see notebooks/dbpedia_compare.ipynb).
FAMILIES = {
    "existence (tc01-03)": ("tc01", "tc02", "tc03"),
    "individual {e} (tc04-06)": ("tc04", "tc05", "tc06"),
    "qualified-exist (tc07-08)": ("tc07", "tc08"),
    "cardinality (tc09-12)": ("tc09", "tc10", "tc11", "tc12"),
}


# ---------------------------------------------------------------------------
# Embedding matrices for the eval splits
# ---------------------------------------------------------------------------

def split_matrix(tokens: list[str], kv, *, dim: int) -> tuple[np.ndarray, int]:
    """Lookup rows for `tokens` in a KeyedVectors-like object; OOV -> zeros."""
    out = np.zeros((len(tokens), dim), dtype=np.float32)
    k2i = kv.key_to_index
    vecs = kv.vectors
    oov = 0
    for i, t in enumerate(tokens):
        j = k2i.get(t)
        if j is None:
            oov += 1
        else:
            out[i] = vecs[j]
    return out, oov


def dict_matrix(tokens: list[str], emb: dict[str, np.ndarray], *, dim: int) -> np.ndarray:
    """Lookup rows for `tokens` in a plain {token: vector} dict; OOV -> zeros."""
    out = np.zeros((len(tokens), dim), dtype=np.float32)
    for i, t in enumerate(tokens):
        v = emb.get(t)
        if v is not None:
            out[i] = v
    return out


# ---------------------------------------------------------------------------
# Scaling lenses
# ---------------------------------------------------------------------------

def _scale_blocks(
    blocks_tr: list[np.ndarray],
    blocks_te: list[np.ndarray],
    scaling: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a scaling lens block-by-block, then concatenate.

    scaling:
      * "raw"         — no scaling (matches dbpedia_compare's eval);
      * "standardize" — per-dim z-score (fit on train); preserves relative
        magnitudes across entities so the cardinality signal survives, but
        removes the hub-norm conditioning problem;
      * "l2row"       — row L2-normalize; removes hub-norm pathology but ERASES
        the count magnitude (so it should hurt cardinality tcs);
      * "standardize+l2row" — l2row then per-dim standardize.
    """
    xs_tr, xs_te = [], []
    for btr, bte in zip(blocks_tr, blocks_te):
        a, b = btr, bte
        if "l2row" in scaling:
            a = _l2rows(a)
            b = _l2rows(b)
        if "standardize" in scaling:
            sc = StandardScaler()
            a = sc.fit_transform(a)
            b = sc.transform(b)
        xs_tr.append(a.astype(np.float32))
        xs_te.append(b.astype(np.float32))
    return np.concatenate(xs_tr, axis=1), np.concatenate(xs_te, axis=1)


def _l2rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _fit_one(x_tr, y_tr, x_te, y_te, seed, max_iter, C):
    clf = LogisticRegression(max_iter=max_iter, random_state=seed, C=C)
    clf.fit(x_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(x_te)))


@dataclass
class Model:
    """A named source of token vectors for evaluation (a kv or a dict)."""
    name: str
    kv: object = None        # KeyedVectors-like (has .key_to_index, .vectors)
    emb: dict = None         # or a {token: np.ndarray} dict
    dim: int = 200

    def matrix(self, tokens: list[str]) -> np.ndarray:
        if self.kv is not None:
            m, _ = split_matrix(tokens, self.kv, dim=self.dim)
            return m
        return dict_matrix(tokens, self.emb, dim=self.dim)


def evaluate_models(
    splits: list[EvalSplit],
    models: list[Model],
    *,
    scaling: str = "raw",
    seed: int = 42,
    max_iter: int = 2000,
    C: float = 1.0,
    n_jobs: int = 12,
) -> dict[str, float]:
    """LogReg test accuracy per split on the concatenation of `models` (a fusion
    when len>1), under a scaling lens. Returns {split_name: acc}."""
    from joblib import Parallel, delayed

    tasks = []
    for sp in splits:
        blocks_tr = [m.matrix(sp.train_tokens) for m in models]
        blocks_te = [m.matrix(sp.test_tokens) for m in models]
        x_tr, x_te = _scale_blocks(blocks_tr, blocks_te, scaling)
        tasks.append((x_tr, sp.y_train, x_te, sp.y_test))
    accs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(x_tr, y_tr, x_te, y_te, seed, max_iter, C)
        for x_tr, y_tr, x_te, y_te in tasks
    )
    return {sp.name: a for sp, a in zip(splits, accs)}


# ---------------------------------------------------------------------------
# Aggregation helpers (per-tc, per-family, normal/hard)
# ---------------------------------------------------------------------------

def split_meta(name: str) -> tuple[str, str, bool]:
    """'tc07/people_hard' -> ('tc07', 'people', True)."""
    tc, domain = name.split("/")
    hard = domain.endswith("_hard")
    return tc, domain.removesuffix("_hard"), hard


def summarize(accs: dict[str, float]) -> dict:
    """Mean over normal/hard splits, and per-family means."""
    import numpy as np
    normal = [a for n, a in accs.items() if not split_meta(n)[2]]
    hard = [a for n, a in accs.items() if split_meta(n)[2]]
    out = {
        "normal": float(np.mean(normal)) if normal else float("nan"),
        "hard": float(np.mean(hard)) if hard else float("nan"),
        "all": float(np.mean(list(accs.values()))),
    }
    for fam, tcs in FAMILIES.items():
        fn = [a for n, a in accs.items() if split_meta(n)[0] in tcs and not split_meta(n)[2]]
        fh = [a for n, a in accs.items() if split_meta(n)[0] in tcs and split_meta(n)[2]]
        out[f"{fam} N"] = float(np.mean(fn)) if fn else float("nan")
        out[f"{fam} H"] = float(np.mean(fh)) if fh else float("nan")
    return out


def per_tc(accs: dict[str, float], *, hard: bool) -> dict[str, float]:
    import numpy as np
    buckets: dict[str, list[float]] = {}
    for n, a in accs.items():
        tc, _dom, h = split_meta(n)
        if h == hard:
            buckets.setdefault(tc, []).append(a)
    return {tc: float(np.mean(v)) for tc, v in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# token -> vector dict on-disk cache (keys json + vectors .npy)
# ---------------------------------------------------------------------------

def save_emb(path: Path, emb: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(emb.keys())
    mat = np.stack([emb[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, 0), np.float32)
    np.save(path.with_suffix(".vecs.npy"), mat)
    path.with_suffix(".keys.json").write_text(json.dumps(keys), encoding="utf-8")


def load_emb(path: Path) -> dict[str, np.ndarray]:
    keys = json.loads(path.with_suffix(".keys.json").read_text(encoding="utf-8"))
    mat = np.load(path.with_suffix(".vecs.npy"))
    return {k: mat[i] for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# Artifact builder: schema, instance types, stage-1 codes, concept-bound init.
# Built once, cached, so finetune-policy experiments don't re-stream graph.nt.
# ---------------------------------------------------------------------------

def build_artifacts(
    *,
    ontology: Path,
    graph: Path,
    out_dir: Path,
    compare_dir: Path,
    corpus_vocab_pkl: Path,
    kinds=("p2", "p3"),
    dim: int = 200,
    pretrain_epochs: int = 5,
    target_norm: float = 8.0,
    seed: int = 42,
) -> dict:
    """Build & cache everything the init/finetune experiments need.

    Caches under out_dir:
      schema.pkl, itypes.pkl, stage1_<kind>.{keys.json,vecs.npy},
      boundinit_<kind>.{keys.json,vecs.npy}, used_norm_<kind>.json,
      build_stats.json.
    Reuses the protograph walks already generated in compare_dir/protographs.
    """
    import pickle
    from _dbpedia_compare import (
        load_schema, load_instance_types, write_protographs, ensure_walks,
        concept_bound_vectors,
    )
    from _synthetic_compare import pretrain_protograph, normalized_stage1_vectors

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    t0 = time.time()

    schema_pkl = out_dir / "schema.pkl"
    if schema_pkl.is_file():
        schema = pickle.loads(schema_pkl.read_bytes())
    else:
        schema = load_schema(ontology)
        schema_pkl.write_bytes(pickle.dumps(schema))
    stats["schema"] = schema.stats
    print(f"[{time.time()-t0:.0f}s] schema: {schema.stats}", flush=True)

    # protographs + walks (reuse compare_dir's if present)
    proto_dir = compare_dir / "protographs"
    proto_paths = write_protographs(schema, proto_dir)
    proto_walks = {
        k: ensure_walks(p, proto_dir / f"walks_{k}.txt", walks_per_entity=200, depth=3, seed=seed)
        for k, p in proto_paths.items() if k in kinds
    }

    itypes_pkl = out_dir / "itypes.pkl"
    if itypes_pkl.is_file():
        itypes = pickle.loads(itypes_pkl.read_bytes())
    else:
        print(f"[{time.time()-t0:.0f}s] loading instance types (pass over graph.nt) ...", flush=True)
        itypes = load_instance_types(graph, schema)
        itypes_pkl.write_bytes(pickle.dumps(itypes))
    stats["itypes"] = itypes.stats
    print(f"[{time.time()-t0:.0f}s] itypes: {itypes.stats}", flush=True)

    with corpus_vocab_pkl.open("rb") as f:
        vocab = set(pickle.load(f)["freq"].keys())
    print(f"[{time.time()-t0:.0f}s] corpus vocab: {len(vocab)} tokens", flush=True)

    for kind in kinds:
        s1_path = out_dir / f"stage1_{kind}"
        bi_path = out_dir / f"boundinit_{kind}"
        norm_path = out_dir / f"used_norm_{kind}.json"
        if bi_path.with_suffix(".keys.json").is_file() and s1_path.with_suffix(".keys.json").is_file():
            print(f"[{time.time()-t0:.0f}s] {kind}: cached, skip", flush=True)
            continue
        pre = pretrain_protograph(proto_walks[kind], dim=dim, epochs=pretrain_epochs, seed=seed)
        stage1, used_norm = normalized_stage1_vectors(pre.wv, target_norm=target_norm)
        del pre
        save_emb(s1_path, stage1)
        norm_path.write_text(json.dumps({"used_norm": used_norm}), encoding="utf-8")
        print(f"[{time.time()-t0:.0f}s] {kind}: stage1 codes {len(stage1)}, used_norm {used_norm:.3f}; "
              f"building bound init ...", flush=True)
        bound = concept_bound_vectors(
            graph, itypes, stage1, vocab,
            direction_tag="rolled", alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
            target_norm=used_norm,
        )
        save_emb(bi_path, bound)
        norms = np.array([float(np.linalg.norm(v)) for v in bound.values()])
        stats[f"boundinit_{kind}"] = {
            "n": len(bound),
            "norm_mean": float(norms.mean()), "norm_median": float(np.median(norms)),
            "norm_p90": float(np.percentile(norms, 90)), "norm_p99": float(np.percentile(norms, 99)),
            "norm_p999": float(np.percentile(norms, 99.9)), "norm_max": float(norms.max()),
        }
        print(f"[{time.time()-t0:.0f}s] {kind}: bound init {len(bound)} vecs, "
              f"norms median {np.median(norms):.1f} p99 {np.percentile(norms,99):.0f} "
              f"max {norms.max():.0f}", flush=True)
        del bound

    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[{time.time()-t0:.0f}s] artifacts done -> {out_dir}", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Init norm policies (the freeze fix: hub rows get norms up to ~64k under one
# global rescale; at LR 0.0025 they never move. A per-row cap unfreezes hubs
# while keeping the cardinality magnitudes of typical (low-degree) entities.)
# ---------------------------------------------------------------------------

def cap_norms(emb: dict[str, np.ndarray], cap: float) -> dict[str, np.ndarray]:
    """Scale down any row whose norm exceeds `cap` to exactly `cap`; leave
    smaller rows untouched (so 1-edge vs 2-edge cardinality survives below cap)."""
    out = {}
    for k, v in emb.items():
        n = float(np.linalg.norm(v))
        out[k] = (v * (cap / n)).astype(np.float32) if n > cap else v
    return out


def rescale_mean(emb: dict[str, np.ndarray], target: float) -> dict[str, np.ndarray]:
    """One global rescale so the mean row norm == target (preserves all ratios)."""
    norms = np.array([float(np.linalg.norm(v)) for v in emb.values()])
    m = float(norms.mean()) if len(norms) else 1.0
    s = np.float32(target / m if m > 0 else 1.0)
    return {k: v * s for k, v in emb.items()}


# ---------------------------------------------------------------------------
# Finetune-policy experiment: load cached artifacts, init, finetune on a
# (sub-sampled) corpus, eval per epoch under one or more scaling lenses.
# ---------------------------------------------------------------------------

from gensim.models.callbacks import CallbackAny2Vec  # noqa: E402


class _MultiLensEval(CallbackAny2Vec):
    """gensim callback: per-epoch LogReg test accuracy under several scaling lenses."""

    def __init__(self, splits, lenses, rows, label, t0, n_jobs=12, seed=42):
        self.splits, self.lenses, self.rows = splits, lenses, rows
        self.label, self.t0, self.n_jobs, self.seed = label, t0, n_jobs, seed
        self.epoch = 0

    def _eval(self, model):
        from gensim.models import KeyedVectors  # noqa
        kv = model.wv
        m = Model("ft", kv=kv, dim=kv.vectors.shape[1])
        rec = {}
        for lens in self.lenses:
            accs = evaluate_models(self.splits, [m], scaling=lens, seed=self.seed, n_jobs=self.n_jobs)
            rec[lens] = {"accs": accs, "summary": summarize(accs)}
        return rec

    def on_epoch_end(self, model):
        self.epoch += 1
        rec = self._eval(model)
        self.rows.append(rec)
        s = rec[self.lenses[0]]["summary"]
        print(f"    [{time.time()-self.t0:6.0f}s] {self.label} ep{self.epoch}: "
              f"[{self.lenses[0]}] normal {s['normal']:.3f} hard {s['hard']:.3f} all {s['all']:.3f}",
              flush=True)


def finetune_run(
    *,
    name: str,
    artifacts_dir: Path,
    walks: Path,
    corpus_vocab_pkl: Path,
    splits,
    kind: str = "p2",              # which protograph codes (p2/p3) — ignored for vanilla
    init: str = "bound",           # "vanilla" | "classic" | "bound"
    norm_policy: str = "global",   # "global" (mean->target) | "cap:<v>" | "renorm:<v>"
    target_norm: float = 8.0,
    finetune_alpha: float = 0.0025,
    min_alpha: float = 0.0001,
    epochs: int = 5,
    dim: int = 200,
    lenses=("raw", "standardize"),
    seed: int = 42,
    workers: int = 20,
    save_kv: Path | None = None,
) -> dict:
    """One finetune run; returns per-epoch accs (incl. epoch 0) under each lens."""
    import pickle
    from _dbpedia_compare import build_model_with_vocab, classic_init

    t0 = time.time()
    with corpus_vocab_pkl.open("rb") as f:
        cv = pickle.load(f)
    freq, n_lines, n_tokens = cv["freq"], cv["n_lines"], cv["n_tokens"]

    alpha = 0.025 if init == "vanilla" else finetune_alpha
    model = build_model_with_vocab(freq, n_lines, n_tokens, dim=dim, alpha=alpha,
                                   min_alpha=min_alpha, seed=seed, workers=workers)

    used_norm = target_norm
    if init != "vanilla":
        schema = pickle.loads((artifacts_dir / "schema.pkl").read_bytes())
        itypes = pickle.loads((artifacts_dir / "itypes.pkl").read_bytes())
        stage1 = load_emb(artifacts_dir / f"stage1_{kind}")
        used_norm = json.loads((artifacts_dir / f"used_norm_{kind}.json").read_text())["used_norm"]
        bound_vectors = None
        if init == "bound":
            bound_vectors = load_emb(artifacts_dir / f"boundinit_{kind}")
            # the cached bound init is already mean-rescaled to used_norm; re-apply policy
            if norm_policy.startswith("cap:"):
                bound_vectors = cap_norms(bound_vectors, float(norm_policy.split(":")[1]))
            elif norm_policy.startswith("renorm:"):
                tgt = float(norm_policy.split(":")[1])
                bound_vectors = {k: (v / max(float(np.linalg.norm(v)), 1e-9) * tgt).astype(np.float32)
                                 for k, v in bound_vectors.items()}
            elif norm_policy.startswith("global:"):
                bound_vectors = rescale_mean(bound_vectors, float(norm_policy.split(":")[1]))
        init_stats = classic_init(model, stage1, itypes, schema,
                                  target_norm=used_norm, bound_vectors=bound_vectors)
        del bound_vectors
        print(f"  {name} init_stats: {init_stats}", flush=True)

    from gensim.models.word2vec import LineSentence
    rows: list = []
    rows.append(_MultiLensEval(splits, list(lenses), [], "", t0)._eval(model))  # epoch 0
    s0 = rows[0][lenses[0]]["summary"]
    print(f"  {name} ep0(init): [{lenses[0]}] normal {s0['normal']:.3f} hard {s0['hard']:.3f} "
          f"all {s0['all']:.3f}", flush=True)
    cb = _MultiLensEval(splits, list(lenses), rows, name, t0)
    model.train(corpus_iterable=LineSentence(str(walks)), total_examples=n_lines, epochs=epochs,
                start_alpha=alpha, end_alpha=min_alpha, callbacks=[cb])
    if save_kv is not None:
        model.wv.save(str(save_kv))
    out = {
        "name": name, "init": init, "kind": kind, "norm_policy": norm_policy,
        "finetune_alpha": alpha, "epochs": epochs, "seconds": round(time.time() - t0, 1),
        "per_epoch": [{lens: rec[lens]["summary"] for lens in lenses} for rec in rows],
        "final_accs": {lens: rows[-1][lens]["accs"] for lens in lenses},
        "init_accs": {lens: rows[0][lens]["accs"] for lens in lenses},
    }
    del model
    return out
