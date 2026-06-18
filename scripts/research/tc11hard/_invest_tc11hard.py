#!/usr/bin/env python3
"""Investigate WHY the per-row norm cap (@16) HURTS tc11_hard while it helps
almost everywhere else (see thesis Table tab:dbpedia_cap_per_tc, where the
capped bound init drops to 0.608 on tc11_hard vs 0.640 un-capped and 0.720
vanilla; the full-corpus table tab:dbpedia_cap_p3 shows the same: 0.697 -> 0.657).

The analysis is run on the `most_specific` concept-bound variant, the only init
mode for which BOTH the no-cap and the cap@16 finetuned models are saved on disk
(output/dbpedia/p2_bound_specific{,_cap16}), so init -> final drift can be
measured directly. The same qualitative effect (cap hurts tc11_hard) holds for
the `uniform` bound that backs the thesis "bound (global)/cap@16" columns
(0.697 -> 0.658 full corpus) and at the 10% sub-corpus (0.640 -> 0.608).

tc11 = qualified cardinality "Person with >=2 distinct recordLabel.RecordLabel";
tc11_hard's negatives are Persons with *exactly 1* record label -> a pure
*counting* discrimination. tc09 (>=2 award) is the cardinality-family control
where the cap HELPS.

Everything is cached to notebooks/dbpedia_tc11hard/{results.json,arrays.npz} so
the notebook renders instantly and is fully reproducible.

Usage:  .venv/bin/python scripts/_invest_tc11hard.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts"))

from _dbpedia_compare import iter_nt_iris  # noqa: E402
from _evaluate import load_labeled_txt  # noqa: E402

ART = ROOT / "notebooks" / "dbpedia_investigate" / "artifacts"
OUT = ROOT / "notebooks" / "dbpedia_tc11hard"
OUT.mkdir(parents=True, exist_ok=True)
GRAPH = ROOT / "dbpedia_graph" / "graph.nt"

VARIANT = "specific"            # most_specific class weighting (complete model pair on disk)
INIT_TAG = "boundinit_specific_full_p2"
DIR_NOCAP = ROOT / "output" / "dbpedia" / "p2_bound_specific"
DIR_CAP = ROOT / "output" / "dbpedia" / "p2_bound_specific_cap16"
CAP = 16.0

RECLAB = "http://dbpedia.org/ontology/recordLabel"
AWARD = "http://dbpedia.org/ontology/award"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# eval splits
# --------------------------------------------------------------------------- #
def domains(tc: str) -> list[str]:
    root = ROOT / "v1" / "dbpedia" / tc
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_split(tc: str, dom: str, hard: bool):
    sub = "train_test_hard" if hard else "train_test"
    base = ROOT / "v1" / "dbpedia" / tc / dom / "5000" / sub
    if not (base / "train.txt").is_file():
        return None
    tr_t, tr_y = load_labeled_txt(base / "train.txt")
    te_t, te_y = load_labeled_txt(base / "test.txt")
    return tr_t, np.asarray(tr_y), te_t, np.asarray(te_y)


def all_entities(tc: str, hard: bool) -> dict[str, int]:
    """Union of train+test entities -> label, over every domain of a tc."""
    out: dict[str, int] = {}
    for dom in domains(tc):
        s = load_split(tc, dom, hard)
        if s is None:
            continue
        tr_t, tr_y, te_t, te_y = s
        for t, y in zip(tr_t, tr_y):
            out[t] = int(y)
        for t, y in zip(te_t, te_y):
            out[t] = int(y)
    return out


# --------------------------------------------------------------------------- #
# init embedding (uncapped) + cap helper
# --------------------------------------------------------------------------- #
log(f"loading init {INIT_TAG} ...")
ikeys = json.loads((ART / f"{INIT_TAG}.keys.json").read_text())
ik2i = {k: i for i, k in enumerate(ikeys)}
ivec = np.load(ART / f"{INIT_TAG}.vecs.npy", mmap_mode="r")
log(f"init: {len(ikeys):,} vectors, dim {ivec.shape[1]}")


def init_vec(e: str, cap: bool):
    j = ik2i.get(e)
    if j is None:
        return None
    v = np.asarray(ivec[j], dtype=np.float64)
    if cap:
        n = np.linalg.norm(v)
        if n > CAP:
            v = v * (CAP / n)
    return v


# --------------------------------------------------------------------------- #
# 1) init-level standardize-LogReg eval, global vs cap@16, per domain
# --------------------------------------------------------------------------- #
def eval_split_init(tc, dom, hard, cap):
    s = load_split(tc, dom, hard)
    if s is None:
        return None
    tr_t, tr_y, te_t, te_y = s

    def build(toks, ys):
        X, Y = [], []
        for t, y in zip(toks, ys):
            v = init_vec(t, cap)
            if v is not None:
                X.append(v)
                Y.append(int(y))
        return np.asarray(X), np.asarray(Y)

    Xtr, Ytr = build(tr_t, tr_y)
    Xte, Yte = build(te_t, te_y)
    if len(Xte) == 0 or len(set(Ytr)) < 2 or len(set(Yte)) < 2:
        return None
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, Ytr)
    return float(accuracy_score(Yte, clf.predict(Xte)))


def init_eval_table(tcs):
    rows = {}
    for tc, hard in tcs:
        g, c, perdom = [], [], {}
        for d in domains(tc):
            ag = eval_split_init(tc, d, hard, False)
            ac = eval_split_init(tc, d, hard, True)
            if ag is None or ac is None:
                continue
            g.append(ag)
            c.append(ac)
            perdom[d] = {"global": ag, "cap16": ac}
        nm = f"{tc}{'_hard' if hard else ''}"
        rows[nm] = {
            "global": float(np.mean(g)),
            "cap16": float(np.mean(c)),
            "delta": float(np.mean(c) - np.mean(g)),
            "per_domain": perdom,
        }
        log(f"init-eval {nm}: global={np.mean(g):.3f} cap16={np.mean(c):.3f} d={np.mean(c)-np.mean(g):+.3f}")
    return rows


# --------------------------------------------------------------------------- #
# 2) per-entity init norms, AUC(norm), tail fractions, for tc11_hard & tc09_hard
# --------------------------------------------------------------------------- #
def per_entity_norms(tc, hard):
    labs = all_entities(tc, hard)
    ent, y, nrm = [], [], []
    for e, lab in labs.items():
        j = ik2i.get(e)
        if j is None:
            continue
        ent.append(e)
        y.append(lab)
        nrm.append(float(np.linalg.norm(ivec[j])))
    return ent, np.asarray(y), np.asarray(nrm)


def norm_stats(tc, hard):
    ent, y, nrm = per_entity_norms(tc, hard)
    capped = np.minimum(nrm, CAP)
    pos, neg = nrm[y == 1], nrm[y == 0]
    out = {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "med_norm_pos": float(np.median(pos)),
        "med_norm_neg": float(np.median(neg)),
        "frac_tail_all": float((nrm > CAP).mean()),
        "frac_tail_pos": float((pos > CAP).mean()),
        "frac_tail_neg": float((neg > CAP).mean()),
        "auc_norm_pre": float(roc_auc_score(y, nrm)),
        "auc_norm_post_cap": float(roc_auc_score(y, capped)),
    }
    log(f"norm-stats {tc}{'_hard' if hard else ''}: AUC(norm) {out['auc_norm_pre']:.3f} "
        f"tail pos={out['frac_tail_pos']:.1%} neg={out['frac_tail_neg']:.1%}")
    return out, (ent, y, nrm)


# --------------------------------------------------------------------------- #
# 3) degree pass over graph.nt (cached): total degree, recordLabel/award counts
# --------------------------------------------------------------------------- #
def degree_pass(targets: set[str]) -> dict[str, dict]:
    cache = OUT / "degree_cache.json"
    if cache.is_file():
        d = json.loads(cache.read_text())
        if set(d.get("entities", {}).keys()) >= targets:
            log(f"degree: loaded cache ({len(d['entities'])} entities)")
            return d["entities"]
    log(f"degree: streaming graph.nt for {len(targets):,} entities ...")
    deg = {e: 0 for e in targets}
    rl = {e: 0 for e in targets}
    aw = {e: 0 for e in targets}
    n = 0
    for s, r, o in iter_nt_iris(GRAPH):
        n += 1
        if n % 20_000_000 == 0:
            log(f"  degree: {n/1e6:.0f}M triples")
        if s in deg:
            deg[s] += 1
            if r == RECLAB:
                rl[s] += 1
            if r == AWARD:
                aw[s] += 1
        if o in deg:
            deg[o] += 1
    ents = {e: {"deg": deg[e], "reclab": rl[e], "award": aw[e]} for e in targets}
    cache.write_text(json.dumps({"entities": ents}))
    log(f"degree: done {n/1e6:.0f}M triples, cached")
    return ents


# --------------------------------------------------------------------------- #
# 4) freeze/drift: cos(init, final) tail/body x pos/neg, no-cap vs cap
# --------------------------------------------------------------------------- #
def drift(tc, hard, degmap):
    from gensim.models import KeyedVectors

    if not hasattr(drift, "_kv"):
        log("loading finetuned models (no-cap, cap) ...")
        drift._kv = (
            KeyedVectors.load(str(DIR_NOCAP / "model.kv"), mmap="r"),
            KeyedVectors.load(str(DIR_CAP / "model.kv"), mmap="r"),
        )
    kv_nc, kv_cap = drift._kv

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan

    labs = all_entities(tc, hard)
    recs = []  # (label, init_norm, deg, cos_nocap, cos_cap)
    for e, lab in labs.items():
        j = ik2i.get(e)
        if j is None or e not in kv_nc or e not in kv_cap:
            continue
        vi = np.asarray(ivec[j], dtype=np.float64)
        ni = np.linalg.norm(vi)
        vic = vi * (CAP / ni) if ni > CAP else vi
        recs.append((
            lab, ni, degmap.get(e, {}).get("deg", 0),
            cos(vi, np.asarray(kv_nc[e], dtype=np.float64)),
            cos(vic, np.asarray(kv_cap[e], dtype=np.float64)),
        ))
    a = np.asarray(recs, dtype=np.float64)
    y, nrm, deg, cnc, ccap = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    tail = nrm > CAP
    summ = {}
    for grp, gm in [("all", np.ones(len(y), bool)), ("pos", y == 1), ("neg", y == 0)]:
        for seg, sm in [("body", ~tail), ("tail", tail)]:
            m = gm & sm
            summ[f"{grp}_{seg}"] = {
                "n": int(m.sum()),
                "cos_nocap": float(np.nanmean(cnc[m])) if m.sum() else float("nan"),
                "cos_cap": float(np.nanmean(ccap[m])) if m.sum() else float("nan"),
            }
    log(f"drift {tc}{'_hard' if hard else ''}: tail no-cap cos={summ['all_tail']['cos_nocap']:.3f} "
        f"-> cap cos={summ['all_tail']['cos_cap']:.3f}")
    return summ, (y, nrm, deg, cnc, ccap, tail.astype(int))


# --------------------------------------------------------------------------- #
# 5) pull pipeline init/final domain-mean accs from saved json + vanilla ref
# --------------------------------------------------------------------------- #
def dmean(accs, tc, hard):
    v = [accs[k] for k in accs if k.startswith(tc + "/") and (k.endswith("_hard") == hard)]
    return float(np.mean(v)) if v else float("nan")


def _load_lens(d, fn):
    j = json.loads((d / fn).read_text())
    return j["standardize"] if "standardize" in j else j


def pipeline_accs(tcs):
    nc_i, nc_f = _load_lens(DIR_NOCAP, "init_accs.json"), _load_lens(DIR_NOCAP, "final_accs.json")
    cp_i, cp_f = _load_lens(DIR_CAP, "init_accs.json"), _load_lens(DIR_CAP, "final_accs.json")
    ref = json.loads((ROOT / "notebooks" / "clean_bound_full" / "ref_evals.json").read_text())
    van = ref["vanilla (full ref)"]
    rows = {}
    for tc, hard in tcs:
        nm = f"{tc}{'_hard' if hard else ''}"
        rows[nm] = {
            "vanilla": dmean(van, tc, hard),
            "nocap_init": dmean(nc_i, tc, hard),
            "nocap_final": dmean(nc_f, tc, hard),
            "cap_init": dmean(cp_i, tc, hard),
            "cap_final": dmean(cp_f, tc, hard),
        }
    return rows


def n_test(tc, dom, hard):
    sub = "train_test_hard" if hard else "train_test"
    p = ROOT / "v1" / "dbpedia" / tc / dom / "5000" / sub / "test.txt"
    return sum(1 for _ in p.open()) if p.is_file() else 0


def per_domain_and_micro(tc, hard):
    """Per-domain final acc (no-cap & cap) + test sizes, with macro (domain-mean,
    the thesis metric) vs micro (entity-weighted) means. Exposes the small-sample
    domain-averaging artifact behind the tc11_hard 'drop'."""
    nc_f = _load_lens(DIR_NOCAP, "final_accs.json")
    cp_f = _load_lens(DIR_CAP, "final_accs.json")
    perdom = {}
    accs_nc, accs_cp, ns = [], [], []
    for dom in domains(tc):
        key = f"{tc}/{dom}{'_hard' if hard else ''}"
        if key not in nc_f or key not in cp_f:
            continue
        nt = n_test(tc, dom, hard)
        perdom[dom] = {"nocap": nc_f[key], "cap": cp_f[key], "n_test": nt,
                       "delta": cp_f[key] - nc_f[key]}
        accs_nc.append(nc_f[key]); accs_cp.append(cp_f[key]); ns.append(nt)
    accs_nc, accs_cp, ns = np.asarray(accs_nc), np.asarray(accs_cp), np.asarray(ns)
    out = {
        "per_domain": perdom,
        "macro_nocap": float(accs_nc.mean()),
        "macro_cap": float(accs_cp.mean()),
        "macro_delta": float(accs_cp.mean() - accs_nc.mean()),
        "micro_nocap": float((accs_nc * ns).sum() / ns.sum()),
        "micro_cap": float((accs_cp * ns).sum() / ns.sum()),
        "micro_delta": float(((accs_cp - accs_nc) * ns).sum() / ns.sum()),
    }
    # leave-one-domain-out: macro delta with each domain removed (find the driver)
    loo = {}
    for i, dom in enumerate([d for d in domains(tc) if d in perdom]):
        m = np.ones(len(accs_nc), bool); m[i] = False
        loo[dom] = float(accs_cp[m].mean() - accs_nc[m].mean())
    out["macro_delta_leave_one_out"] = loo
    log(f"micro/macro {tc}{'_hard' if hard else ''}: macroΔ={out['macro_delta']:+.3f} "
        f"microΔ={out['micro_delta']:+.3f}")
    return out


def tail_body_final(tc, hard, degmap):
    """Re-fit the standardize-LogReg per domain on the FINAL embeddings, record
    per-test-entity correctness, and split by init-norm tail/body. Shows where the
    cap's effect actually lands (the freeze/drift confound check)."""
    from gensim.models import KeyedVectors
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not hasattr(drift, "_kv"):
        drift._kv = (
            KeyedVectors.load(str(DIR_NOCAP / "model.kv"), mmap="r"),
            KeyedVectors.load(str(DIR_CAP / "model.kv"), mmap="r"),
        )
    kv_nc, kv_cap = drift._kv

    def per_entity_correct(kv):
        out = {}
        for dom in domains(tc):
            s = load_split(tc, dom, hard)
            if s is None:
                continue
            trt, trY, tet, teY = s

            def emb(toks, Y):
                X, YY, E = [], [], []
                for t, y in zip(toks, Y):
                    if t in kv:
                        X.append(np.asarray(kv[t], dtype=np.float64)); YY.append(int(y)); E.append(t)
                return np.asarray(X), np.asarray(YY), E

            Xtr, Ytr, _ = emb(trt, trY)
            Xte, Yte, Ete = emb(tet, teY)
            if len(Xte) == 0 or len(set(Ytr)) < 2:
                continue
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
            clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, Ytr)
            pred = clf.predict(Xte)
            for e, yt, pr in zip(Ete, Yte, pred):
                j = ik2i.get(e)
                if j is None:
                    continue
                out[e] = (int(yt == pr), float(np.linalg.norm(ivec[j])) > CAP)
        return out

    cnc, ccap = per_entity_correct(kv_nc), per_entity_correct(kv_cap)
    common = set(cnc) & set(ccap)
    a = np.asarray([(cnc[e][0], ccap[e][0], cnc[e][1]) for e in common])
    tail = a[:, 2].astype(bool)
    res = {}
    for seg, m in [("all", np.ones(len(a), bool)), ("body", ~tail), ("tail", tail)]:
        s = a[m]
        res[seg] = {
            "n": int(len(s)),
            "nocap_acc": float(s[:, 0].mean()) if len(s) else float("nan"),
            "cap_acc": float(s[:, 1].mean()) if len(s) else float("nan"),
        }
    log(f"tail/body {tc}{'_hard' if hard else ''}: all microΔ={res['all']['cap_acc']-res['all']['nocap_acc']:+.3f}")
    return res


# --------------------------------------------------------------------------- #
# run everything
# --------------------------------------------------------------------------- #
def main():
    tcs = [("tc11", False), ("tc11", True), ("tc09", False), ("tc09", True),
           ("tc04", False), ("tc01", True)]

    results = {
        "meta": {
            "variant": VARIANT,
            "init_tag": INIT_TAG,
            "cap": CAP,
            "dir_nocap": str(DIR_NOCAP.relative_to(ROOT)),
            "dir_cap": str(DIR_CAP.relative_to(ROOT)),
            "note": "most_specific bound; same cap-hurts-tc11_hard effect as the uniform "
                    "bound in thesis Table tab:dbpedia_cap_per_tc",
        }
    }

    log("== pipeline init/final accs ==")
    results["pipeline_accs"] = pipeline_accs(tcs)

    log("== init-level standardize-LogReg, global vs cap ==")
    results["init_eval"] = init_eval_table(tcs)

    log("== per-entity norm stats ==")
    norm_arrays = {}
    results["norm_stats"] = {}
    for tc in ("tc11", "tc09"):
        st, arr = norm_stats(tc, True)
        results["norm_stats"][f"{tc}_hard"] = st
        ent, y, nrm = arr
        norm_arrays[f"{tc}_hard_y"] = y
        norm_arrays[f"{tc}_hard_norm"] = nrm

    log("== degree pass ==")
    targets = set(all_entities("tc11", True)) | set(all_entities("tc09", True))
    degmap = degree_pass(targets)

    # degree correlations
    results["degree_corr"] = {}
    for tc, relkey, relname in [("tc11", "reclab", "recordLabel"), ("tc09", "award", "award")]:
        labs = all_entities(tc, True)
        es = [e for e in labs if e in ik2i and e in degmap]
        nrm = np.asarray([np.linalg.norm(ivec[ik2i[e]]) for e in es])
        dg = np.asarray([degmap[e]["deg"] for e in es])
        rc = np.asarray([degmap[e][relkey] for e in es])
        yy = np.asarray([labs[e] for e in es])
        pos_rc = rc[yy == 1]
        neg_rc = rc[yy == 0]
        results["degree_corr"][f"{tc}_hard"] = {
            "rel": relname,
            "spearman_norm_degree": float(spearmanr(nrm, dg).correlation),
            f"spearman_norm_{relname}count": float(spearmanr(nrm, rc).correlation),
            "rel_count_pos_median": float(np.median(pos_rc)),
            "rel_count_neg_median": float(np.median(neg_rc)),
            "rel_count_pos_ge2_frac": float((pos_rc >= 2).mean()),
            "rel_count_neg_eq1_frac": float((neg_rc == 1).mean()),
            "deg_pos_median": float(np.median(dg[yy == 1])),
            "deg_neg_median": float(np.median(dg[yy == 0])),
        }
        log(f"degree-corr {tc}_hard: norm~deg={results['degree_corr'][f'{tc}_hard']['spearman_norm_degree']:.3f} "
            f"norm~{relname}={results['degree_corr'][f'{tc}_hard'][f'spearman_norm_{relname}count']:.3f}")

    log("== freeze/drift ==")
    results["drift"] = {}
    drift_arrays = {}
    for tc in ("tc11", "tc09"):
        summ, arr = drift(tc, True, degmap)
        results["drift"][f"{tc}_hard"] = summ
        y, nrm, deg, cnc, ccap, tail = arr
        drift_arrays[f"{tc}_hard_drift"] = np.column_stack([y, nrm, deg, cnc, ccap, tail])

    # -- corrected: the "drop" is a small-domain + lens + variant artifact --------
    log("== per-domain / micro-vs-macro (the domain-averaging artifact) ==")
    results["per_domain_final"] = {f"{tc}_hard": per_domain_and_micro(tc, True) for tc in ("tc11", "tc09")}

    log("== lens sensitivity (standardize vs raw) ==")
    nc_all = json.loads((DIR_NOCAP / "final_accs.json").read_text())
    cp_all = json.loads((DIR_CAP / "final_accs.json").read_text())
    results["lens_sensitivity"] = {}
    for tc in ("tc11", "tc09"):
        row = {}
        for lens in ("standardize", "raw"):
            dn = dmean(nc_all[lens], tc, True)
            dc = dmean(cp_all[lens], tc, True)
            row[lens] = {"nocap": dn, "cap": dc, "delta": dc - dn}
        results["lens_sensitivity"][f"{tc}_hard"] = row
        log(f"lens {tc}_hard: standardizeΔ={row['standardize']['delta']:+.3f} rawΔ={row['raw']['delta']:+.3f}")

    log("== cross-variant tc11_hard delta (uniform / specific / decay) ==")
    ref = json.loads((ROOT / "notebooks" / "clean_bound_full" / "ref_evals.json").read_text())

    def dm_std(d):
        j = _load_lens(ROOT / "output" / "dbpedia" / d, "final_accs.json")
        return dmean(j, "tc11", True)

    variants = {}
    # uniform: no-cap from ref_evals, cap from p2bound_cap16
    uni_nc = dmean(ref["p2_bound (uniform, no-cap)"], "tc11", True) if "p2_bound (uniform, no-cap)" in ref else float("nan")
    variants["uniform"] = {"nocap": uni_nc, "cap": dm_std("p2bound_cap16"), "delta": dm_std("p2bound_cap16") - uni_nc}
    variants["most_specific"] = {"nocap": dm_std("p2_bound_specific"), "cap": dm_std("p2_bound_specific_cap16"),
                                 "delta": dm_std("p2_bound_specific_cap16") - dm_std("p2_bound_specific")}
    variants["decay"] = {"nocap": dm_std("p2_bound_decay"), "cap": dm_std("p2_bound_decay_cap16"),
                         "delta": dm_std("p2_bound_decay_cap16") - dm_std("p2_bound_decay")}
    results["variant_sensitivity"] = variants
    for k, v in variants.items():
        log(f"variant {k}: nocap={v['nocap']:.3f} cap={v['cap']:.3f} Δ={v['delta']:+.3f}")

    log("== tail/body final-acc decomposition (freeze/drift confound check) ==")
    results["tail_body_final"] = {f"{tc}_hard": tail_body_final(tc, True, degmap) for tc in ("tc11", "tc09")}

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(OUT / "arrays.npz", **norm_arrays, **drift_arrays)
    log(f"WROTE {OUT/'results.json'} and {OUT/'arrays.npz'}")
    print("\n==== SUMMARY ====")
    print(json.dumps(results["pipeline_accs"]["tc11_hard"], indent=2))
    print(json.dumps(results["norm_stats"]["tc11_hard"], indent=2))
    print(json.dumps(results["drift"]["tc11_hard"], indent=2))


if __name__ == "__main__":
    main()
