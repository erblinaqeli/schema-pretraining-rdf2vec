# Why we normalize the protograph init (and what "normalize" means here)

This note explains the claim in [notebooks/synthetic.ipynb Part 2](notebooks/synthetic.ipynb)
that the transferred protograph vectors must be **normalized at initialization**. It defines every
term it uses and works through concrete examples. The implementation lives in
[scripts/_synthetic_compare.py](scripts/_synthetic_compare.py).

---

## 1. Terms used in this note

Read these once; the rest of the document leans on them.

- **Protograph.** A small graph built only from the *ontology* (classes + relations + the
  `subClassOf` hierarchy), with no instance data. Pretraining on walks over the protograph teaches
  the model the *schema*, independent of any particular entity.

- **Instance walks.** Random walks over the real data graph `graph.nt` (the entities and their
  edges). This is what we *finetune* on.

- **Skip-gram / SGNS.** The word2vec training objective (Skip-Gram with Negative Sampling). It keeps
  two matrices: `wv.vectors` (the "center" vectors you read out as embeddings) and `syn1neg` (the
  "context" vectors used during training). Training nudges these so that tokens appearing together in
  a walk get a high dot product, and random ("negative") pairs get a low one.

- **Sigmoid `σ`.** The squashing function `σ(x) = 1 / (1 + e^(−x))`. SGNS pushes `σ(center · context)`
  toward 1 for real co-occurrences and toward 0 for negatives. The **gradient** of each update is
  proportional to `(label − σ(center · context))`.

- **Sigmoid saturation.** When `center · context` is large (because the vectors are long and aligned),
  `σ(·)` is already ≈ 1, so `(label − σ)` ≈ 0 and **the update is almost zero**. A "saturated" vector
  barely moves during training. We use this *on purpose* to protect a good initialization.

- **L2 norm `‖v‖`.** The length of a vector, `sqrt(Σ vᵢ²)`. "Unit-normalize" means rescale a vector
  to length 1 (`v / ‖v‖`). "Normalize to norm 8" means rescale to length 8 (`8 · v / ‖v‖`).

- **Hadamard product `⊙`.** Elementwise multiplication of two vectors: `(a ⊙ b)ᵢ = aᵢ · bᵢ`. We use it
  to *bind* a relation to a neighbour class into a single vector — see "code" vs "component" below.

- **`roll`.** Cyclically shift a vector's entries by one position (`numpy.roll(v, 1)`). Applying it to a
  relation code gives a *different* vector for the same relation, which is how we distinguish an
  **outgoing** edge (`e —r→ o`) from an **incoming** one (`s —r→ e`).

---

## 2. The two building blocks: **code** vs **component**

These are the two words that cause the most confusion, so they get their own section.

### A **code** is one pretrained vector for one schema symbol

After pretraining skip-gram on the protograph walks, you get **exactly one vector per class and per
relation**. Each such vector is a **code** (as in "codeword" — a fixed-length encoding of one symbol).

```
code(C_390)  →  the vector for the class    C_390
code(P_19)   →  the vector for the relation P_19
```

Codes are the **atoms** of the construction. There is nothing smaller. The same code is reused across
every entity that touches that class or relation.

### A **component** is one additive term in an entity's init, built *out of* codes

An entity `e`'s initialization vector is a **sum of terms**. Each term is a **component**, and each
component is assembled by combining codes (via `⊙`, `roll`, averaging). The full formula:

```
init(e) = α · ⟦ mean of e's own class codes ⟧                            ← own-class component
        + β · Σ ⟦ code(r) ⊙ code(c) ⟧      over edges e —r→ o, c ∈ cls(o) ← outgoing components
        + γ · Σ ⟦ roll(code(r)) ⊙ code(c) ⟧ over edges s —r→ e, c ∈ cls(s) ← incoming components
        + δ · Σ ⟦ code(r) ⟧                 over every edge touching e      ← bare-relation components
```

with `α = β = γ = δ = 1`. The `⟦ ⟧` brackets mean "unit-normalize this component" (the "hats" in the
notebook formula). `cls(o)` is the set of classes of `o`, **ancestor-materialized** — i.e. if `o` is a
`C_390`, it also counts as every superclass of `C_390`, so deep leaf types still project onto the
discriminative superclass.

| component | built from | reads as |
|---|---|---|
| own-class (α) | mean of `e`'s own class codes | "what kind of thing `e` is" |
| outgoing (β)  | `code(r) ⊙ code(c)` per edge `e —r→ o`, per class `c` of `o` | "`e` has an `r`-edge to a `c`-thing" |
| incoming (γ)  | `roll(code(r)) ⊙ code(c)` per edge `s —r→ e` | "a `c`-thing points at `e` via `r`" |
| bare-relation (δ) | `code(r)` per incident edge | "`e` participates in relation `r`" |

**One-line distinction:** a *code* is a reusable atom (one per ontology symbol); a *component* is one
term in one entity's sum, made by binding codes together. Normalization ("the hat") is applied **per
component**, not per code.

### Worked example

Entity `e` is typed `C_152`. It has two edges, `e —P_19→ o₁` and `e —P_19→ o₂`, where both `o₁` and
`o₂` are typed `C_390`.

- **Codes** involved (3 atoms): `code(C_152)`, `code(P_19)`, `code(C_390)`.
- **Components** built for `init(e)`:
  - own-class: `⟦ code(C_152) ⟧` — once
  - outgoing: `⟦ code(P_19) ⊙ code(C_390) ⟧` — added **twice**, once per edge
  - bare-relation: `code(P_19)` — added twice
- Because the outgoing component is added twice, `init(e)` is **~2× longer** in that direction than it
  would be for an entity with a single `P_19 → C_390` edge. **That magnitude is the cardinality
  signal** that the counting test cases (tc09–tc12) need. We must not destroy it later.

---

## 3. Where normalization happens, and in what order

There are **two** places, and they are easy to conflate. Keep them separate.

### Stage A — normalize the **codes** ([`normalized_stage1_vectors`](scripts/_synthetic_compare.py#L378))

Every pretrained code is rescaled to **norm 8** (`TARGET_NORM = 8`): unit-normalize, then multiply by
8. After this stage, *every individual code sits at length 8.*

```python
v = vecs[idx] * (target_norm / n)   # each code → length 8
```

This is the dict the **classic** init (`*_classic` variants) consumes directly.

### Stage B — build the **bound** init ([`concept_bound_vectors`](scripts/_synthetic_compare.py#L403))

The concept-bound init (`*_bound` variants) re-normalizes **per component to unit length** while
assembling the sum (the `unit()` calls in the code):

- each looked-up code → `unit()` → length 1
- the own-class mean → `unit()` → length 1
- **each** `code(r) ⊙ code(c)` pair → `unit()` → length 1

So in the bound path the Stage-A norm-8 does *not* carry through — what enters the sum is a set of
**unit-length component vectors**. They are then **accumulated** (`acc[ent] += w · vec`, weights = 1),
so repeated `(r, c)` pairs add the same unit vector multiple times and the entity's vector grows in
that direction (the cardinality magnitude from the example above).

Finally, **one global rescale** is applied so the **mean** entity norm becomes 8 — a *single shared
scalar*, not a per-vector normalization ([lines 516–518](scripts/_synthetic_compare.py#L516)):

```python
mean_norm = mean(‖v‖ for all entities)
scale     = 8 / mean_norm        # ONE number, applied to everyone
return {ent: v * scale for ent, v in acc.items()}
```

A 2-edge entity stays longer than a 1-edge entity after this step — exactly what we want.

> **Corrected mental model.** It is **not** "each code is unit-normed then the sum is normed to 8."
> It is: *each **component** is unit-normed → components are summed with accumulation → one global
> scalar sets the **mean** entity norm to 8.*

---

## 4. Why normalize at all — "normalization matters twice"

This is Finding #4 of the notebook. It is one idea (control vector length deliberately) applied for
two distinct reasons.

### Reason 1 — strip pretrain-frequency artifacts

A skip-gram vector's **length grows with how often its token appeared** during pretraining. A class
that shows up in many protograph walks ends up with a longer code than a rare class — an artifact of
the walk distribution, **not** of meaning. If you feed those raw codes into the Hadamard sums, the
spurious lengths distort the binding. Unit-normalizing first makes only the **direction** (the
semantics) count, so a frequent class and a rare class contribute on equal footing.

### Reason 2 — saturate the SGNS sigmoid so finetuning can't overwrite the init ("protection")

We pretrained to put schema knowledge into the init. We do **not** want a few epochs of instance-walk
SGD to erase it back toward the vanilla solution. The lever is vector length:

- We set the init to length 8 in **both** `wv.vectors` **and** `syn1neg`.
- For any consistent context pair, `center · context` is then large → `σ(·)` saturates near 1 →
  gradient `(label − σ)` ≈ 0 → **the update nearly vanishes**.
- Measured effect: each update is ≈ `LR · 8 = 0.0025 · 8 ≈ 0.02` against rows of length 8, so after 5
  epochs `mean cos(epoch0, epoch5) ≈ 0.995`. **The init barely moves.**

This is "protected finetuning": large, uniform norms turn finetuning into a near-no-op that *preserves*
the transferred signal.

**Control that proves the mechanism.** The same class-mean init finetuned at the *from-scratch* LR
`0.025` instead *un-freezes* (`cos ≈ 0.66`) and drifts back to vanilla quality (tc09: 0.623 vs vanilla
0.635). So the frozen behavior is caused by the protective LR + norm, not by a pipeline bug.

### The critical caveat — the final rescale must be **global**, never per-vector

The second half of the "matters twice" finding is a **warning**. The bound init deliberately encodes
**counts in magnitude** (the accumulated components). Therefore:

- ✅ rescale the **whole set** by one shared factor so the *mean* norm is 8 (keeps protection, keeps
  relative lengths),
- ❌ never unit-normalize each entity vector to 8 individually — that would put a 1-edge and a 2-edge
  entity on the same sphere and **erase the cardinality feature** (tc09–tc12).

---

## 5. Scaling to real graphs: the per-row norm cap (`cap_norms`)

Everything above is the **synthetic** recipe ([scripts/_synthetic_compare.py](scripts/_synthetic_compare.py),
[notebooks/synthetic.ipynb Part 2](notebooks/synthetic.ipynb)), where the single global rescale
of §3 is all you need. The reason it suffices is that on the synthetic graph every entity has a
*similar* number of edges, so the per-entity norms are tightly clustered: setting the **mean** to 8
also keeps **every** entity near 8 — uniform enough to protect, with no entity wildly over-protected.

On **real DBpedia** that assumption breaks, and a second normalization step is needed: `cap_norms`,
introduced in [scripts/_dbpedia_investigate.py](scripts/_dbpedia_investigate.py#L353) and used by the
full-corpus ablation in [notebooks/clean_bound_full.ipynb](notebooks/clean_bound_full.ipynb).

### The issue it solves — a heavy-tailed degree distribution freezes hub entities

The bound init **accumulates** one component per incident edge (§2). On a real graph the degree
distribution is heavy-tailed: most entities have a handful of edges, but **hub** entities (popular
countries, common types, "United States"-style nodes) have *thousands*. Their init vectors therefore
accumulate thousands of components and end up with **enormous norms — up to ~64k** after the global
rescale.

The global rescale only fixes the **mean** norm; it does nothing about the **spread**. So while the
typical entity sits near the target, hubs sit thousands of times higher — and that collides head-on
with the protection mechanism of §4 (Reason 2):

- Protection works *because* a length-8 row saturates the sigmoid → gradient ≈ 0 → the row barely
  moves. That is the goal.
- A length-**64000** row is saturated against *everything*, so its gradient is **exactly** 0 → it
  **never moves at all**. That is no longer protection, it is **freezing**.

A frozen hub cannot be corrected by finetuning **even where the instance corpus disagrees with its
init**. Concretely this blocks the recovery that vanilla gets for free on the individual-`{e}` family
(tc04–tc06): vanilla *learns* those hubs from the corpus, but the bound model's hubs are stuck at their
init (which carries no `{e}`-task signal). On full DBpedia this drags the no-cap bound variants
**below** vanilla (decay 0.811, most_specific 0.806 vs vanilla 0.823), purely from the frozen `{e}`
family (`indiv-H` 0.772 vs vanilla 0.869).

### The fix — clamp the tail, leave the body untouched

```python
def cap_norms(emb, cap):                       # scripts/_dbpedia_investigate.py
    for k, v in emb.items():
        n = ‖v‖
        out[k] = v * (cap / n) if n > cap else v   # only rows LONGER than cap shrink
    return out
```

`cap_norms(emb, cap=16)` rescales **only the rows whose norm exceeds the cap (16)** down to exactly 16,
and **leaves every shorter row exactly as it was**. It is applied *after* the global mean-rescale
([_invest_clean_bound_full.py](scripts/_invest_clean_bound_full.py#L192) via `norm_policy="cap:16"`),
and it threads the needle between the two failure modes:

- **Hubs are unfrozen.** A norm-16 row is still protected (fairly saturated) but **not** dead — at LR
  `0.0025` it can move again when the corpus pushes on it, so the `{e}` family recovers.
- **Cardinality survives.** The whole point of accumulation (§2 worked example) is that a 2-edge entity
  is longer than a 1-edge entity. Those typical, low-degree entities have norms **well below 16**, so
  the cap **never touches them** — the 1-edge-vs-2-edge magnitude (tc09–tc12) is preserved. This is why
  the cap clamps *only the tail* instead of L2-normalizing every row, which would repeat the §4
  never-per-vector mistake and erase the counts.

> **Why a cap and not just a smaller global target?** Lowering the global rescale target would shrink
> the hubs but also crush the cardinality magnitudes of normal entities toward zero. The cap is
> *non-uniform on purpose*: it touches only the pathological tail and leaves the part of the
> distribution that actually carries signal alone.

### What it buys — the cap is the decisive ingredient (full-corpus result)

In the full-DBpedia ablation, the per-row cap, **not** the class-weighting scheme, is what carries the
method past vanilla:

| variant | final acc | vs vanilla |
|---|---|---|
| vanilla (full ref) | 0.823 | — |
| `p2_bound_decay` (no cap) | 0.811 | −0.012 |
| `p2_bound_specific` (no cap) | 0.806 | −0.017 |
| **`p2_bound_decay_cap16`** | **0.860** | **+0.037** |
| `p2_bound_specific_cap16` | 0.849 | +0.026 |

Adding the cap lifts decay **+0.049** (0.811 → 0.860) and most_specific **+0.043** (0.806 → 0.849) —
every bound variant goes from *below* vanilla to clearly above it. The weighting scheme (decay vs
most_specific) is worth ~1 point on top; **the cap is worth ~5.**

### Related alternative — compress the tail at the source

The same hub-tail problem can instead be tamed *before* the global rescale:
[`compress="sqrt"` / `"log1p"`](scripts/_dbpedia_compare.py) applies a per-row monotone squashing of the
norm during the build, pulling the heavy tail in at the source so that (per its docstring) "the
post-hoc per-row cap is no longer needed." `cap_norms` (a hard ceiling) and `compress` (a soft monotone
squash) are two solutions to the **same** frozen-hub issue; the `clean_bound_full` ablation uses the cap.

---

## 6. End-to-end summary

1. **Pretrain** skip-gram on protograph walks → one **code** (vector) per class and per relation.
2. **Stage A:** rescale every code to norm 8 — removes pretrain-frequency noise; this set feeds the
   classic init.
3. **Stage B (bound init):** for each entity, build **components** (own-class mean; `code(r) ⊙ code(c)`
   per outgoing edge; `roll(code(r)) ⊙ code(c)` per incoming edge; bare `code(r)`), **unit-normalize
   each component**, and **sum with accumulation** (repeats add → magnitude encodes counts).
4. **Global rescale:** one shared scalar sets the **mean** entity norm to 8 — large enough to saturate
   the SGNS sigmoid (so finetuning preserves the init), uniform enough to be frequency-neutral, and
   *not* per-vector (so cardinality magnitudes survive).
5. **Protected finetune** on instance walks at LR `0.0025` with the init mirrored into `syn1neg` — the
   curves stay within a few points of the epoch-0 init.
6. **On a real graph (DBpedia), add a per-row norm cap.** A heavy-tailed degree distribution gives hub
   entities norms up to ~64k after the global rescale, which *freezes* them at the protected LR;
   `cap_norms(·, 16)` clamps only those over-long rows back to 16 and leaves the rest untouched —
   unfreezing the hubs while preserving the cardinality magnitudes. The decisive ingredient on full
   DBpedia (§5).

**The unifying principle:** in SGNS, **vector length is not free** — it controls both how much
pretrain frequency leaks in and how strongly the init resists finetuning. So we set it deliberately at
initialization rather than letting training, or a careless per-vector normalization, decide it for us.
On a real, heavy-tailed graph the same principle adds one rule: clamp the pathological norm tail (the
hub freeze) without flattening the rest (the cardinality signal) — §5.
