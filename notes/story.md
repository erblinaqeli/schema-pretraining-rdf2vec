# Storytelling Plot — *Schema-Based Pre-Training for RDF2Vec*

A story map of the thesis. The spine is a single claim that every experiment
sharpens:

> **The value of schema pre-training for RDF2Vec is set not by *how much* schema
> you pretrain, but by *what the initialization encodes* — and, on real graphs,
> by whether the fine-tuning policy lets that encoding survive.**

Each beat below gives the **motivation** (why we ran it / the open question) and
the **conclusion** (what it settled), so the through-line is visible end to end.

---

**Note — DLCC test-case families (which TC tests what).** The results lean on
three families, grouped by *what decides the label*. The focus families are
**relation-centric**, **cardinality-centric**, and **class-centric**; tc01–tc06
are baseline **existence** tasks outside the three.

| TC   | Family                | Constructor         | Label decided by                       |
| ---- | --------------------- | ------------------- | -------------------------------------- |
| tc01 | Existence             | ∃r.⊤                | has an outgoing r-edge                  |
| tc02 | Existence             | ∃r⁻.⊤               | has an incoming r-edge                  |
| tc03 | Existence             | ∃r.⊤ ⊔ ∃r⁻.⊤        | has an r-edge in either direction       |
| tc04 | Enumerated-individual | ∃R.{e} ⊔ ∃R⁻.{e}    | connection to one specific entity e     |
| tc05 | Enumerated-individual | ∃R1.(∃R2.{e}) ⊔ inv | two-hop path to one specific entity e   |
| tc06 | Enumerated-individual | ∃r.{e}              | relation r to one specific entity e     |
| tc07 | Relation-centric      | ∃r.T                | a neighbour's class on an outgoing edge |
| tc08 | Relation-centric      | ∃r⁻.T               | a neighbour's class on an incoming edge |
| tc09 | Cardinality           | ≥2r.⊤               | ≥2 outgoing r-edges                     |
| tc10 | Cardinality           | ≥2r⁻.⊤              | ≥2 incoming r-edges                     |
| tc11 | Cardinality           | ≥2r.T               | ≥2 outgoing r-edges to type-T           |
| tc12 | Cardinality           | ≥2r⁻.T              | ≥2 incoming r-edges from type-T         |
| tc13 | Class-centric         | D ⊓ ∃p.R1 vs R2     | object's range-side subclass (R1 vs R2) |
| tc14 | Class-centric         | D1 vs D2 ⊓ ∃p.R     | the entity's own (domain-side) class    |
| tc15 | Class-centric         | ∃p.R_sub vs ∃p.W    | filler in-range (valid) vs out-of-range |

## Act 1 — Port MASCHInE to RDF2Vec ("classic transfer") … and watch it fail

- **What we built.** Two protographs of increasing schema content: **P1**
  (one prototriple per relation from declared domain/range) and **P2** (P1 +
  one-sided direct-subclass substitutions). Pre-train skip-gram on protograph
  walks → class & relation codes. Transfer: seed each typed instance from its
  class code (MASCHInE `most_specific` rule), then fine-tune on instance walks.
  (A third protograph, **P3**, that bakes in the *full* subClassOf hierarchy as
  explicit bidirectional edges enters later, in Act 1.6, as the decisive coverage
  control.)
- **Motivation.** This is the faithful RDF2Vec adaptation of MASCHInE — the
  natural first thing to try, and the baseline everything later must beat.
- **The headline failure (synthetic).** The classic transfer matches `vanilla`
  on existence tasks (tc01–06) and **adds essentially nothing** on the
  relation-centric tasks (tc07–12, stuck ~0.54–0.69). It clearly helps on **one**
  case, tc14, where the label *is* the entity's own class (reaches ~1.0 already
  at initialization).
- **Why it fails — four structural reasons** (each later becomes a repair):
  1. **Own-class init carries no signal for relation-centric tasks** — on
     tc07–12 positives and negatives share the same own class; the deciding
     info is in *neighbours'* classes, which the rule never touches.
     - *Why we judge it:* the labels are *independent of own class* — on tc07
       every labeled entity (positive and negative) is the same leaf class, so
       any own-class rule maps them to **one** code. Act 3's LDA diagnostic shows
       classic collapsing tc07's 400 test entities onto just **two** init
       vectors, curves flat at chance from epoch 0 (no separable signal to
       preserve — *not* a freezing artifact).
     - *Control — the noise test (it's an information gap, not a tie artifact):*
       the collapse maps opposite-label instances to the *same* vector, but the
       barrier isn't the coincidence itself. **Sweep random init noise σ=0…0.5**
       and the collapsed vectors become distinct — yet **held-out epoch-0 accuracy
       stays at chance** (~0.49 across the sweep on tc07–12). Jitter can't
       manufacture a distinction the init never encoded, so the class-mean init
       carries **no label-aligned separation at all**: its only axis is own-class,
       which both labels share. (Act 4 reuses this same sweep in a second role —
       to rule out a generic-regularization reading of the bound method.)
     - *Where & how addressed:* the **β/γ terms** of concept-bound (Act 2) inject
       the *neighbours'* (r, class) pairs the rule never sees; Act 3 lifts
       tc07–12 from ~0.62 → **0.905**.
  2. **Skip-gram summation destroys relation–class binding** — an SGNS vector is
     a bag of context vectors, so "∃r.C" (a *pair*) is erased by plain summation.
     - *Why we judge it:* an explicit **bag-of-(relation × object-class) oracle
       reaches 1.00 on tc07 while the *unbound* bag saturates at 0.75** — the
       features are present, only the *binding* is missing.
     - *Where & how addressed:* the **Hadamard binding** in β/γ (Act 2) makes
       (r, C) pairs quasi-orthogonal so a linear probe can read them; Act 3
       confirms the separability margin exists *before* fine-tuning.
  3. **Coverage gaps** — instances are typed with deep leaf classes absent from
     P1/P2, so they fall back to random init.
     - *Why we judge it:* P1 holds only declared domain/range classes and P2 only
       their direct subclasses, while instances carry deep *leaf* types — the
       per-class init-coverage table (results chapter) shows most leaf lookups
       reverting to random vectors.
     - *Where & how addressed:* **Act 1.6** settles it — climbing strategies and
       full-coverage **P3** all reach 100% yet none beats `most_specific`, ruling
       coverage *count* out as the driver. Coverage re-enters only *inside* the
       binding (Act 3: P2/P3 beat P1 on qualified tc07/11/12, since β/γ can bind
       only classes that have pretrained codes).
  4. **Magnitude is fragile** — cardinality (counting edges) can only live in the
     *norm*, which per-vector normalization erases and free fine-tuning randomizes.
     - *Why we judge it:* a count survives only as magnitude (‖1 edge‖ = 1.00 vs
       ‖2 edges‖ = 2.00); per-vector normalization maps both to 1.00, making
       tc09–12 unsolvable *from the init* — Act 3 shows classic clamps every
       entity to norm 8 (no count info).
     - *Where & how addressed:* the **δ term** accumulates relation codes so
       counts survive as magnitude, kept by the **global** (not per-vector)
       target-norm precondition (Act 2); the **direction-aware cardinality
       ablation** (Act 4) confirms the count — and its direction — is the
       mechanism (exact-count oracle confirmed).
- **A rival hypothesis — forgetting — deferred to Act 1.5.** The four reasons say
  the init is *task-useless to begin with*; a reader might object that the real
  culprit is instead **forgetting** — the transferred vectors *do* drift hard in
  fine-tuning, so perhaps a real signal simply gets washed out. We take that
  hypothesis up head-on in **Act 1.5**, where the two policy fixes that stop the
  drift both fail to move tc07–12 — establishing that forgetting is real but only
  bites *where the init already carries the answer*, never on the relation-centric
  tasks.
- **Conclusion.** The classic transfer is a *better starting point* with **no
  lasting accuracy gain**. The failure is structural, not a tuning accident
  (confirmed later by grid/LR/walk searches that never move tc07–12).

## Act 1.5 — Try to save classic: rule out forgetting and anchoring

- **Motivation — forgetting is real, so test it first.** The init is now the only
  suspect left — but before blaming *what* it encodes, rule out the cheaper
  hypothesis: maybe classic's signal *is* there and merely gets **lost in
  fine-tuning**. The drift is real and large: classic retains far less of its init
  than bound will (cosine <0.2, forgetting visible entity by entity; the Act 3
  drift plots make it concrete). So "the signal was there but washed out" is a
  live suspect — if true, a gentler fine-tuning *policy*, not a new init, would
  rescue it. We test the two natural policy fixes on the *classic* transfer,
  varying one knob at a time off a fixed comparison budget (dim 200, 100 instance
  walks/entity, depth 3, 5+5 epochs, codes unit-normalized → target norm 8, seed
  42; `vanilla` and the un-rescued transfer as the reference anchors).
- **Rescue 1 — slow the drift (lower the fine-tuning LR).** Drop the from-scratch
  rate 0.025 → a protected 0.0025 (1/10). *Result:* it does **not** rescue classic
  — it *underfits*. tc07–12 stays flat at **both** rates (~0.45–0.70), and the low
  rate even *hurts* the rest (`p1_classic` 0.757→0.609; P2/P3 roughly flat). The
  only task protection helps is tc14, where the init already *is* the answer
  (1.000 vs 0.986). At its *best* rate classic merely **ties** vanilla
  (~0.76 vs 0.736).
- **Rescue 2 — pin to the schema (protograph anchoring).** Add an L2 pull-back to
  the schema init, `λ‖v_e − v⁰_e‖²`, applied as a proximal step at the end of each
  fine-tune epoch (`v_e ← (1−λ)·v_e + λ·v⁰_e`); λ=0 is the plain transfer, λ=1
  resets every instance to its class code each epoch. *Result:* mean accuracy
  **decreases monotonically in λ on every protograph** (P1 0.740→0.710, P2
  0.768→0.695, P3 0.738→0.668). It helps on **exactly one** task (tc14) and drags
  everything else below vanilla at high λ. Set λ=0.
- **Why both fail — and why that matters.** Both fixes attack *drift*, and both
  miss the relation-centric tasks completely. That is the decisive evidence that
  classic's tc07–12 failure is **not** a forgetting problem: by the four
  structural reasons in Act 1, the init never encoded a separable signal there in
  the first place — there is **nothing to forget**, so no fine-tuning policy can
  surface a signal that was never present. Classic fails because its starting
  point is **bad and task-useless**, not because a good one gets washed out. If
  forgetting were the cause, stopping the drift would recover the signal; it
  doesn't, because there is none. Forgetting *is* real — but it only **bites where
  the init already carries the answer** (tc14 erodes 1.000 → 0.986 unprotected),
  never on tc07–12. The genuine fix is to change *what the init encodes*
  (concept-bound, Act 2), not how gently it is fine-tuned.
- **Conclusion → the init, not the dynamics.** The fine-tuning policy is
  exonerated: neither slowing the drift nor pinning the init recovers tc07–12,
  because there is nothing to preserve. So the deficiency lives in the
  **initialization itself** — and the most *obvious* thing wrong with it is
  coverage: P1/P2 never give most entities a class code at all. That is the next
  rescue we try (Act 1.6). (Both knobs return in Act 4 in a second role: the
  protected LR as a precondition the later method requires, anchoring as a
  retained ablation.)

## Act 1.6 — Rule out "just cover more entities" as the fix

- **Motivation.** With the dynamics exonerated (Act 1.5), the init's most obvious
  flaw is **coverage** (reason #3): P1/P2 leave many leaf classes uninitialized
  (P1 seeds only **58.5%** of typed instances, P2 **88.1%**), so the rest fall
  back to *random* vectors. The cheap hypothesis is that *this alone* sinks
  classic — give every entity a class code and it works. We test it without
  leaving the class-mean framework two ways: add climbing strategies (`all_init`,
  `average_hier`, `weight_hierarchy`) that synthesize a code from ancestors for
  every uncovered leaf; and **now introduce P3** (P1 + the *full* subClassOf
  hierarchy baked into the pre-training corpus as explicit bidirectional edges),
  which gives every leaf class its **own** distributionally-learned code at 100%
  coverage. We then run a head-to-head varying *only which class vectors seed an
  instance*. (The thesis reports these coverage numbers later, in the results
  chapter, but the *logic* belongs here.)
- **The counter-intuitive result.** All routes to full coverage reach **100%**
  but **none beats `most_specific`**: aggressive filling even *hurts* (P1: 0.752
  → 0.565–0.621).
- **Why the climbing strategies fail — over-smoothing.** Climbing replaces a
  missing leaf code with a *shared ancestor*, collapsing distinct instances onto
  a few ancestor vectors (`average_hier`, the most uniform blend, is the worst).
- **Why P3 is the *decisive* control.** P3 is **not** over-smoothed — by baking
  the subclass hierarchy into the pre-training corpus, every leaf gets its **own**
  distributionally-learned embedding, not a copied ancestor. Yet P3's 100%
  coverage *still* does not lift `most_specific` above its P2 value (**0.742 vs
  0.769**). Clean, full coverage with distinct codes — and it changes nothing.
  That is the sharpest possible proof the coverage *count* isn't the driver.
- **Conclusion — the puzzle that's left.** Every obvious, classic-compatible fix
  is now exhausted: gentler fine-tuning (Act 1.5) and fuller coverage (here) both
  fail, and `most_specific` stays the default. What survives is sharper and less
  obvious — the deficiency is neither *how many* entities the init covers nor
  *how* it is fine-tuned, but **what a single class-mean code can express**:
  own-class membership, and nothing about an entity's neighbours or edge counts.
  That is the open problem we take up next.

---

## Act 2 — The proposed method: Concept-Bound Initialization

- **The idea.** Stop treating protograph vectors as *initial values to be
  improved by fine-tuning*; treat them as **symbols** and *explicitly construct*,
  per entity, a superposition of the one-hop schema concepts it satisfies — then
  *protect* it during fine-tuning.
- **The construction** (Eq. with α=β=γ=δ=1), four terms = the four failure modes,
  each one a repair:
  - **α — own classes**: mean of the entity's *materialized* class codes (its
    types + all transitive superclasses; materialization makes the discriminating
    superclass linearly accessible).
  - **β — outgoing ∃r.C**: sum of **Hadamard-bound** (relation ⊙ neighbour-class)
    pairs → fixes the binding problem (VSA binding makes pairs quasi-orthogonal,
    so a linear probe can read "has r to a C-thing").
  - **γ — incoming ∃r⁻.C**: same, with a circular **roll** on the relation so in-
    and out-edges land in distinguishable subspaces.
  - **δ — cardinalities**: accumulate plain relation codes so *edge counts*
    survive as **magnitude**; incoming edges tagged with `roll` so directions
    don't alias.
  - **Fallback**: instances the construction doesn't cover (e.g. isolated nodes)
    revert to the classic class-mean rule with `all_init` hierarchy-climbing.
- **The preconditions that make it work** (without them the fancy terms do
  nothing):
  - **Unit-normalize** every protograph code (use as symbols, not
    quality-weighted estimates).
  - **Shared *global* target norm (mean norm → 8)** — saturates the SGNS sigmoid
    to slow erosion; *global* (one scalar for the matrix) so the δ-term's
    magnitude differences survive (a per-vector norm would erase cardinality).
    (The *value* 8 itself is an unablated choice — flagged as an open TODO in the
    thesis.)
  - **Mirror into both SGNS matrices** (input + `syn1neg`) so the random context
    matrix doesn't immediately pull the init apart.
  - **Protected learning rate (0.0025, 1/10 of from-scratch)** — keeps the
    transferred geometry intact.
- **Motivation, restated.** Compute the neighbour-aware features *directly* at
  init time instead of hoping fine-tuning discovers them from walk co-occurrence.
- **Conclusion.** A pipeline where pre-training supplies quasi-orthogonal symbols
  and the construction assembles them into a readable one-hop schema feature
  vector; fine-tuning only adds instance-specific detail on top.

---

## Act 3 — Synthetic results: the method does what it was designed to

- **Headline win.** Concept-bound **solves the relation-centric tasks everything
  else failed**: tc07–12 mean 0.626 (vanilla) / 0.623 (best classic) → **0.905**
  (`p2_bound`); biggest jumps tc07 0.583→0.993, tc12 0.662→0.980. Mean over all
  15 TCs: 0.735 (vanilla) → **0.914** (`p2_bound`).
- **Coverage still matters — but now *within* the binding.** Among bound
  variants P2/P3 beat P1 on qualified tasks (tc07 0.825 vs 0.993/0.985), because
  β/γ can only bind classes that *have* pretrained codes. (Among classic variants
  coverage no longer separates them — confirming coverage alone is inert.)
- **No collateral damage.** Class-membership tasks (tc05, tc06, tc14) are
  preserved; the bound terms cost only a few points where classic was already
  near-perfect (tc14, where the label *is* the own class).
- **Runtime motivation & conclusion.** Is the gain expensive? **No** — the bound
  construction adds ~10% of pipeline time (a few percent over classic, which
  shares walks/vocab/fine-tuning); fine-tuning dominates. ≈29-point accuracy gain
  for marginal overhead.
- **Diagnostics — *why* it works (the "show, don't tell"):**
  - **Use LDA, not PCA.** The discriminative direction is often *low-variance*,
    so PCA hides a perfectly separable boundary (tc07: separable at 0.998 but
    looks mixed in PCA). Label-aware LDA is the honest lens.
  - **Classic collapses entities onto class means** — tc07's 400 test entities
    share just *two* init vectors; a collapsed init can't encode a label that is
    independent of own class, so the classic curves stay flat at chance. (Classic
    is fine-tuned at the *high* from-scratch rate, so this is **not** a freezing
    effect — there is simply no separable signal to preserve.)
  - **The margin exists *before* fine-tuning** — bound classes are near-bimodal
    along LD1 at epoch 0; the classifier just reads off a margin the init already
    provides (curves start high, stay flat).
  - **Magnitude structure is real** — bound produces entity-specific norms (the
    *footprint* of accumulated count features), while the classic init clamps
    every entity to exactly norm 8 (no count info) and vanilla's norms are
    uninformative. *Caveat:* the raw norm alone doesn't separate the classes
    (means 9.0 vs 8.4) — the count signal is read off the **direction** of the
    accumulated features, with the norm spread merely its footprint.
- **Drift / forgetting, visualized (confirms Act 1.5).** The forgetting Act 1.5
  established, now shown side by side with the method: classic vectors travel far
  (arrows long, cosine <0.2 — forgetting seen entity by entity), while bound under
  protected LR barely moves (cosine ≥0.85). Preserving the init is exactly what
  keeps the label-consistent geometry from washing out.

---

## Act 4 — Ablations: attribute the gains to mechanisms, not luck

Each isolates one knob; together they rule out "it's just regularization."

- **Fine-tuning learning rate.** *Motivation:* is the protected low LR a free
  lunch? (Its *first* role — failing to rescue classic from "forgetting" — is in
  Act 1.5; here it is tested as the bound method's knob.) *Conclusion:* No — it's
  a **precondition the bound method requires**, not a universal good. Transplanted
  to classic it *underfits* (`p1_classic` collapses 0.758→0.609); even at its
  *best* rate the classic transfer only *matches* vanilla. Schema buys a better
  start, not a lasting gain, once fine-tuning is free to move.
- **Protograph anchoring (λ pull-back to init).** Covered in full as a rescue
  attempt in **Act 1.5** (forcing the schema to survive is inert at best, harmful
  at high λ — set λ=0). Listed here only as one of the knobs that, alongside the
  noise ablation, rules out a generic-regularization explanation of the gains.
- **Initialization noise (jitter).** *Motivation:* originally to break ties when
  many instances share one class code; could it also regularize? *Conclusion:*
  σ=0 is best for everything that carries signal; bound degrades smoothly with
  noise; the original tie-breaking motivation is *obsolete* under bound (entities
  already differ by construction). Noise is strictly harmful. *The same σ-sweep's
  other role* — the information-gap control behind Act 1's reason #1, where jitter
  makes the collapsed classic vectors distinct yet leaves epoch-0 held-out
  accuracy at chance on tc07–12 — is presented there; here it doubles as the
  ablation that rules out an "it's just a regularizer" reading of the bound gains.
- **Reducing fine-tuning walks (the standout).** *Motivation:* the bound init's
  epoch-0 accuracy ≈ its final accuracy, and fine-tuning mostly *erodes* it — so
  can a schema-informed init *buy* walk budget? *Conclusion:* **Yes,
  dramatically.** Bound is insensitive to budget and even *improves* as it
  shrinks (every skipped SGNS update is drift that doesn't happen):
  quarter-budget `p2_bound` **beats full-budget vanilla by ~30 points** on the
  focus set while spending ~3.6× less fine-tuning time. Exceptions (tc03, tc06)
  are exactly where the signal is irreducibly instance-level and no schema init
  can substitute.
- **Direction-aware cardinality.** *Motivation:* is the `roll` direction tag a
  real design choice or an implementation detail? *Conclusion:* Direction is
  **real, label-relevant** info: a direction-*blind* count term doesn't help
  (0.838 ≈ count-free 0.845), but tagging direction (`rolled`) lifts to 0.912
  (+7.4 pts, up to +14 on tc10). The `roll` itself is *nothing special* — a
  random ±1 `keyed` tag ties it; any fixed decorrelating norm-preserving
  transform works. Negation fails differently (encodes out−in in one channel).
  An exact-count oracle confirms direction carries the information.

---

## Act 5 — Reality check on DBpedia: a puzzle, diagnosed and fixed

- **The puzzle (the hook).** *A state-of-the-art initializer that loses the
  race.* On 89 real DLCC-DBpedia splits, concept-bound is the **best
  *initialization*** by far (epoch-0 0.769 vs vanilla 0.500, classic 0.638) — the
  construction works on a real, noisy schema — **but loses end-to-end** (final
  0.803 vs classic 0.844, vanilla 0.831). Wins concentrate exactly where predicted
  (hard existence + cardinality splits, e.g. tc09-hard 0.837 vs 0.774); losses
  concentrate on tc04–06 (the enumerated-individual `{e}` families). Also:
  **P3 ≈ P2 here** — protograph depth is *not* the bottleneck. *(Setup detail:
  `owl:Thing` is deleted first — a class everyone belongs to carries no signal
  and, kept, becomes a hub that drowns the protograph walks.)*
- **Diagnosis — why synthetic gains don't transfer (three stacked mechanisms of
  the *corpus and benchmark*, not the construction):**
  1. **No headroom.** The 1.16B-token walk corpus lets vanilla jump 0.50→0.815 in
     the *first epoch*; the tc07–12 families it once struggled with are already
     solved on the *regular* splits (~0.886). ~0.11 headroom left vs the synthetic
     suite's ~0.37.
  2. **Benchmark mass is degree-shaped.** A *degree-only* probe (4 numbers,
     **no embeddings**) scores 0.791 regular / 0.739 hard — *beating trained
     vanilla on the hard splits* (0.700). On a real KG constructors can't be
     decorrelated from degree (≥2r literally implies higher degree). Schema
     channels can only add at the margin above this — which they do, on the hard
     splits.
  3. **Protected fine-tune freezes the bound model.** The construction creates a
     **pathological norm tail** (median ~1.8, max ~65,000); at LR 0.0025 hub rows
     barely move. The freeze *preserves* the constructor wins but *blocks* the
     recovery vanilla gets on tc04–06 — and that one family costs more than all
     the constructor wins add.
  - **Verdict:** *not a broken init — a **policy** problem.* The fix lives in the
    init's norm geometry and the fine-tuning policy, not the binding construction.
- **The fix that carries the act — tame the norm tail:**
  - **Norm capping (cap@16).** Put a ceiling on each entity row so frozen hubs can
    learn again. **"One ceiling on a few thousand hub vectors turns the best
    initializer into the best single embedding"** — 0.847 (10%), 0.849 (full,
    edging classic's 0.848 and clearing vanilla's 0.823); the `{e}` family
    recovers 0.808→0.913 (10% sub-corpus). Crucially, *raising the LR instead is
    not a substitute* — it makes the score *fall below* its own init.
  - **Build-time compression (`log1p`) — and its control (`sqrt`).** Squash the
    norm *as the init is built*, so the tail never forms. **`log1p` is the
    keeper** (matches the cap, no threshold; norms top out ~47). **`sqrt` is the
    cautionary control** — compresses too gently, a residual tail survives, the
    hubs re-freeze, the `{e}` sink returns. Accuracy moves **in lock-step** with
    the surviving tail (0.811 → 0.841 → 0.853) — **the cleanest proof that the
    norm tail, not the binding, was the mechanism all along.**
- **…and three controls confirm the diagnosis (no further lever):**
  - **Specificity (IDF) weighting** of the class term is a **null** (−0.002):
    gains on leaf-type splits cancel losses on inherited-superclass splits — the
    class *prior* is not a lever, the *magnitude* is (the classic-side analogue,
    per-code normalization, is likewise a no-op since DLCC entities are near
    single-typed).
  - **Hierarchy weighting** (`decay`, `most_specific`) is task-dependent on
    synthetic but only *second-order* (~1 pt) on DBpedia where the cap dominates
    (~4 pts) — though **`decay`+cap@16 is the best single bound embedding at
    0.860** (+0.037 over vanilla), its edge on exactly the hard/cardinality splits
    the construction targets.
  - **Cap-size & LR sweeps:** the cap is **robust** (broad plateau, τ=16 marginal
    best — a safe unfreeze, not a delicate knob), and once the tail is gone the
    **protected LR was already optimal** (higher rates cut loss but lower
    hard-split accuracy — textbook over-fitting of a saturated corpus).
- **Why the trade-off costs more on a real graph (mechanism → consequence):**
  - *Mechanism:* classic keeps each class compact & separable (every entity ≈ its
    class code); bound **trades** separation for relation info. DBpedia's **loose
    relations** (reused across domains/ranges — `city` links orgs *and* people)
    blur classes that classic keeps apart, so the trade costs more here than on
    the strict synthetic graph.
  - *Consequence (GEval clustering / classification / analogies):* **the DLCC
    ranking does not transfer.** Classic wins clustering (compact classes),
    everyone flattens on classification (a fitted model recovers the signal
    wherever it sits), analogies are mixed. **The best initialization is
    task-dependent.**

---

## Act 6 — Conclusion: what the whole arc proves

- **The thesis claim, earned.** The payoff of protograph pre-training for RDF2Vec
  is governed by **what the initialization encodes**, not how much schema is
  pretrained:
  - *Class-mean / classic* encodes **own-class membership** → helps only where
    the label *is* own-class (tc14; the class-flavored DBpedia families).
  - *Concept-bound* encodes **one-hop schema features** (bound (r, neighbour-
    class) pairs + directed counts) → solves exactly the constructor families
    built from those features (synthetic +0.28 over vanilla on tc07–12; hard
    DBpedia splits).
- **The mechanisms, not magic.** Ablations attribute gains to identifiable
  causes — binding, direction tagging (~7 pts), normalization + protection as
  *preconditions* — not generic regularization: the **noise** ablation rules out
  the simplest "it's just a regularizer" explanation, and the separate
  **anchoring** ("force the schema to survive") knob is inert at best, harmful at
  worst.
- **The real-graph lesson.** A strong init is necessary but not sufficient: on a
  rich corpus the **fine-tuning policy and the init's norm geometry** decide
  whether the encoded schema survives. Capping / `log1p` compression turns the
  best initializer into the best end-to-end single embedding; gains land where
  schema carries signal (hard splits) and vanish where the signal is
  instance-level (`{e}` families) or already free in the degree distribution.
- **Honest limits / open ends.** Multi-seed stds for all synthetic tables; an
  aggregated hyper-parameter-search table for the classic transfer; DBpedia P3 +
  direction-tag ablations; and a *selective/relaxed protection* run to address the
  tc04–06 blind spot (let `{e}`-family entities learn while keeping the hubs
  tamed).

---

### One-line spine to keep in view

**Schema gives RDF2Vec a better *starting point*; only the concept-bound
construction turns that head start into the *right features*; and only a
norm-aware fine-tuning policy lets those features survive contact with a real
graph.**
