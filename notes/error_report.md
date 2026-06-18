# Error Report — `latex/thesis.tex`

Full read-through of `latex/thesis.tex` (4022 lines) plus every `\input`-ed table
(`assets/*.tex`, `tables/*.tex`). Each problem below was checked against the
thesis's own numbers and cross-references; every proposed fix is grounded in the
thesis text itself (the competing number/section is cited).

Severity: **HIGH** = substantive correctness (wrong number, wrong conclusion,
broken render); **MEDIUM** = real inconsistency a careful reader/examiner will
catch; **LOW** = cosmetic / style / consistency.

Line numbers are from the current `thesis.tex`. The document compiles with **no
undefined references and no multiply-defined labels** (checked `thesis.log`), and
every `\includegraphics`/`\input` target exists, so the issues below are content
issues, not build errors.

---

## HIGH severity

> **Status: all 7 HIGH items (P1–P7) fixed in `latex/thesis.tex`.** The document
> recompiles cleanly (no errors, no new undefined references). MEDIUM/LOW items
> below are not yet applied.

### P1 — Wrong number: capped bound init "0.847" should be **0.853** (§4.7.3 Norm Capping) — ✅ FIXED (0.847→0.853, line 2778)
- **Location:** line 2766 (point (i) under Fig. `fig:dbpedia_ft010`).
- **Quote:** "with the hubs capped, the bound embedding climbs from $0.786$ to **$0.847$** and becomes the best single embedding on the sub-corpus".
- **Problem:** Every other report of this exact metric (cap@16 final accuracy, 10 % sub-corpus, standardize lens, flat mean over 89 splits) gives **0.853**, not 0.847:
  Table `tab:dbpedia_compress` cap@16 Final = **0.853** (line 2931); Table `tab:dbpedia_idf` cap@16 uniform Final = **0.853** (line 2997); the trend chain at line 2908 reads "$0.811 \rightarrow 0.841 \rightarrow 0.853$"; line 2937 cites "matching the capped $0.849$–$0.850$" on full corpus; the LR-schedule table protected row = **0.853** (line 3192). The init value 0.786 in the prose matches the tables exactly, isolating the **final** value as the error.
- **Fix:** Change `0.847` → `0.853` to agree with Tables `tab:dbpedia_compress`/`tab:dbpedia_idf` and the surrounding prose.

### P2 — tc14 mischaracterized as a "counting" task (§4.7.6 + Table `tab:hierarchy_synth`) — ✅ FIXED (prose, caption, row label relabeled "own-class membership")
- **Location:** prose lines 3027–3030; Table 4.19 caption line 3038; Table 4.19 row label line 3048 ("tc14 (counting / stability)").
- **Quote:** "In the counting tasks tc09 ($\geq 2r$) and **tc14**, what matters is **an edge count** … most\_specific wins tc14 ($0.981$)"; caption "drop it when the task counts edges **(tc09, tc14)**".
- **Problem:** tc14 is **not** a counting task. The thesis defines it everywhere else as a domain-side subclass / own-class-membership task: Table `tab:tc13-tc16-motivation` (lines 719–725) "which subclass of the declared domain appears as the subject"; line 1596–1599 lists tc14 under "Class-membership tasks … because the label *is* the entity's own class"; line 2091–2092 "effectively a class-membership task whose most-specific class initialization is already the answer"; line 3386 "the single test case tc14". The cardinality/counting tasks are tc09–tc12 (line 976). The empirical result (most\_specific wins tc14) is correct, but the *reason* is that the label is the entity's own most-specific class — not edge counting.
- **Fix:** Stop grouping tc14 with tc09. Reword the prose so the edge-count rationale applies to tc09 only, and tc14 is won by most\_specific because the answer is the entity's own most-specific class (dropping generic superclasses sharpens it). Relabel the Table 4.19 row from "tc14 (counting / stability)" to e.g. "tc14 (own-class membership)" and fix the caption phrase "when the task counts edges (tc09, tc14)".

### P3 — "The four terms correspond exactly to the four failure modes" is not a clean mapping (§3.4.1) — ✅ FIXED (remapped; mode 3 attributed to all_init/P3 fallback)
- **Location:** line 1046 (prose after Eq. `eq:bound_init`).
- **Quote:** "The four terms correspond **exactly** to the four failure modes of Section~\ref{sec:classic\_limits}: the $\alpha$-term is the classic own-class signal; the $\beta$- and $\gamma$-terms encode qualified existential restrictions …; the $\delta$-term accumulates plain relation embeddings".
- **Problem:** The mapping is not one-to-one and partly self-contradicts §3.3.1:
  (a) Failure mode 1 (lines 959–964) says the own-class signal *is the problem*; yet the $\alpha$-term *is* "the classic own-class signal", i.e. it reproduces the failure rather than repairing it. The neighbour-class repair of mode 1 is actually done by the $\beta/\gamma$ terms.
  (b) Failure mode 3 ("Coverage gaps", lines 971–975) has **no** corresponding term — coverage is closed *outside* the equation by the `all_init`/P3 fallback (the thesis says so at lines 1052–1055). So three terms cover modes 1+2 and 4, and mode 3 is handled by a fallback, contradicting "correspond exactly".
- **Fix:** Replace "correspond exactly to the four failure modes" with the true mapping: $\beta/\gamma$ supply the neighbour-class signal (mode 1) and restore binding (mode 2); $\delta$ preserves cardinality magnitude (mode 4); $\alpha$ carries over the classic own-class signal as a base; the coverage gap (mode 3) is closed separately by the `all_init`/P3 fallback (lines 1052–1055).

### P4 — Broken citation: doubled author name (§2.4) — ✅ FIXED (`\citet`, line 563)
- **Location:** line 563.
- **Quote:** "Jain et al.~\citep{jain2021semantics} show that embeddings …".
- **Problem:** With `natbib`/`chicagoa`, `\citep` renders "(Jain et al., 2021)", so the hand-written "Jain et al." prefix produces **"Jain et al. (Jain et al., 2021)"**. Line 276 does it correctly with `\citet`.
- **Fix:** Replace "Jain et al.~\citep{jain2021semantics}" with `\citet{jain2021semantics}` (matching line 276).

### P5 — Same method, two means + flipped P2/P3 ranking across tables (§4.4.1 vs Table 4.1) — ✅ FIXED (footnote added; "direct evidence"→"evidence")
- **Location:** §4.4.1 line 1757 vs Table `tab:synthetic_final` line 1518; data in `assets/init_strategies_table.tex`.
- **Quote:** line 1757 "P3's complete coverage does not lift `most_specific` above its P2 value (**$0.742$** vs.\ $0.769$)"; Table 4.1 footnote says all classic transfers use `most_specific`, and there `p3_classic = 0.774 > p2_classic = 0.768`.
- **Problem:** The identical nominal config (classic / `most_specific`, tc01–tc15) is **0.742** in the init-strategy table but **0.774** in the headline table — a 3.2 pp gap that **exceeds the thesis's own ≈3 pp noise floor** (lines 1466–1467) and **flips** the P2-vs-P3 ordering (P3<P2 in §4.4.1, P3>P2 in Table 4.1). The two are different single-seed runs (Table 4.1: 5 epochs @ lr 0.025; init-strategy table: 5+5 epochs, λ=0.5), but the text never says so, so the conclusion "coverage does not help P3" appears to contradict the headline table where `p3_classic` is the *best* classic variant.
- **Fix:** State at line 1757 that `tab:init_strategies` is an independent run (different epochs/λ) from `tab:synthetic_final`, so its `most_specific`-P3 value (0.742) differs from `p3_classic` (0.774) by ~3.2 pp; or reconcile the two P3 figures. The "Tellingly … direct evidence" wording (lines 1757–1758) should be softened.

### P6 — "entity-pool" coverage level is defined and promised but never reported (§3.2.3 / §4.4) — ✅ FIXED (entity-pool level dropped from definition and pointer; "two levels")
- **Location:** definition lines 856–859; forward pointer line 1657 ("at the vocabulary, entity-pool, and training-set levels"); §4.4 reports only two tables (line 1660 "the two tables tell the same story").
- **Quote:** line 856 "At the **entity-pool** level, we count the full positive and negative candidate sets (1000 entities each) …"; line 1657 "We report the initialization coverage … at the vocabulary, **entity-pool**, and training-set levels".
- **Problem:** Only a vocabulary table (`tab:vocab_init_coverage_tc01_tc15`) and a training-set table (`tab:train_class_init_coverage_tc01_tc15`) are ever given. No entity-pool (1000/1000) numbers/table/figure exist (the string "entity-pool" appears only in the definition and the forward pointer).
- **Fix:** Either add the promised entity-pool coverage numbers in §4.4, or drop the entity-pool level from the definition (lines 856–859) and the pointer (line 1657), keeping the two levels actually reported.

### P7 — tc11/tc12 labelled "$\exists r.C$" but they are qualified-cardinality ($\geq 2r.T$) (§3.4.3) — ✅ FIXED (tc07/tc08 kept as ∃r.C; tc11/tc12 described as ≥2r.T, line 1147)
- **Location:** line 1140.
- **Quote:** "This is exactly the form taken by the DLCC concepts $\exists r.C$ (**tc07/tc08, tc11/tc12**) and $\exists r^{-}.C$."
- **Problem:** Table `tab:dlcc-test-cases` (lines 512–518) defines tc11 = $\geq 2r.T$ and tc12 = $\geq 2r^{-1}.T$ — qualified *cardinality*, not qualified existential. Only tc07 ($\exists r.T$) and tc08 ($\exists r^{-1}.T$) are $\exists r.C$ tasks.
- **Fix:** Restrict the $\exists r.C$/$\exists r^{-}.C$ list to tc07/tc08; describe tc11/tc12 as the qualified-cardinality tasks ($\geq 2r.T$ / $\geq 2r^{-1}.T$) whose $(r,C)$ pairing the binding terms *also* feed.

---

## MEDIUM severity

> **Status: all MEDIUM items (P8–P14) fixed in `latex/thesis.tex`.** Document
> recompiles cleanly; renamed labels resolve on the second pass.

### P8 — tc04 corpus size: "roughly twenty times" vs "≈15×" elsewhere (§4.6.3) — ✅ FIXED ("twenty"→"fifteen")
- **Location:** line 2055.
- **Quote:** "(whose instance-walk corpus is **roughly twenty times** larger than the others)".
- **Problem:** Three other places state ~15×: runtime figure caption line 1633 ("$\approx$15$\times$"), runtime table caption (`runtime_per_tc_table.tex`), and stability appendix line 3548 ("about fifteen times larger"). The runtime data supports ~13–15× (tc04 vanilla 286.8 s vs non-tc04 mean 22.0 s), not 20×.
- **Fix:** Change "twenty" → "fifteen".

### P9 — "Class-membership tasks are preserved" heading lists enumerated-individual tasks (§4.2)
- **Location:** bold heading line 1594; body lines 1594–1600.
- **Quote:** "**Class-membership tasks are preserved.** On tc05, tc06, and tc14 the bound variants stay high".
- **Problem:** Only tc14 is class-membership. tc05 ($\exists R_1.(\exists R_2.\{e\})\dots$) and tc06 ($\exists r.\{e\}$) are **enumerated-individual ($\{e\}$)** tasks — the thesis itself repeatedly calls tc04–tc06 "the enumerated-individual family ($\{e\}$ constructors)" (lines 2548, 2672, 2852) and reserves "class-membership" for tc14 (lines 526–528, 2091).
- **Fix:** Reword the heading to cover "tasks the classic transfer already solves" (existence/enumerated-individual + tc14), or split tc05/tc06 out as enumerated-individual and keep "class-membership" for tc14 alone.

### P10 — Jitter figure caption "all nine variant families" but the table lists six (§4.6.4)
- **Location:** Fig. `fig:jitter` caption line 2166; companion Table `tab:jitter` lines 2150–2157.
- **Quote:** "shown per test case (tc07, tc09--tc12), for **all nine variant families**."
- **Problem:** Table `tab:jitter` has exactly **6** rows (p1/p2/p3 × {bound, classic}); "nine" matches neither the six table rows nor the thesis's usual "seven variants" (lines 1488, 1929).
- **Fix:** Make the count consistent — change "nine" to "six" to match the six rows of `tab:jitter` (or state explicitly which nine series the figure plots).

### P11 — Failure mode 1's "neighbour classes" claim is wrong for tc09/tc10 (§3.3.1)
- **Location:** lines 959–964.
- **Quote:** "On tc07--tc12 … The decision-relevant information sits in the **classes of the entity's neighbours**".
- **Problem:** tc09 = $\geq 2r.\top$ and tc10 = $\geq 2r^{-1}.\top$ (lines 504–510) are **type-agnostic** cardinality tasks; the label is the *edge count*, not the neighbour class. Failure mode 4 (lines 976–980) classifies tc09–tc12 as counting tasks, so mode 1's blanket "neighbour classes" over tc07–tc12 conflicts with mode 4 for tc09/tc10.
- **Fix:** Restrict "classes of the entity's neighbours" to the typed cases (tc07, tc08, tc11, tc12); note that for tc09/tc10 the signal is the edge count (consistent with failure mode 4).

### P12 — "the cap lifts every weighting about 4 points above vanilla" overstates it (§4.7.6)
- **Location:** line 3062.
- **Quote:** "The cap is what lifts **every weighting about $4$ points above vanilla**."
- **Problem:** Per Table `tab:dbpedia_hierarchy`, only `decay`+cap@16 (0.860) is ~3.7 pp above vanilla (0.823). `uniform`+cap@16 (0.850) is +0.027 and `most-specific`+cap@16 (0.849) is +0.026 — about 2.6–2.7 pp. The "≈4 points" is the *cap-vs-global-rescale* effect (caption line 3080; e.g. uniform 0.808→0.850 = +0.042), not cap-vs-vanilla.
- **Fix:** Reframe as the cap effect over the *global-rescale base* (~4 pts, per caption line 3080), or correct to "about 3 to 4 points above vanilla (+0.026 to +0.037)".

### P13 — Placeholder labels `\label{sec}` and `\label{fig}` (§2.2)
- **Location:** line 283 (`\label{sec}`); line 338 (`\label{fig}`, referenced as `Figure~\ref{fig}` at line 341).
- **Problem:** Generic, collision-prone labels inconsistent with the descriptive convention (`sec:lit_dlcc`, `fig:protograph_construction`, …). `\label{fig}` is actively referenced, so any future reuse of "fig" silently breaks the cross-reference. `\label{sec}` is unused.
- **Fix:** Rename to e.g. `sec:rdf2vec` (line 283) and `fig:rdf2vec_example` (line 338), updating `\ref{fig}` at line 341.

### P14 — Method name casing: "RDF2vec" vs the dominant "RDF2Vec" (title page + 5 spots)
- **Location:** title line 55; lines 195, 198, 655, 666, 795.
- **Quote (title):** "\title{Schema-based pre-training for **RDF2vec**}".
- **Problem:** The abstract (line 66) and ~68 body occurrences use "RDF2Vec"; only 6 use lowercase "RDF2vec", including the **title page** (most prominent location). The canonical name (`groth_rdf2vec_2016`, "{RDF2Vec}") uses the capital V.
- **Fix:** Normalize all to "RDF2Vec" (title + lines 195, 198, 655, 666, 795).

---

## LOW severity

> **Status: LOW items P15–P25, P27, P28 and the logic note N1 fixed in
> `latex/thesis.tex` / asset captions. P26 (orphan files) is DEFERRED** — those
> files are still `\input` by `thesis_old.tex` and were not authored in this
> task, so deleting them is left to the author's decision (they do not affect the
> live `thesis.tex` output).

### P15 — `0.913` vs Table `tab:dbpedia_compress` `0.914` ({e} family, §4.7.3) — ✅ FIXED (→0.914)
- **Location:** line 2773.
- **Quote:** "the enumerated-individual family (tc04--tc06), stuck at $0.808$ … **recovers to $0.913$**".
- **Problem:** Table `tab:dbpedia_compress` cap@16 `{e}` column = **0.914** (line 2931), and line 2947 itself cites "the cap's $0.914$". The un-capped 0.808 matches the table, so only the capped figure is off by 0.001.
- **Fix:** `0.913` → `0.914`.

### P16 — Stale label `tab:tc13-tc16-motivation` for a TC13–TC15 table (§3.1)
- **Location:** `\label` line 698; `\ref` line 691; caption "TC13--TC15" line 697.
- **Problem:** Label names TC16, but the table (and the whole thesis) has no TC16; the three rows are TC13/TC14/TC15.
- **Fix:** Rename label to `tab:tc13-tc15-motivation` and update the `\ref` at line 691.

### P17 — Terminology: "class-mean transfer" (6×) vs the dominant "classic transfer" (§4.8 / appendix)
- **Location:** lines 3302, 3311, 3336, 3736, 3951, 3982 (plus "classic class-mean transfer(s)" at 3306, 3690, 3955).
- **Problem:** The bare term "class-mean transfer" is inconsistent with "classic transfer", used throughout (lines 148, 933, 1230, 3251, …).
- **Fix:** Standardize on "classic transfer" / "classic init".

### P18 — Protograph prototriple notation inconsistent: `(p_{C_d}, r, p_{C_r})` vs `(C_d, r, C_r)`
- **Location:** lines 591–596 (use $p_{C_d}$, $p_{C_r}$) vs line 762 (uses $C_d$, $C_r$).
- **Problem:** The same P1 prototriple is written with and without the $p$-subscript protograph-node notation.
- **Fix:** Pick one notation and use it in both places.

### P19 — Grammar: stray comma splitting subject and verb (§3.2.2)
- **Location:** lines 833–834.
- **Quote:** "Each class (including every leaf)**,** thus receives its \emph{own} embedding".
- **Fix:** Remove the comma after the parenthetical: "Each class (including every leaf) thus receives …".

### P20 — Runtime: "the concept-bound construction itself takes 2.4 s" attributes the whole Init stage (§4.3)
- **Location:** line 1607.
- **Problem:** The runtime-table description (lines 1642–1647) and caption define the Init stage as "transfer of pretrained codes **plus** the concept-bound construction". The p2_bound mean Init is 2.39 s, of which the classic transfer already costs ~0.14 s (p2_classic Init), so the construction *proper* is ~2.25 s.
- **Fix:** Attribute the 2.4 s to the initialization stage (transfer + construction), or subtract the 0.14 s transfer to report the construction alone as ~2.25 s.

### P21 — Misleading runtime-table column header "Init (w/ pretrain)"
- **Location:** `assets/runtime_per_tc_table.tex` header (also caption in main text).
- **Problem:** Pretrain is a *separate* column and Total = Pretrain + Init + Fine-tuning (e.g. tc01 p2_bound 2.04 + 2.07 + 17.92 = 22.03 for every row), so the Init column does **not** include pretraining.
- **Fix:** Rename the column "Init" (drop "(w/ pretrain)").

### P22 — KGE acronym defined twice; KGE/Jain-et-al. criticism repeated (§2.1 vs §2.4)
- **Location:** line 270 ("Knowledge graph embedding (KGE)") vs line 555 (re-defines it); same surveys (`ji_survey_2022`, `sardina_survey_2024`) and the Jain et al. point repeated.
- **Fix:** Define KGE once (line 270); at line 555 reference rather than re-state the framing.

### P23 — Conclusion: ARI "0.88" paired with "a much lower vanilla floor" misleads (§5)
- **Location:** lines 3408–3409.
- **Quote:** "Clustering rewards the classic transfer most clearly, with adjusted Rand index **up to $0.88$ against a much lower vanilla floor**".
- **Problem:** Per §4.8 (lines 3312–3313) the 0.88 is on **CAMAP**, where vanilla is **0.78** (not "much lower"). The "much lower" floor (0.06) is on **CC**, where the classic transfer reaches only ~0.72. The implied single-task 0.88-vs-floor gap does not exist on one task.
- **Fix:** Reword so the per-task pairing is clear: classic leads on all three (CC ~0.72, CCB 0.85, CAMAP up to 0.88); the gap over vanilla is largest on the shared-parent tasks (vanilla 0.06 on CC, 0.69 on CCB), not on CAMAP (vanilla 0.78).

### P24 — `init_strategies_pertc` reports tc14 = 1.000 with an unstated learning rate (Appendix)
- **Location:** `assets/init_strategies_pertc_table.tex` tc14 row (all 12 strategy cells = 1.000).
- **Problem:** Table 4.1 and the LR table report classic/`most_specific` tc14 = 0.982–0.996 at the from-scratch rate; tc14 = 1.000 matches the **protected** rate 0.0025 (LR table). The init-strategy captions never state the fine-tuning rate, so a reader cannot tell which run this is, and 1.000 looks inconsistent with the from-scratch classic tc14 reported elsewhere.
- **Fix:** State the fine-tuning learning rate in the `tab:init_strategies` / `tab:init_strategies_pertc` captions (as the anchoring table caption does), so the tc14 = 1.000 figure is unambiguous.

### P25 — Three different "vanilla over tc01–tc15" baselines: 0.735 / 0.736 / 0.739
- **Location:** Table 4.1 mean row line 1518 (0.735, quoted at 1715); LR ablation Fig./prose lines 1974, 2002 (0.736); init-strategies Table 4.4 (0.739, quoted at line 1774).
- **Problem:** All three describe the same quantity (mean LogReg accuracy over tc01–tc15) but differ, with no note that they are independent single-seed runs (the spread is <0.5 pp, within the noise floor, but unexplained).
- **Fix:** Add a one-line note where the divergent baselines first appear that the vanilla baseline is recomputed per experiment (independent single-seed runs), so its mean fluctuates 0.735–0.739, well within the ≈3 pp floor of `app:stability-analysis`. Alternatively harmonize.

### P26 — Orphan/stale table files contradict the live narrative — ⚠️ DEFERRED (still `\input` by `thesis_old.tex`; author to decide on deletion)
- **Location:** not `\input` anywhere in `thesis.tex`: `assets/anchoring_lambda_table.tex`, `tables/acc_norm_vs_nonorm.tex`, `tables/acc_ms_norm_vs_nonorm.tex`.
- **Problem:** `anchoring_lambda_table.tex` carries a **15-TC** anchoring sweep (P2 λ=0 = 0.768) with a caption claiming accuracy "decreases monotonically in λ on every protograph" — contradicting the live **14-TC** analysis (P2 λ=0 = 0.741; best mean P1 at λ=0.1 = 0.747 > 0.740, i.e. *non*-monotonic; lines 2069–2074, `anchoring_per_tc_tables.tex`). It is still `\input` by `thesis_old.tex`, so it is a latent contradiction. The two `tables/acc_*` files are superseded by `assets/norm_ablation_pertc_table.tex` (mean |Δ| 0.020 vs the orphans' 0.022).
- **Fix:** Delete the stale orphan files (or reconcile their numbers with the live source) so no draft can render contradictory anchoring/normalization numbers.

### P27 — Abstract/headline 0.808 is the standardize-lens value; not flagged as such
- **Location:** abstract line 82, intro line 166, conclusion line 3400 ("$0.808 \rightarrow 0.850$").
- **Problem:** Table `tab:dbpedia_per_tc` (the raw-lens headline) lists `p2_bound` = **0.803**, while 0.808 is the *standardize-lens* value from Table `tab:dbpedia_cap_p3`. The body flags that the two lenses "should not be compared directly" (line 2749), but the abstract/intro/conclusion quietly use 0.808 without that qualification.
- **Fix:** Either use the raw-lens 0.803 (consistent with the headline table) or note that 0.808→0.850 is on the standardize lens.

### P28 — "≈0.5 point lower" understates the lens gap (§4.7.3)
- **Location:** line 2749.
- **Quote:** "its baselines sit **$\approx 0.5$ point lower**".
- **Problem:** Full-corpus vanilla is 0.823 (standardize, Table `tab:dbpedia_hierarchy`) vs 0.831 (raw, Table `tab:dbpedia_per_tc`) — a 0.8 pp gap, not 0.5.
- **Fix:** Change "≈0.5 point" to "≈0.8 point" (or "about 1 point").

---

## Logic/argument note (not an outright error, worth a sentence)

### N1 — Vanilla "absorbs schema for free" yet classic transfer still beats vanilla on DBpedia
- **Location:** diagnosis lines 2593–2600 vs headline lines 2539–2542.
- **Observation:** §4.7.2 argues DBpedia walks traverse `rdf:type`, so "vanilla skip-gram learns [class membership] as ordinary co-occurrences … vanilla absorbs it for free". But §4.7.1 reports the classic transfer (which only copies own-class vectors) still wins the overall mean (0.844 vs vanilla 0.831). If vanilla already learns class membership for free, the residual classic edge (+0.013, only just above the cited ≈0.003 DBpedia std) deserves one explanatory sentence; as written the two claims sit in mild tension.
- **Fix:** Add a clause reconciling them (e.g. the classic copy still sharpens/*initializes* the class geometry the corpus only learns approximately), or note the +0.013 is small relative to the headroom argument.

---

## Verified consistent (spot-checks that passed — no action needed)

To bound the scope, these high-traffic numbers were recomputed and **match**:
the Table 4.1 means (vanilla 0.735, p1/p2/p3_classic 0.759/0.768/0.774,
p1/p2/p3_bound 0.890/0.914/0.909) and the 0.28 headline gain (tc07–tc12:
0.905−0.626); the 89-split count (Σ of the `n` column in `tab:dbpedia_cap_p3` =
89); the norm-distribution table (median 1.80, max 65 339; vanilla max 17.7);
the degree-probe table (regular 0.791, hard 0.739 > vanilla 0.700); the headroom
arithmetic (0.37 synthetic, 0.11 DBpedia); the cap-size sweep (τ16 best, 0.785);
the LR-schedule table (loss 24.1→19.8, hard 0.748→0.737); the decay+cap@16 =
0.860 (+0.037 over vanilla, +0.010 over uniform cap); the LR-ablation,
norm-ablation, anchoring per-TC, reduced-walks, and direction-tag tables all
match their prose. The stability numbers (mean std 0.0112, max 0.0218 on tc08
vanilla) are consistent. No leftover `\missing`/TODO markers; no em-dash in body
text; no undefined references or duplicate labels.
