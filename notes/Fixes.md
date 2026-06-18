# Fixes.md — Review of `latex/thesis.tex`

Full read-through of the thesis (4 121 lines) plus an exhaustive cross-check of every
number, table, figure caption and citation against the experiment data in the repo
(`notebooks/`, `output/`, `scripts/`, `latex/assets/`). Figures were verified by
reading the script/notebook that **generates** each asset (the images themselves can't
be diffed), so a caption is flagged when it disagrees with what the code actually plots.

**Mechanical checks that passed** — all 78 referenced figures/`\input` tables exist on
disk (none missing); all 34 `\cite` keys are defined in `references.bib`; no `\missing`
markers remain; no duplicate `\label` definitions.

**Totals:** 8 critical · ~17 medium · ~36 low.

Severity legend: **[C]** fix before submission · **[M]** factual/caption error, fix ·
**[L]** polish / rounding / reproducibility.

---

## 1. Critical — fix before submission

### 1.1 [C] The Abstract is placeholder template text
[thesis.tex:64-93](latex/thesis.tex#L64-L93) — the "Abstract" chapter is still the
DWS/scrbook template boilerplate (`Your thesis must contain an abstract … \blockcquote{zobel2004}{An abstract is typically …}`).
There is **no actual abstract** of this work — no mention of RDF2Vec, protographs, the
concept-bound init, or any result.
**Fix:** write a real 50–200 word abstract (schema-based pre-training for RDF2Vec → P1/P2/P3
protographs → classic transfer's structural failure on relation-centric DLCC → concept-bound
initialization → headline results, e.g. DBpedia `p2_bound_decay_cap16` 0.860, +0.037 over vanilla).
Remove the `zobel2014`/`zobel2004` citations from the abstract.

### 1.2 [C] Corrupted LaTeX subscripts in the P2 construction
[thesis.tex:621-623](latex/thesis.tex#L621-L623) — markdown-style asterisks leaked into
math mode, so the subscripts render as multiplication:
`$C'*d$`, `$C'*r$`, `$(p*{C'*d}, r, p*{C_r})$`, `$(p*{C_d}, r, p_{C'_r})$`.
Line 618 already shows the intended form `$(p_{C_d}, r, p_{C_r})$`.
**Fix:** replace `*` with `_`:
`$C'_d \sqsubseteq C_d$ and $C'_r \sqsubseteq C_r$`; `$(p_{C'_d}, r, p_{C_r})$ and $(p_{C_d}, r, p_{C'_r})$`.

### 1.3 [C] The Introduction understates the thesis scope
[thesis.tex:165-167](latex/thesis.tex#L165-L167) and contribution list
[thesis.tex:183-190](latex/thesis.tex#L183-L190) say "**Two** protograph construction
strategies … P1 … and P2" and never mention **P3** (a core construct, [thesis.tex:842-865](latex/thesis.tex#L842))
or the **concept-bound initialization** — which the body presents as *the* proposed method
([thesis.tex:1011-1013](latex/thesis.tex#L1011)). The Introduction reads as if the thesis
only does classic P1/P2 transfer.
**Fix:** mention P3 (three protographs) and add the concept-bound init as the central
proposed method/contribution.

### 1.4 [C] Runtime prose numbers come from a superseded run and contradict the cited table
[thesis.tex:1632-1636](latex/thesis.tex#L1632-L1636) — "p2_bound … **43.0 s** end-to-end,
fine-tuning **32.9 s (76 %)**, bound construction **4.2 s**; pretraining P1 0.6 / P2 4.3 / P3 2.8 s".
These reproduce **only** from the deprecated `notebooks/runtime/results.mixed-hardware.bak.json`.
The table the same sentence cites (`tab:runtime_per_tc`, [runtime_per_tc_table.tex](latex/assets/runtime_per_tc_table.tex))
is generated from `output/synthetic_benchmark/summary.json` and gives the **p2_bound mean row
total = 25.49 s, fine-tuning = 20.75 s**, and tc01 pretrain P1 0.26 / P2 2.04 / P3 1.79 s.
So the prose is ~1.7–2× the table it points to — same quantity, two values.
**Fix:** regenerate the prose numbers from `summary.json` (the table's source) and state the
aggregation (table mean = tc01–12 excl tc04, n=11): ≈25.5 s total, ≈20.8 s fine-tuning,
≈2.4 s init; tc01 pretrain 0.26 / 2.04 / 1.79 s.

### 1.5 [C] The anchoring appendix table is a *different experiment* than the figure/prose it backs
The appendix table [anchoring_per_tc_tables.tex](latex/assets/anchoring_per_tc_tables.tex)
(`\input` at [thesis.tex:3989](latex/thesis.tex#L3989)) says it "gives the complete
per-test-case results … summarized in Figure `fig:anchoring_lambda_sweep`". Its **mean row is
λ=0 → 0.746, λ=1 → 0.716**, reproducing the *non*-last-epoch sweep
(`notebooks/anchoring/anchoring_final_acc_per_tc.csv`). But the figure `new_anchoring.png` is
byte-identical to the **last-epoch** sweep, and the prose
([thesis.tex:2093-2126](latex/thesis.tex#L2093-L2126)) reports the much sharper last-epoch
collapse (λ=1 → 0.649/0.596/0.564 per P1/P2/P3, ≈0.603 averaged; tc01 0.950→0.584, tc06 0.984→0.599).
The included table (0.716) and the figure/prose (≈0.603) are inconsistent.
**Fix:** regenerate `anchoring_per_tc_tables.tex` from the **last-epoch** data
(`notebooks/anchoring/anchor_last_epoch/…`) so the mean row reads λ=0 ≈0.748, λ=1 ≈0.603
and tc01/tc06 λ=1 read 0.584/0.599, matching the figure and prose.

### 1.6 [C] 41 em-dashes (`---`) violate the project no-em-dash rule
Project rule: never write `---` in `thesis.tex`. There are 41 real (non-comment) occurrences:
lines 1654, 1774, 2078, 2096, 2108, 2119, 2132, 2205, 2206, 2428, 2611, 2612, 2619, 2621,
2626, 2635, 2636, 2696, 2772, 2987, 2988, 3002, 3036, 3052, 3053, 3072, 3103, 3110, 3111,
3113, 3150, 3170, 3171, 3215, 3216, 3230, 3251, 3921, 3922, 4037, 4039.
**Fix:** replace each with an en-dash `--`, a colon, or a comma/parenthesis restructure.
(Line 4 `% ---- FAST-DRAFT TOGGLE ----` is a comment; leave it.)

### 1.7 [C] Front/back-matter placeholders and a malformed declaration table
- **Date mismatch + placeholder:** title page `\date{June 18, 2026}` [thesis.tex:57](latex/thesis.tex#L57)
  vs declaration `Mannheim, den XX.~XXXX 2024` [thesis.tex:4122](latex/thesis.tex#L4122)
  (unfilled day/month, wrong year).
- **Malformed AI-tools row:** [thesis.tex:4113](latex/thesis.tex#L4113) the ChatGPT row has only
  3 cells in a 4-column table — the "Useful?" rating `+` lands in the "Where?" column. Add a `&`:
  `ChatGPT & Related work structure creation & Ch.~\ref{ch:related_work} & + \\`.
- **Typos:** "Genaration" → "Generation" twice ([thesis.tex:4114-4115](latex/thesis.tex#L4114-L4115)).
- **Template not specialized:** declaration still says "Bachelor-, Master-, Seminar-, oder
  Projektarbeit" ([thesis.tex:4091](latex/thesis.tex#L4091)) and has an empty signature/date —
  specialize to "Masterarbeit" and fill in the date.
- **Confirm author:** `\author{Erblina Qeli}` / matriculation 1980490 ([thesis.tex:56-57](latex/thesis.tex#L56-L57)).
  (Erblina Qeli is a genuine committer in the git log, so likely correct — just confirm before submission.)

---

## 2. Medium — factual / caption errors

### 2.1 [M] Jitter prose mixes a single-variant start with an all-variant endpoint
[thesis.tex:2196-2200](latex/thesis.tex#L2196-L2200) — "mean **0.906 → 0.647** for `p2_bound`
as σ goes 0→0.5 across probe and pipeline". The start 0.906 is p2-specific, but **0.647 is the
pipeline mean over *all* bound protographs (p1+p2+p3)** (`notebooks/random_jitter/summary.json`
`pipe_mean_bound_on_sigma05`=0.6473). The true p2_bound σ=0.5 endpoint is ≈0.685 (pipeline) /
0.683 (combined).
**Fix:** use "0.906 → 0.685 for `p2_bound` (pipeline)", or restate as the all-bound mean "0.892 → 0.647".

### 2.2 [M] GEval classification: "vanilla best on Cities" is wrong
[thesis.tex:3367](latex/thesis.tex#L3367) — recomputed from `notebooks/geval.ipynb` (Part 2, classification):
Cities best = `p2_classic` 0.691; vanilla 0.682 is second by 0.009. Vanilla is best only on
Metacritic Movies (0.657).
**Fix:** "competitive across all five and best on Metacritic Movies (within 0.009 of the top on Cities)".

### 2.3 [M] `fig:dbpedia6_final` caption miscounts variants/columns
[thesis.tex:2471-2477](latex/thesis.tex#L2471-L2477) — `scripts/plot_dbpedia6_thesis.py`
plots 6 bars: vanilla, p1/p2/p3_classic, **p2_bound**, cap16. So it is **not** "six headline
variants" (the headline set has 3 bound variants; only p2_bound is shown), and the columns
shared with `tab:dbpedia_per_tc` are **five** (vanilla + 3 classic + p2_bound), not "four".
**Fix:** "for vanilla, the three classic variants, p2_bound, and the capped bound init"; change "four shared columns" → "five".

### 2.4 [M] `fig:dbpedia_epochs` caption: crossover is at epoch 1, not 2–3
[thesis.tex:2546-2549](latex/thesis.tex#L2546-L2549) — recomputed from
`notebooks/dbpedia_compare/results.json`: on the overall mean, p2_bound (0.804 @ ep1) is already
behind vanilla 0.815 / p1_classic 0.827 / p2_classic 0.832 at **epoch 1**. Also the figure shows
two panels (normal | hard) of 4 variants and **no overall-mean panel**, and p2_bound is **never**
overtaken on the hard panel.
**Fix:** say the crossover is at epoch 1 on the overall mean (never on the hard splits); describe the two panels; "bound variants" → "the (p2) bound variant".

### 2.5 [M] `fig:dbpedia_ft010`: 0.853 is not the value plotted (0.847)
[thesis.tex:2778-2780](latex/thesis.tex#L2778-L2780) attributes "climbs … to 0.853" to the figure,
but the figure's cap@16 curve comes from `exp3_ft010.json` and ends at **0.847** (peak 0.851 @ ep4).
The 0.853 is a different run (`exp6_ideas234`) that backs the later tables.
**Fix:** change "0.853" → "0.847" in the figure sentence, or decouple ("its converged value of
0.853 — a different cap@16 run — backs the per-test-case tables").

### 2.6 [M] Norm-ablation caption overclaims monotonic descent
[thesis.tex:4025-4028](latex/thesis.tex#L4025-L4028) — "every classic curve now descends
**monotonically** over the five epochs". In the source `output/synthetic_benchmark_ms_nonorm`,
only **2 of 45** classic runs are strictly monotone; 43 have a late rebound. The main-text prose
([thesis.tex:2055](latex/thesis.tex#L2055)) already makes the correct weaker claim ("all 45 fall
from the first epoch onward").
**Fix:** weaken the caption to match: "every classic curve falls from the first epoch onward
(steep initial drop)", not "monotonically".

### 2.7 [M] Figs 5.1 / 5.2 mislabel the full set as "relation-centric focus"
[thesis.tex:1557](latex/thesis.tex#L1557), [thesis.tex:1565](latex/thesis.tex#L1565) — "the
relation-centric focus test cases **tc01–tc15**". The thesis defines the focus set as **tc07–tc12**
([thesis.tex:1599](latex/thesis.tex#L1599)); these figures span all 15.
**Fix:** "the fifteen synthetic DLCC test cases (tc01–tc15)" — drop "relation-centric focus".

### 2.8 [M] Drift definition missing the absolute-value bars in one place
[thesis.tex:1253-1254](latex/thesis.tex#L1253-L1254) writes `$1 - \cos(\mathbf{v}_0,\mathbf{v}_t)$`,
but every other occurrence (lines 1951, 1968, 3833, 3914) and the generating code
(`scripts/plot/cosine_drift_from_run.py` → `1.0 - np.abs(cos)`) use `$1 - |\cos(\cdot)|$`.
**Fix:** add the `|·|` bars at line 1253.

### 2.9 [M] Table 5.1 `p1_classic` / `p2_classic` columns are not reproducible from any committed run
Table 5.1 ([thesis.tex:1526-1545](latex/thesis.tex#L1526-L1545)): vanilla + all three `*_bound`
columns reproduce **exactly** from `notebooks/synthetic_compare/results.json`, and `p3_classic`
reproduces **exactly** from `output/synthetic_benchmark/summary.json` — but `p1_classic`/`p2_classic`
match **no** committed file (closest is summary.json, off by up to 0.033, e.g. tc15 0.585 vs 0.553,
tc08 0.560 vs 0.530). Deviations are within the ~3 pp noise floor (conclusions unaffected), but the
columns can't be regenerated. (`lr_ablation_table.tex` even notes "the p1/p2 values match the reruns
of Table 5.1 to within run-to-run noise" — i.e. they are an uncommitted rerun.)
**Fix:** regenerate the classic columns from one committed run (the same `summary.json` that backs
`p3_classic`), or add a footnote that the classic columns are a separate from-scratch rerun.

### 2.10 [M] One stale cell in the runtime table
[runtime_per_tc_table.tex](latex/assets/runtime_per_tc_table.tex) tc01 `p1_classic` row reads
Init 0.13 / FT 17.92 / Total 18.32, but current `summary.json` gives 0.20 / 15.66 / 16.13 — the
**only** cell of 84 that doesn't reproduce. It also shifts the p1_classic mean row (21.62 → 21.43).
**Fix:** re-run `scripts/plot_runtime_per_tc.py` to regenerate from current `summary.json`.

---

## 3. Missing descriptions / parameters

- **[L] Stability subset not named** — [thesis.tex:3648](latex/thesis.tex#L3648). The table lists
  only 11 cases (tc01–03, tc05–12); tc04 and tc13–15 are absent (genuinely not in the data, not
  dropped). State the subset: "11 representative cases (tc01–03, tc05–12; tc04 excluded by
  convention, tc13–16 not part of the stability subset)".
- **[L] GEval regression run but unmentioned** — Section 5.9 reports clustering/classification/analogies
  but `notebooks/geval_regression.ipynb` produced full RMSE tables. Add a one-line note that
  regression was also run but scoped out (continuous targets, not schema-aligned).
- **[L] `1600 training entities` over-generalized** — [thesis.tex:886](latex/thesis.tex#L886) says
  "1600 training entities (800/800) that DLCC provides"; true only for tc01–12. tc13–15 use 400
  (200+200) — the table caption [thesis.tex:1698](latex/thesis.tex#L1698) already states this. Scope line 886.
- **[L] `mean cosine ≈ -0.04`** ([thesis.tex:1197](latex/thesis.tex#L1197)) and the **`sqrt`/`log1p`
  max-norms 979 / 47** ([thesis.tex:2943-2945](latex/thesis.tex#L2943-L2945)) are not reproducible
  from committed artifacts (live only in unsaved `.kv` files). Persist them or mark as illustrative.
- **[L] `~29 million IRI triples`** ([thesis.tex:2452](latex/thesis.tex#L2452)) — not traceable to
  any committed log/script. Point to the script that emits it, or soften to an approximation.
- **[L] Three different vanilla baselines unexplained** — 0.735 (Table 5.1), 0.736 (LR ablation),
  0.739 (init strategies) are three independent runs agreeing within noise; add a one-line note so
  it doesn't read as inconsistent.

---

## 4. Low — polish, rounding, reproducibility

- **[L] Generic / stale labels:** `\label{sec}` ([thesis.tex:309](latex/thesis.tex#L309), never
  referenced) and `\label{fig}` ([thesis.tex:364](latex/thesis.tex#L364), referenced at 367) — rename
  to `sec:lit_rdf2vec` / `fig:rdf2vec_example`. `\label{fig:embedding_drift_tc01_tc12}`
  ([thesis.tex:1969](latex/thesis.tex#L1969)) names tc01–tc12 but the figure shows tc01/tc07/tc10.
  Table label `tab:tc13-tc16-motivation` ([thesis.tex:724](latex/thesis.tex#L724)) names tc16 but the
  table is TC13–TC15 → rename `tab:tc13-tc15-motivation`. Unused alias labels: `ch:schema_pretraining`,
  `ch:results`, `sec:nw50`, `sec:results_nw50_classic` (defined, never `\ref`'d).
- **[L] Unused package:** `\usepackage{lipsum}` ([thesis.tex:27](latex/thesis.tex#L27)) — remove.
- **[L] Name capitalization:** the Introduction/title write "RDF2vec"; the rest writes "RDF2Vec"
  (the paper's form). Standardize on "RDF2Vec".
- **[L] Norm table caption** [thesis.tex:2694](latex/thesis.tex#L2694) — "Entity-row L2-norm
  distribution": the numbers are computed over **all** vocab rows (entities+classes+predicates), so
  the vanilla max 17.7 is a class node. Reword to "per-row (vocabulary) L2-norm distribution".
- **[L] LDA appendix caption** [thesis.tex:3730](latex/thesis.tex#L3730) — "first principal component
  (PC1, vertical)"; the script plots the PC **orthogonal to LD1** (panels labelled "PC⊥"). Reword.
- **[L] Epoch-curve count** [thesis.tex:1568](latex/thesis.tex#L1568) — "after each of the 5
  fine-tuning epochs"; the plot shows 6 points (epoch 0 init + 5). Reword to "from initialization
  (epoch 0) through epoch 5".
- **[L] Rounding nits (cosmetic, no effect on conclusions):**
  tc12 p2_bound gain endpoint — clarify "tc12 (0.662 → 0.980, p2_bound)" ([thesis.tex:1602](latex/thesis.tex#L1602));
  `direction_pipeline` negated mean 0.832 vs round(0.8325)=0.833 ([thesis.tex:2408](latex/thesis.tex#L2408));
  `dbpedia_cap_p3` tc10 p2_bound 0.842 vs data 0.841 ([thesis.tex:2842](latex/thesis.tex#L2842));
  LR-ablation fig prints p1 0.757 vs table 0.758; reduced-walks "21.9 s" vs actual full-budget
  vanilla 21.4 s ([thesis.tex:2241](latex/thesis.tex#L2241)).
- **[L] Approximation bounds slightly off:** vocab caption "up to three tokens" — TC04 differs by 4
  ([thesis.tex:3609](latex/thesis.tex#L3609)); "P2 raises … by roughly 23–40 pp" — TC14 gains only
  ~15 pp ([thesis.tex:1687](latex/thesis.tex#L1687)); diagnostics "deviate … by at most a few tenths
  of a point" — actual max ≈2.3 pp ([thesis.tex:1824](latex/thesis.tex#L1824)); stability "noise at
  most ≈2 pp" — fixed-walk max is 2.7 pp ([thesis.tex:3700](latex/thesis.tex#L3700)).
- **[L] Jitter figure caption** [thesis.tex:2192](latex/thesis.tex#L2192) — "**Mean** final accuracy …";
  the script plots one line **per test case**, no averaging. Drop "Mean".
- **[L] Citations:** `zobel2014` and `zobel2004` are duplicate entries for the same Zobel book cited
  in consecutive sentences (will disappear with the abstract rewrite); consider keeping one.
  `groth_rdf2vec_2016` is correct (Ristoski & Paulheim; "Groth" is the ISWC editor) — optionally
  rename the key to `ristoski2016rdf2vec` for clarity.
- **[L] Bound cosine spread** ([thesis.tex:1881-1885](latex/thesis.tex#L1881-L1885)) — "cosine
  typically ≥0.85" is backed by the re-run cache where tc07 p3_bound = 0.795 (<0.85) while the figure
  panel shows 0.944 (different source). The "typically" hedge keeps it true; optionally note the range.
- **[L] Orphan source tables** `latex/tables/acc_norm_vs_nonorm.tex` and `acc_ms_norm_vs_nonorm.tex`
  are **not** `\input` into the thesis; their captions invent a "0.014 noise floor" that conflicts with
  the 3 pp convention. Harmless while unused; fix the wording if ever included.

---

## Appendix: data-provenance map (for regeneration)

Which committed file actually backs each major table — useful when regenerating to close §2.9, §1.4, §1.5.

| Table / figure | Backing data |
|---|---|
| Table 5.1 vanilla + `*_bound` | `notebooks/synthetic_compare/results.json` (exact) |
| Table 5.1 `p3_classic` | `output/synthetic_benchmark/summary.json` (exact) |
| Table 5.1 `p1/p2_classic` | **uncommitted rerun** — not reproducible (§2.9) |
| LR-ablation table (0.025 cols) | `output/synthetic_benchmark` ; (0.0025) `synthetic_compare/results.json` + `notebooks/p3_classic/results.json` |
| Norm-ablation (in-doc) | `output/synthetic_benchmark_ms_norm` vs `…_ms_nonorm` |
| Init-strategies / anchoring appendix table | `notebooks/init_strategies/…` ; `notebooks/anchoring/anchoring_final_acc_per_tc.csv` (non-last-epoch — §1.5) |
| Anchoring figure + prose | `notebooks/anchoring/anchor_last_epoch/…` (last-epoch) |
| Jitter table/figure | `notebooks/random_jitter/summary.json` + `pipeline_results.json` |
| Reduced-walks | `method_proposal/data/` (`@100`), `…/nw50/`, `…/nw25/` |
| Runtime table + figures | `output/synthetic_benchmark/summary.json` |
| Runtime **prose** (1632-1636) | `notebooks/runtime/results.mixed-hardware.bak.json` (**superseded** — §1.4) |
| Direction tables | `notebooks/direction_aware_cardinality/pipeline_grid_results.json` (+ notebook cell for the oracle) |
| DBpedia headline + degree probe | `notebooks/dbpedia_compare/results.json`, `degree_probe.json` |
| DBpedia norms table | `notebooks/dbpedia_compare/*.kv` + `dbpedia_investigate/artifacts/build_stats.json` |
| DBpedia cap / ft010 figure | `notebooks/dbpedia_investigate/exp3_ft010.json` |
| DBpedia compress / IDF / hierarchy / LR | `dbpedia_investigate/exp6_ideas234`, `output/dbpedia/p2bound_*`, `p2_bound_*` |
| GEval clustering/classification/analogies | `notebooks/geval.ipynb` |
| Stability appendix | `notebooks/5_times_varywalks/results.json` (+ `5_times/` fixed-walk) |
