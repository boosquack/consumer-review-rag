---
title: 'EDA notebook with written insights'
type: 'feature'
created: '2026-09-16'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '0272cd1b316bc24d3b4aee5e24b5f2d0b48366d8'
context:
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/eda-charts.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CAP-2 has no deliverable. `notebooks/` is empty, so there is no analyst view of rating, volume, sentiment, and theme patterns per brand, and the README has no measured EDA highlights to draw on.

**Approach:** Put the aggregations (VADER scoring, monthly series with small-sample flags, sentiment/rating agreement, top terms by rating band) in an importable `src/eda.py`, so story 7's app reuses them. `notebooks/01_eda.ipynb` imports those functions, draws every chart in `eda-charts.md` with a takeaway based on measured numbers, documents the cleaning rules and quality report, and keeps a running insights cell.

## Boundaries & Constraints

**Always:** Read `config.CLEAN_PARQUET` and `config.CLEAN_REPORT_JSON`; put new constants (small-sample threshold, VADER cutoffs, extra stopwords) in `src/config.py`. Score sentiment on the original `review_text`. Base every takeaway on numbers the notebook prints, and say when a pattern is weak. Frame findings as describing the sampled reviews, not the customer population, and call VADER a lexicon heuristic. Commit-ready notebook: executed top to bottom, with outputs saved so GitHub renders the charts. Leave changes uncommitted.

**Never:** Add a dependency (pandas, matplotlib, seaborn, scikit-learn, vaderSentiment, jupyter, and nbconvert are installed). Reimplement cleaning in the notebook or re-run the download. Do topic modeling (stretch only). Write index, retrieval, generation, or app code. Hand-edit `SPEC.md`, its companions, `stories.yaml`, `.memlog.md`, or `AGENTS.md`. Print full review bodies in bulk (short illustrative snippets are fine).

**Decisions (planning, 2026-09-16):**
- **Time series:** monthly per brand across the full 2005–2023 range. Months with fewer than 30 reviews are drawn as de-emphasized markers and excluded from the average-rating trend line. 2023 is labelled partial (data ends 2023-09-01).
- **Sentiment bands:** standard VADER compound cutoffs: ≥0.05 positive, ≤−0.05 negative, otherwise neutral.
- **Top terms:** TF-IDF over (brand × rating band) documents, where 1–2★ is low and 4–5★ is high and 3★ is excluded. Use English stopwords plus brand-name and generic product tokens so the terms show themes, not "shampoo" or "pantene".

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | Parquet + report present | Notebook executes clean; all 6 charts render with takeaways | N/A |
| Missing parquet | `reviews_clean.parquet` absent | First data cell fails with a message naming the two pipeline commands | Raise with hint |
| Small-sample month | A brand-month with n < 30 | Flagged `small_sample=True`; excluded from trend line, still plotted de-emphasized | N/A |
| Undated rows | `review_date` null (0 in real data) | Excluded from time series, counted in sentiment/rating/terms; undated count printed | N/A |
| Empty band | A brand has no rows in one rating band | Top-terms returns an empty list for that cell, no crash | N/A |
| Empty/odd text | Text that VADER scores 0.0 | Classified neutral | N/A |

</frozen-after-approval>

## Code Map

- `src/config.py` -- Add EDA constants next to "Cleaning rules": `EDA_MIN_MONTH_REVIEWS = 30`, `VADER_POS = 0.05`, `VADER_NEG = -0.05`, extra stopwords, `NOTEBOOK_EDA`. Do not touch brand patterns or Groq constants.
- `src/eda.py` -- New. Pure functions over the clean DataFrame: `load_clean()` (raises with the hint if missing), `add_sentiment()`, `monthly_stats()` (count, mean rating, `small_sample`), `sentiment_rating_agreement()`, `top_terms_by_band()`, plus small plotting helpers the app can reuse. The docstring follows `src/clean.py`.
- `src/clean.py` -- Reuse only: `config.UNDATED_BUCKET` and the `date_bucket` column; do not modify.
- `notebooks/01_eda.ipynb` -- New. The first cell puts the repo root on `sys.path` (the notebook's working directory is `notebooks/`), then sections for data and cleaning rules (store-only brand match, dedup of ≥50-character normalized text, from story 2), charts 1–6 each followed by a takeaway cell, then the insights cell.
- `tests/test_eda.py` -- New. Stdlib `unittest` with inline DataFrame fixtures and no parquet, matching `tests/test_clean.py`.
- Parquet facts (measured): 53,872 rows; `review_date` is `datetime64[us, UTC]`; `rating` is int64; ~42–53% of brand-months have <30 reviews (mostly 2005–2012). Mean VADER compound: H&S 0.45, Herbal Essences 0.55, Pantene 0.49. 32% of 1★ reviews score positive, which is material for chart 5. Full-corpus VADER takes ~3 s, so no cache.

## Tasks & Acceptance

**Execution:**
- [x] `src/config.py` -- add the EDA constants -- constants live only here.
- [x] `src/eda.py` -- implement the aggregations and plot helpers -- app reuse, testability.
- [x] `tests/test_eda.py` -- cover every matrix row except happy path/notebook -- matrix requires tests.
- [x] `notebooks/01_eda.ipynb` -- build, execute in place, write takeaways from printed numbers -- CAP-2.

**Acceptance Criteria:**
- Given a fresh kernel, when `nbconvert --execute` runs the notebook, then it exits 0 with no error outputs.
- Given the executed notebook, when read, then each of the 6 `eda-charts.md` charts has a figure followed by a 1–2 sentence takeaway citing a measured number, and a final insights cell lists 3–5 README-ready bullets.
- Given the notebook, when searched, then it defines no cleaning, sentiment, or aggregation logic that duplicates `src/`.

## Implementation Notes

- **Files:** `src/config.py` (EDA constants), `src/eda.py`, `tests/test_eda.py`, `notebooks/01_eda.ipynb` (new, executed, outputs saved, ~455 KB). No dependency added.
- **Stopwords beyond the Code Map wording:** besides brand and generic product tokens, filler praise ("like", "just", "good", "love"…), contraction fragments, and `br` (4,346 reviews carry `<br>` markup) — without them filler led all six cells.
- **Small-sample share measured lower than planned:** 31.7–38.5% of *observed* brand-months have n < 30 (planning's 42–53% counted empty months); they hold only 1.8–6.9% of reviews.
- **Takeaways are hardcoded markdown** checked against printed output; a data rebuild requires re-checking them by hand.
- **Orchestrator verification:** 68 tests OK (16 new); fresh `nbconvert --execute` to a scratch copy exits clean with 0 error outputs and sequential execution counts.

**Review pass 1 patches (2026-09-16).** Applied by the implementation subagent; 12 patch entries, nothing deferred.
- `src/eda.py` -- `term_mention_share` matches an optional plural `s`; `top_terms_by_band` keeps accented words; empty term cells read "no terms".
- `tests/test_eda.py` -- 16 to 22 tests: ranking/truncation, per-brand sentiment shares, mention shares, peak months, trend summary, accented tokens.
- `notebooks/01_eda.ipynb` -- mention shares printed per brand; Chart 2–6 takeaways and insights rewritten to printed numbers (itchy is not an H&S complaint theme, greasy is shared, smell(s) leans 4–5★, Herbal Essences' 2015→2022 drop is 0.10); Chart 5/6 takeaways cut to 2 sentences; labels built from config.
- **Re-verified by the orchestrator:** 74 tests OK; fresh `nbconvert --execute` to a scratch copy exits 0 with 0 error outputs; takeaway numbers spot-checked against printed tables.
- **Not committed:** AGENTS.md forbids commits unless the maintainer asks, which overrides the workflow's local-commit step.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-16). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V, `VO` = its Other findings). Every finding logged below (grouped by shared root cause); nothing routed to intent_gap or bad_spec, so no loopback.

| ID | Finding | Verdict | Route | Evidence |
|---|---|---|---|---|
| B1+E17 | Chart 6 / insight 4 present pooled mention shares as brand-specific | medium | patch | Measured per brand: itchy H&S 6.8% vs pooled 2.7%; greasy highest for Herbal Essences (6.1%), so "Pantene greasy" is wrong. |
| B2 | `term_mention_share` misses plurals | medium | patch | Measured: bottle(s) 16.4% vs 11.8% low; smell(s) 16.6% low vs 21.6% high reverses the "equal in both bands" claim. |
| B3+E15+E16 | Insight 2 overstates rating drift and calls it consistent/small against noise | medium | patch | Printed yearly table: Herbal Essences 2015 4.25 → 2022 4.15 (0.10); printed monthly std 0.15–0.24 is below the claimed drop. |
| B6+E18 | Takeaways claim beyond printed evidence (campaign-shaped, 2015 plateau, product-line names, dandruff vocabulary) | medium | patch | Pantene and H&S 4–5★ lists are mostly generic words; one burst month does not show a pattern. Violates the frozen "base every takeaway on printed numbers". |
| B5 | Chart 5 takeaway 3 sentences; Chart 6 run-on; misquoted snippet | low | patch | AC requires 1–2 sentences; printed snippet reads "Thw smell of the product is great however". Direct correction. |
| V1 | Top-term ranking/truncation untested | medium | patch | Pre-verified: `argsort()[:n]` keeps all 16 eda tests green. |
| V2 | `sentiment_summary` shares untested | medium | patch | Pre-verified: `normalize="columns"` keeps tests green; Chart 4 numbers depend on it. |
| V3 | `term_mention_share` untested | medium | patch | Pre-verified by search: no test calls it; backs the headline complaint theme. |
| V4+B8 | `peak_months`, `trend_summary` (and other helpers) untested | medium | patch | Pre-verified: `idxmin` / `mean_rating` swaps pass. Plot-helper smoke tests from B8 rejected: low, visual, no crash shown on reachable data. |
| E4 | ASCII-only token pattern drops accented words | low | patch | Probe returned [] for "champú suavísimo café"; 0.65% of reviews carry accented letters. One-argument direct correction. |
| E5 | Empty top-terms cell labelled "no reviews" when reviews existed | low | patch | Real for all-stopword cells; label change is a direct correction. |
| B10+E11 | `misread.sample(5)` raises on <5 rows | low | patch | Real mechanism; `min()` is a direct correction. |
| B7+E14 | Notebook hardcodes n<30, 2023 partial, 3 subplot rows | low | patch | Labels go wrong if config changes; f-strings from config are direct corrections. Colour-dict KeyError part rejected: brand set is locked. |
| B4+E9 | Partial final month (2023-09 holds 1 Pantene review) and shading past data end | low | rejected | Real but cosmetic: the chart already labels 2023 partial; trimming needs added logic. |
| B9 | Stopwords remove content words ("head", "pro"); plurals split weight | low | rejected | Trade-off documented in config comment; stemming adds complexity for a cosmetic top-term effect. |
| B11 | `config.NOTEBOOK_EDA` unused | low | rejected | The Code Map specified it; no named harm. |
| B12+VO1 | Hardcoded takeaways can drift after a rebuild | low | rejected | Corpus is a fixed snapshot; recorded in Implementation Notes. Automated prose checks add machinery. |
| E1 | NaN `review_text` breaks `" ".join` | false | rejected | Parquet has 0 null `review_text`; `src/clean.py` drops empty and non-string bodies. |
| E8 | NaN/None text miscounted in `term_mention_share` | false | rejected | Same: 0 null texts reach EDA. |
| E13 | `date_range` None crashes notebook | false | rejected | Requires every row undated; measured undated 0. |
| E10 | `small_sample_share` inflated on `complete_months` output | false | rejected | No caller passes `complete_months` output; the notebook passes `monthly_stats`. |
| E2+E3 | Untyped empty `monthly_stats` frame breaks `plot_monthly_rating` | low | rejected | Only reachable with all-undated data (measured 0); fix adds dtype/guard branches. |
| E6 | Negative `n` slices wrong | low | rejected | No caller passes negative n; fix is a guard. |
| E7 | Empty word gives 100% share | low | rejected | No caller passes an empty word; fix is a guard. |
| E12 | Bare StopIteration when kernel runs outside repo | low | rejected | Loud failure in an unsupported setup; the notebook lives in the repo. |

## Verification

**Commands:**
- `uv run python -m unittest -v` -- expected: 52 existing plus the new eda tests, all OK
- `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb` -- expected: exit 0
- `grep -c '"output_type": "error"' notebooks/01_eda.ipynb` -- expected: 0

**Manual checks:**
- Open the notebook and confirm each takeaway's number matches the printed output above it.
