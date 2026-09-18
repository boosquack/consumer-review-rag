---
title: 'Evaluation gold set, metrics, and results write-up'
type: 'feature'
created: '2026-09-17'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '153d933a52971c79a96bbb47d442d74fb50c5bca'
context:
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/evaluation.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CAP-5 has no implementation. Retrieval and generation work on hand-picked examples, but nothing measures them. The known weaknesses have no numbers yet: thin answers to broad questions, brand names in the query crowding retrieval, and over-generalizing from one review. The README and application cannot yet report honest results.

**Approach:** Write 18 gold questions to `eval/gold_questions.json`, mixing factual, comparative, filtered, and unanswerable questions. `src/evaluate.py` runs them through `retrieve()` and `answer()` and scores automated metrics: retrieval hit rate, citation validity, uncited rate, and refusal accuracy. It saves every answer with its sources to `eval/answers.json`. Faithfulness is graded per question with notes. `eval/results.md` reports every number and discusses every failure.

## Boundaries & Constraints

**Always:** Label relevance independently of the embedding model. Each answerable question carries a case-insensitive regex `relevant_pattern`, and a retrieved review is relevant when its original `review_text` matches. Confirm each pattern matches at least 10 reviews within the question's filters, and print that count. Pace Groq calls to stay under 8K tokens/min, using the measured usage and honoring `rate_limited`. Freeze the gold set (a `frozen_at` timestamp in the file) before the first scored run. Grade faithfulness against the saved `eval/answers.json`, not a fresh run. Report failures with their question id and evidence. Leave changes uncommitted.

**Never:** Change `src/retrieve.py`, `src/generate.py`, `src/index.py`, or the prompt to improve the scores. Pick or tune gold questions after seeing the system's answers. Claim population-level conclusions. Add a dependency. Hand-edit `SPEC.md`, its companions, `stories.yaml`, `.memlog.md`, or `AGENTS.md`.

**Decisions (planning, 2026-09-17):**
- **Gold set (18):** 14 answerable: 12 specific (factual, comparative, brand- and band-filtered, including the headline "what do 1-star Pantene reviewers complain about" and story 4's miss query 3) and 2 broad theme questions. 4 unanswerable (outside the corpus, e.g. prices in Japan, ingredient safety claims, competitor brands, events after 2023).
- **Retrieval metrics:** hit@k (any of the top k=6 relevant) and precision@k (share of the 6 that are relevant), per question and averaged over answerable questions.
- **Faithfulness grader (maintainer, 2026-09-17):** the implementing agent grades each saved answer against its cited reviews, pass/fail with a note. `results.md` states plainly that this is agent judgement, not an independent grader.
- **Scope (maintainer, 2026-09-17):** measure the current system only. Known weaknesses (brand-name crowding, thin broad answers, over-generalization) are reported as limitations, not fixed here.
- **Generation metrics:** citation validity (every citation is among that call's sources), uncited rate, refusal accuracy on unanswerable questions, and false-refusal rate on answerable ones.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Full run | Index, parquet, key present | Per-question table and totals printed; `eval/answers.json` written | N/A |
| Retrieval only | `--retrieval-only` | Hit@k and precision@k, no Groq calls | N/A |
| Rate limited | `answer()` returns `rate_limited` | Wait, retry once, then record the question as `rate_limited` and exclude it from generation metrics, reported | N/A |
| Weak pattern | Pattern matches < 10 reviews within filters | Exit 1 naming the question id before any Groq call | Raise |
| Invalid gold file | Missing field, unknown brand or band, duplicate id | Exit 1 naming the problem | Raise |
| No hits | Filters return no reviews | Hit@k 0 for that question, `no_reviews` recorded | N/A |

</frozen-after-approval>

## Code Map

- `src/retrieve.py` -- `retrieve(query, k, brand, rating_band)` returns hits with `review_id`, `review_text`, `brand`, `rating`, `distance`. Raises ValueError on bad filters and RuntimeError on a stale index.
- `src/generate.py` -- `answer(question, brand, rating_band, k, *, client, retriever)` returns `Answer` with `status`, `text`, `citations`, `sources`, `invalid_citations`, `uncited`, `usage`. Pass the same `retriever` so retrieval runs once per question. It never raises for API failures.
- `eval/retrieval_checks.md` -- Story 4 hand checks; reuse queries 1, 3, 4, 7, 8 as gold candidates with patterns. Measured pattern counts: headache 72, hair loss 234, leaking or broken bottle 465, color-treated 511, sulfate 713.
- `eval/gold_questions.json` -- New. A list of `{id, question, type, brand|null, rating_band|null, answerable, relevant_pattern|null, notes}`.
- `src/evaluate.py` -- New. `main() -> int` and `sys.exit(main())`; pure scoring functions (`hit_at_k`, `precision_at_k`, `citation_valid`, `summarize`) separate from I/O; `--retrieval-only`. Put the pacing and the gold/answers paths in `src/config.py`.
- `eval/answers.json` -- New, committed. The graded run: question, status, text, citation numbers, sources (review_id, brand, rating, first 300 characters), usage, and timestamp.
- `eval/results.md` -- New. Setup (model, k, date, corpus size), a metrics table, a per-question table, faithfulness grades with notes, and failure commentary by category (retrieval misses, crowding, over-generalization, refusals), plus limitations.
- `tests/test_evaluate.py` -- New. Stdlib `unittest` over the scoring functions, gold validation, and pacing, with a fake retriever and fake answer. No network.

## Tasks & Acceptance

**Execution:**
- [x] `eval/gold_questions.json` -- write and freeze 18 questions with patterns -- CAP-5 gold set.
- [x] `src/config.py`, `src/evaluate.py` -- validation, runner, scoring, JSON output, CLI -- CAP-5 metrics.
- [x] `tests/test_evaluate.py` -- every matrix row plus scoring functions -- matrix requires tests.
- [x] Run `--retrieval-only`, then the full run; grade faithfulness as the agent -- measured evidence.
- [x] `eval/results.md` -- numbers, per-question table, failure commentary -- CAP-5 write-up.

**Acceptance Criteria:**
- Given the full run, when `src.evaluate` prints its summary, then hit@k, precision@k, citation validity, uncited rate, refusal accuracy, and false-refusal rate appear, and they match `eval/results.md`.
- Given `eval/results.md`, when read, then every question below hit, citation, or faithfulness pass is named with evidence, and the grader of faithfulness is stated.
- Given `eval/gold_questions.json` and `eval/answers.json`, when compared, then `frozen_at` precedes the run timestamp and every graded question id matches the frozen set.

## Implementation Notes

- **Files:** `eval/gold_questions.json`, `src/evaluate.py`, `tests/test_evaluate.py` (35 tests), `eval/answers.json`, `eval/faithfulness.json`, `eval/results.md` (new); `src/config.py` (evaluation constants). `src/retrieve.py`, `src/generate.py`, `src/index.py`, and the prompt are unchanged.
- **Deviations from the Code Map:** the gold file is an object `{frozen_at, questions}` so the freeze timestamp lives in the file; `eval/faithfulness.json` was added so the evaluator can print agent-graded faithfulness; `--rescore` reprints metrics from the saved files without Groq.
- **Measured run (2026-09-16T16:12:28Z, k=6, gpt-oss-120b):** hit@6 0.86 (12/14); precision@6 0.70; citation validity 1.00 (13/13); uncited 0.00; refusal accuracy 1.00 (4/4); false-refusal 0.07 (1/14, q02); faithfulness, agent-graded, 14/18 overall and 9/13 of answered; 13,496 tokens, 99.9 s paced sleep, no rate limits.
- **Failures:** retrieval misses q02 (story 4's known miss, 0/6, then a false refusal) and q14 (six one-line "I love Pantene" reviews, a content-free answer); headline q01 precision 2/6; faithfulness fails q05 ("safe" not stated), q07 (brand comparison from a 4-vs-2 split in 6 hits), q10 ("didn't help itchy scalp" read as "made it itchy"), and q12 (misattributed and inverted citations). Automated citation validity cannot see q12's semantic errors, and the uncited flag missed q01's prose "Review 6". Regex label noise in both directions (q03, q05, q07, q09, q11) is documented, not corrected.
- **Orchestrator verification:** 190 tests OK; `--rescore` reproduces the summary; gold file mtime 16:10:28Z < `frozen_at` 16:12:00Z < run 16:12:28Z; spot-checked q02, q05, and q14 answers against their saved sources, and the grades are consistent with the notes. The freeze precedes the run by only 28 s, which shows ordering, not that the questions were written without seeing any system output.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-17). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V, `VO` = its Other findings). Every finding is logged below, grouped by shared root cause. Nothing routed to intent_gap or bad_spec, so there is no loopback. No patch needs a new Groq run or a grade change.

| ID | Finding | Verdict | Route | Evidence |
|---|---|---|---|---|
| B5+VO1 | Headline faithfulness 14/18 counts refusals as automatic passes | medium | patch | Confirmed by `--rescore`: 5 refused questions are graded pass; answered-only 9/13 appears only in hand-written results.md. |
| B6+E6 | False-refusal rate ignores `no_reviews` while refusal accuracy counts it | low | patch | Confirmed asymmetry in `summarize`; a direct correction to use `DECLINE_STATUSES`. |
| B8+E1 | A full run silently orphans the grades, even when every call errors | medium | patch | Confirmed: `answers.json` is overwritten unconditionally, and `load_grades` then returns None. Backup-before-overwrite adds no CLI surface. |
| B10+E12+B17 | Freeze and "code unchanged" claims cannot be verified | medium | patch | `frozen_at` is self-declared (28 s before `run_at`); rescore checks only ids and frozen_at. A gold hash and commit SHA on future runs, stated as absent for the current run. |
| B15+E10+E11 | Pacer forgets the measured max after the window; retry sleep uncounted | low | patch | Docstring contradicts code; direct correction. |
| V1+V2+V3+V4+B16 | Rescore-with-grades, full-run pacing, id-mismatch, and mid-run retriever failure untested | medium | patch | Pre-verified by mutation: dropping the run_at comparison, setting `pacer = None`, deleting the id check, or removing the except all keep 35 tests green. V4 filed as defer by the layer; patched because it is a test-only addition. |
| B1+E13 | q01 content-free count differs between results.md and faithfulness.json | low | patch | Saved sources: [1]–[3] anger, [4]–[5] concrete, [6] price remark; results.md says 4 and 2. |
| B2 | q11 corrected precision stated as 0.50–0.67 | low | patch | One false negative and one false positive cancel to exactly 0.50. |
| B11 | Gold set not blind: questions reused from story 4 with outcomes seen | medium | patch | The spec's Code Map directed reuse of story 4 queries; the honest fix is a stated limitation in results.md. |
| B4 | Relevance labeled on the full review while the model sees the chunk | low | patch | Identical for unsplit reviews (98.3%); differs for split ones such as q11 [4]. Documented as a limitation rather than re-scored. |
| B3 | Saved 300-character snippets make grades unauditable | low | rejected | Every saved source carries `review_id`, so its full text can be looked up in the parquet; stated in results.md as part of B4's patch. |
| B12 | Gold set drops the "expected supporting review(s)" in evaluation.md | false | rejected | evaluation.md says "review(s) **or theme**", and the approved frozen decision chose regex theme patterns. |
| B13 | No per-type breakdown | low | rejected | Nice to have; 1–4 questions per type are too few for per-type rates to mean much. |
| B14 | Citation flags assume `hits` and `result.sources` share order | false | rejected | `answer()` returns the same `hits` list it was given as `sources`, unreordered. The `None` display case is not reached. |
| B9+E2 | A mid-run crash discards finished answers | low | rejected | A full run takes about 3 minutes; checkpointing adds I/O branches. The crash path now exits 1 cleanly and preserves the graded file (V4, B8). |
| B16 (repo test) | Gold-file test pinned to 18 questions | low | rejected | The frozen intent fixes the gold set at 18; the pin guards that decision. |
| E3+E4+E5+E7 | Missing parquet, corrupt JSON, missing `pass`, unhashable brand give tracebacks | low | rejected | Inputs are agent-written, frozen files or pipeline outputs; loud failure is acceptable for an offline runner, and each fix is a guard. |
| E8 | Null rating in a source hit aborts the run | false | rejected | `retrieve._collapse` casts `int(meta["rating"])`; Chroma stores ints. |
| E9 | Rescore labels use the current `RETRIEVAL_K`, not the saved `k` | low | rejected | k has not changed; the fix threads a parameter for a hypothetical config edit. |

## Verification

**Commands:**
- `uv run python -m unittest -v` -- expected: 155 existing plus the new tests, all OK, no network
- `uv run python -m src.evaluate --retrieval-only` -- expected: hit@k and precision@k per question, exit 0
- `uv run python -m src.evaluate` -- expected: full summary, `eval/answers.json` written, exit 0
