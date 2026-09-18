# Evaluation results (story 6, CAP-5)

These are measured results for the current system over an 18-question hand-written gold set, and they include the failures. With only 18 questions and 6 reviews per question, they show how the system behaves on these questions. They say nothing about all reviewers or all questions.

**The faithfulness grades are the implementing agent's judgement (Claude), not an independent grader.** The automated metrics (hit@k, precision@k, citation validity, uncited rate, refusal accuracy, false-refusal rate) are computed by `src/evaluate.py`.

## Setup

| Item | Value |
|---|---|
| Gold set | `eval/gold_questions.json`, 18 questions: 14 answerable (12 specific, 2 broad) and 4 unanswerable. Frozen at `2026-09-16T16:12:00Z`, before the first scored run |
| Graded run | `eval/answers.json`, `run_at` `2026-09-16T16:12:28+00:00` (questions answered 16:12:41 to 16:14:44 UTC) |
| Corpus | 53,872 cleaned reviews (Pantene 28,977, Herbal Essences 15,532, Head & Shoulders 9,363), indexed as 55,001 chunks |
| Retrieval | `all-MiniLM-L6-v2`, cosine, k = 6, brand and rating-band pre-filters |
| Generation | Groq `openai/gpt-oss-120b`, temperature 0.1, reasoning effort low |
| Usage | 13,496 total tokens over 18 calls (at most 1,152 per call). 0 rate-limited, 0 errors. The pacer waited 99.9 s in total |
| Code | `src/retrieve.py`, `src/generate.py`, `src/index.py`, and the prompt were not changed for this evaluation |

Reproduce:

```sh
uv run python -m src.evaluate --retrieval-only   # retrieval metrics, no Groq calls
uv run python -m src.evaluate                    # full run; backs up the previous eval/answers.json, then replaces it
uv run python -m src.evaluate --rescore          # re-print these numbers from the saved run and eval/faithfulness.json
```

**How relevance is labelled.** An answerable question has a case-insensitive regex, `relevant_pattern`. A retrieved review counts as relevant when its original `review_text` matches that regex. The embedding model plays no part in the label. Before any Groq call, each pattern was checked to match at least 10 reviews within its filters. The counts ranged from 126 (q11) to 16,211 (q08). A regex label is a rough proxy, and its known errors are listed under [Label noise](#label-noise).

## Metrics

| Metric | Result | Definition |
|---|---|---|
| Retrieval hit@6 | **0.86** (12/14) | Share of answerable questions with at least one relevant review in the top 6 |
| Retrieval precision@6 | **0.70** (mean of 14) | Share of the 6 retrieved reviews that are relevant, averaged over answerable questions |
| Citation validity | **1.00** (13/13) | Share of answered questions where every parsed citation is one of that call's sources and no out-of-range number was removed |
| Uncited rate | **0.00** (0/13) | Share of answered questions with no parsed citation |
| Refusal accuracy | **1.00** (4/4) | Share of unanswerable questions answered with the refusal sentence |
| False-refusal rate | **0.07** (1/14) | Share of answerable questions that were refused or returned no reviews |
| Faithfulness (agent-graded), overall | **0.78** (14/18) | Pass means no invented or overstated claims relative to the cited reviews. Includes the 5 refusals, all graded pass |
| Faithfulness (agent-graded), answered only | **0.69** (9/13) | The same grades over the 13 questions that were answered. This is the more meaningful figure, because a faithful refusal passes automatically |

No question was rate-limited or errored, so no question was left out of the generation metrics.

## Per question

"Rel" is the number of the 6 retrieved reviews that match the question's pattern.

| id | Type | Filters | Question | Rel | hit | P@6 | Status | Cited | Faithful |
|---|---|---|---|---|---|---|---|---|---|
| q01 | filtered | Pantene, low | what do 1-star Pantene reviewers complain about | 2/6 | 1 | 0.33 | answered | 1,2,3,5 | pass (thin) |
| q02 | factual | Herbal Essences | Herbal Essences smell is too strong or gives headaches | 0/6 | **0** | 0.00 | **refused** | none | pass (faithful refusal, wrong outcome) |
| q03 | filtered | low | Does the shampoo make hair fall out or cause hair loss? | 5/6 | 1 | 0.83 | answered | 1–6 | pass |
| q04 | factual | none | Did the package arrive leaking or with a broken bottle? | 6/6 | 1 | 1.00 | answered | 1–6 | pass |
| q05 | filtered | Pantene, high | Is Pantene safe for color-treated hair? | 5/6 | 1 | 0.83 | answered | 1,2,4 | **fail** |
| q06 | factual | Head & Shoulders | Does Head & Shoulders actually get rid of dandruff? | 6/6 | 1 | 1.00 | answered | 1,3,4,6 | pass |
| q07 | comparative | low | Which brand gets more complaints about residue or buildup left on the hair? | 6/6 | 1 | 1.00 | answered | 1–6 | **fail** |
| q08 | comparative | none | How do reviewers describe the scent of Pantene compared with Herbal Essences? | 6/6 | 1 | 1.00 | answered | 1,2,4,5 | pass (thin) |
| q09 | filtered | mid | Is it worth the price or too expensive for the size? | 4/6 | 1 | 0.67 | answered | 2,3,5,6 | pass |
| q10 | filtered | Head & Shoulders, low | Does Head & Shoulders cause an itchy, burning, or irritated scalp? | 6/6 | 1 | 1.00 | answered | 1–6 | **fail** |
| q11 | filtered | Pantene, low | Do reviewers say the new Pantene formula is worse than the old one? | 3/6 | 1 | 0.50 | answered | 1,2,3,5 | pass |
| q12 | factual | none | What do reviewers say about sulfates in these shampoos? | 6/6 | 1 | 1.00 | answered | 1–6 | **fail** |
| q13 | broad | low | What are the most common complaints in low-rated reviews? | 4/6 | 1 | 0.67 | answered | 1–6 | pass |
| q14 | broad | Pantene, high | What do people like most about Pantene? | 0/6 | **0** | 0.00 | answered | 1–6 | pass (empty) |
| u01 | unanswerable | none | How much does Pantene shampoo cost in Japan? | – | – | – | refused | none | pass |
| u02 | unanswerable | none | Have clinical studies proven the ingredients in Head & Shoulders safe for long-term use? | – | – | – | refused | none | pass |
| u03 | unanswerable | none | What do reviewers say about the scent of Redken All Soft shampoo? | – | – | – | refused | none | pass |
| u04 | unanswerable | none | How did reviewers react to the Herbal Essences products launched in 2025? | – | – | – | refused | none | pass |

## Faithfulness grades

The grades and notes also live in `eval/faithfulness.json`, which is tied to this run's `run_at`. Each answer was read against the chunk of each cited review that the model saw.

- **q01, pass.** The claims match [1][2][3][5], and the price remark about review 6 is hedged. The answer is thin, though. Three of the 6 hits are angry reviews with no content. The model skipped the one concrete "dry and brittle hair" complaint in [4]. It also wrote "Review 6" in prose instead of `[6]`, so that citation was not parsed.
- **q02, pass.** The refusal is faithful, because none of the 6 retrieved reviews mentions a strong scent or headaches. It is still the wrong outcome for a user, because the corpus does contain such reviews (see retrieval misses).
- **q03, pass.** All cited reviews report hair loss. The opening "Yes" reads as a general verdict drawn from 6 low-band reviews, and it drops [1]'s own hedge ("cant 100% sure it causes this").
- **q04, pass.** All 6 cited reviews describe leaking, cracked, or busted bottles.
- **q05, FAIL.** The answer says reviewers call Pantene "safe" for color-treated hair, citing [1][2][4]. None of them says safe. They say it is great or works well on colored hair. All the evidence is 5-star because of the high-band filter, so a "Yes" verdict is one-sided by construction.
- **q06, pass.** The quotes are accurate, and [3][4] do say the dandruff is gone. "Eliminates" is too strong for [1] ("helped in past") and [6] ("helps"), and the answer omits the doubtful 3-star [2].
- **q07, FAIL.** "Pantene receives more complaints about residue … than Herbal Essences" is a claim about brand frequency, based on a 4-to-2 split among 6 retrieved reviews. Top-6 similarity cannot measure frequency. [1] is also about a hairspray, not a shampoo.
- **q08, pass.** The quotes are accurate and credited to the right brands. The answer is thin, skips [6] (a Pantene review saying there was barely any fragrance), and makes no real comparison.
- **q09, pass.** The split verdict is supported by [2][3] against [5][6]. The quote "good value" comes only from [3], but it is cited to [2][3] together.
- **q10, FAIL.** The answer says [1][2][3][5] report the shampoo "left their scalp itchy or made it worse". Those four actually say it *did not help* an itchy scalp they already had. Only [6] (dry and itchy after use) and [4] (burning) describe irritation caused by the product, so the conclusion that it "can cause" irritation rests on 2 reviews, not 6.
- **q11, pass.** [1][2] say the formula changed for the worse, and [3] says it is not the formula Pantene used for 20 years. [5] fits "past results" only loosely.
- **q12, FAIL.** There are two errors. [2] is cited for sulfates "drying hair and stripping oils", but it is about a product mislabelled as sulfate-free. [3] ("Don't buy non Sulfate shampoo") is read as concern about sulfates, which is the opposite of what it says.
- **q13, pass.** Each listed complaint appears in its cited reviews, and the answer states no frequencies. Six reviews cannot establish "most common", and the answer leaves out the hair loss in [3].
- **q14, pass.** The answer is faithful but empty. All 6 hits are generic "I love Pantene" reviews, so it cannot say *what* people like. "Use them regularly" fits only [1][3][6], yet it is cited to all six.
- **u01–u04, pass.** All four were correct refusals. None of the retrieved reviews covers prices in Japan, clinical studies, Redken, or 2025 launches. For u04 the prompt shows no review dates, so the model could not have checked dates anyway.

## Failures by category

### Retrieval misses (hit@6 = 0)

- **q02 (brand-name crowding).** 0 of 6 hits mention a strong scent or headaches, even though 335 Herbal Essences reviews match the pattern. The hits are short reviews that repeat "Herbal Essences" plus "smell", and 4 of them praise the scent. This reproduces the story 4 miss (query 3) exactly: the brand name in the query pulls toward reviews that repeat it, and the brand filter already restricts to that brand. Generation then refused, so this miss is also the only **false refusal** (1/14). The refusal was faithful to what was retrieved, so the harm comes from retrieval, not from the model.
- **q14 (short generic praise).** 0 of 6 hits say what reviewers like. All six are one-line "I love Pantene" reviews that sit closest to a short, brand-heavy question. Brand crowding and short-review bias act together here. The model did not refuse. It returned an empty "people love Pantene" answer with six citations, and citation validity and uncited rate cannot catch that.

### Low precision (hit, but under half relevant)

- **q01 (headline demo question), P@6 = 0.33.** Of the 6 hits, 3 are content-free angry reviews ([1][2][3]), 2 name concrete complaints ([4] dry and brittle hair, [5] dull, staticky hair after a formula change), and 1 is an unclear price remark ([6]). The regex labels only [4] and [5] as relevant. The model used [5] and skipped [4]. A meta question like "what do they complain about" has no single semantic target, so top-6 similarity cannot give a spread of themes. Story 4 predicted this.
- **q11, P@6 = 0.50.** Correcting the known label errors leaves it at 0.50: [3] is a false negative and [4] is a false positive, so they cancel (see label noise). Two hits ([5], [6]) say only that the product is worse than other Pantene products.

### Over-generalization (faithfulness failures)

- **q07.** The answer turns a 4-to-2 brand split among 6 retrieved reviews into "Pantene receives more complaints". The prompt's rule against generalizing did not stop a frequency claim on a comparative question.
- **q05.** A 5-star-only sample plus the question "is it safe" produced "reviewers say … safe", a word no reviewer used.
- **q10.** The answer merges "didn't help my itchy scalp" (4 reviews) into "made my scalp itchy", which changes what the reviewers said.
- **q03 and q06 (passed, with caveats).** Both open with a "Yes" verdict drawn from 6 filtered reviews and soften or drop a hedge or a dissenting review.

### Misreading a source

- **q12.** One citation is attached to a claim its review does not make ([2]), and one review is read backwards ([3]). Both citations are *valid* by the automated metric, which checks only that a citation number maps to a retrieved source. **Citation validity 1.00 does not mean the citations support the claims.** Only the manual grading caught this.

### Refusals

- 4/4 unanswerable questions were refused. u01 (prices in Japan) and u03 (Redken) both retrieved topically close reviews about Pantene prices and other shampoos' scents, and the model still declined.
- 1/14 answerable questions were refused (q02), caused by the retrieval miss above.
- No model refused while the retrieved reviews did contain the answer.

### Citation format

- q01 wrote "Review 6" in prose. The citation parser does not recognise that, so the claim counts as uncited, yet the answer still has other citations, so uncited rate (a per-answer flag) stays 0. The prompt forbids this format, and the model broke the rule once in 13 answers.

## Label noise

The regex labels are a rough proxy. These errors turned up while grading, and they are reported as found. **The measured numbers above were not adjusted.**

- **False negatives, so precision is understated:** q03 [3] "I started to loss hair" (the pattern expects "lose/losing/lost hair"). q09 [1] "Is it worth the money? Yes" and [6] "Not worth the money" (the pattern has "value" and "price" but not "worth"). q11 [3] "NOT the formula that Pantene has been using" (no "new/old formula" phrase).
- **False positives, so precision is overstated:** q05 [3] matches "color treated hair", but the reviewer says they have never colored their hair. q11 [4] matches on "changing the formula" at the end of a long review about a skin reaction, and the chunk the model saw is mostly an unrelated lawsuit note. q07 [1] matches "residue", but the product is a hairspray. q08's pattern (any scent word, 16,211 matches) and q06's ("dandruff", 2,506) are loose enough that 1.00 precision there is an upper bound.
- Counting only these known errors, q03 and q09 would rise to 1.00, q11 would stay at 0.50 (one false negative and one false positive cancel), and q05 would fall to 0.67. Neither retrieval miss (q02, q14) is affected: their hits clearly do not answer the question.

## Limitations

- **Small sample.** 18 questions, 14 answerable, 4 unanswerable. A single question moves hit@6 by 0.07 and refusal accuracy by 0.25. These are not estimates of performance on user traffic, and nothing here supports claims about the wider population of reviewers.
- **Known, unfixed weaknesses** (reported, not fixed, per the story scope): brand names in the query crowd out on-topic reviews (q02, q14), broad questions get thin answers (q01, q14), and comparative or filtered questions invite over-generalization (q05, q07, q10).
- **The gold set is not blind.** Several questions reuse story 4 hand-check queries whose retrieval outcomes had already been seen: q01 (query 1, partly), q02 (query 3, the known miss), q03 (query 4), q04 (query 7), q05 (query 8), q06 (query 2), q09 (close to query 6), and q14 (the same short-generic-praise pattern as query 5). The gold set was frozen before any scored run of these exact questions, but it was not written without knowledge of how the system behaves.
- **One agent did everything.** The same agent built the system, wrote the questions, wrote each `relevant_pattern` after checking its match count in the corpus, and then graded faithfulness. Nobody else checked the work. The stretch goal of an LLM grader or a human second grader was not done.
- **Relevance labels and the model see different text.** Relevance is labelled on the full original `review_text`, but the model sees only the retrieved chunk. The two differ only for the 1.72% of reviews that were split. q11 [4] is one: its full text matches "changing the formula", but the chunk the model saw is mostly about a lawsuit. `eval/answers.json` saves the first 300 characters of each source. The full text of any saved source can be looked up in `data/processed/reviews_clean.parquet` by `review_id`.
- **Provenance of this run is partly unrecorded.** `frozen_at` is a timestamp the gold file declares about itself, not a verified one. This graded run was saved before the evaluator started recording `gold_sha256` (a hash of the gold file) and `git_commit`/`git_dirty`, so `eval/answers.json` has neither. `--rescore` can check only that the question ids and `frozen_at` match, not that patterns or question text are unchanged. Future full runs record both, and `--rescore` rejects a gold-hash mismatch. The run's `pacing_slept_s` (99.9 s) also comes from the earlier pacer, which estimated the next call from the largest call still inside the 60 s window rather than the largest call of the whole run. No call was rate-limited, so the fix that counts retry sleeps in that figure does not change it.
- **Automated generation metrics are shallow.** Citation validity checks only that citation numbers map to sources. It does not check that the cited review supports the claim (q12). Uncited rate is a per-answer flag and misses individual uncited claims (q01).
- **Regex relevance labels** are noisy in both directions (see above), and precision@6 counts missing results as not relevant.
- **Single run.** Temperature is 0.1, not 0, and the Groq model can change server-side, so a re-run may produce different answers. The graded answers are those saved in `eval/answers.json`.
- **The rating band cannot express exactly 1 star**, so q01's "1-star" question is answered from 1- and 2-star reviews.
- **The prompt shows no review dates**, so time-bound questions (u04) can only be refused, not checked.
