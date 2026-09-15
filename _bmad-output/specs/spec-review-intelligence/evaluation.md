# Evaluation

Protect this phase even if the week gets tight.

## Gold set: `eval/gold_questions.json`

- 15–20 hand-written questions, mixed:
  - factual ("what do reviewers say about the scent of X")
  - comparative ("which brand has more complaints about residue")
  - several unanswerable questions, to test refusal
- Each question records its expected supporting review(s) or theme.

## Metrics: `src/evaluate.py`

| Metric | Method |
|---|---|
| Retrieval hit rate | For questions with a known relevant review, whether it appears in the top-k |
| Answer faithfulness | Manual checklist: the answer stays within the retrieved reviews with no invented claims; pass/fail per question with notes |
| Refusal behavior | Correctly declines the unanswerable questions |

## Write-up: `eval/results.md`

Numbers plus honest commentary on every failure.
