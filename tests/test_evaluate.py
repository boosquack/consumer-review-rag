"""I/O matrix and scoring tests for src.evaluate (story 6).

No network, no index, no model: the retriever and answerer are fakes, the review
frame is built in memory, and gold/answers paths point at a temp directory.

Run with: uv run python -m unittest -v
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import config, evaluate
from src.generate import Answer


def gold_question(qid="q01", answerable=True, **overrides):
    question = {
        "id": qid,
        "question": f"question {qid} about dandruff",
        "type": "factual" if answerable else "unanswerable",
        "brand": None,
        "rating_band": None,
        "answerable": answerable,
        "relevant_pattern": "dandruff" if answerable else None,
        "notes": "",
    }
    question.update(overrides)
    return question


def gold(*questions, frozen_at="2026-01-01T00:00:00Z"):
    return {"frozen_at": frozen_at, "questions": list(questions)}


def make_hits(texts, brand="Pantene", rating=1):
    return [
        {
            "review_id": f"rid{i}",
            "brand": brand,
            "rating": rating,
            "product_name": "Shampoo",
            "review_date": "2021-01-01",
            "review_title": "t",
            "review_text": text,
            "chunk_text": text,
            "distance": 0.1 * i,
        }
        for i, text in enumerate(texts, start=1)
    ]


def review_frame(n_matching=12, brand="Pantene", rating=1):
    rows = [{"review_id": f"m{i}", "brand": brand, "rating": rating, "review_text": "DANDRUFF gone"} for i in range(n_matching)]
    rows += [{"review_id": f"x{i}", "brand": "Head & Shoulders", "rating": 5, "review_text": "dandruff"} for i in range(20)]
    rows += [{"review_id": f"o{i}", "brand": brand, "rating": rating, "review_text": "nice scent"} for i in range(5)]
    return pd.DataFrame(rows)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def __call__(self, query, k=None, brand=None, rating_band=None):
        self.calls.append((query, k, brand, rating_band))
        return self.hits


class FakeAnswerer:
    """Returns queued Answers; builds sources from the injected retriever like answer() does."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, question, brand, rating_band, k, *, retriever):
        self.calls.append(question)
        hits = retriever(question, k=k, brand=brand, rating_band=rating_band)
        spec = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        status, numbers, extra = spec
        if not hits:
            return Answer(status="no_reviews", text="none")
        return Answer(
            status=status,
            text="answer [1]" if status == "answered" else status,
            citations=[hits[n - 1] for n in numbers],
            sources=hits,
            uncited=status == "answered" and not numbers,
            usage={"total_tokens": 1000} if status != "rate_limited" else {},
            **extra,
        )


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ValidateGoldTest(unittest.TestCase):
    def assertGoldError(self, data, fragment):
        with self.assertRaises(evaluate.GoldError) as caught:
            evaluate.validate_gold(data)
        self.assertIn(fragment, str(caught.exception))

    def test_valid_gold_returns_questions(self):
        questions = evaluate.validate_gold(gold(gold_question("q01"), gold_question("u01", answerable=False)))
        self.assertEqual([q["id"] for q in questions], ["q01", "u01"])

    def test_missing_field_named(self):
        question = gold_question("q07")
        del question["relevant_pattern"]
        self.assertGoldError(gold(question), "q07 is missing field(s): relevant_pattern")

    def test_unknown_brand_named(self):
        self.assertGoldError(gold(gold_question("q02", brand="Dove")), "q02 has unknown brand 'Dove'")

    def test_unknown_band_named(self):
        self.assertGoldError(gold(gold_question("q03", rating_band="top")), "q03 has unknown rating_band 'top'")

    def test_duplicate_id_named(self):
        self.assertGoldError(gold(gold_question("q01"), gold_question("q01")), "Duplicate gold question id: q01")

    def test_missing_frozen_at(self):
        self.assertGoldError({"questions": [gold_question()]}, "frozen_at")

    def test_answerable_without_pattern(self):
        self.assertGoldError(gold(gold_question("q04", relevant_pattern=None)), "q04 is answerable but has no relevant_pattern")

    def test_invalid_regex(self):
        self.assertGoldError(gold(gold_question("q05", relevant_pattern="(unclosed")), "q05 has an invalid relevant_pattern")

    def test_unknown_type_and_type_answerable_mismatch(self):
        self.assertGoldError(gold(gold_question("q06", type="vibes")), "q06 has unknown type")
        self.assertGoldError(gold(gold_question("q06", type="unanswerable")), "q06: type 'unanswerable'")

    def test_repo_gold_file_is_valid_and_sized(self):
        data = evaluate.load_gold(config.EVAL_GOLD_JSON)
        questions = data["questions"]
        self.assertEqual(len(questions), 18)
        self.assertEqual(sum(not q["answerable"] for q in questions), 4)
        self.assertEqual(sum(q["type"] == "broad" for q in questions), 2)


class PatternCountTest(unittest.TestCase):
    def test_counts_case_insensitive_within_filters(self):
        questions = [gold_question("q01", brand="Pantene", rating_band="low")]
        counts = evaluate.pattern_counts(questions, review_frame(n_matching=12))
        self.assertEqual(counts, {"q01": 12})

    def test_unfiltered_counts_all_and_skips_unanswerable(self):
        questions = [gold_question("q01"), gold_question("u01", answerable=False)]
        self.assertEqual(evaluate.pattern_counts(questions, review_frame(n_matching=12)), {"q01": 32})

    def test_weak_pattern_names_question(self):
        with self.assertRaises(evaluate.GoldError) as caught:
            evaluate.check_pattern_counts({"q01": 50, "q09": 9})
        self.assertIn("q09 (9)", str(caught.exception))
        self.assertNotIn("q01", str(caught.exception))
        evaluate.check_pattern_counts({"q01": 10})


class ScoringTest(unittest.TestCase):
    def test_hit_at_k(self):
        self.assertEqual(evaluate.hit_at_k([False, False, True], 6), 1)
        self.assertEqual(evaluate.hit_at_k([False] * 6, 6), 0)
        self.assertEqual(evaluate.hit_at_k([], 6), 0)
        self.assertEqual(evaluate.hit_at_k([False, True], 1), 0)

    def test_precision_at_k_counts_missing_results_as_not_relevant(self):
        self.assertAlmostEqual(evaluate.precision_at_k([True, False, True, True, False, False], 6), 0.5)
        self.assertAlmostEqual(evaluate.precision_at_k([True, True], 6), 2 / 6)
        self.assertEqual(evaluate.precision_at_k([], 6), 0.0)
        with self.assertRaises(ValueError):
            evaluate.precision_at_k([True], 0)

    def test_relevance_uses_original_review_text(self):
        hits = make_hits(["No more DANDRUFF", "smells nice"])
        hits[1]["chunk_text"] = "dandruff"
        self.assertEqual(evaluate.relevance_flags(hits, "dandruff"), [True, False])

    def test_citation_valid(self):
        self.assertTrue(evaluate.citation_valid(["a"], ["a", "b"], []))
        self.assertTrue(evaluate.citation_valid([], ["a"], []))
        self.assertFalse(evaluate.citation_valid(["z"], ["a", "b"], []))
        self.assertFalse(evaluate.citation_valid(["a"], ["a"], [9]))

    def test_summarize_generation_metrics(self):
        def rec(qid, answerable, status, hit=1, precision=0.5, valid=True, uncited=False):
            record = {"id": qid, "answerable": answerable, "status": status, "citation_valid": valid, "uncited": uncited, "retrieval": {}}
            if answerable:
                record["retrieval"] = {"hit": hit, "precision": precision}
            return record

        records = [
            rec("q1", True, "answered", hit=1, precision=1.0),
            rec("q2", True, "answered", hit=0, precision=0.0, valid=False, uncited=True),
            rec("q3", True, "refused", hit=1, precision=0.5),
            rec("q4", True, "rate_limited", hit=1, precision=0.5),
            rec("u1", False, "refused"),
            rec("u2", False, "answered"),
        ]
        summary = evaluate.summarize(records, grades={"q1": {"pass": True}, "q2": {"pass": False}, "q4": {"pass": True}})
        self.assertEqual(summary["hit_at_k"], {"value": 0.75, "n": 3, "of": 4})
        self.assertAlmostEqual(summary["precision_at_k"]["value"], 0.5)
        # Answered and scored: q1, q2, u2 (q4 is excluded).
        self.assertEqual(summary["citation_validity"], {"value": 2 / 3, "n": 2, "of": 3})
        self.assertEqual(summary["uncited_rate"], {"value": 1 / 3, "n": 1, "of": 3})
        self.assertEqual(summary["refusal_accuracy"], {"value": 0.5, "n": 1, "of": 2})
        self.assertEqual(summary["false_refusal_rate"], {"value": 1 / 3, "n": 1, "of": 3})
        self.assertEqual(summary["excluded"], [{"id": "q4", "status": "rate_limited"}])
        # Grades for an excluded question do not count.
        self.assertEqual(summary["faithfulness"], {"value": 0.5, "n": 1, "of": 2})
        # Answered-only: q1, q2 graded (u2 answered but ungraded).
        self.assertEqual(summary["faithfulness_answered"], {"value": 0.5, "n": 1, "of": 2})
        self.assertEqual(summary["ungraded"], ["q3", "u1", "u2"])

    def test_faithfulness_answered_excludes_refusals(self):
        records = [
            {"id": "q1", "answerable": True, "status": "answered", "citation_valid": True, "uncited": False, "retrieval": {"hit": 1, "precision": 1.0}},
            {"id": "u1", "answerable": False, "status": "refused", "citation_valid": True, "uncited": False, "retrieval": {}},
        ]
        summary = evaluate.summarize(records, grades={"q1": {"pass": False}, "u1": {"pass": True}})
        self.assertEqual(summary["faithfulness"], {"value": 0.5, "n": 1, "of": 2})
        self.assertEqual(summary["faithfulness_answered"], {"value": 0.0, "n": 0, "of": 1})
        self.assertEqual(summary["ungraded"], [])
        self.assertIn("faithfulness, answered only (agent-graded): 0.00 (0/1)", evaluate.format_summary(summary))

    def test_no_reviews_on_answerable_counts_as_false_refusal(self):
        records = [
            {"id": "q1", "answerable": True, "status": "no_reviews", "citation_valid": True, "uncited": False, "retrieval": {"hit": 0, "precision": 0.0}},
            {"id": "q2", "answerable": True, "status": "answered", "citation_valid": True, "uncited": False, "retrieval": {"hit": 1, "precision": 1.0}},
        ]
        self.assertEqual(evaluate.summarize(records)["false_refusal_rate"], {"value": 0.5, "n": 1, "of": 2})

    def test_summarize_retrieval_only_has_no_generation_metrics(self):
        records = [{"id": "q1", "answerable": True, "retrieval": {"hit": 1, "precision": 1 / 6}}]
        summary = evaluate.summarize(records)
        self.assertNotIn("citation_validity", summary)
        self.assertEqual(summary["hit_at_k"]["value"], 1.0)


class TokenPacerTest(unittest.TestCase):
    def test_first_call_never_waits(self):
        clock = Clock()
        pacer = evaluate.TokenPacer(budget=7000, window_s=60, estimate=3000, clock=clock, sleep=clock.sleep)
        self.assertEqual(pacer.wait(), 0.0)
        self.assertEqual(clock.sleeps, [])

    def test_waits_until_window_frees_budget_using_measured_usage(self):
        clock = Clock()
        pacer = evaluate.TokenPacer(budget=7000, window_s=60, estimate=100, clock=clock, sleep=clock.sleep)
        pacer.record(3000)
        clock.now = 10
        pacer.record(3000)
        clock.now = 20
        # 6000 used + 3000 expected (largest measured) > 7000: wait for the first call to expire.
        waited = pacer.wait()
        self.assertGreater(waited, 0)
        self.assertGreaterEqual(clock.now, 60)
        self.assertLess(clock.now, 61)
        self.assertEqual(len(pacer.events), 1)

    def test_expected_uses_run_max_after_window_expires(self):
        clock = Clock()
        pacer = evaluate.TokenPacer(budget=7000, window_s=60, estimate=3000, clock=clock, sleep=clock.sleep)
        pacer.record(1200)
        pacer.record(800)
        clock.now = 500
        pacer.wait()
        self.assertEqual(pacer.events, [])
        self.assertEqual(pacer.expected(), 1200)

    def test_rate_limit_retry_sleep_counts_as_slept(self):
        clock = Clock()
        pacer = evaluate.TokenPacer(clock=clock, sleep=clock.sleep)
        evaluate.run_question(
            gold_question(), retriever=FakeRetriever(make_hits(["dandruff"])),
            answerer=FakeAnswerer(("rate_limited", [], {})), pacer=pacer, k=6, sleep=clock.sleep,
        )
        self.assertGreaterEqual(pacer.slept_s, config.EVAL_RATE_LIMIT_WAIT_S)

    def test_no_wait_when_budget_allows(self):
        clock = Clock()
        pacer = evaluate.TokenPacer(budget=7000, window_s=60, estimate=3000, clock=clock, sleep=clock.sleep)
        pacer.record(2000)
        clock.now = 5
        self.assertEqual(pacer.wait(), 0.0)


class RunQuestionTest(unittest.TestCase):
    def test_rate_limited_waits_retries_once_then_records(self):
        clock = Clock()
        hits = make_hits(["dandruff"] * 6)
        answerer = FakeAnswerer(("rate_limited", [], {}))
        record = evaluate.run_question(
            gold_question(), retriever=FakeRetriever(hits), answerer=answerer,
            pacer=evaluate.TokenPacer(clock=clock, sleep=clock.sleep), k=6, sleep=clock.sleep,
        )
        self.assertEqual(record["status"], "rate_limited")
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(len(answerer.calls), 2)
        self.assertIn(config.EVAL_RATE_LIMIT_WAIT_S, clock.sleeps)

    def test_rate_limited_then_answered_on_retry(self):
        clock = Clock()
        answerer = FakeAnswerer(("rate_limited", [], {}), ("answered", [1], {}))
        record = evaluate.run_question(
            gold_question(), retriever=FakeRetriever(make_hits(["dandruff"])), answerer=answerer,
            pacer=None, k=6, sleep=clock.sleep,
        )
        self.assertEqual(record["status"], "answered")
        self.assertEqual(record["attempts"], 2)

    def test_retrieval_runs_once_and_sources_saved(self):
        retriever = FakeRetriever(make_hits(["dandruff gone " + "x" * 400, "nice scent"]))
        record = evaluate.run_question(
            gold_question(brand="Pantene", rating_band="low"), retriever=retriever,
            answerer=FakeAnswerer(("answered", [2, 1], {})), pacer=None, k=6,
        )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][1:], (6, "Pantene", "low"))
        self.assertEqual(record["retrieval"]["relevant"], [True, False])
        self.assertEqual(record["retrieval"]["hit"], 1)
        self.assertAlmostEqual(record["retrieval"]["precision"], 1 / 6)
        self.assertEqual(record["citation_numbers"], [2, 1])
        self.assertTrue(record["citation_valid"])
        self.assertEqual(len(record["sources"][0]["review_text"]), config.EVAL_SNIPPET_CHARS)
        self.assertEqual(record["sources"][1]["relevant"], False)

    def test_invalid_citation_marks_invalid(self):
        record = evaluate.run_question(
            gold_question(), retriever=FakeRetriever(make_hits(["dandruff"])),
            answerer=FakeAnswerer(("answered", [1], {"invalid_citations": [7]})), pacer=None, k=6,
        )
        self.assertFalse(record["citation_valid"])
        self.assertEqual(record["invalid_citations"], [7])

    def test_no_hits_scores_zero_and_records_no_reviews(self):
        clock = Clock()
        pacer = evaluate.TokenPacer(clock=clock, sleep=clock.sleep)
        record = evaluate.run_question(
            gold_question(), retriever=FakeRetriever([]), answerer=FakeAnswerer(("answered", [], {})),
            pacer=pacer, k=6,
        )
        self.assertEqual(record["retrieval"]["hit"], 0)
        self.assertEqual(record["retrieval"]["precision"], 0.0)
        self.assertTrue(record["retrieval"]["no_reviews"])
        self.assertEqual(record["status"], "no_reviews")
        self.assertEqual(pacer.events, [])

    def test_retrieval_only_makes_no_answer_call(self):
        record = evaluate.run_question(gold_question(), retriever=FakeRetriever(make_hits(["dandruff"])), answerer=None, pacer=None, k=6)
        self.assertNotIn("status", record)
        self.assertEqual(record["retrieval"]["hit"], 1)


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.gold_path = root / "gold.json"
        self.answers_path = root / "answers.json"
        for name, value in {
            "EVAL_GOLD_JSON": self.gold_path,
            "EVAL_ANSWERS_JSON": self.answers_path,
            "EVAL_FAITHFULNESS_JSON": root / "faithfulness.json",
        }.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(evaluate, "git_state", return_value={"git_commit": "abc123", "git_dirty": False})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.faithfulness_path = root / "faithfulness.json"

    def write_gold(self, data):
        self.gold_path.write_text(json.dumps(data), encoding="utf-8")

    def run_main(self, argv, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = evaluate.main(argv, **kwargs)
        return code, out.getvalue(), err.getvalue()

    def test_full_run_prints_summary_and_writes_answers(self):
        self.write_gold(gold(gold_question("q01"), gold_question("u01", answerable=False)))
        answerer = FakeAnswerer(("answered", [1], {}), ("refused", [], {}))
        clock = Clock()
        code, out, _ = self.run_main(
            [], retriever=FakeRetriever(make_hits(["dandruff", "scent"])), answerer=answerer,
            reviews=review_frame(), sleep=clock.sleep, clock=clock,
        )
        self.assertEqual(code, 0)
        for label in ("hit@6", "precision@6", "citation validity", "uncited rate", "refusal accuracy", "false-refusal rate"):
            self.assertIn(label, out)
        self.assertIn("q01: 32", out)
        saved = json.loads(self.answers_path.read_text(encoding="utf-8"))
        self.assertEqual([r["id"] for r in saved["questions"]], ["q01", "u01"])
        self.assertLess(evaluate.parse_timestamp(saved["gold_frozen_at"]), evaluate.parse_timestamp(saved["run_at"]))
        self.assertEqual(saved["summary"]["refusal_accuracy"]["value"], 1.0)

        # Rescore reads the saved run without retrieval or generation.
        code, out, _ = self.run_main(["--rescore"])
        self.assertEqual(code, 0)
        self.assertIn("refusal accuracy (unanswerable): 1.00 (1/1)", out)
        self.assertIn("faithfulness: not graded", out)

    def test_retrieval_only_makes_no_groq_calls_and_writes_nothing(self):
        self.write_gold(gold(gold_question("q01")))
        with mock.patch("src.generate.answer", side_effect=AssertionError("no Groq calls")):
            code, out, _ = self.run_main(["--retrieval-only"], retriever=FakeRetriever(make_hits(["dandruff"])), reviews=review_frame())
        self.assertEqual(code, 0)
        self.assertIn("hit@6: 1.00 (1/1)", out)
        self.assertNotIn("citation validity", out)
        self.assertFalse(self.answers_path.exists())

    def test_weak_pattern_exits_1_before_any_answer_call(self):
        self.write_gold(gold(gold_question("q09", brand="Pantene", rating_band="low")))
        answerer = FakeAnswerer(("answered", [1], {}))
        retriever = FakeRetriever(make_hits(["dandruff"]))
        code, _, err = self.run_main([], retriever=retriever, answerer=answerer, reviews=review_frame(n_matching=9))
        self.assertEqual(code, 1)
        self.assertIn("q09 (9)", err)
        self.assertEqual(answerer.calls, [])
        self.assertEqual(retriever.calls, [])

    def test_invalid_gold_exits_1_naming_problem(self):
        self.write_gold(gold(gold_question("q01"), gold_question("q01")))
        code, _, err = self.run_main(["--retrieval-only"], retriever=FakeRetriever([]), reviews=review_frame())
        self.assertEqual(code, 1)
        self.assertIn("Duplicate gold question id: q01", err)

    def test_future_frozen_at_exits_1(self):
        self.write_gold(gold(gold_question("q01"), frozen_at="2999-01-01T00:00:00Z"))
        code, _, err = self.run_main(["--retrieval-only"], retriever=FakeRetriever([]), reviews=review_frame())
        self.assertEqual(code, 1)
        self.assertIn("future", err)

    def test_all_questions_errored_exits_1(self):
        self.write_gold(gold(gold_question("q01")))
        code, out, _ = self.run_main(
            [], retriever=FakeRetriever(make_hits(["dandruff"])), answerer=FakeAnswerer(("error", [], {})),
            reviews=review_frame(), sleep=lambda s: None,
        )
        self.assertEqual(code, 1)
        self.assertIn("q01 (error)", out)

    def full_run(self, *questions, answers=(("answered", [1], {}),), **kwargs):
        self.write_gold(gold(*questions))
        clock = Clock()
        kwargs.setdefault("retriever", FakeRetriever(make_hits(["dandruff", "scent"])))
        return self.run_main(
            [], answerer=FakeAnswerer(*answers), reviews=review_frame(), sleep=clock.sleep, clock=clock, **kwargs
        ), clock

    def test_rescore_rejects_answers_from_another_gold_set(self):
        self.write_gold(gold(gold_question("q01")))
        self.answers_path.write_text(json.dumps({"run_at": "x", "gold_frozen_at": "other", "model": "m", "k": 6, "questions": [{"id": "q01"}]}))
        code, _, err = self.run_main(["--rescore"])
        self.assertEqual(code, 1)
        self.assertIn("do not match", err)

    def test_rescore_rejects_question_id_mismatch_with_same_frozen_at(self):
        self.write_gold(gold(gold_question("q01")))
        self.answers_path.write_text(json.dumps({"run_at": "x", "gold_frozen_at": "2026-01-01T00:00:00Z", "model": "m", "k": 6, "questions": [{"id": "q99"}]}))
        code, _, err = self.run_main(["--rescore"])
        self.assertEqual(code, 1)
        self.assertIn("do not match", err)

    def test_rescore_prints_faithfulness_for_matching_grades_only(self):
        (code, _, _), _ = self.full_run(gold_question("q01"), gold_question("u01", answerable=False),
                                        answers=(("answered", [1], {}), ("refused", [], {})))
        self.assertEqual(code, 0)
        run_at = json.loads(self.answers_path.read_text(encoding="utf-8"))["run_at"]
        grades = {"q01": {"pass": False, "note": ""}, "u01": {"pass": True, "note": ""}}
        self.faithfulness_path.write_text(json.dumps({"answers_run_at": run_at, "grades": grades}))
        code, out, _ = self.run_main(["--rescore"])
        self.assertEqual(code, 0)
        self.assertIn("faithfulness (agent-graded): 0.50 (1/2)", out)
        self.assertIn("faithfulness, answered only (agent-graded): 0.00 (0/1)", out)

        self.faithfulness_path.write_text(json.dumps({"answers_run_at": "stale", "grades": grades}))
        code, out, _ = self.run_main(["--rescore"])
        self.assertIn("faithfulness: not graded", out)
        self.assertNotIn("faithfulness (agent-graded)", out)

    def test_rescore_reports_ungraded_ids(self):
        (code, _, _), _ = self.full_run(gold_question("q01"), gold_question("q02"))
        run_at = json.loads(self.answers_path.read_text(encoding="utf-8"))["run_at"]
        self.faithfulness_path.write_text(json.dumps({"answers_run_at": run_at, "grades": {"q01": {"pass": True}}}))
        _, out, _ = self.run_main(["--rescore"])
        self.assertIn("ungraded: q02", out)

    def test_rescore_rejects_gold_sha256_mismatch(self):
        (code, _, _), _ = self.full_run(gold_question("q01"))
        saved = json.loads(self.answers_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["gold_sha256"], evaluate.file_sha256(self.gold_path))
        self.assertEqual(saved["git_commit"], "abc123")
        self.assertFalse(saved["git_dirty"])
        # Same ids and frozen_at, edited pattern.
        self.write_gold(gold(gold_question("q01", relevant_pattern="dandruff|flakes")))
        code, _, err = self.run_main(["--rescore"])
        self.assertEqual(code, 1)
        self.assertIn("gold_sha256 mismatch", err)

    def test_full_run_paces_over_budget(self):
        questions = [gold_question(f"q0{i}") for i in range(1, 4)]
        with mock.patch.object(config, "EVAL_TOKENS_PER_MIN", 1500):
            (code, _, _), clock = self.full_run(*questions)
        self.assertEqual(code, 0)
        # Each call measures 1,000 tokens, so calls 2 and 3 wait for the window.
        self.assertEqual(len(clock.sleeps), 2)
        self.assertGreater(json.loads(self.answers_path.read_text(encoding="utf-8"))["pacing_slept_s"], 100)

    def test_retriever_raising_mid_run_exits_1_without_writing(self):
        class Failing(FakeRetriever):
            def __call__(self, query, k=None, brand=None, rating_band=None):
                if self.calls:
                    raise RuntimeError("index is stale")
                return super().__call__(query, k, brand, rating_band)

        (code, _, err), _ = self.full_run(gold_question("q01"), gold_question("q02"), retriever=Failing(make_hits(["dandruff"])))
        self.assertEqual(code, 1)
        self.assertIn("index is stale", err)
        self.assertFalse(self.answers_path.exists())

    def test_overwrite_backs_up_previous_answers(self):
        self.answers_path.write_text(json.dumps({"run_at": "2026-09-16T16:12:28+00:00", "questions": []}))
        previous = self.answers_path.read_bytes()
        (code, out, _), _ = self.full_run(gold_question("q01"))
        self.assertEqual(code, 0)
        backup = self.answers_path.with_name("answers.2026-09-16T16-12-28-00-00.json")
        self.assertEqual(backup.read_bytes(), previous)
        self.assertNotEqual(self.answers_path.read_bytes(), previous)
        self.assertIn("backed up", out)

    def test_all_error_run_does_not_write_or_replace_answers(self):
        self.answers_path.write_text("graded run")
        (code, _, err), _ = self.full_run(gold_question("q01"), answers=(("error", [], {}),))
        self.assertEqual(code, 1)
        self.assertIn("not written", err)
        self.assertEqual(self.answers_path.read_text(), "graded run")
        self.assertEqual(list(self.answers_path.parent.glob("answers.*.json")), [])


class GitStateTest(unittest.TestCase):
    def test_tolerates_missing_git(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("git")):
            self.assertEqual(evaluate.git_state(), {"git_commit": None, "git_dirty": None})



class CompareGradesTests(unittest.TestCase):
    def test_lists_only_pass_fail_differences_for_ids_both_graded(self):
        agent = {"q01": {"pass": True}, "q02": {"pass": False}, "q03": {"pass": True}}
        human = {"q01": {"pass": True}, "q02": {"pass": True}, "q04": {"pass": False}}
        self.assertEqual(
            evaluate.compare_grades(agent, human),
            [{"id": "q02", "agent": False, "human": True}],
        )

if __name__ == "__main__":
    unittest.main()
