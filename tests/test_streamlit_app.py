"""I/O matrix tests for app/streamlit_app.py (story 7).

Headless via streamlit.testing.v1.AppTest. ``src.generate.answer`` and
``src.index.build_index`` are patched at their source modules so the app's
fresh script exec picks up the fakes; no network and no real index build.

Run with: uv run python -m unittest -v
"""

import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config, generate

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
QUESTION = "what do 1-star Pantene reviewers complain about"


def make_hits(n: int = 3) -> list[dict]:
    brands = list(config.BRANDS)
    return [
        {
            "review_id": f"rid{i:02d}",
            "brand": brands[i % len(brands)],
            "rating": 1 + i % 5,
            "product_name": "Shampoo",
            "review_date": "2021-03-04",
            "review_title": f"title {i}",
            "review_text": f"Full original review {i}.",
            "chunk_text": f"chunk {i}",
            "distance": 0.1 * i,
        }
        for i in range(1, n + 1)
    ]


def _submit(at: AppTest, question: str = QUESTION) -> AppTest:
    at.text_input[0].input(question)
    at.button[0].click()
    return at.run()


class StreamlitAppTest(unittest.TestCase):
    def setUp(self):
        # index.build_index is only ever called through the module reference
        # (`index.build_index(...)`), so patching the module attribute here is
        # visible to the app script's own fresh exec.
        patcher = mock.patch("src.index.build_index", return_value={"status": "skipped"})
        self.addCleanup(patcher.stop)
        patcher.start()
        # _ensure_index is @st.cache_resource; its cache is a process-global
        # singleton that would otherwise leak a "built" result across tests
        # that patch build_index differently (e.g. to raise).
        st.cache_resource.clear()

    def test_loads_and_shows_eda_panel(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertIn("EDA highlights", "\n".join(h.value for h in at.header))

    def test_answered_query_renders_text_and_sources(self):
        hits = make_hits()
        result = generate.Answer(
            status=generate.STATUS_ANSWERED,
            text="Reviewers complain about dryness [1].",
            citations=[hits[0]],
            sources=hits,
        )
        with mock.patch("src.generate.answer", return_value=result) as fake_answer:
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        fake_answer.assert_called_once()
        self.assertIn(result.text, "\n".join(m.value for m in at.markdown))
        self.assertTrue(any("Sources (3 reviews)" in e.label for e in at.expander))
        # The badge must land on source 1 (the actual citation) and nowhere else.
        source_lines = [m.value for m in at.markdown if m.value.startswith("**[")]
        self.assertEqual(len(source_lines), 3)
        self.assertIn("✓ cited", source_lines[0])
        self.assertNotIn("✓ cited", source_lines[1])
        self.assertNotIn("✓ cited", source_lines[2])

    def test_uncited_answer_shows_caption(self):
        hits = make_hits()
        result = generate.Answer(
            status=generate.STATUS_ANSWERED, text="A thin answer with no citation.", uncited=True, sources=hits
        )
        with mock.patch("src.generate.answer", return_value=result):
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        self.assertIn("Note: this answer cites no review.", [c.value for c in at.caption])

    def test_no_hits_shows_warning(self):
        result = generate.Answer(status=generate.STATUS_NO_REVIEWS, text=generate.NO_REVIEWS_TEXT)
        with mock.patch("src.generate.answer", return_value=result):
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        self.assertIn(generate.NO_REVIEWS_TEXT, [w.value for w in at.warning])

    def test_refusal_shows_info(self):
        result = generate.Answer(status=generate.STATUS_REFUSED, text=config.REFUSAL_TEXT, sources=make_hits())
        with mock.patch("src.generate.answer", return_value=result):
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        self.assertIn(config.REFUSAL_TEXT, [i.value for i in at.info])

    def test_rate_limited_shows_error(self):
        result = generate.Answer(
            status=generate.STATUS_RATE_LIMITED, text=generate.RATE_LIMITED_TEXT, sources=make_hits()
        )
        with mock.patch("src.generate.answer", return_value=result):
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        self.assertIn(generate.RATE_LIMITED_TEXT, [e.value for e in at.error])

    def test_missing_key_shows_error(self):
        # answer() always sets sources=hits for this status (src/generate.py:190),
        # so the fake must too, to exercise the real error+sources composite.
        result = generate.Answer(status=generate.STATUS_ERROR, text=generate.MISSING_KEY_TEXT, sources=make_hits())
        with mock.patch("src.generate.answer", return_value=result):
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        self.assertIn(generate.MISSING_KEY_TEXT, [e.value for e in at.error])
        self.assertTrue(any("Sources (3 reviews)" in e.label for e in at.expander))

    def test_answer_raising_shows_error(self):
        # _with_review raises RuntimeError for a stale index; the submit handler
        # must catch it and show st.error, not let the script crash.
        with mock.patch("src.generate.answer", side_effect=RuntimeError("index is stale, rebuild it")):
            at = _submit(AppTest.from_file(APP_PATH, default_timeout=30).run())
        self.assertFalse(at.exception)
        self.assertIn("index is stale, rebuild it", [e.value for e in at.error])

    def test_index_build_failure_shows_error_and_stops(self):
        with mock.patch("src.index.build_index", side_effect=OSError("disk full")):
            at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertIn("disk full", [e.value for e in at.error])
        # st.stop() must prevent the query form from rendering at all.
        self.assertEqual(len(at.text_input), 0)

    def test_deployed_secret_copied_to_environ_before_answer_call(self):
        result = generate.Answer(status=generate.STATUS_NO_REVIEWS, text=generate.NO_REVIEWS_TEXT)
        seen = {}

        def fake_answer(question, **kwargs):
            import os

            seen["key"] = os.environ.get(config.GROQ_API_KEY_ENV)
            return result

        with mock.patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop(config.GROQ_API_KEY_ENV, None)
            with mock.patch("src.generate.answer", side_effect=fake_answer):
                at = AppTest.from_file(APP_PATH, default_timeout=30)
                at.secrets[config.GROQ_API_KEY_ENV] = "from-secrets-test-key"
                at = _submit(at.run())
        self.assertFalse(at.exception)
        self.assertEqual(seen.get("key"), "from-secrets-test-key")


if __name__ == "__main__":
    unittest.main()
