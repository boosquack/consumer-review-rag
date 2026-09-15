"""I/O matrix tests for src.groq_smoke (story 1). No network: the Groq client is mocked.

Run with: uv run python -m unittest -v
"""

import contextlib
import io
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import groq
import httpx

from src import config, groq_smoke

FAKE_KEY = "fake-test-key-DO-NOT-PRINT-1234567890"
UNSET = object()
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def _status_error(status: int, exc_text: str, body_message: str) -> groq.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", CHAT_URL))
    return groq.APIStatusError(exc_text, response=response, body={"error": {"message": body_message}})


def _raw(content, status: int = 200) -> SimpleNamespace:
    """A stand-in for the SDK's raw response: measured status plus parse()."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
    completion = SimpleNamespace(choices=[choice], model=config.GROQ_MODEL, usage=None)
    return SimpleNamespace(status_code=status, parse=lambda: completion)


class GroqSmokeMatrixTest(unittest.TestCase):
    def run_main(self, key=FAKE_KEY, client: mock.MagicMock | None = None):
        """Run main() with a key (or UNSET) and a mocked client.

        Returns (exit code, stdout, stderr, Groq class mock, load_dotenv mock).
        """
        out, err = io.StringIO(), io.StringIO()
        groq_cls = mock.MagicMock(return_value=client or mock.MagicMock())
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(groq_smoke, "load_dotenv") as load_dotenv,
            mock.patch.object(groq_smoke.groq, "Groq", groq_cls),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            if key is UNSET:
                os.environ.pop(config.GROQ_API_KEY_ENV, None)
            else:
                os.environ[config.GROQ_API_KEY_ENV] = key
            code = groq_smoke.main()
        return code, out.getvalue(), err.getvalue(), groq_cls, load_dotenv

    def client_returning(self, raw=None, error=None) -> mock.MagicMock:
        client = mock.MagicMock()
        create = client.chat.completions.with_raw_response.create
        if error is not None:
            create.side_effect = error
        else:
            create.return_value = raw
        return client

    def test_happy_path_prints_measured_status_model_reply_finish_reason_and_exits_0(self):
        for status in (200, 202):
            with self.subTest(status=status):
                client = self.client_returning(_raw("hello from Groq", status))
                code, out, err, _, load_dotenv = self.run_main(client=client)

                self.assertEqual(code, 0)
                self.assertIn(f"HTTP status: {status}", out)
                self.assertIn(config.GROQ_MODEL, out)
                self.assertIn("hello from Groq", out)
                self.assertIn("finish reason: stop", out)
                kwargs = client.chat.completions.with_raw_response.create.call_args.kwargs
                self.assertEqual(kwargs["model"], config.GROQ_MODEL)
                load_dotenv.assert_called_once_with(config.ENV_FILE)
                self.assertNotIn(FAKE_KEY, out + err)

    def test_missing_key_makes_no_call_names_variable_and_env_example_exits_1(self):
        for key in (UNSET, "", "   "):
            with self.subTest(key="unset" if key is UNSET else repr(key)):
                code, out, err, groq_cls, _ = self.run_main(key)
                self.assertEqual(code, 1)
                groq_cls.assert_not_called()
                self.assertIn(config.GROQ_API_KEY_ENV, err)
                self.assertIn(".env.example", err)

    def test_api_error_prints_status_and_groq_body_message_without_key_exits_1(self):
        cases = [
            (401, "Invalid API Key"),
            (404, "The model `llama-3.3-70b-versatile` does not exist or you do not have access to it."),
            (429, "Rate limit reached for model"),
        ]
        for status, body_message in cases:
            with self.subTest(status=status):
                error = _status_error(status, "sdk exception text", body_message)
                code, out, err, _, _ = self.run_main(client=self.client_returning(error=error))

                self.assertEqual(code, 1)
                self.assertIn(f"HTTP status: {status}", err)
                self.assertIn(f"Groq error: {body_message}", err)
                self.assertNotIn("sdk exception text", err)
                self.assertNotIn(FAKE_KEY, out + err)

    def test_empty_or_none_reply_exits_1(self):
        for content in ("", "   ", None):
            with self.subTest(content=repr(content)):
                code, _, err, _, _ = self.run_main(client=self.client_returning(_raw(content)))
                self.assertEqual(code, 1)
                self.assertIn("Empty reply from model.", err)

    def test_connection_error_exits_1_without_key(self):
        error = groq.APIConnectionError(request=httpx.Request("POST", CHAT_URL))
        code, out, err, _, _ = self.run_main(client=self.client_returning(error=error))

        self.assertEqual(code, 1)
        self.assertIn("Connection error", err)
        self.assertNotIn(FAKE_KEY, out + err)


class ConfigTest(unittest.TestCase):
    def test_paths_anchor_at_repo_root(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.assertEqual(config.PROJECT_ROOT, repo_root)
        self.assertEqual(config.ENV_FILE, repo_root / ".env")

    def test_brand_patterns_are_case_insensitive_without_flags_and_have_no_groups(self):
        samples = {
            "Head & Shoulders": ["HEAD & SHOULDERS", "head and shoulders", "Head n' Shoulders"],
            "Pantene": ["PANTENE Pro-V", "pantene"],
            "Herbal Essences": ["HERBAL ESSENCES", "herbal essence"],
        }
        for brand, pattern in config.BRANDS.items():
            with self.subTest(brand=brand):
                compiled = re.compile(pattern)
                self.assertEqual(compiled.groups, 0)
                for text in samples[brand]:
                    self.assertIsNotNone(compiled.search(text), text)


if __name__ == "__main__":
    unittest.main()
