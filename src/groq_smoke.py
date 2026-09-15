"""Phase 0 smoke test: one trivial Groq chat completion with config.GROQ_MODEL.

Run with: uv run python -m src.groq_smoke

Exit 0 on a non-empty reply; exit 1 if the key is missing or the API errors.
The API key is never printed.
"""

import os
import sys

import groq
from dotenv import load_dotenv

from src import config


def main() -> int:
    # Explicit path: load_dotenv() without one fails its frame lookup under -m.
    load_dotenv(config.ENV_FILE)
    api_key = os.environ.get(config.GROQ_API_KEY_ENV, "").strip()
    print(f"{config.GROQ_API_KEY_ENV} present: {bool(api_key)}")
    if not api_key:
        print(
            f"{config.GROQ_API_KEY_ENV} is not set. Copy .env.example to .env "
            "and add your key.",
            file=sys.stderr,
        )
        return 1

    client = groq.Groq(api_key=api_key)
    print(f"model: {config.GROQ_MODEL}")
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            messages=[
                {"role": "user", "content": "Reply with exactly: hello from Groq"}
            ],
        )
    except groq.APIStatusError as exc:
        print(f"HTTP status: {exc.status_code}", file=sys.stderr)
        print(f"Groq error: {_error_message(exc)}", file=sys.stderr)
        return 1
    except groq.APIConnectionError as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1

    response = raw.parse()
    choice = response.choices[0]
    reply = (choice.message.content or "").strip()
    print(f"HTTP status: {raw.status_code}")
    print(f"response model: {response.model}")
    print(f"reply: {reply!r}")
    print(f"finish reason: {choice.finish_reason}")
    if response.usage is not None:
        print(f"usage: {response.usage.model_dump(exclude_none=True)}")
    if not reply:
        print("Empty reply from model.", file=sys.stderr)
        return 1
    return 0


def _error_message(exc: groq.APIStatusError) -> str:
    """Return Groq's error message from the response body, falling back to str(exc)."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(exc)


if __name__ == "__main__":
    sys.exit(main())
