"""Evaluate retrieval and generation against the frozen gold set (CAP-5).

Run with:
  uv run python -m src.evaluate --retrieval-only   # hit@k and precision@k, no Groq calls
  uv run python -m src.evaluate                    # full run, writes eval/answers.json
  uv run python -m src.evaluate --rescore          # re-print metrics from eval/answers.json

Relevance is labelled without the embedding model: a retrieved review is relevant
when its original ``review_text`` matches the question's case-insensitive
``relevant_pattern``. Every pattern must match at least
``config.EVAL_MIN_PATTERN_MATCHES`` reviews within the question's filters, checked
before any Groq call. Groq calls are paced under the free-tier tokens/min cap
using the measured usage of earlier calls. Faithfulness is graded separately,
against the saved answers, in ``config.EVAL_FAITHFULNESS_JSON``.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src import config

REQUIRED_FIELDS = ("id", "question", "type", "brand", "rating_band", "answerable", "relevant_pattern", "notes")
# Statuses that say nothing about answer quality; excluded from generation metrics.
EXCLUDED_STATUSES = ("rate_limited", "error")
DECLINE_STATUSES = ("refused", "no_reviews")


class GoldError(ValueError):
    """The gold file is invalid or a pattern is too weak to score against."""


# --- Gold set ----------------------------------------------------------------------------


def parse_timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def validate_gold(data: Any) -> list[dict[str, Any]]:
    """Return the gold questions, raising GoldError naming the first problem."""
    if not isinstance(data, dict):
        raise GoldError("Gold file must be an object with 'frozen_at' and 'questions'.")
    frozen_at = data.get("frozen_at")
    if not isinstance(frozen_at, str) or not frozen_at:
        raise GoldError("Gold file has no 'frozen_at' timestamp; freeze it before a scored run.")
    try:
        parse_timestamp(frozen_at)
    except ValueError as error:
        raise GoldError(f"Gold 'frozen_at' is not an ISO timestamp: {frozen_at!r}") from error
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise GoldError("Gold file has no 'questions' list.")

    seen: set[str] = set()
    for position, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise GoldError(f"Gold question #{position} is not an object.")
        label = question.get("id", f"#{position}")
        missing = [name for name in REQUIRED_FIELDS if name not in question]
        if missing:
            raise GoldError(f"Gold question {label} is missing field(s): {', '.join(missing)}")
        qid = question["id"]
        if not isinstance(qid, str) or not qid:
            raise GoldError(f"Gold question #{position} has a blank or non-string id.")
        if qid in seen:
            raise GoldError(f"Duplicate gold question id: {qid}")
        seen.add(qid)
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise GoldError(f"Gold question {qid} has a blank question.")
        if question["type"] not in config.EVAL_QUESTION_TYPES:
            raise GoldError(f"Gold question {qid} has unknown type {question['type']!r}.")
        if question["brand"] is not None and question["brand"] not in config.BRANDS:
            raise GoldError(f"Gold question {qid} has unknown brand {question['brand']!r}.")
        if question["rating_band"] is not None and question["rating_band"] not in config.RATING_BANDS:
            raise GoldError(f"Gold question {qid} has unknown rating_band {question['rating_band']!r}.")
        if not isinstance(question["answerable"], bool):
            raise GoldError(f"Gold question {qid} has a non-boolean 'answerable'.")
        if question["answerable"] == (question["type"] == "unanswerable"):
            raise GoldError(f"Gold question {qid}: type 'unanswerable' must match answerable=false.")
        pattern = question["relevant_pattern"]
        if question["answerable"]:
            if not isinstance(pattern, str) or not pattern:
                raise GoldError(f"Gold question {qid} is answerable but has no relevant_pattern.")
            try:
                re.compile(pattern)
            except re.error as error:
                raise GoldError(f"Gold question {qid} has an invalid relevant_pattern: {error}") from error
        elif pattern is not None:
            raise GoldError(f"Gold question {qid} is unanswerable and must have relevant_pattern null.")
    return questions


def load_gold(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GoldError(f"No gold file at {path}.") from error
    except json.JSONDecodeError as error:
        raise GoldError(f"Gold file {path} is not valid JSON: {error}") from error
    validate_gold(data)
    return data


def pattern_matcher(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern, re.IGNORECASE)
    return lambda text: bool(compiled.search(str(text)))


def pattern_counts(questions: Sequence[dict[str, Any]], reviews: Any) -> dict[str, int]:
    """Count reviews whose review_text matches each answerable pattern, within its filters."""
    counts = {}
    for question in questions:
        if not question["answerable"]:
            continue
        frame = reviews
        if question["brand"] is not None:
            frame = frame[frame["brand"] == question["brand"]]
        if question["rating_band"] is not None:
            frame = frame[frame["rating"].isin(config.RATING_BANDS[question["rating_band"]])]
        counts[question["id"]] = int(frame["review_text"].map(pattern_matcher(question["relevant_pattern"])).sum())
    return counts


def check_pattern_counts(counts: dict[str, int], minimum: int | None = None) -> None:
    minimum = config.EVAL_MIN_PATTERN_MATCHES if minimum is None else minimum
    weak = [f"{qid} ({count})" for qid, count in counts.items() if count < minimum]
    if weak:
        raise GoldError(
            f"Weak relevant_pattern: fewer than {minimum} matching reviews within filters for "
            + ", ".join(weak)
        )


# --- Scoring (pure) ----------------------------------------------------------------------


def relevance_flags(hits: Sequence[dict[str, Any]], pattern: str) -> list[bool]:
    matches = pattern_matcher(pattern)
    return [matches(hit["review_text"]) for hit in hits]


def hit_at_k(flags: Sequence[bool], k: int) -> int:
    """1 when any of the top k is relevant, else 0."""
    return int(any(flags[:k]))


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Share of k that is relevant; missing results count as not relevant."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return sum(bool(flag) for flag in flags[:k]) / k


def citation_valid(cited_ids: Sequence[str], source_ids: Sequence[str], invalid_citations: Sequence[int]) -> bool:
    """True when every citation maps to one of that call's sources and none was dropped as out of range."""
    return not invalid_citations and all(review_id in source_ids for review_id in cited_ids)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"value": (numerator / denominator) if denominator else None, "n": numerator, "of": denominator}


def summarize(records: Sequence[dict[str, Any]], grades: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate per-question records into the reported metrics."""
    answerable = [r for r in records if r["answerable"]]
    summary: dict[str, Any] = {
        "questions": len(records),
        "answerable": len(answerable),
        "hit_at_k": _rate(sum(r["retrieval"]["hit"] for r in answerable), len(answerable)),
        "precision_at_k": {
            "value": (sum(r["retrieval"]["precision"] for r in answerable) / len(answerable)) if answerable else None,
            "of": len(answerable),
        },
    }
    generated = [r for r in records if r.get("status") is not None]
    if not generated:
        return summary

    scored = [r for r in generated if r["status"] not in EXCLUDED_STATUSES]
    answered = [r for r in scored if r["status"] == "answered"]
    unanswerable = [r for r in scored if not r["answerable"]]
    answerable_scored = [r for r in scored if r["answerable"]]
    summary.update(
        {
            "excluded": [{"id": r["id"], "status": r["status"]} for r in generated if r["status"] in EXCLUDED_STATUSES],
            "citation_validity": _rate(sum(bool(r["citation_valid"]) for r in answered), len(answered)),
            "uncited_rate": _rate(sum(bool(r["uncited"]) for r in answered), len(answered)),
            "refusal_accuracy": _rate(sum(r["status"] in DECLINE_STATUSES for r in unanswerable), len(unanswerable)),
            "false_refusal_rate": _rate(
                sum(r["status"] in DECLINE_STATUSES for r in answerable_scored), len(answerable_scored)
            ),
        }
    )
    if grades:
        # Overall includes refusals (a faithful refusal passes); answered-only is the
        # stricter figure, over answers that make claims.
        graded = [grades[r["id"]] for r in scored if r["id"] in grades]
        graded_answered = [grades[r["id"]] for r in answered if r["id"] in grades]
        summary["faithfulness"] = _rate(sum(bool(grade["pass"]) for grade in graded), len(graded))
        summary["faithfulness_answered"] = _rate(
            sum(bool(grade["pass"]) for grade in graded_answered), len(graded_answered)
        )
        summary["ungraded"] = [r["id"] for r in scored if r["id"] not in grades]
    return summary


# --- Pacing ------------------------------------------------------------------------------


class TokenPacer:
    """Keep measured Groq token usage under a per-window budget.

    Before each call, ``wait()`` sleeps until the tokens recorded in the last
    window plus the expected cost of the next call (the largest call measured so
    far in the run, or ``estimate`` before any) fit the budget.
    """

    def __init__(
        self,
        budget: int | None = None,
        window_s: float | None = None,
        estimate: int | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.budget = config.EVAL_TOKENS_PER_MIN if budget is None else budget
        self.window_s = config.EVAL_WINDOW_S if window_s is None else window_s
        self.estimate = config.EVAL_TOKEN_ESTIMATE if estimate is None else estimate
        self.clock = clock
        self.sleep = sleep
        self.events: list[tuple[float, int]] = []
        self.slept_s = 0.0
        self.max_measured = 0

    def _expire(self, now: float) -> None:
        self.events = [(stamp, tokens) for stamp, tokens in self.events if now - stamp < self.window_s]

    def expected(self) -> int:
        return self.max_measured if self.max_measured > 0 else self.estimate

    def wait(self) -> float:
        waited = 0.0
        expected = self.expected()
        while True:
            now = self.clock()
            self._expire(now)
            used = sum(tokens for _, tokens in self.events)
            if not self.events or used + expected <= self.budget:
                self.slept_s += waited
                return waited
            delay = self.events[0][0] + self.window_s - now + 0.05
            self.sleep(delay)
            waited += delay

    def record(self, tokens: int) -> None:
        tokens = max(int(tokens or 0), 0)
        self.max_measured = max(self.max_measured, tokens)
        self.events.append((self.clock(), tokens))


# --- Running -----------------------------------------------------------------------------


def _usage_tokens(usage: dict[str, Any]) -> int:
    if not usage:
        return 0
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    return int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_question(
    question: dict[str, Any],
    *,
    retriever: Callable[..., list[dict[str, Any]]],
    answerer: Callable[..., Any] | None,
    pacer: TokenPacer | None,
    k: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Retrieve once, score retrieval, and (unless answerer is None) generate and score the answer."""
    hits = retriever(question["question"], k=k, brand=question["brand"], rating_band=question["rating_band"])
    flags = relevance_flags(hits, question["relevant_pattern"]) if question["answerable"] else [None] * len(hits)
    record: dict[str, Any] = {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "brand": question["brand"],
        "rating_band": question["rating_band"],
        "answerable": question["answerable"],
        "retrieval": {"n_hits": len(hits), "relevant": flags},
    }
    if question["answerable"]:
        record["retrieval"]["hit"] = hit_at_k(flags, k)
        record["retrieval"]["precision"] = precision_at_k(flags, k)
    if not hits:
        record["retrieval"]["no_reviews"] = True
    if answerer is None:
        return record

    # Generation reuses these exact hits, so retrieval runs once per question and the
    # scored sources are the ones the model saw.
    def fixed(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return hits

    attempts = 0
    while True:
        attempts += 1
        if hits and pacer is not None:
            pacer.wait()
        result = answerer(question["question"], question["brand"], question["rating_band"], k, retriever=fixed)
        if hits and pacer is not None:
            pacer.record(_usage_tokens(result.usage))
        if result.status != "rate_limited" or attempts >= 2:
            break
        sleep(config.EVAL_RATE_LIMIT_WAIT_S)
        if pacer is not None:
            pacer.slept_s += config.EVAL_RATE_LIMIT_WAIT_S

    source_ids = [hit["review_id"] for hit in result.sources]
    cited_ids = [hit["review_id"] for hit in result.citations]
    record.update(
        {
            "status": result.status,
            "text": result.text,
            "citation_numbers": [source_ids.index(rid) + 1 if rid in source_ids else None for rid in cited_ids],
            "invalid_citations": list(result.invalid_citations),
            "uncited": bool(result.uncited),
            "citation_valid": citation_valid(cited_ids, source_ids, result.invalid_citations),
            "sources": [
                {
                    "number": number,
                    "review_id": hit["review_id"],
                    "brand": hit["brand"],
                    "rating": int(hit["rating"]),
                    "relevant": flags[number - 1] if number - 1 < len(flags) else None,
                    "review_text": str(hit["review_text"])[: config.EVAL_SNIPPET_CHARS],
                }
                for number, hit in enumerate(result.sources, start=1)
            ],
            "usage": dict(result.usage),
            "attempts": attempts,
            "timestamp": _now(),
        }
    )
    return record


# --- Output ------------------------------------------------------------------------------


def _fmt_rate(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("value") is None:
        return "n/a"
    counts = f" ({metric['n']}/{metric['of']})" if "n" in metric else f" (mean of {metric['of']})"
    return f"{metric['value']:.2f}{counts}"


def format_table(records: Sequence[dict[str, Any]], counts: dict[str, int]) -> str:
    header = f"{'id':<4} {'type':<12} {'brand':<16} {'band':<4} {'matches':>7} {'rel/hits':>8} {'hit':>3} {'P@k':>4}"
    generated = any(r.get("status") is not None for r in records)
    if generated:
        header += f"  {'status':<12} {'cited':<12} {'valid':<5} {'uncited':<7}"
    lines = [header, "-" * len(header)]
    for r in records:
        retrieval = r["retrieval"]
        relevant = sum(bool(flag) for flag in retrieval["relevant"])
        line = (
            f"{r['id']:<4} {r['type']:<12} {(r['brand'] or '-'):<16} {(r['rating_band'] or '-'):<4} "
            f"{counts.get(r['id'], '-')!s:>7} "
            + (f"{relevant}/{retrieval['n_hits']}" if r["answerable"] else f"-/{retrieval['n_hits']}").rjust(8)
            + f" {retrieval.get('hit', '-')!s:>3} "
            + (f"{retrieval['precision']:.2f}" if r["answerable"] else " -  ")
        )
        if generated:
            cited = ",".join(str(n) for n in r.get("citation_numbers", [])) or "-"
            valid = str(r["citation_valid"]) if r.get("status") == "answered" else "-"
            uncited = str(r["uncited"]) if r.get("status") == "answered" else "-"
            line += f"  {r['status']:<12} {cited:<12} {valid:<5} {uncited:<7}"
        if retrieval.get("no_reviews"):
            line += "  no_reviews"
        lines.append(line)
    return "\n".join(lines)


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"questions: {summary['questions']} ({summary['answerable']} answerable)",
        f"hit@{config.RETRIEVAL_K}: {_fmt_rate(summary['hit_at_k'])}",
        f"precision@{config.RETRIEVAL_K}: {_fmt_rate(summary['precision_at_k'])}",
    ]
    if "citation_validity" in summary:
        lines += [
            f"citation validity: {_fmt_rate(summary['citation_validity'])}",
            f"uncited rate: {_fmt_rate(summary['uncited_rate'])}",
            f"refusal accuracy (unanswerable): {_fmt_rate(summary['refusal_accuracy'])}",
            f"false-refusal rate (answerable): {_fmt_rate(summary['false_refusal_rate'])}",
            "excluded from generation metrics: "
            + (", ".join(f"{e['id']} ({e['status']})" for e in summary["excluded"]) or "none"),
        ]
        if "faithfulness" in summary:
            lines.append(f"faithfulness (agent-graded): {_fmt_rate(summary['faithfulness'])}")
            lines.append(f"faithfulness, answered only (agent-graded): {_fmt_rate(summary['faithfulness_answered'])}")
            if summary["ungraded"]:
                lines.append("ungraded: " + ", ".join(summary["ungraded"]))
        else:
            lines.append("faithfulness: not graded for this run")
    return "\n".join(lines)


def load_grades(path: Path, run_at: str) -> dict[str, Any] | None:
    """Return per-id grades when the grades file grades this exact answers run."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("answers_run_at") != run_at:
        return None
    return data.get("grades") or None


def compare_grades(agent: dict[str, Any], human: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ids both graders graded where their pass/fail differs."""
    return [
        {"id": qid, "agent": bool(agent[qid]["pass"]), "human": bool(human[qid]["pass"])}
        for qid in human
        if qid in agent and bool(agent[qid]["pass"]) != bool(human[qid]["pass"])
    ]


# --- CLI ---------------------------------------------------------------------------------


def _load_reviews() -> Any:
    import pandas as pd

    return pd.read_parquet(config.CLEAN_PARQUET, columns=["review_id", "brand", "rating", "review_text"])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    """Current commit and dirty flag; both None when git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit or None, "git_dirty": dirty}


def backup_path(path: Path, run_at: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z]+", "-", run_at).strip("-") or "unknown"
    return Path(path).with_name(f"{Path(path).stem}.{safe}{Path(path).suffix}")


def _rescore(gold: dict[str, Any]) -> int:
    path = config.EVAL_ANSWERS_JSON
    if not path.exists():
        print(f"No saved answers at {path}. Run: uv run python -m src.evaluate", file=sys.stderr)
        return 1
    saved = json.loads(path.read_text(encoding="utf-8"))
    gold_ids = [q["id"] for q in gold["questions"]]
    saved_ids = [r["id"] for r in saved["questions"]]
    if saved_ids != gold_ids or saved.get("gold_frozen_at") != gold["frozen_at"]:
        print("Saved answers do not match the frozen gold set; re-run the evaluation.", file=sys.stderr)
        return 1
    if saved.get("gold_sha256") is not None and saved["gold_sha256"] != file_sha256(config.EVAL_GOLD_JSON):
        print("The gold file changed since these answers were saved (gold_sha256 mismatch).", file=sys.stderr)
        return 1
    grades = load_grades(config.EVAL_FAITHFULNESS_JSON, saved["run_at"])
    print(f"run_at: {saved['run_at']}  model: {saved['model']}  k: {saved['k']}")
    print(format_table(saved["questions"], saved.get("pattern_counts", {})))
    print()
    print(format_summary(summarize(saved["questions"], grades)))
    human = load_grades(config.EVAL_FAITHFULNESS_HUMAN_JSON, saved["run_at"])
    if human:
        human_summary = summarize(saved["questions"], human)
        print()
        print(f"faithfulness (human-graded): {_fmt_rate(human_summary['faithfulness'])}")
        print(f"faithfulness, answered only (human-graded): {_fmt_rate(human_summary['faithfulness_answered'])}")
        if human_summary["ungraded"]:
            print(f"ungraded by human: {', '.join(human_summary['ungraded'])}")
        diffs = compare_grades(grades or {}, human)
        print(f"agent/human disagreements: {len(diffs)}")
        for diff in diffs:
            print(f"  {diff['id']}: agent={'pass' if diff['agent'] else 'fail'}, human={'pass' if diff['human'] else 'fail'}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    retriever: Callable[..., list[dict[str, Any]]] | None = None,
    answerer: Callable[..., Any] | None = None,
    reviews: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retrieval-only", action="store_true", help="score retrieval only; no Groq calls")
    mode.add_argument("--rescore", action="store_true", help="re-print metrics from the saved answers")
    args = parser.parse_args(argv)

    try:
        gold = load_gold(config.EVAL_GOLD_JSON)
        run_at = datetime.now(timezone.utc)
        if parse_timestamp(gold["frozen_at"]) > run_at:
            raise GoldError(f"Gold frozen_at {gold['frozen_at']} is in the future.")
        if args.rescore:
            return _rescore(gold)
        questions = gold["questions"]
        counts = pattern_counts(questions, _load_reviews() if reviews is None else reviews)
        print("relevant_pattern matches within filters:")
        for qid, count in counts.items():
            print(f"  {qid}: {count}")
        check_pattern_counts(counts)
    except GoldError as error:
        print(error, file=sys.stderr)
        return 1

    if retriever is None:
        from src.retrieve import retrieve as retriever
    if answerer is None and not args.retrieval_only:
        from src.generate import answer as answerer
    k = config.RETRIEVAL_K
    pacer = None if args.retrieval_only else TokenPacer(clock=clock, sleep=sleep)

    records = []
    try:
        for question in questions:
            record = run_question(
                question,
                retriever=retriever,
                answerer=None if args.retrieval_only else answerer,
                pacer=pacer,
                k=k,
                sleep=sleep,
            )
            records.append(record)
            status = f" -> {record['status']}" if "status" in record else ""
            print(f"[{len(records)}/{len(questions)}] {question['id']}{status}", flush=True)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    print()
    print(format_table(records, counts))
    print()
    summary = summarize(records)
    print(format_summary(summary))
    if args.retrieval_only:
        return 0

    # A run where no question produced a scoreable answer (e.g. missing key) is a
    # failure, and must not replace a graded answers file.
    if all(r["status"] in EXCLUDED_STATUSES for r in records):
        print(f"\nno question produced a scoreable answer; {config.EVAL_ANSWERS_JSON} not written", file=sys.stderr)
        return 1

    payload = {
        "run_at": run_at.isoformat(timespec="seconds"),
        "gold_frozen_at": gold["frozen_at"],
        "gold_sha256": file_sha256(config.EVAL_GOLD_JSON),
        **git_state(),
        "model": config.GROQ_MODEL,
        "temperature": config.GROQ_TEMPERATURE,
        "reasoning_effort": config.GROQ_REASONING_EFFORT,
        "k": k,
        "pattern_counts": counts,
        "pacing_slept_s": round(pacer.slept_s, 1) if pacer else 0.0,
        "summary": summary,
        "questions": records,
    }
    path = config.EVAL_ANSWERS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Keep the previous run (and any faithfulness grades tied to its run_at) recoverable.
        try:
            previous_run_at = str(json.loads(path.read_text(encoding="utf-8")).get("run_at") or "unknown")
        except (json.JSONDecodeError, AttributeError):
            previous_run_at = "unknown"
        backup = backup_path(path, previous_run_at)
        backup.write_bytes(path.read_bytes())
        print(f"\nbacked up previous answers to {backup}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
