"""Write a blind grading sheet for hand-grading the saved answers (story 6 follow-up).

Reads eval/answers.json and the gold set, and writes one Markdown file with each
answered question, its answer, and the full review text of every source. The agent's
grades and notes are deliberately left out so the human grade is independent.

Run with: uv run python -m src.grading_sheet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src import config


def full_texts(review_ids: Sequence[str], reviews: pd.DataFrame) -> dict[str, str]:
    """Map review_id to the unmodified review_text for the ids that exist."""
    subset = reviews[reviews["review_id"].isin(set(review_ids))]
    return dict(zip(subset["review_id"], subset["review_text"].astype(str)))


def render_question(record: dict[str, Any], texts: dict[str, str]) -> str:
    filters = f"brand={record.get('brand')}, rating_band={record.get('rating_band')}"
    lines = [
        f"## {record['id']}: {record['question']}",
        f"Filters: {filters}",
        "",
        "**Answer**",
        "",
        record["text"],
        "",
        "**Sources** (full review text; the model saw the retrieved chunk, which differs only for long reviews)",
        "",
    ]
    for source in record["sources"]:
        text = texts.get(source["review_id"], source["review_text"])
        lines.append(f"[{source['number']}] {source['brand']}, {source['rating']}★")
        lines.append("")
        lines.append("> " + text.replace("\n", "\n> "))
        lines.append("")
    lines.append("**Your grade:** pass / fail  \n**Note:**")
    lines.append("")
    return "\n".join(lines)


def render_gold(gold: dict[str, Any]) -> str:
    lines = ["# Gold questions to review (cut, keep, or rewrite)", ""]
    for question in gold["questions"]:
        kind = "answerable" if question["answerable"] else "unanswerable"
        lines.append(f"- **{question['id']}** ({question['type']}, {kind}): {question['question']}")
        lines.append(f"  - filters: brand={question.get('brand')}, rating_band={question.get('rating_band')}")
        if question.get("relevant_pattern"):
            lines.append(f"  - relevant_pattern: `{question['relevant_pattern']}`")
        lines.append(f"  - notes: {question.get('notes', '')}")
    lines.append("")
    return "\n".join(lines)


def build_sheet(saved: dict[str, Any], gold: dict[str, Any], reviews: pd.DataFrame) -> str:
    answered = [r for r in saved["questions"] if r.get("sources")]
    ids = [s["review_id"] for r in answered for s in r["sources"]]
    texts = full_texts(ids, reviews)
    parts = [
        "# Blind faithfulness grading sheet",
        "",
        f"Run: {saved['run_at']}. Pass means every claim is stated by the cited reviews, with no invented "
        "or overstated claims; a refusal passes when the retrieved reviews really do not answer the question. Record grades in eval/faithfulness_human.json (same schema as "
        "eval/faithfulness.json; all 18 need grades, refusals included).",
        "",
    ]
    parts += [render_question(r, texts) for r in answered]
    parts.append(render_gold(gold))
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    saved = json.loads(config.EVAL_ANSWERS_JSON.read_text(encoding="utf-8"))
    gold = json.loads(config.EVAL_GOLD_JSON.read_text(encoding="utf-8"))
    reviews = pd.read_parquet(config.CLEAN_PARQUET, columns=["review_id", "review_text"])
    sheet = build_sheet(saved, gold, reviews)
    path = Path(config.EVAL_GRADING_SHEET)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sheet, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
