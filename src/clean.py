"""Clean filtered raw reviews into the target-schema parquet plus a quality report.

Run with: uv run python -m src.clean

Input is the git-ignored JSONL written by ``data/download_data.py``; output is
``data/processed/reviews_clean.parquet`` in the ``data-schema.md`` target schema.
The notebook, index, and app import these functions rather than reimplementing
them. ``review_text`` is kept byte-identical to the source for display and
citation; the normalized copy used for embedding lives in its own column.

Exit 0 when the parquet is written; exit 1 when the input is missing or no row
survives cleaning.
"""

import json
import math
import re
import sys
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from src import config

# Target schema (data-schema.md) plus the two derived columns the spec requires:
# the separate normalized embedding text and the date bucket that holds undated rows.
COLUMNS = [
    "review_id",
    "brand",
    "product_name",
    "rating",
    "review_title",
    "review_text",
    "review_text_normalized",
    "review_date",
    "date_bucket",
    "verified",
]

DROP_REASONS = (
    "no_brand_match",
    "empty_text",
    "invalid_rating",
    "duplicate_review_id",
    "duplicate_text",
)

_WHITESPACE = re.compile(r"\s+")
_ID_SEPARATOR = "\x1f"


def normalize_text(text: str) -> str:
    """Return text stripped and whitespace-collapsed, for embedding and dedup."""
    return _WHITESPACE.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Return the casefolded, whitespace-collapsed form used to spot copy-paste spam."""
    return normalize_text(text).casefold()


def make_review_id(record: dict[str, Any]) -> str:
    """Derive a stable short id; the source has none and citations must survive reruns.

    Hashing user, product, timestamp and body keeps the id identical across runs,
    which row numbers and UUIDs would not.
    """
    timestamp = record.get("timestamp")
    parts = [
        str(record.get("user_id") or ""),
        str(record.get("parent_asin") or ""),
        "" if timestamp is None else str(timestamp),
        str(record.get("text") or ""),
    ]
    digest = blake2b(
        _ID_SEPARATOR.join(parts).encode("utf-8"),
        digest_size=config.REVIEW_ID_DIGEST_BYTES,
    )
    return digest.hexdigest()


def parse_timestamp_ms(value: Any) -> datetime | None:
    """Convert an epoch-milliseconds timestamp to UTC, or None when unusable.

    None, 0, non-numeric values, and anything outside the configured year bounds
    all return None so the row lands in the undated bucket rather than being dropped.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    try:
        moment = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    if not config.REVIEW_YEAR_MIN <= moment.year <= config.REVIEW_YEAR_MAX:
        return None
    return moment


def date_bucket(moment: datetime | None) -> str:
    """Return the YYYY-MM bucket for a review, or the undated bucket."""
    return config.UNDATED_BUCKET if moment is None else moment.strftime("%Y-%m")


def parse_rating(value: Any) -> int | None:
    """Return the rating as an int in 1-5, or None when it is missing or invalid.

    NaN and infinity reach here whenever json.loads sees them, and int() raises
    on both, so they are screened out before the conversion.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    rating = int(value)
    if rating != value or not 1 <= rating <= 5:
        return None
    return rating


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank and malformed lines.

    A stray byte or a line that is valid JSON but not an object skips that line
    rather than ending a pass over tens of thousands of good rows.
    """
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def clean_records(
    records: Iterable[dict[str, Any]], sample: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw review records into the target schema and a measured quality report.

    Returns the cleaned frame and a report carrying total rows, rows per brand,
    date range, undated count, and the percent dropped per reason.
    """
    rows: list[dict[str, Any]] = []
    drops = {reason: 0 for reason in DROP_REASONS}
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    total_input = 0

    for record in records:
        total_input += 1

        # A source field can be any JSON type, so type-check before use: an
        # unhashable brand or a non-string body would otherwise abort the whole
        # run rather than costing one counted drop.
        brand = record.get("brand")
        if not isinstance(brand, str) or brand not in config.BRANDS:
            drops["no_brand_match"] += 1
            continue

        raw_text = record.get("text")
        text = raw_text if isinstance(raw_text, str) else ""
        normalized = normalize_text(text)
        if not normalized:
            drops["empty_text"] += 1
            continue

        rating = parse_rating(record.get("rating"))
        if rating is None:
            drops["invalid_rating"] += 1
            continue

        review_id = make_review_id(record)
        if review_id in seen_ids:
            drops["duplicate_review_id"] += 1
            continue

        key = normalized.casefold()
        if len(key) >= config.DEDUP_MIN_TEXT_CHARS:
            if key in seen_texts:
                drops["duplicate_text"] += 1
                continue
            seen_texts.add(key)

        seen_ids.add(review_id)
        moment = parse_timestamp_ms(record.get("timestamp"))
        rows.append(
            {
                "review_id": review_id,
                "brand": brand,
                "product_name": record.get("product_name") or "",
                "rating": rating,
                "review_title": record.get("title") or "",
                # Byte-identical to the source for display and citation.
                "review_text": text,
                "review_text_normalized": normalized,
                "review_date": moment,
                "date_bucket": date_bucket(moment),
                "verified": bool(record.get("verified_purchase")),
            }
        )

    frame = _build_frame(rows)
    return frame, build_report(frame, drops, total_input, sample)


def _build_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the typed target-schema frame, including for an empty result."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        frame = frame.astype(
            {
                "review_id": "object",
                "brand": "object",
                "product_name": "object",
                "rating": "int64",
                "review_title": "object",
                "review_text": "object",
                "review_text_normalized": "object",
                "date_bucket": "object",
                "verified": "bool",
            }
        )
    else:
        frame["rating"] = frame["rating"].astype("int64")
        frame["verified"] = frame["verified"].astype("bool")
    frame["review_date"] = pd.to_datetime(frame["review_date"], utc=True)
    return frame.reset_index(drop=True)


def build_report(
    frame: pd.DataFrame, drops: dict[str, int], total_input: int, sample: bool = False
) -> dict[str, Any]:
    """Assemble the measured quality report the spec's quality bar requires."""
    dated = frame.loc[frame["review_date"].notna(), "review_date"]
    dropped_total = sum(drops.values())

    def pct(count: int) -> float:
        return round(100.0 * count / total_input, 2) if total_input else 0.0

    return {
        "sample": sample,
        "total_input": total_input,
        "total_rows": int(len(frame)),
        "dropped_total": dropped_total,
        "dropped_pct": pct(dropped_total),
        "drops": {
            reason: {"count": count, "pct": pct(count)} for reason, count in drops.items()
        },
        "rows_per_brand": {
            brand: int((frame["brand"] == brand).sum()) for brand in config.BRANDS
        },
        "date_range": {
            "min": dated.min().isoformat() if not dated.empty else None,
            "max": dated.max().isoformat() if not dated.empty else None,
        },
        "undated": int((frame["date_bucket"] == config.UNDATED_BUCKET).sum()),
    }


def format_report(report: dict[str, Any]) -> str:
    """Render the report as the plain-text block printed by the entry point."""
    scope = "SAMPLE (bounded run, not the full corpus)" if report["sample"] else "full corpus"
    lines = [
        "=== Data quality report ===",
        f"scope: {scope}",
        f"rows in:  {report['total_input']:,}",
        f"rows out: {report['total_rows']:,}",
        f"dropped:  {report['dropped_total']:,} ({report['dropped_pct']}%)",
        "drops by reason:",
    ]
    for reason, stats in report["drops"].items():
        lines.append(f"  {reason}: {stats['count']:,} ({stats['pct']}%)")
    lines.append("rows per brand:")
    for brand, count in report["rows_per_brand"].items():
        lines.append(f"  {brand}: {count:,}")
    date_range = report["date_range"]
    lines.append(f"date range: {date_range['min']} .. {date_range['max']}")
    lines.append(f"undated: {report['undated']:,}")
    return "\n".join(lines)


def clean_file(
    input_path: Path | None = None,
    output_path: Path | None = None,
    sample: bool | None = None,
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the filtered JSONL into parquet and return the frame and report.

    The report is written beside the parquet so the measured numbers can be read
    back without repeating the multi-hour download.
    """
    input_path = input_path or config.RAW_FILTERED_JSONL
    output_path = output_path or config.CLEAN_PARQUET
    if sample is None:
        # The manifest describes the default raw JSONL only, so an explicitly
        # supplied input is never labelled from it.
        sample = _manifest_says_sample() if input_path == config.RAW_FILTERED_JSONL else False

    frame, report = clean_records(read_jsonl(input_path), sample=sample)
    if frame.empty:
        return frame, report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    if report_path is None:
        report_path = (
            config.CLEAN_REPORT_JSON
            if output_path == config.CLEAN_PARQUET
            else output_path.with_name(output_path.stem + "_quality_report.json")
        )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return frame, report


def _manifest_says_sample() -> bool:
    """Report whether the raw JSONL was produced by a bounded --limit run."""
    if not config.RAW_MANIFEST_JSON.exists():
        return False
    try:
        manifest = json.loads(config.RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(manifest.get("sample"))


def main() -> int:
    if not config.RAW_FILTERED_JSONL.exists():
        print(
            f"Missing {config.RAW_FILTERED_JSONL}. Run: "
            "uv run python data/download_data.py",
            file=sys.stderr,
        )
        return 1

    frame, report = clean_file()
    if frame.empty:
        print("No rows survived cleaning; parquet not written.", file=sys.stderr)
        if config.CLEAN_PARQUET.exists():
            # Nothing is deleted here, but the next story would otherwise read a
            # stale corpus believing this run produced it.
            print(
                f"WARNING: {config.CLEAN_PARQUET} is left over from an earlier run "
                "and does not reflect the current input.",
                file=sys.stderr,
            )
        print(format_report(report), file=sys.stderr)
        return 1

    print(format_report(report))
    print(f"wrote: {config.CLEAN_PARQUET}")
    print(f"wrote: {config.CLEAN_REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
