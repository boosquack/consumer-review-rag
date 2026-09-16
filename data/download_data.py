"""Stream Amazon Reviews 2023 and filter it to the locked P&G haircare brands.

Run with: uv run python data/download_data.py [--limit N] [--refresh]

Two streaming passes, neither of which stores a source file whole:

1. Metadata pass (2.8 GB) -- map ``parent_asin`` -> canonical brand and product
   title, using the metadata ``store`` field only. Title-only matches are
   third-party listings that merely name the brand, so they are not mapped.
2. Reviews pass (11 GB) -- keep only reviews whose ``parent_asin`` is in that
   map, writing them to a git-ignored JSONL under ``data/raw/`` with ``brand``
   and ``product_name`` joined in.

Output is written to a ``.part`` file and promoted only on a complete pass, so
an interrupted stream never leaves a half-written corpus behind. Exit 0 on
success; exit 1 on a stream failure, naming the bytes read.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

# Run as a script from the repo root, so put the root (not data/) on sys.path
# before importing the project's namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

CHUNK_BYTES = 1 << 20
REQUEST_TIMEOUT = (30, 120)

# Cheap prescreens run on raw bytes so that only candidate lines are JSON-parsed.
# Whatever they match is confirmed against the parsed record, never trusted alone.
_STORE_PRESCREEN = re.compile(rb'"store"\s*:\s*"((?:[^"\\]|\\.)*)"')
_PARENT_ASIN_PRESCREEN = re.compile(rb'"parent_asin"\s*:\s*"([^"]{1,32})"')

_BRAND_PATTERNS = {brand: re.compile(pattern) for brand, pattern in config.BRANDS.items()}


class NoReviewsMatched(RuntimeError):
    """A complete reviews pass matched nothing, so there is no corpus to promote."""

    def __init__(self, products: int, reviews_seen: int) -> None:
        super().__init__(
            f"no reviews matched {products:,} mapped products after {reviews_seen:,} reviews"
        )


class StreamInterrupted(RuntimeError):
    """A source stream failed partway through; carries the bytes read so far."""

    def __init__(self, url: str, bytes_read: int, cause: Exception) -> None:
        super().__init__(f"stream failed after {bytes_read:,} bytes from {url}: {cause}")
        self.url = url
        self.bytes_read = bytes_read
        self.cause = cause


def brand_for_store(store: str) -> str | None:
    """Return the canonical brand whose pattern matches this `store` value.

    A store naming two brands takes the first in config.BRANDS order, which is
    deterministic across runs.
    """
    if not store:
        return None
    for brand, pattern in _BRAND_PATTERNS.items():
        if pattern.search(store):
            return brand
    return None


def stream_lines(
    url: str, total_bytes: int, desc: str, session: requests.Session
) -> Iterator[bytes]:
    """Yield raw JSONL lines from a streamed HTTP body, showing byte progress.

    Raises StreamInterrupted, naming the bytes read, if the connection drops.
    """
    bytes_read = 0
    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            # Trust the served length over the constant, which is only an ETA.
            declared = response.headers.get("Content-Length") if response.headers else None
            expected = int(declared) if declared and declared.isdigit() else None
            remainder = b""
            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                file=sys.stderr,
            ) as progress:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    bytes_read += len(chunk)
                    progress.update(len(chunk))
                    lines = (remainder + chunk).split(b"\n")
                    remainder = lines.pop()
                    for line in lines:
                        if line.strip():
                            yield line
            if remainder.strip():
                yield remainder
            if expected is not None and bytes_read < expected:
                # A body can end early without raising; treating that as success
                # would cache a truncated corpus as a complete one.
                raise StreamInterrupted(
                    url, bytes_read, EOFError(f"body ended early, expected {expected:,} bytes")
                )
    except requests.RequestException as exc:
        raise StreamInterrupted(url, bytes_read, exc) from exc


def build_brand_map(session: requests.Session) -> dict[str, dict[str, str]]:
    """Stream product metadata and map parent_asin -> {brand, product_name}."""
    brand_map: dict[str, dict[str, str]] = {}
    for line in stream_lines(
        config.DATASET_META_URL, config.DATASET_META_BYTES, "metadata", session
    ):
        prescreen = _STORE_PRESCREEN.search(line)
        if prescreen is None or brand_for_store(prescreen.group(1).decode("utf-8", "replace")) is None:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        brand = brand_for_store(record.get("store") or "")
        parent_asin = record.get("parent_asin")
        if brand is None or not parent_asin:
            continue
        brand_map[parent_asin] = {
            "brand": brand,
            "product_name": record.get("title") or "",
        }
    return brand_map


def load_brand_map(session: requests.Session, refresh: bool) -> dict[str, dict[str, str]]:
    """Return the cached brand map, rebuilding it from the network when needed."""
    if config.RAW_BRAND_MAP_JSON.exists() and not refresh:
        try:
            cached = json.loads(config.RAW_BRAND_MAP_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A map truncated by an interrupted write would otherwise traceback on
            # every later run until someone deleted the file by hand.
            print(f"brand map: cached copy unreadable ({exc}); rebuilding", file=sys.stderr)
        else:
            print(f"brand map: cached, {len(cached):,} products")
            return cached

    brand_map = build_brand_map(session)
    config.RAW_BRAND_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Same write-then-promote discipline as the filtered JSONL below.
    map_part = config.RAW_BRAND_MAP_JSON.with_name(config.RAW_BRAND_MAP_JSON.name + ".part")
    map_part.write_text(json.dumps(brand_map), encoding="utf-8")
    map_part.replace(config.RAW_BRAND_MAP_JSON)
    per_brand: dict[str, int] = {}
    for entry in brand_map.values():
        per_brand[entry["brand"]] = per_brand.get(entry["brand"], 0) + 1
    print(f"brand map: {len(brand_map):,} products matched on `store`")
    for brand in config.BRANDS:
        print(f"  {brand}: {per_brand.get(brand, 0):,} products")
    return brand_map


def filter_reviews(
    brand_map: dict[str, dict[str, str]],
    session: requests.Session,
    limit: int | None,
) -> dict[str, int]:
    """Stream reviews, keeping those on mapped products, into the filtered JSONL.

    Writes to `<output>.part` and promotes it only after a complete pass.
    """
    out_path = config.RAW_FILTERED_JSONL
    part_path = out_path.with_name(out_path.name + ".part")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    matched = 0
    no_brand_match = 0
    with part_path.open("w", encoding="utf-8") as handle:
        for line in stream_lines(
            config.DATASET_REVIEWS_URL, config.DATASET_REVIEWS_BYTES, "reviews", session
        ):
            seen += 1
            prescreen = _PARENT_ASIN_PRESCREEN.search(line)
            if prescreen is None or prescreen.group(1).decode("utf-8", "replace") not in brand_map:
                no_brand_match += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = brand_map.get(record.get("parent_asin", ""))
            if entry is None:
                no_brand_match += 1
                continue
            record["brand"] = entry["brand"]
            record["product_name"] = entry["product_name"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            matched += 1
            if limit is not None and matched >= limit:
                break

    if matched == 0:
        # Promoting this would cache an empty corpus that every later run reports
        # as a valid cache.
        part_path.unlink(missing_ok=True)
        raise NoReviewsMatched(len(brand_map), seen)

    part_path.replace(out_path)
    return {"reviews_seen": seen, "matched": matched, "no_brand_match": no_brand_match}


def _cache_covers(limit: int | None) -> bool:
    """Report whether the cached JSONL already satisfies the requested scope.

    Existence alone is not enough: a cached `--limit 200` sample must not be
    served to a caller who asked for the full corpus.
    """
    if not config.RAW_MANIFEST_JSON.exists():
        # Pre-manifest or hand-made file: only a bounded request can trust it.
        return limit is not None
    try:
        manifest = json.loads(config.RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    cached_limit = manifest.get("limit")
    if cached_limit is None:
        return True  # a full corpus covers every request
    return limit is not None and limit <= cached_limit


def write_manifest(counts: dict[str, int], limit: int | None, brand_map_size: int) -> None:
    """Record how the filtered JSONL was produced, so the report can label a sample."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_repo": config.DATASET_HF_REPO,
        "reviews_url": config.DATASET_REVIEWS_URL,
        "meta_url": config.DATASET_META_URL,
        "products_in_brand_map": brand_map_size,
        "reviews_seen": counts["reviews_seen"],
        "matched": counts["matched"],
        "no_brand_match": counts["no_brand_match"],
        "limit": limit,
        "sample": limit is not None,
    }
    config.RAW_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N matching reviews; the report labels the output a sample",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch even when cached output is present",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        print("--limit must be a positive integer.", file=sys.stderr)
        return 1

    if config.RAW_FILTERED_JSONL.exists() and not args.refresh and _cache_covers(args.limit):
        print(f"cached: {config.RAW_FILTERED_JSONL} (use --refresh to re-fetch)")
        return 0

    session = requests.Session()
    try:
        brand_map = load_brand_map(session, args.refresh)
        if not brand_map:
            print("No products matched the brand patterns; nothing to filter.", file=sys.stderr)
            return 1
        counts = filter_reviews(brand_map, session, args.limit)
    except StreamInterrupted as exc:
        print(f"Stream interrupted: {exc}", file=sys.stderr)
        print("Partial output kept as .part and not promoted.", file=sys.stderr)
        return 1
    except NoReviewsMatched as exc:
        print(f"No reviews matched: {exc}", file=sys.stderr)
        print("Nothing promoted; check the brand patterns in src/config.py.", file=sys.stderr)
        return 1
    finally:
        session.close()

    write_manifest(counts, args.limit, len(brand_map))
    label = " (sample)" if args.limit is not None else ""
    print(f"reviews scanned: {counts['reviews_seen']:,}")
    print(f"skipped, no brand match: {counts['no_brand_match']:,}")
    print(f"reviews kept{label}: {counts['matched']:,} -> {config.RAW_FILTERED_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
