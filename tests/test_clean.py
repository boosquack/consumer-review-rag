"""I/O matrix tests for src.clean and data/download_data.py (story 2).

No network: the requests session is mocked and every fixture is inline.

Run with: uv run python -m unittest -v
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from src import clean, config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_download_module():
    """Import data/download_data.py, which is a script rather than a package member."""
    path = REPO_ROOT / "data" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_data = _load_download_module()

LONG_TEXT = (
    "This shampoo left my scalp calm for the first time in months and the scent is mild."
)


def review(**overrides):
    """A raw review record as data/download_data.py writes it, with brand joined in."""
    record = {
        "rating": 5.0,
        "title": "Great",
        "text": LONG_TEXT,
        "asin": "B001",
        "parent_asin": "P001",
        "user_id": "U1",
        "timestamp": 1_616_743_454_733,  # 2021-03-26 UTC, epoch ms
        "helpful_vote": 2,
        "verified_purchase": True,
        "brand": "Pantene",
        "product_name": "Pantene Pro-V Shampoo",
    }
    record.update(overrides)
    return record


class CleanMatrixTest(unittest.TestCase):
    def test_happy_path_yields_target_schema_unique_ids_int_ratings_canonical_brands(self):
        records = [
            review(user_id="U1", brand="Pantene", rating=5.0),
            review(user_id="U2", brand="Head & Shoulders", rating=1.0, text=LONG_TEXT + " a"),
            review(user_id="U3", brand="Herbal Essences", rating=3.0, text=LONG_TEXT + " b"),
        ]
        frame, report = clean.clean_records(records)

        self.assertEqual(list(frame.columns), clean.COLUMNS)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame["review_id"].nunique(), 3)
        self.assertEqual(frame["rating"].dtype.kind, "i")
        self.assertTrue(frame["rating"].between(1, 5).all())
        self.assertTrue(set(frame["brand"]).issubset(config.BRANDS))
        self.assertEqual(report["total_rows"], 3)
        self.assertEqual(report["dropped_total"], 0)
        self.assertEqual(report["rows_per_brand"]["Pantene"], 1)

    def test_review_text_stays_byte_identical_and_normalized_copy_is_separate(self):
        raw = "  Loved   it.\n\nSo  soft!  "
        frame, _ = clean.clean_records([review(text=raw)])

        self.assertEqual(frame.loc[0, "review_text"], raw)
        self.assertEqual(frame.loc[0, "review_text_normalized"], "Loved it. So soft!")

    def test_unknown_product_without_brand_is_dropped_as_no_brand_match(self):
        for brand in (None, "", "Dove"):
            with self.subTest(brand=brand):
                frame, report = clean.clean_records([review(brand=brand)])
                self.assertTrue(frame.empty)
                self.assertEqual(report["drops"]["no_brand_match"]["count"], 1)

    def test_empty_or_whitespace_text_is_dropped_as_empty_text(self):
        for text in ("", "   ", "\n\t", None):
            with self.subTest(text=repr(text)):
                frame, report = clean.clean_records([review(text=text)])
                self.assertTrue(frame.empty)
                self.assertEqual(report["drops"]["empty_text"]["count"], 1)

    def test_missing_or_out_of_range_timestamp_keeps_row_in_undated_bucket(self):
        cases = {
            "null": None,
            "zero": 0,
            "pre-2000": 946_684_799_000 - 86_400_000,
            "post-2026": 1_798_761_600_000,  # year 2027
            "non-numeric": "2021-03-26",
        }
        for label, timestamp in cases.items():
            with self.subTest(case=label):
                frame, report = clean.clean_records([review(timestamp=timestamp)])
                self.assertEqual(len(frame), 1, "undated rows are kept, not dropped")
                self.assertTrue(frame["review_date"].isna().all())
                self.assertEqual(frame.loc[0, "date_bucket"], config.UNDATED_BUCKET)
                self.assertEqual(report["undated"], 1)
                self.assertIsNone(report["date_range"]["min"])

    def test_valid_timestamp_parses_to_utc_and_month_bucket(self):
        frame, report = clean.clean_records([review()])
        self.assertEqual(frame.loc[0, "date_bucket"], "2021-03")
        self.assertEqual(report["undated"], 0)
        self.assertTrue(report["date_range"]["min"].startswith("2021-03-26"))

    def test_repeat_review_id_is_dropped(self):
        frame, report = clean.clean_records([review(), review()])
        self.assertEqual(len(frame), 1)
        self.assertEqual(report["drops"]["duplicate_review_id"]["count"], 1)

    def test_duplicate_long_normalized_text_is_dropped_case_and_whitespace_insensitive(self):
        duplicate = review(user_id="U2", timestamp=1_600_000_000_000, text=f"  {LONG_TEXT.upper()}  ")
        frame, report = clean.clean_records([review(), duplicate])

        self.assertGreaterEqual(len(clean.dedup_key(LONG_TEXT)), config.DEDUP_MIN_TEXT_CHARS)
        self.assertEqual(len(frame), 1)
        self.assertEqual(report["drops"]["duplicate_text"]["count"], 1)

    def test_short_duplicate_text_is_exempt_and_kept(self):
        short = "Love it!"
        self.assertLess(len(short), config.DEDUP_MIN_TEXT_CHARS)
        records = [
            review(user_id="U1", text=short),
            review(user_id="U2", text=short),
        ]
        frame, report = clean.clean_records(records)

        self.assertEqual(len(frame), 2)
        self.assertEqual(report["drops"]["duplicate_text"]["count"], 0)

    def test_invalid_rating_is_dropped(self):
        for rating in (None, 0, 6, 2.5, "five"):
            with self.subTest(rating=repr(rating)):
                frame, report = clean.clean_records([review(rating=rating)])
                self.assertTrue(frame.empty)
                self.assertEqual(report["drops"]["invalid_rating"]["count"], 1)

    def test_review_ids_are_identical_across_runs(self):
        records = [review(user_id=f"U{i}", text=f"{LONG_TEXT} {i}") for i in range(5)]
        first, _ = clean.clean_records(records)
        second, _ = clean.clean_records(records)
        self.assertEqual(list(first["review_id"]), list(second["review_id"]))

    def test_report_covers_totals_brands_dates_undated_and_percent_per_reason(self):
        records = [
            review(user_id="U1"),
            review(user_id="U2", text=LONG_TEXT + " x", timestamp=None),
            review(user_id="U3", text=""),
            review(user_id="U4", brand="Dove", text=LONG_TEXT + " y"),
        ]
        _, report = clean.clean_records(records)

        self.assertEqual(report["total_input"], 4)
        self.assertEqual(report["total_rows"], 2)
        self.assertEqual(report["dropped_total"], 2)
        self.assertEqual(report["dropped_pct"], 50.0)
        self.assertEqual(report["drops"]["empty_text"]["pct"], 25.0)
        self.assertEqual(report["undated"], 1)
        self.assertEqual(set(report["rows_per_brand"]), set(config.BRANDS))

        text = clean.format_report(report)
        for fragment in ("rows out", "empty_text", "Pantene", "date range", "undated"):
            self.assertIn(fragment, text)

    def test_sample_run_is_labeled_in_the_report(self):
        _, report = clean.clean_records([review()], sample=True)
        self.assertTrue(report["sample"])
        self.assertIn("SAMPLE", clean.format_report(report))

    def test_clean_file_round_trips_through_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reviews_filtered.jsonl"
            target = Path(tmp) / "reviews_clean.parquet"
            records = [review(user_id=f"U{i}", text=f"{LONG_TEXT} {i}") for i in range(3)]
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
            )

            frame, report = clean.clean_file(source, target, sample=False)
            self.assertTrue(target.exists())

            import pandas as pd

            reloaded = pd.read_parquet(target)
            self.assertEqual(list(reloaded.columns), clean.COLUMNS)
            self.assertEqual(len(reloaded), 3)
            self.assertEqual(report["total_rows"], 3)
            self.assertEqual(list(reloaded["review_id"]), list(frame["review_id"]))


class _FakeResponse:
    """Stands in for a streamed requests response, optionally failing mid-body."""

    def __init__(self, chunks, fail_after=None, headers=None):
        self.chunks = chunks
        self.fail_after = fail_after
        self.headers = {} if headers is None else headers

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index == self.fail_after:
                raise requests.ConnectionError("connection reset by peer")
            yield chunk


class _FakeSession:
    """Serves canned bodies per URL and records which URLs were requested."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        return self.bodies[url]

    def close(self):
        return None


META_LINES = [
    # Matched on `store`: these three become the brand map.
    {"store": "Pantene", "title": "Pantene Pro-V Shampoo", "parent_asin": "P001"},
    {"store": "Head & Shoulders", "title": "H&S Classic Clean", "parent_asin": "P002"},
    {"store": "Herbal Essences", "title": "Bio:Renew Argan Oil", "parent_asin": "P003"},
    # Title names a brand but `store` does not: a third-party listing, not mapped.
    {"store": "SomeReseller", "title": "Pantene Pro-V bundle", "parent_asin": "P004"},
    {"store": "shiyeen", "title": "Hair Chalk", "parent_asin": "P005"},
]

REVIEW_LINES = [
    {"rating": 5.0, "title": "Great", "text": LONG_TEXT, "parent_asin": "P001",
     "user_id": "U1", "timestamp": 1_616_743_454_733, "verified_purchase": True},
    {"rating": 1.0, "title": "Bad", "text": LONG_TEXT + " b", "parent_asin": "P002",
     "user_id": "U2", "timestamp": 1_616_743_454_733, "verified_purchase": False},
    {"rating": 4.0, "title": "Fine", "text": LONG_TEXT + " c", "parent_asin": "P003",
     "user_id": "U3", "timestamp": 1_616_743_454_733, "verified_purchase": True},
    # Unmapped products: title-only match and an unrelated one.
    {"rating": 5.0, "title": "Nice", "text": LONG_TEXT + " d", "parent_asin": "P004",
     "user_id": "U4", "timestamp": 1_616_743_454_733, "verified_purchase": True},
    {"rating": 2.0, "title": "Meh", "text": LONG_TEXT + " e", "parent_asin": "P999",
     "user_id": "U5", "timestamp": 1_616_743_454_733, "verified_purchase": True},
]


def _body(records, chunks=3, fail_after=None, declared_bytes=None):
    """Encode records as a JSONL body split across chunks, to exercise line splicing.

    `declared_bytes` sets a Content-Length larger than the body to simulate a
    server that closes the connection early without raising.
    """
    blob = ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")
    size = max(1, len(blob) // chunks)
    pieces = [blob[i : i + size] for i in range(0, len(blob), size)]
    headers = {"Content-Length": str(declared_bytes)} if declared_bytes else {}
    return _FakeResponse(pieces, fail_after=fail_after, headers=headers)


class DownloadMatrixTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.jsonl = root / "reviews_filtered.jsonl"
        self.part = root / "reviews_filtered.jsonl.part"
        patches = {
            "RAW_FILTERED_JSONL": self.jsonl,
            "RAW_BRAND_MAP_JSON": root / "brand_map.json",
            "RAW_MANIFEST_JSON": root / "manifest.json",
        }
        for name, value in patches.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, argv, session):
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(download_data.requests, "Session", return_value=session),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = download_data.main(argv)
        return code, out.getvalue(), err.getvalue()

    def session(
        self,
        meta_fail_after=None,
        review_fail_after=None,
        meta_lines=None,
        review_declared_bytes=None,
    ):
        return _FakeSession(
            {
                config.DATASET_META_URL: _body(
                    META_LINES if meta_lines is None else meta_lines, fail_after=meta_fail_after
                ),
                config.DATASET_REVIEWS_URL: _body(
                    REVIEW_LINES,
                    fail_after=review_fail_after,
                    declared_bytes=review_declared_bytes,
                ),
            }
        )

    def written_records(self):
        return [json.loads(line) for line in self.jsonl.read_text(encoding="utf-8").splitlines()]

    def test_happy_path_writes_store_matched_reviews_with_brand_and_product_name(self):
        code, out, _ = self.run_main([], self.session())

        self.assertEqual(code, 0)
        records = self.written_records()
        self.assertEqual([r["parent_asin"] for r in records], ["P001", "P002", "P003"])
        self.assertEqual(
            [r["brand"] for r in records],
            ["Pantene", "Head & Shoulders", "Herbal Essences"],
        )
        self.assertEqual(records[0]["product_name"], "Pantene Pro-V Shampoo")
        self.assertIn("reviews kept", out)
        self.assertFalse(self.part.exists(), "the .part file is promoted, not left behind")

    def test_title_only_match_is_not_mapped_so_its_reviews_are_skipped(self):
        self.run_main([], self.session())
        self.assertNotIn("P004", [r["parent_asin"] for r in self.written_records()])

    def test_unknown_product_is_skipped(self):
        self.run_main([], self.session())
        self.assertNotIn("P999", [r["parent_asin"] for r in self.written_records()])

    def test_cached_raw_skips_the_network_and_refresh_re_fetches(self):
        self.jsonl.write_text("", encoding="utf-8")
        # The manifest is what marks the cache a complete pass; existence alone
        # must not qualify, or a truncated corpus would be served as a full one.
        config.RAW_MANIFEST_JSON.write_text(
            json.dumps({"limit": None, "sample": False}), encoding="utf-8"
        )

        session = self.session()
        code, out, _ = self.run_main([], session)
        self.assertEqual(code, 0)
        self.assertEqual(session.requested, [], "no network pass when the JSONL is cached")
        self.assertIn("cached", out)

        fresh = self.session()
        code, _, _ = self.run_main(["--refresh"], fresh)
        self.assertEqual(code, 0)
        self.assertEqual(len(fresh.requested), 2)
        self.assertEqual(len(self.written_records()), 3)

    def test_limit_bounds_the_run_and_labels_the_output_a_sample(self):
        code, out, _ = self.run_main(["--limit", "2"], self.session())

        self.assertEqual(code, 0)
        self.assertEqual(len(self.written_records()), 2)
        manifest = json.loads(config.RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
        self.assertTrue(manifest["sample"])
        self.assertEqual(manifest["limit"], 2)
        self.assertIn("(sample)", out)

        _, report = clean.clean_records(
            (record for record in self.written_records()), sample=manifest["sample"]
        )
        self.assertIn("SAMPLE", clean.format_report(report))

    def test_interrupted_review_stream_exits_non_zero_and_never_promotes_the_part_file(self):
        code, _, err = self.run_main([], self.session(review_fail_after=1))

        self.assertEqual(code, 1)
        self.assertIn("Stream interrupted", err)
        self.assertIn("bytes", err)
        self.assertFalse(self.jsonl.exists(), "partial output is never promoted")
        self.assertTrue(self.part.exists(), "the partial pass stays as .part")

    def test_interrupted_metadata_stream_exits_non_zero_before_the_reviews_pass(self):
        session = self.session(meta_fail_after=1)
        code, _, err = self.run_main([], session)

        self.assertEqual(code, 1)
        self.assertIn("Stream interrupted", err)
        self.assertEqual(session.requested, [config.DATASET_META_URL])

    def test_brand_for_store_matches_canonical_brands_only(self):
        self.assertEqual(download_data.brand_for_store("PANTENE"), "Pantene")
        self.assertEqual(download_data.brand_for_store("head and shoulders"), "Head & Shoulders")
        self.assertIsNone(download_data.brand_for_store("SomeReseller"))
        self.assertIsNone(download_data.brand_for_store(""))

    def test_multi_brand_store_takes_the_first_brand_in_config_order(self):
        # The frozen decision says `store` settles the brand; config order is what
        # makes that deterministic when one store names two of them.
        self.assertEqual(download_data.brand_for_store("Pantene & Herbal Essences"), "Pantene")
        self.assertEqual(
            download_data.brand_for_store("Head & Shoulders and Pantene"), "Head & Shoulders"
        )

    def test_cached_brand_map_is_reused_without_re_streaming_metadata(self):
        config.RAW_BRAND_MAP_JSON.write_text(
            json.dumps({"P001": {"brand": "Pantene", "product_name": "Cached Pantene"}}),
            encoding="utf-8",
        )
        session = self.session()
        code, out, _ = self.run_main([], session)

        self.assertEqual(code, 0)
        self.assertEqual(session.requested, [config.DATASET_REVIEWS_URL])
        self.assertIn("cached", out)
        records = self.written_records()
        self.assertEqual([r["product_name"] for r in records], ["Cached Pantene"])

    def test_unreadable_cached_brand_map_is_rebuilt_instead_of_tracebacking(self):
        config.RAW_BRAND_MAP_JSON.write_text('{"P001": {"brand"', encoding="utf-8")
        session = self.session()
        code, _, err = self.run_main([], session)

        self.assertEqual(code, 0)
        self.assertIn("unreadable", err)
        self.assertEqual(len(session.requested), 2)
        self.assertEqual(len(self.written_records()), 3)

    def test_empty_brand_map_exits_1_before_the_reviews_pass(self):
        unmatched = [{"store": "SomeReseller", "title": "Hair Chalk", "parent_asin": "P005"}]
        session = self.session(meta_lines=unmatched)
        code, _, err = self.run_main([], session)

        self.assertEqual(code, 1)
        self.assertIn("No products matched", err)
        self.assertNotIn(config.DATASET_REVIEWS_URL, session.requested)
        self.assertFalse(self.jsonl.exists())

    def test_zero_matching_reviews_is_an_error_and_nothing_is_promoted(self):
        brand_map = {"P_ABSENT": {"brand": "Pantene", "product_name": "Nothing reviews this"}}
        config.RAW_BRAND_MAP_JSON.write_text(json.dumps(brand_map), encoding="utf-8")
        code, _, err = self.run_main([], self.session())

        self.assertEqual(code, 1)
        self.assertIn("No reviews matched", err)
        self.assertFalse(self.jsonl.exists(), "an empty corpus is never promoted")
        self.assertFalse(self.part.exists())

    def test_body_ending_early_is_treated_as_an_interruption(self):
        session = self.session(review_declared_bytes=10_000_000)
        code, _, err = self.run_main([], session)

        self.assertEqual(code, 1)
        self.assertIn("ended early", err)
        self.assertFalse(self.jsonl.exists(), "a truncated pass is never promoted")

    def test_cached_sample_is_not_served_to_a_full_corpus_request(self):
        self.run_main(["--limit", "2"], self.session())
        self.assertEqual(len(self.written_records()), 2)

        refetch = self.session()
        code, _, _ = self.run_main([], refetch)

        self.assertEqual(code, 0)
        self.assertIn(
            config.DATASET_REVIEWS_URL,
            refetch.requested,
            "a bounded sample does not satisfy a full run",
        )
        self.assertEqual(len(self.written_records()), 3)
        manifest = json.loads(config.RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
        self.assertFalse(manifest["sample"])

    def test_full_corpus_cache_satisfies_a_bounded_request(self):
        self.run_main([], self.session())
        session = self.session()
        code, out, _ = self.run_main(["--limit", "2"], session)

        self.assertEqual(code, 0)
        self.assertIn("cached", out)
        self.assertEqual(session.requested, [])

    def test_manifest_records_the_unmapped_skip_count(self):
        self.run_main([], self.session())
        manifest = json.loads(config.RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
        # Two of the five review lines are on unmapped products (P004, P999).
        self.assertEqual(manifest["no_brand_match"], 2)
        self.assertEqual(manifest["matched"], 3)


class CleanGuardTest(unittest.TestCase):
    """Malformed source values cost one counted drop, never the whole run."""

    def test_non_finite_rating_is_counted_not_raised(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                frame, report = clean.clean_records([review(rating=value)])
                self.assertTrue(frame.empty)
                self.assertEqual(report["drops"]["invalid_rating"]["count"], 1)

    def test_non_string_text_is_counted_as_empty_not_raised(self):
        for value in (123, ["a"], {"b": 1}):
            with self.subTest(value=repr(value)):
                frame, report = clean.clean_records([review(text=value)])
                self.assertTrue(frame.empty)
                self.assertEqual(report["drops"]["empty_text"]["count"], 1)

    def test_unhashable_brand_is_counted_as_no_brand_match_not_raised(self):
        frame, report = clean.clean_records([review(brand=["Pantene"])])
        self.assertTrue(frame.empty)
        self.assertEqual(report["drops"]["no_brand_match"]["count"], 1)

    def test_non_object_json_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "raw.jsonl"
            source.write_text(
                "\n".join(["[1, 2, 3]", '"a string"', "42", json.dumps(review())]) + "\n",
                encoding="utf-8",
            )
            records = list(clean.read_jsonl(source))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["brand"], "Pantene")


class CleanFileLabelTest(unittest.TestCase):
    """clean_file derives the sample label from the manifest when not told."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "reviews_filtered.jsonl"
        self.source.write_text(json.dumps(review()) + "\n", encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        for name, value in {
            "RAW_FILTERED_JSONL": self.source,
            "RAW_MANIFEST_JSON": self.manifest,
            "CLEAN_PARQUET": self.root / "reviews_clean.parquet",
            "CLEAN_REPORT_JSON": self.root / "quality_report.json",
        }.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_sample_manifest_labels_the_report_a_sample(self):
        self.manifest.write_text(json.dumps({"sample": True, "limit": 200}), encoding="utf-8")
        _, report = clean.clean_file()
        self.assertTrue(report["sample"])
        self.assertIn("SAMPLE", clean.format_report(report))

    def test_full_manifest_labels_the_report_a_full_corpus(self):
        self.manifest.write_text(json.dumps({"sample": False, "limit": None}), encoding="utf-8")
        _, report = clean.clean_file()
        self.assertFalse(report["sample"])
        self.assertIn("full corpus", clean.format_report(report))

    def test_missing_or_corrupt_manifest_defaults_to_full_corpus(self):
        _, report = clean.clean_file()
        self.assertFalse(report["sample"])

        self.manifest.write_text("{not json", encoding="utf-8")
        _, report = clean.clean_file()
        self.assertFalse(report["sample"])

    def test_explicit_input_path_is_never_labelled_from_the_global_manifest(self):
        self.manifest.write_text(json.dumps({"sample": True}), encoding="utf-8")
        other = self.root / "elsewhere.jsonl"
        other.write_text(json.dumps(review()) + "\n", encoding="utf-8")

        _, report = clean.clean_file(other, self.root / "other.parquet")
        self.assertFalse(report["sample"])

    def test_the_quality_report_is_written_next_to_the_parquet(self):
        clean.clean_file()
        written = json.loads(config.CLEAN_REPORT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(written["total_rows"], 1)
        self.assertIn("rows_per_brand", written)


class CleanMainTest(unittest.TestCase):
    """The documented exit-code contract of `uv run python -m src.clean`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "reviews_filtered.jsonl"
        self.parquet = self.root / "reviews_clean.parquet"
        for name, value in {
            "RAW_FILTERED_JSONL": self.source,
            "RAW_MANIFEST_JSON": self.root / "manifest.json",
            "CLEAN_PARQUET": self.parquet,
            "CLEAN_REPORT_JSON": self.root / "quality_report.json",
        }.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = clean.main()
        return code, out.getvalue(), err.getvalue()

    def test_missing_input_exits_1_and_names_the_download_command(self):
        code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("data/download_data.py", err)
        self.assertFalse(self.parquet.exists())

    def test_no_surviving_rows_exits_1_without_writing_a_parquet(self):
        self.source.write_text(json.dumps(review(text="")) + "\n", encoding="utf-8")
        code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("No rows survived", err)
        self.assertFalse(self.parquet.exists())

    def test_a_failed_reclean_warns_that_an_existing_parquet_is_stale(self):
        self.source.write_text(json.dumps(review()) + "\n", encoding="utf-8")
        self.assertEqual(self.run_main()[0], 0)

        self.source.write_text(json.dumps(review(text="")) + "\n", encoding="utf-8")
        code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("WARNING", err)
        self.assertIn("earlier run", err)

    def test_happy_path_exits_0_and_reports_both_written_files(self):
        self.source.write_text(json.dumps(review()) + "\n", encoding="utf-8")
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("wrote:", out)
        self.assertIn("rows out", out)
        self.assertTrue(self.parquet.exists())
        self.assertTrue(config.CLEAN_REPORT_JSON.exists())


class ConfigConstantsTest(unittest.TestCase):
    def test_pipeline_paths_live_under_the_project_data_dirs(self):
        self.assertEqual(config.RAW_FILTERED_JSONL.parent, config.DATA_RAW_DIR)
        self.assertEqual(config.CLEAN_PARQUET.parent, config.DATA_PROCESSED_DIR)
        self.assertTrue(config.DATASET_REVIEWS_URL.startswith(config.DATASET_RESOLVE_BASE))
        self.assertIn(config.DATASET_HF_REPO, config.DATASET_RESOLVE_BASE)


if __name__ == "__main__":
    unittest.main()
