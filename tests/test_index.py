"""I/O matrix tests for src.index (story 4).

No model, no network, and never the real .chroma/ store: a whitespace tokenizer
with character offsets, a deterministic bag-of-words embedder, and Chroma clients
backed by memory or a temp dir stand in for the real pieces.

Run with: uv run python -m unittest -v
"""

import hashlib
import math
import re
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import chromadb
import pandas as pd

from src import config, index

_WORD = re.compile(r"\S+")
_DIM = 32


class FakeTokenizer:
    """Whitespace tokenizer shaped like a Hugging Face fast tokenizer call."""

    def __init__(self, special_tokens: int = 0, pattern: str = r"\S+"):
        self.special_tokens = special_tokens
        self.pattern = re.compile(pattern)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, verbose=True):
        offsets = [(match.start(), match.end()) for match in self.pattern.finditer(text)]
        return {"offset_mapping": offsets, "input_ids": list(range(len(offsets)))}

    def num_special_tokens_to_add(self):
        return self.special_tokens


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors; records every batch it embeds."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [embed_text(text) for text in texts]


def embed_text(text: str) -> list[float]:
    vector = [0.0] * _DIM
    vector[0] = 0.01  # keeps a wordless text from being a zero vector
    for word in _WORD.findall(text.lower()):
        slot = 1 + int(hashlib.md5(word.encode()).hexdigest(), 16) % (_DIM - 1)
        vector[slot] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def failing_embedder(texts):
    raise AssertionError("embedder must not be called")


def review_frame(rows):
    """Clean-schema frame from (review_id, brand, rating, review_text[, date]) tuples."""
    records = []
    for row in rows:
        review_id, brand, rating, text = row[:4]
        date = row[4] if len(row) > 4 else "2021-03-04"
        records.append(
            {
                "review_id": review_id,
                "brand": brand,
                "product_name": f"{brand} Shampoo",
                "rating": rating,
                "review_title": f"title {review_id}",
                "review_text": text,
                "review_text_normalized": " ".join(text.split()),
                "review_date": pd.NaT if date is None else pd.Timestamp(date, tz="UTC"),
                "date_bucket": config.UNDATED_BUCKET if date is None else date[:7],
                "verified": True,
            }
        )
    return pd.DataFrame(records)


def ephemeral_client():
    return chromadb.EphemeralClient(settings=index.chroma_settings())


def unique_name():
    return f"test-{uuid.uuid4().hex[:12]}"


def silent(_message):
    pass


class ChunkReviewTest(unittest.TestCase):
    def test_short_review_is_one_whole_chunk(self):
        chunks = index.chunk_review("fits in the window", FakeTokenizer(), max_tokens=4, overlap=1)
        self.assertEqual(chunks, [index.Chunk("fits in the window", 0, 1)])

    def test_long_review_splits_into_exact_overlapping_slices(self):
        words = [f"w{i}" for i in range(23)]
        text = "  ".join(words)
        chunks = index.chunk_review(text, FakeTokenizer(), max_tokens=10, overlap=3)

        self.assertGreater(len(chunks), 1)
        tokens = [chunk.text.split() for chunk in chunks]
        for position, chunk in enumerate(chunks):
            self.assertIn(chunk.text, text)
            self.assertEqual(chunk.chunk_index, position)
            self.assertEqual(chunk.n_chunks, len(chunks))
            self.assertLessEqual(len(tokens[position]), 10)
        for previous, following in zip(tokens, tokens[1:]):
            self.assertEqual(previous[-3:], following[:3])
        self.assertEqual(tokens[0][0], "w0")
        self.assertEqual(tokens[-1][-1], "w22")

    def test_next_window_starts_on_a_word_boundary(self):
        # Punctuation is its own token glued to the word before it, like a subword.
        glued = FakeTokenizer(pattern=r"\w+|[^\w\s]")
        text = "aa bb, cc dd, ee ff"
        chunks = index.chunk_review(text, glued, max_tokens=3, overlap=1)
        # Plain stepping would open windows on "," twice; both move to the next word.
        self.assertEqual([chunk.text for chunk in chunks], ["aa bb,", "cc dd,", "ee ff"])

    def test_exactly_max_tokens_is_not_split(self):
        chunks = index.chunk_review("a b c d", FakeTokenizer(), max_tokens=4, overlap=1)
        self.assertEqual(len(chunks), 1)

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            index.chunk_review("a b", FakeTokenizer(), max_tokens=4, overlap=4)

    def test_budget_subtracts_special_tokens(self):
        self.assertEqual(index.content_token_budget(FakeTokenizer(2)), config.CHUNK_MAX_TOKENS - 2)


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        self.frame = review_frame([("a", "Pantene", 1, "too dry"), ("b", "Pantene", 5, "soft", None)])

    def test_order_independent_and_sensitive_to_ids(self):
        fingerprint = index.corpus_fingerprint(self.frame)
        self.assertEqual(index.corpus_fingerprint(self.frame.iloc[::-1]), fingerprint)
        self.assertEqual(index.corpus_fingerprint(self.frame.set_index("review_id", drop=False)), fingerprint)
        renamed = self.frame.assign(review_id=["a", "c"])
        self.assertNotEqual(index.corpus_fingerprint(renamed), fingerprint)
        self.assertTrue(fingerprint.startswith("2:"))

    def test_same_review_id_with_changed_content_changes_fingerprint(self):
        fingerprint = index.corpus_fingerprint(self.frame)
        changes = {
            "brand": "Herbal Essences",
            "rating": 3,
            "review_text_normalized": "too dry!",
            "product_name": "Other Shampoo",
            "review_date": pd.Timestamp("2022-01-01", tz="UTC"),
        }
        for column, value in changes.items():
            changed = self.frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column):
                self.assertEqual(changed["review_id"].tolist(), ["a", "b"])
                self.assertNotEqual(index.corpus_fingerprint(changed), fingerprint)


class BuildIndexTest(unittest.TestCase):
    def setUp(self):
        self.client = ephemeral_client()
        self.name = unique_name()
        self.frame = review_frame(
            [
                ("r1", "Pantene", 1, "made my hair  dry and brittle"),
                ("r2", "Head & Shoulders", 5, "cleared my dandruff fast", None),
                ("r3", "Herbal Essences", 4, "smells lovely"),
            ]
        )

    def tearDown(self):
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass

    def build(self, frame=None, embedder=None, **kwargs):
        return index.build_index(
            self.frame if frame is None else frame,
            client=self.client,
            embedder=embedder or FakeEmbedder(),
            tokenizer=FakeTokenizer(),
            collection_name=self.name,
            log=silent,
            **kwargs,
        )

    def test_build_writes_collection_with_metadata(self):
        with mock.patch.object(index, "load_model", side_effect=AssertionError("no model in tests")):
            summary = self.build()

        self.assertEqual(summary["status"], "built")
        self.assertEqual(summary["n_reviews"], 3)
        self.assertEqual(summary["n_chunks"], 3)
        self.assertGreaterEqual(summary["seconds"], 0.0)
        collection = self.client.get_collection(self.name)
        self.assertEqual(collection.count(), 3)
        self.assertTrue(index.is_current(collection, index.corpus_fingerprint(self.frame)))

        stored = collection.get(ids=["r1:0", "r2:0"], include=["metadatas", "documents"])
        by_id = dict(zip(stored["ids"], zip(stored["metadatas"], stored["documents"])))
        meta, document = by_id["r1:0"]
        self.assertEqual(document, "made my hair dry and brittle")  # normalized text is embedded
        self.assertEqual(meta["rating"], 1)
        self.assertEqual(meta["brand"], "Pantene")
        self.assertEqual(meta["review_date"], "2021-03-04T00:00:00+00:00")
        self.assertEqual((meta["chunk_index"], meta["n_chunks"]), (0, 1))
        self.assertEqual(by_id["r2:0"][0]["review_date"], "")  # undated rows keep a string

    def test_build_logs_count_and_timing(self):
        messages = []
        index.build_index(
            self.frame,
            client=self.client,
            embedder=FakeEmbedder(),
            tokenizer=FakeTokenizer(),
            collection_name=self.name,
            log=messages.append,
        )
        self.assertTrue(any("3 chunks for 3 reviews in" in message for message in messages))

    def test_up_to_date_index_skips_embedding(self):
        self.build()
        messages = []
        summary = index.build_index(
            self.frame,
            client=self.client,
            embedder=failing_embedder,
            tokenizer=FakeTokenizer(),
            collection_name=self.name,
            log=messages.append,
        )
        self.assertEqual(summary["status"], "skipped")
        self.assertIn("skipping embedding", messages[-1])

    def test_skip_does_not_load_the_model(self):
        self.build()
        with mock.patch.object(index, "load_model", side_effect=AssertionError("model loaded")):
            summary = index.build_index(self.frame, client=self.client, collection_name=self.name, log=silent)
        self.assertEqual(summary["status"], "skipped")

    def test_rebuild_flag_forces_embedding(self):
        self.build()
        embedder = FakeEmbedder()
        summary = self.build(embedder=embedder, rebuild=True)
        self.assertEqual(summary["status"], "built")
        self.assertTrue(embedder.calls)
        self.assertEqual(self.client.get_collection(self.name).count(), 3)

    def test_stale_index_is_rebuilt(self):
        self.build()
        changed = review_frame([("r1", "Pantene", 1, "dry"), ("r9", "Pantene", 2, "greasy roots")])
        embedder = FakeEmbedder()
        summary = self.build(frame=changed, embedder=embedder)

        self.assertEqual(summary["status"], "built")
        self.assertTrue(embedder.calls)
        collection = self.client.get_collection(self.name)
        self.assertEqual(sorted(collection.get()["ids"]), ["r1:0", "r9:0"])

    def test_interrupted_build_is_not_current(self):
        collection = self.client.create_collection(self.name, metadata={"fingerprint": "building"})
        self.assertFalse(index.is_current(collection, index.corpus_fingerprint(self.frame)))
        self.assertEqual(self.build()["status"], "built")

    def test_changed_settings_trigger_rebuild(self):
        changes = {
            "CHUNK_OVERLAP_TOKENS": config.CHUNK_OVERLAP_TOKENS + 1,
            "EMBED_MODEL": "some/other-model",
            "CHUNK_MAX_TOKENS": config.CHUNK_MAX_TOKENS + 1,
        }
        for setting, value in changes.items():
            self.build()
            with self.subTest(setting=setting), mock.patch.object(config, setting, value):
                self.assertEqual(self.build()["status"], "built")

    def test_rebuild_log_warns_old_files_stay_on_disk(self):
        self.build()
        messages = []
        index.build_index(
            self.frame, client=self.client, embedder=FakeEmbedder(), tokenizer=FakeTokenizer(),
            collection_name=self.name, rebuild=True, log=messages.append,
        )
        self.assertTrue(any("rm -rf .chroma" in message for message in messages))

    def test_failed_chunking_keeps_existing_index(self):
        self.build()

        def broken_tokenizer(*args, **kwargs):
            raise RuntimeError("tokenizer failed")

        with self.assertRaises(RuntimeError):
            index.build_index(
                self.frame, client=self.client, embedder=FakeEmbedder(), tokenizer=broken_tokenizer,
                collection_name=self.name, rebuild=True, log=silent,
            )
        self.assertEqual(self.client.get_collection(self.name).count(), 3)

    def test_special_tokens_shrink_stored_chunks(self):
        long_text = " ".join(f"word{i}" for i in range(40))
        frame = review_frame([("long", "Pantene", 2, long_text)])
        with mock.patch.object(config, "CHUNK_MAX_TOKENS", 10), mock.patch.object(config, "CHUNK_OVERLAP_TOKENS", 2):
            index.build_index(
                frame, client=self.client, embedder=FakeEmbedder(), tokenizer=FakeTokenizer(special_tokens=2),
                collection_name=self.name, log=silent,
            )
            limit = config.CHUNK_MAX_TOKENS - 2
        documents = self.client.get_collection(self.name).get()["documents"]
        self.assertGreater(len(documents), 1)
        for document in documents:
            self.assertLessEqual(len(document.split()), limit)
        self.assertTrue(any(len(document.split()) == limit for document in documents))

    def test_long_review_is_stored_as_multiple_chunks(self):
        long_text = " ".join(f"word{i}" for i in range(40))
        frame = review_frame([("long", "Pantene", 2, long_text), ("short", "Pantene", 5, "fine")])
        with mock.patch.object(config, "CHUNK_MAX_TOKENS", 16), mock.patch.object(config, "CHUNK_OVERLAP_TOKENS", 4):
            summary = self.build(frame=frame)

        self.assertEqual(summary["n_split_reviews"], 1)
        stored = self.client.get_collection(self.name).get(where={"review_id": "long"})
        self.assertGreater(len(stored["ids"]), 1)
        self.assertEqual({meta["n_chunks"] for meta in stored["metadatas"]}, {len(stored["ids"])})
        for document in stored["documents"]:
            self.assertIn(document, long_text)
            self.assertLessEqual(len(document.split()), 16)

    def test_batches_respect_client_max_batch_size(self):
        frame = review_frame([(f"r{i}", "Pantene", 5, f"review number {i}") for i in range(7)])
        embedder = FakeEmbedder()
        with mock.patch.object(self.client, "get_max_batch_size", return_value=3):
            self.build(frame=frame, embedder=embedder)
        self.assertEqual([len(batch) for batch in embedder.calls], [3, 3, 1])
        self.assertEqual(self.client.get_collection(self.name).count(), 7)

    def test_empty_frame_raises(self):
        with self.assertRaises(ValueError):
            self.build(frame=self.frame.iloc[0:0])


class PersistenceTest(unittest.TestCase):
    def test_reopened_store_skips_embedding(self):
        frame = review_frame([("r1", "Pantene", 1, "too dry"), ("r2", "Pantene", 5, "soft")])
        with tempfile.TemporaryDirectory() as tmp:
            first = index.persistent_client(Path(tmp))
            index.build_index(frame, client=first, embedder=FakeEmbedder(), tokenizer=FakeTokenizer(), log=silent)
            chromadb.api.client.SharedSystemClient.clear_system_cache()

            reopened = index.persistent_client(Path(tmp))
            summary = index.build_index(
                frame, client=reopened, embedder=failing_embedder, tokenizer=FakeTokenizer(), log=silent
            )
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.assertEqual(summary["status"], "skipped")


class MainTest(unittest.TestCase):
    def test_missing_parquet_exits_1_with_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "reviews_clean.parquet"
            with mock.patch.object(config, "CLEAN_PARQUET", missing), mock.patch("sys.stderr") as stderr:
                code = index.main([])
        self.assertEqual(code, 1)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("uv run python -m src.clean", written)


if __name__ == "__main__":
    unittest.main()
