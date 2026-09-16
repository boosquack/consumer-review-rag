"""I/O matrix tests for src.retrieve (story 4).

Built on the same fakes as tests/test_index.py: no model, no network, and an
in-memory Chroma collection rather than the real .chroma/ store.

Run with: uv run python -m unittest -v
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chromadb

from src import config, index, retrieve
from tests.test_index import (
    FakeEmbedder,
    FakeTokenizer,
    ephemeral_client,
    review_frame,
    silent,
    unique_name,
)


class BuildWhereTest(unittest.TestCase):
    def test_no_filters(self):
        self.assertIsNone(retrieve.build_where())

    def test_brand_only(self):
        self.assertEqual(retrieve.build_where(brand="Pantene"), {"brand": "Pantene"})

    def test_band_only(self):
        self.assertEqual(retrieve.build_where(rating_band="low"), {"rating": {"$in": [1, 2]}})

    def test_both(self):
        self.assertEqual(
            retrieve.build_where("Pantene", "mid"),
            {"$and": [{"brand": "Pantene"}, {"rating": {"$in": [3]}}]},
        )

    def test_unknown_brand_names_allowed_values(self):
        with self.assertRaises(ValueError) as caught:
            retrieve.build_where(brand="Dove")
        for name in config.BRANDS:
            self.assertIn(name, str(caught.exception))

    def test_unknown_band_names_allowed_values(self):
        with self.assertRaises(ValueError) as caught:
            retrieve.build_where(rating_band="1-star")
        for name in config.RATING_BANDS:
            self.assertIn(name, str(caught.exception))


class RetrieveTest(unittest.TestCase):
    def setUp(self):
        self.client = ephemeral_client()
        self.name = unique_name()
        long_text = "dandruff " + " ".join(f"filler{i}" for i in range(30)) + " dandruff flakes itchy"
        self.frame = review_frame(
            [
                ("p1", "Pantene", 1, "Left my hair\n\n dry and  brittle, dandruff too"),
                ("p2", "Pantene", 2, "greasy roots and dandruff flakes"),
                ("p3", "Pantene", 5, "dandruff gone, soft hair"),
                ("p4", "Pantene", 3, "okay smell, dandruff same"),
                ("h1", "Head & Shoulders", 1, "itchy scalp dandruff flakes"),
                ("h2", "Head & Shoulders", 5, "cleared dandruff flakes itchy scalp", None),
                ("h3", "Head & Shoulders", 2, long_text),
                ("e1", "Herbal Essences", 5, "smells like flowers"),
            ]
        )
        with mock.patch.object(config, "CHUNK_MAX_TOKENS", 12), mock.patch.object(config, "CHUNK_OVERLAP_TOKENS", 2):
            index.build_index(
                self.frame,
                client=self.client,
                embedder=FakeEmbedder(),
                tokenizer=FakeTokenizer(),
                collection_name=self.name,
                log=silent,
            )
        self.collection = self.client.get_collection(self.name)
        self.reviews = retrieve.reviews_by_id(self.frame)
        self.embedder = FakeEmbedder()

    def tearDown(self):
        self.client.delete_collection(self.name)

    def run_query(self, query, **kwargs):
        return retrieve.retrieve(
            query, collection=self.collection, embedder=self.embedder, reviews=self.reviews, **kwargs
        )

    def test_filtered_query_hits_all_match_filters(self):
        hits = self.run_query("dandruff flakes", brand="Pantene", rating_band="low")
        self.assertEqual({hit["review_id"] for hit in hits}, {"p1", "p2"})
        for hit in hits:
            self.assertEqual(hit["brand"], "Pantene")
            self.assertIn(hit["rating"], (1, 2))

    def test_band_filter_alone(self):
        hits = self.run_query("dandruff", rating_band="high", k=8)
        self.assertEqual({hit["review_id"] for hit in hits}, {"p3", "h2", "e1"})

    def test_hits_are_distinct_best_first_and_at_most_k(self):
        hits = self.run_query("dandruff flakes itchy", k=3)
        ids = [hit["review_id"] for hit in hits]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)
        distances = [hit["distance"] for hit in hits]
        self.assertEqual(distances, sorted(distances))

    def test_hit_carries_original_text_and_all_fields(self):
        hits = self.run_query("dry brittle", brand="Pantene", rating_band="low", k=1)
        hit = hits[0]
        self.assertEqual(hit["review_id"], "p1")
        original = self.frame.set_index("review_id").loc["p1", "review_text"]
        self.assertEqual(hit["review_text"].encode("utf-8"), original.encode("utf-8"))
        self.assertEqual(hit["chunk_text"], "Left my hair dry and brittle, dandruff too")
        self.assertEqual(hit["review_title"], "title p1")
        self.assertEqual(hit["product_name"], "Pantene Shampoo")
        self.assertEqual(hit["review_date"], "2021-03-04T00:00:00+00:00")
        self.assertIsInstance(hit["distance"], float)
        self.assertEqual(
            set(hit),
            {"review_id", "brand", "rating", "product_name", "review_date", "review_title",
             "review_text", "chunk_text", "distance"},
        )

    def test_long_review_returns_one_hit(self):
        stored = self.collection.get(where={"review_id": "h3"})
        self.assertGreater(len(stored["ids"]), 1)
        hits = self.run_query("dandruff flakes itchy", brand="Head & Shoulders", k=6)
        ids = [hit["review_id"] for hit in hits]
        self.assertEqual(ids.count("h3"), 1)
        self.assertEqual(sorted(ids), ["h1", "h2", "h3"])

    def test_overfetch_widens_when_chunks_crowd_out_reviews(self):
        # Every chunk of the long review h3 outranks h1 and h2 for these words, so
        # the first 2-chunk fetch holds one review and the fetch must widen.
        query = "filler9 filler10 filler19 filler20 filler29 filler5 filler15 filler25"
        with mock.patch.object(config, "RETRIEVAL_OVERFETCH", 1), \
                mock.patch.object(self.collection, "query", wraps=self.collection.query) as spy:
            hits = self.run_query(query, brand="Head & Shoulders", k=2)
        self.assertEqual(len({hit["review_id"] for hit in hits}), 2)
        self.assertEqual(hits[0]["review_id"], "h3")
        self.assertGreater(spy.call_count, 1)

    def test_collapse_keeps_the_closest_chunk(self):
        query = "filler12 filler13 filler14 filler15"
        hits = self.run_query(query, brand="Head & Shoulders", k=1)
        self.assertEqual(hits[0]["review_id"], "h3")
        for word in query.split():
            self.assertIn(word, hits[0]["chunk_text"])
        chunks = self.collection.query(
            query_embeddings=[FakeEmbedder()([query])[0]],
            n_results=10,
            where={"review_id": "h3"},
            include=["distances"],
        )
        self.assertGreater(len(chunks["distances"][0]), 1)
        self.assertAlmostEqual(hits[0]["distance"], min(chunks["distances"][0]), places=6)

    def test_default_k_from_config(self):
        hits = self.run_query("dandruff")
        self.assertEqual(len(hits), config.RETRIEVAL_K)

    def test_no_matching_rows_returns_empty_list(self):
        self.assertEqual(self.run_query("dandruff", brand="Herbal Essences", rating_band="low"), [])

    def test_blank_query_raises_without_embedding(self):
        for query in ("", "   \n\t"):
            with self.assertRaises(ValueError):
                self.run_query(query)
        self.assertEqual(self.embedder.calls, [])

    def test_bad_filters_raise_without_embedding(self):
        with self.assertRaises(ValueError) as caught:
            self.run_query("dandruff", brand="pantene")
        self.assertIn("Head & Shoulders", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.run_query("dandruff", rating_band="bad")
        self.assertIn("low", str(caught.exception))
        self.assertEqual(self.embedder.calls, [])

    def test_hit_missing_from_parquet_raises_stale_hint(self):
        reviews = retrieve.reviews_by_id(self.frame[self.frame["review_id"] != "p1"])
        with self.assertRaises(RuntimeError) as caught:
            retrieve.retrieve(
                "dry brittle", k=1, brand="Pantene", rating_band="low",
                collection=self.collection, embedder=self.embedder, reviews=reviews,
            )
        self.assertIn("uv run python -m src.index", str(caught.exception))


class DefaultResourcesTest(unittest.TestCase):
    """The un-injected path: on-disk store, parquet, and cached model loaders."""

    def setUp(self):
        self.clear_caches()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.chroma_dir = root / ".chroma"
        self.parquet = root / "reviews_clean.parquet"
        self.frame = review_frame(
            [
                ("p1", "Pantene", 1, "Left my hair\n dry and brittle"),
                ("p2", "Pantene", 5, "soft and shiny"),
            ]
        )
        self.frame.to_parquet(self.parquet)
        self.patches = contextlib.ExitStack()
        self.patches.enter_context(mock.patch.object(config, "CHROMA_DIR", self.chroma_dir))
        self.patches.enter_context(mock.patch.object(config, "CLEAN_PARQUET", self.parquet))
        self.patches.enter_context(mock.patch.object(index, "load_model", return_value=object()))
        self.patches.enter_context(mock.patch.object(index, "model_embedder", return_value=FakeEmbedder()))

    def tearDown(self):
        self.patches.close()
        self.clear_caches()
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.tmp.cleanup()

    @staticmethod
    def clear_caches():
        retrieve._default_collection.cache_clear()
        retrieve._default_embedder.cache_clear()
        retrieve._default_reviews.cache_clear()

    def test_cli_prints_hit_from_default_store_and_parquet(self):
        index.build_index(index.load_reviews(), embedder=FakeEmbedder(), tokenizer=FakeTokenizer(), log=silent)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = retrieve.main(["dry brittle", "--brand", "Pantene", "--band", "low", "-k", "1"])
        self.assertEqual(code, 0)
        # The original review_text (newline flattened for display), not the normalized copy.
        self.assertIn("Left my hair  dry and brittle", stdout.getvalue())
        self.assertIn("p1", stdout.getvalue())

    def test_index_that_is_not_current_raises_rebuild_hint(self):
        client = index.persistent_client()
        client.create_collection(config.CHROMA_COLLECTION, metadata={"fingerprint": "building"})
        embedder = FakeEmbedder()
        with self.assertRaises(RuntimeError) as caught:
            retrieve.retrieve("dry", embedder=embedder)
        self.assertIn("uv run python -m src.index", str(caught.exception))
        self.assertEqual(embedder.calls, [])

    def test_index_from_older_parquet_raises_rebuild_hint(self):
        index.build_index(index.load_reviews(), embedder=FakeEmbedder(), tokenizer=FakeTokenizer(), log=silent)
        self.frame.assign(rating=[2, 5]).to_parquet(self.parquet)
        with self.assertRaises(RuntimeError) as caught:
            retrieve.retrieve("dry", embedder=FakeEmbedder())
        self.assertIn("uv run python -m src.index", str(caught.exception))


class MissingIndexTest(unittest.TestCase):
    def setUp(self):
        retrieve._default_collection.cache_clear()

    def tearDown(self):
        retrieve._default_collection.cache_clear()

    def test_absent_chroma_dir_raises_with_index_command(self):
        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "CHROMA_DIR", Path(tmp) / ".chroma"):
                with self.assertRaises(FileNotFoundError) as caught:
                    retrieve.retrieve("dandruff", embedder=embedder, reviews=review_frame([]))
        self.assertIn("uv run python -m src.index", str(caught.exception))
        self.assertEqual(embedder.calls, [])

    def test_store_without_collection_raises_with_index_command(self):
        import chromadb

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "CHROMA_DIR", Path(tmp)):
                with self.assertRaises(FileNotFoundError) as caught:
                    retrieve.retrieve("dandruff", embedder=FakeEmbedder(), reviews=review_frame([]))
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.assertIn("uv run python -m src.index", str(caught.exception))

    def test_cli_reports_missing_index_and_exits_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "CHROMA_DIR", Path(tmp) / ".chroma"), mock.patch("sys.stderr") as stderr:
                code = retrieve.main(["dandruff"])
        self.assertEqual(code, 1)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("uv run python -m src.index", written)


if __name__ == "__main__":
    unittest.main()
