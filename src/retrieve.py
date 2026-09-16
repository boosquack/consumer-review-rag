"""Semantic top-k review retrieval with brand and rating-band pre-filters (CAP-3).

Run with: uv run python -m src.retrieve "question" [--brand NAME] [--band low|mid|high] [-k N]

The query is embedded with the same model as the index; filters are applied
inside Chroma (``where``) before ranking, so every hit satisfies them. Chunk hits
collapse to the best-scoring chunk per review, so ``k`` counts reviews. Each hit
carries the original ``review_text`` joined from the parquet by ``review_id``,
byte-identical to the clean corpus, plus the matching ``chunk_text`` and its
cosine distance. There is no relevance cutoff here; generation owns that rule.

The app and generation import ``retrieve()``; neither reimplements it.
"""

import argparse
import sys
from functools import lru_cache
from typing import Any, Sequence

import pandas as pd

from src import config, index

_MISSING_INDEX_HINT = (
    "No review index at {path}. Build it first with:\n"
    "  uv run python -m src.index"
)


# --- Validation and filters --------------------------------------------------------------


def build_where(brand: str | None = None, rating_band: str | None = None) -> dict[str, Any] | None:
    """Return the Chroma ``where`` pre-filter, raising ValueError on unknown values."""
    clauses = []
    if brand is not None:
        if brand not in config.BRANDS:
            allowed = ", ".join(repr(name) for name in config.BRANDS)
            raise ValueError(f"Unknown brand {brand!r}. Allowed: {allowed}, or None.")
        clauses.append({"brand": brand})
    if rating_band is not None:
        if rating_band not in config.RATING_BANDS:
            allowed = ", ".join(repr(name) for name in config.RATING_BANDS)
            raise ValueError(f"Unknown rating_band {rating_band!r}. Allowed: {allowed}, or None.")
        clauses.append({"rating": {"$in": list(config.RATING_BANDS[rating_band])}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# --- Cached resources ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _default_collection() -> Any:
    from chromadb.errors import NotFoundError

    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(_MISSING_INDEX_HINT.format(path=config.CHROMA_DIR))
    client = index.persistent_client()
    try:
        collection = client.get_collection(config.CHROMA_COLLECTION, embedding_function=None)
    except NotFoundError as error:
        raise FileNotFoundError(_MISSING_INDEX_HINT.format(path=config.CHROMA_DIR)) from error
    # Refuse a half-built index, one built with other model or chunk settings, or
    # one built from a different parquet, rather than serve it silently.
    if not index.is_current(collection, index.corpus_fingerprint(_default_reviews())):
        raise RuntimeError(
            f"The review index at {config.CHROMA_DIR} is incomplete or out of date "
            "for the current parquet and settings. Rebuild it with:\n"
            "  uv run python -m src.index"
        )
    return collection


@lru_cache(maxsize=1)
def _default_embedder() -> index.Embedder:
    return index.model_embedder(index.load_model())


@lru_cache(maxsize=1)
def _default_reviews() -> pd.DataFrame:
    return reviews_by_id(index.load_reviews())


def reviews_by_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Index the clean frame by review_id for the display-text join."""
    return frame.set_index("review_id", drop=False)


# --- Retrieval ---------------------------------------------------------------------------


def retrieve(
    query: str,
    k: int | None = None,
    brand: str | None = None,
    rating_band: str | None = None,
    *,
    collection: Any = None,
    embedder: index.Embedder | None = None,
    reviews: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return up to k distinct reviews for the query, best first.

    ``collection``, ``embedder``, and ``reviews`` (a frame indexed by review_id)
    default to the cached on-disk index, model, and parquet; tests inject fakes.
    Raises ValueError for a blank query or unknown filter before any embedding,
    and FileNotFoundError naming the index command when no index exists.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-blank string.")
    k = config.RETRIEVAL_K if k is None else k
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    where = build_where(brand, rating_band)

    collection = _default_collection() if collection is None else collection
    reviews = _default_reviews() if reviews is None else reviews
    embedder = _default_embedder() if embedder is None else embedder

    vector = list(embedder([query])[0])
    total = collection.count()
    n_results = min(k * config.RETRIEVAL_OVERFETCH, total)
    best: dict[str, dict[str, Any]] = {}
    while n_results > 0:
        result = collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = result["ids"][0]
        best = _collapse(result["metadatas"][0], result["documents"][0], result["distances"][0])
        # Stop once k reviews are found, or when Chroma returned fewer chunks than
        # asked (the filter is exhausted) or the whole collection was scanned.
        if len(best) >= k or len(ids) < n_results or n_results >= total:
            break
        n_results = min(n_results * 2, total)

    hits = sorted(best.values(), key=lambda hit: hit["distance"])[:k]
    return [_with_review(hit, reviews) for hit in hits]


def _collapse(
    metadatas: list[dict[str, Any]], documents: list[str], distances: list[float]
) -> dict[str, dict[str, Any]]:
    """Keep the closest chunk per review_id."""
    best: dict[str, dict[str, Any]] = {}
    for meta, document, distance in zip(metadatas, documents, distances):
        review_id = meta["review_id"]
        if review_id not in best or distance < best[review_id]["distance"]:
            best[review_id] = {
                "review_id": review_id,
                "brand": meta["brand"],
                "rating": int(meta["rating"]),
                "product_name": meta["product_name"],
                "review_date": meta["review_date"],
                "chunk_text": document,
                "distance": float(distance),
            }
    return best


def _with_review(hit: dict[str, Any], reviews: pd.DataFrame) -> dict[str, Any]:
    review_id = hit["review_id"]
    if review_id not in reviews.index:
        raise RuntimeError(
            f"Index hit {review_id} is not in the clean parquet; the index is stale. "
            "Rebuild it with: uv run python -m src.index"
        )
    row = reviews.loc[review_id]
    return {
        "review_id": review_id,
        "brand": hit["brand"],
        "rating": hit["rating"],
        "product_name": hit["product_name"],
        "review_date": hit["review_date"],
        "review_title": row["review_title"],
        "review_text": row["review_text"],
        "chunk_text": hit["chunk_text"],
        "distance": hit["distance"],
    }


# --- CLI ---------------------------------------------------------------------------------


def format_hit(rank: int, hit: dict[str, Any], width: int = 240) -> str:
    text = hit["review_text"].replace("\n", " ")
    snippet = text if len(text) <= width else text[: width - 3] + "..."
    date = hit["review_date"][:10] or "undated"
    return (
        f"{rank}. [{hit['brand']} | {hit['rating']}* | {date} | d={hit['distance']:.3f}] "
        f"{hit['review_id']}\n   {hit['product_name'][:80]}\n   {snippet}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--brand", choices=list(config.BRANDS))
    parser.add_argument("--band", choices=list(config.RATING_BANDS))
    parser.add_argument("-k", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        hits = retrieve(args.query, k=args.k, brand=args.brand, rating_band=args.band)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    if not hits:
        print("No reviews match these filters.")
    for rank, hit in enumerate(hits, start=1):
        print(format_hit(rank, hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
