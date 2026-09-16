"""Embed the clean review corpus into a persistent ChromaDB collection (CAP-3).

Run with: uv run python -m src.index [--rebuild]

Input is ``data/processed/reviews_clean.parquet`` from ``src/clean.py``; output is
the ``config.CHROMA_COLLECTION`` collection under ``.chroma/``. The normalized
review text is embedded locally with ``config.EMBED_MODEL``; the original
``review_text`` is never embedded or altered, and retrieval joins it back from
the parquet by ``review_id``.

A review is embedded whole unless its normalized text overflows the model's
token window; only then is it split into overlapping chunks, each an exact slice
of the normalized text. The collection metadata carries a fingerprint of the
parquet plus the model and chunk settings, so an unchanged corpus is never
re-embedded and a changed one is never served stale.

``build_index()`` is importable (the app may need to build at startup) and takes
an injectable embedder, tokenizer, and client so tests never load the model.

Exit 0 when the index is built or already current; exit 1 when the parquet is
missing or empty.
"""

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src import config

Embedder = Callable[[list[str]], Sequence[Sequence[float]]]
# A Hugging Face fast tokenizer, or anything called the same way:
# tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
Tokenizer = Callable[..., Any]

# Metadata values that mark a collection as unusable until rebuilt.
_BUILDING = "building"

_MISSING_PARQUET_HINT = (
    "{path} not found. Build it first with:\n"
    "  uv run python data/download_data.py\n"
    "  uv run python -m src.clean"
)


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    n_chunks: int


# --- Chunking ------------------------------------------------------------------------


def chunk_review(
    text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap: int,
) -> list[Chunk]:
    """Split normalized text into <= max_tokens windows, or return it whole if it fits.

    ``max_tokens`` counts content tokens only (no special tokens). Windows advance
    by ``max_tokens - overlap`` and are cut on the tokenizer's character offsets,
    so every chunk is ``text[start:end]`` for some offsets, never re-joined tokens.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"overlap must be in [0, {max_tokens}), got {overlap}")

    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)[
        "offset_mapping"
    ]
    if len(offsets) <= max_tokens:
        return [Chunk(text=text, chunk_index=0, n_chunks=1)]

    step = max_tokens - overlap
    spans = []
    start = 0
    while True:
        end = min(start + max_tokens, len(offsets))
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end == len(offsets):
            break
        # Start the next window on a word boundary: a slice that opens mid-word
        # (a "##" subword or glued punctuation) re-tokenizes differently and can
        # overflow the window by a token. Giving up some overlap is cheaper.
        start += step
        while start < end and offsets[start][0] == offsets[start - 1][1]:
            start += 1
    return [
        Chunk(text=text[lo:hi], chunk_index=i, n_chunks=len(spans))
        for i, (lo, hi) in enumerate(spans)
    ]


# --- Fingerprint ----------------------------------------------------------------------


def corpus_fingerprint(frame: pd.DataFrame) -> str:
    """Return row count plus a hash of everything the index stores per review.

    Reviews are hashed in review_id order, each as its review_id, the embedded
    ``review_text_normalized``, and the filter metadata (brand, rating,
    product_name, review_date). review_id alone is not enough: it does not cover
    brand, rating, product name, or normalization, so a change to any of those
    must still mark the index stale.
    """
    digest = hashlib.sha256()
    # argsort by column position also works on a frame indexed by review_id.
    order = frame["review_id"].to_numpy().argsort(kind="stable")
    for row in frame.iloc[order].itertuples(index=False):
        fields = (
            str(row.review_id),
            str(row.review_text_normalized),
            str(row.brand),
            str(int(row.rating)),
            _text_or_empty(row.product_name),
            iso_date(row.review_date),
        )
        digest.update("\x1f".join(fields).encode("utf-8"))
        digest.update(b"\x1e")
    return f"{len(frame)}:{digest.hexdigest()[:32]}"


def iso_date(value: Any) -> str:
    """ISO string for a review date, or "" when undated (Chroma rejects None)."""
    return "" if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()


def _text_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def index_settings(fingerprint: str) -> dict[str, Any]:
    """Return the collection metadata that must match for an index to be reused."""
    return {
        "fingerprint": fingerprint,
        "embed_model": config.EMBED_MODEL,
        "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
    }


def is_current(collection: Any, fingerprint: str) -> bool:
    """True when the collection was fully built from this corpus with these settings."""
    metadata = collection.metadata or {}
    expected = index_settings(fingerprint)
    return all(metadata.get(key) == value for key, value in expected.items())


# --- Clients and model ------------------------------------------------------------------


def chroma_settings() -> Any:
    from chromadb.config import Settings

    return Settings(anonymized_telemetry=False)


def persistent_client(path: Path | None = None) -> Any:
    """Open the on-disk Chroma store with telemetry disabled."""
    import chromadb

    return chromadb.PersistentClient(
        path=str(path or config.CHROMA_DIR), settings=chroma_settings()
    )


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load and cache the sentence-transformers model (shared with src.retrieve)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def model_embedder(model: Any) -> Embedder:
    """Wrap a SentenceTransformer as a batch embedder of L2-normalized vectors."""

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    return embed


def content_token_budget(tokenizer: Any) -> int:
    """Tokens per chunk left after the model's special tokens ([CLS], [SEP])."""
    special = tokenizer.num_special_tokens_to_add() if hasattr(tokenizer, "num_special_tokens_to_add") else 0
    return config.CHUNK_MAX_TOKENS - special


# --- Build ------------------------------------------------------------------------------


def load_reviews(path: Path | None = None) -> pd.DataFrame:
    """Read the clean parquet, raising with the pipeline commands when it is missing."""
    path = Path(path or config.CLEAN_PARQUET)
    if not path.exists():
        raise FileNotFoundError(_MISSING_PARQUET_HINT.format(path=path))
    return pd.read_parquet(path)


def chunk_records(
    frame: pd.DataFrame, tokenizer: Tokenizer, max_tokens: int, overlap: int
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return (ids, documents, metadatas) for every chunk of every review."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        review_date = iso_date(row.review_date)
        for chunk in chunk_review(row.review_text_normalized, tokenizer, max_tokens, overlap):
            ids.append(f"{row.review_id}:{chunk.chunk_index}")
            documents.append(chunk.text)
            # Chroma rejects None metadata values, so every field has a concrete type.
            metadatas.append(
                {
                    "brand": str(row.brand),
                    "rating": int(row.rating),
                    "product_name": _text_or_empty(row.product_name),
                    "review_date": review_date,
                    "review_id": str(row.review_id),
                    "chunk_index": chunk.chunk_index,
                    "n_chunks": chunk.n_chunks,
                }
            )
    return ids, documents, metadatas


def build_index(
    frame: pd.DataFrame | None = None,
    *,
    client: Any = None,
    embedder: Embedder | None = None,
    tokenizer: Tokenizer | None = None,
    collection_name: str | None = None,
    rebuild: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build the collection unless it already matches the corpus; return a summary.

    The model is loaded only if embedding is actually needed, so an up-to-date
    index is confirmed without touching it. The fingerprint is written last, so an
    interrupted build is never mistaken for a complete one.
    """
    frame = load_reviews() if frame is None else frame
    if frame.empty:
        raise ValueError("The clean corpus has no rows; nothing to index.")
    client = persistent_client() if client is None else client
    name = collection_name or config.CHROMA_COLLECTION
    fingerprint = corpus_fingerprint(frame)

    existing = _get_collection(client, name)
    if existing is not None and not rebuild and is_current(existing, fingerprint):
        count = existing.count()
        log(f"Index '{name}' is up to date ({count:,} chunks, fingerprint {fingerprint}); skipping embedding.")
        return {"status": "skipped", "n_reviews": len(frame), "n_chunks": count, "seconds": 0.0}

    started = time.perf_counter()
    # Load the model and chunk before deleting anything, so a failure here leaves
    # a working index in place.
    if embedder is None or tokenizer is None:
        model = load_model()
        embedder = embedder or model_embedder(model)
        tokenizer = tokenizer or model.tokenizer
    max_tokens = content_token_budget(tokenizer)
    ids, documents, metadatas = chunk_records(frame, tokenizer, max_tokens, config.CHUNK_OVERLAP_TOKENS)
    n_split = sum(1 for meta in metadatas if meta["chunk_index"] == 0 and meta["n_chunks"] > 1)

    if existing is not None:
        reason = "--rebuild requested" if rebuild else "corpus or settings changed"
        log(
            f"Rebuilding index '{name}' ({reason}). Chroma leaves the old collection's "
            "files on disk; for a clean rebuild, run: rm -rf .chroma"
        )
        client.delete_collection(name)

    log(f"Embedding {len(frame):,} reviews as {len(ids):,} chunks ({n_split:,} reviews split).")

    collection = client.create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"fingerprint": _BUILDING},
        embedding_function=None,
    )
    batch = max(1, min(client.get_max_batch_size(), 4096))
    for lo in range(0, len(ids), batch):
        hi = lo + batch
        collection.add(
            ids=ids[lo:hi],
            embeddings=[list(vector) for vector in embedder(documents[lo:hi])],
            documents=documents[lo:hi],
            metadatas=metadatas[lo:hi],
        )
        log(f"  added {min(hi, len(ids)):,}/{len(ids):,}")
    collection.modify(metadata=index_settings(fingerprint))

    seconds = time.perf_counter() - started
    log(f"Built index '{name}': {collection.count():,} chunks for {len(frame):,} reviews in {seconds:.1f} s.")
    return {
        "status": "built",
        "n_reviews": len(frame),
        "n_chunks": len(ids),
        "n_split_reviews": n_split,
        "seconds": seconds,
    }


def _get_collection(client: Any, name: str) -> Any:
    from chromadb.errors import NotFoundError

    try:
        return client.get_collection(name, embedding_function=None)
    except NotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="re-embed even if the index is current")
    args = parser.parse_args(argv)

    try:
        frame = load_reviews()
        summary = build_index(frame, rebuild=args.rebuild)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    if summary["status"] == "built":
        print(f"wrote: {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
