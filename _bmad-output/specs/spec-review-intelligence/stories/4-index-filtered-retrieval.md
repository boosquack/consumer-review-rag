---
title: 'Index and filtered retrieval'
type: 'feature'
created: '2026-09-16'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '95d2b19b53b21fe6a01a2c901756ee983846abb5'
context:
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/rag-design.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CAP-3 has no implementation. There is no vector index over the cleaned corpus, so there is nothing for generation (story 5), evaluation (story 6), or the app (story 7) to retrieve from, and no evidence yet that a question plus brand and rating filters returns relevant reviews.

**Approach:** `src/index.py` embeds the normalized review text locally with `all-MiniLM-L6-v2` and writes it to a persistent ChromaDB collection under `.chroma/`, with filterable metadata. `src/retrieve.py` embeds a query and returns the top-k reviews, optionally pre-filtered by brand and rating band. Relevance is hand-checked on 8 queries and recorded.

## Boundaries & Constraints

**Always:** Take model names, paths, k, and chunk settings from `src/config.py`. Embed `review_text_normalized`; return the original `review_text` for display and citation. Persist the index, and skip re-embedding when it already matches the parquet. Disable Chroma telemetry. Report measured build time, index size, and relevance judgements, misses included. Leave changes uncommitted.

**Never:** Write generation, prompt, Groq, evaluation-metric, or app code. Add a dependency (chromadb 1.5.9, sentence-transformers 6.0.1, and torch are installed). Commit `.chroma/` or data. Download the model or open Chroma in unit tests. Hand-edit `SPEC.md`, its companions, `stories.yaml`, `.memlog.md`, or `AGENTS.md`.

**Decisions (planning, 2026-09-16):**
- **Chunking (resolves SPEC open question 2):** split a review only when its normalized text exceeds the model's 256-token window. That affects 928 reviews (1.72%); every other review is embedded whole. Split into ≤256-token windows with overlap, cut on tokenizer offsets so each chunk is an exact slice of the normalized text. Results collapse to the best-scoring chunk per review, so k counts reviews, not chunks.
- **Filters:** `brand` is one canonical name or None. `rating_band` is `low` (1–2★), `mid` (3★), `high` (4–5★), or None. Both are metadata pre-filters (Chroma `where`), not post-filters.
- **Scoring:** cosine space, normalized embeddings; each hit carries its distance. There is no relevance cutoff here; story 5 owns the empty-retrieval rule.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Build | Parquet present, no index | Collection written; count and timing printed | N/A |
| Up-to-date index | Index matches parquet fingerprint | Skips embedding, exits 0; `--rebuild` forces it | N/A |
| Stale index | Parquet changed since build | Rebuilds instead of serving stale vectors | N/A |
| Long review | Normalized text > 256 tokens | Multiple chunks; one hit per review in results | N/A |
| Filtered query | brand and/or rating_band | Every hit matches the filters | N/A |
| No match | Filters match no rows | Empty list, no crash | N/A |
| Blank query | `""` or whitespace only | No embedding call | ValueError |
| Bad filter | Unknown brand or band | ValueError naming the allowed values | Raise |
| Missing index | `.chroma/` absent | Error naming `uv run python -m src.index` | Raise with hint |

</frozen-after-approval>

## Code Map

- `src/config.py` -- Has `CHROMA_DIR`, `CLEAN_PARQUET`, `BRANDS`, `EDA_LOW_RATINGS`/`EDA_HIGH_RATINGS`. Add `EMBED_MODEL`, `EMBED_BATCH_SIZE`, `CHROMA_COLLECTION`, `CHUNK_MAX_TOKENS=256`, `CHUNK_OVERLAP_TOKENS`, `RETRIEVAL_K=6`, and `RATING_BANDS` (point the two EDA constants at it so there is one source of truth).
- `src/index.py` -- New. `main() -> int`, `sys.exit(main())`, errors to stderr, following `src/clean.py`. Pure `chunk_review()` over an injected tokenizer. `build_index()` takes an injectable embedder and client so tests and the story 7 app can call it. Batch `add` under Chroma's max batch size. Collection metadata stores a parquet fingerprint (row count plus a hash of the sorted `review_id`s). Metadata keys: `brand`, `rating` (int), `product_name`, `review_date` (ISO string, `""` if undated), `review_id`, `chunk_index`, `n_chunks`. Chroma rejects None values.
- `src/retrieve.py` -- New. `retrieve(query, k=None, brand=None, rating_band=None) -> list[dict]`. Each hit has `review_id`, `brand`, `rating`, `product_name`, `review_date`, `review_title`, `review_text` (original, joined from the parquet by `review_id`), `chunk_text`, and `distance`, best first. Over-fetch chunks (e.g. 3k), collapse by `review_id`, then cut to k. Cache the model, client, and parquet. The CLI prints hits for a query with `--brand`/`--band`/`-k`.
- `src/eda.py` -- Reads `config.EDA_LOW_RATINGS`/`EDA_HIGH_RATINGS`; keep them valid, do not edit the file.
- `eval/retrieval_checks.md` -- New. 8 queries mixing brand, band, and unfiltered questions, e.g. "what do 1-star Pantene reviewers complain about". For each: filters, the top-6 results (brand, rating, short snippet), and a relevant/partly/not verdict with a one-line reason. This is the agent's judgement, and the file says so. It feeds story 6's gold set.
- `tests/test_index.py`, `tests/test_retrieve.py` -- New. Stdlib `unittest`; a fake whitespace tokenizer with offsets, a fake deterministic embedder, and a Chroma `EphemeralClient`/temp dir. Follow `tests/test_eda.py`.
- Measured: all-MiniLM-L6-v2 on MPS loads in ~21 s the first time (download is cached after) and encodes 4,000 reviews in 4.1 s, so the full corpus should take ~1 min. Token p50/p90/p99 are 28/112/318. A 4,000-row Chroma store is 23 MB.

## Tasks & Acceptance

**Execution:**
- [x] `src/config.py` -- retrieval constants and `RATING_BANDS` -- single source of truth.
- [x] `src/index.py` -- chunking, fingerprinted build, CLI -- CAP-3 index.
- [x] `src/retrieve.py` -- filtered top-k with per-review collapse, CLI -- CAP-3 retrieval.
- [x] `tests/test_index.py`, `tests/test_retrieve.py` -- cover every matrix row -- matrix requires tests.
- [x] Run the full build, then write `eval/retrieval_checks.md` from real results -- CAP-3 success needs hand-tested evidence.

**Acceptance Criteria:**
- Given a built index, when `src.index` runs again in a fresh process, then it exits 0 without calling the embedder on documents and reports the skip.
- Given a fresh process, when `retrieve()` runs a filtered query, then it returns ≤k distinct `review_id`s whose original `review_text` is byte-identical to the parquet row.
- Given `eval/retrieval_checks.md`, when read, then it lists 8 queries with filters, results, and per-query verdicts, including any misses.

## Implementation Notes

- Measured full build after the review fixes (`rm -rf .chroma` first, MPS): 53,872 reviews -> 55,001 chunks (928 reviews split), 66.0 s (74.2 s on the first clean build before the fixes). Up-to-date rerun: skip in 1.9 s wall, model never loaded.
- Measured full index size (rebuilt after the fixes): 244 MB (`chroma.sqlite3` 144 MB + one 100 MB HNSW segment dir). Over GitHub's 100 MB file limit, as the deployment risk predicted.
- `--rebuild` / stale rebuild calls `delete_collection`, but Chroma 1.5.9 leaves the old HNSW segment dir and sqlite pages on disk (observed 345 MB after one rebuild). `rm -rf .chroma` before a rebuild reclaims it; the rebuild log line now says so. Not auto-cleaned (would mean touching Chroma internals). The model load and chunking now run before `delete_collection`, so a failure there leaves the old index intact.
- Token window: `CHUNK_MAX_TOKENS=256` includes [CLS]/[SEP], so chunks hold 254 content tokens (this is what yields exactly 928 split reviews). `CHUNK_OVERLAP_TOKENS=32`. Window starts are moved forward past glued subword/punctuation tokens so a slice re-tokenizes within the window; verified all 2,057 split chunks are <= 256 tokens with specials and are exact substrings of the normalized text.
- Fingerprint = row count + sha256 over reviews in `review_id` order of `review_id`, `review_text_normalized`, `brand`, `rating`, `product_name`, and `review_date`. This supersedes the Code Map's "row count plus a hash of the sorted `review_id`s": `clean.make_review_id` hashes user, product, timestamp, and raw text only, so brand, rating, product-name, or normalization changes would otherwise reuse stale vectors and filter metadata. Stored with `embed_model`, `chunk_max_tokens`, `chunk_overlap_tokens`; any mismatch rebuilds. Written only after the last batch, so an interrupted build is rebuilt. A change to chunking *code* alone does not change the fingerprint; use `--rebuild`.
- `retrieve()` over-fetches `k * RETRIEVAL_OVERFETCH` (3) chunks and doubles the fetch when collapsing leaves fewer than k reviews and more rows may exist. A hit whose `review_id` is not in the parquet raises RuntimeError pointing at `src.index` (stale index) rather than being dropped. On first use, the default collection is checked with `index.is_current` against the cached parquet; a half-built, other-settings, or older-parquet index raises RuntimeError naming `uv run python -m src.index`.
- For story 7: `retrieve()` caches the collection, parquet, and model with `lru_cache`. After an in-process rebuild, clear `_default_collection` and `_default_reviews` (or restart) so queries do not run against the deleted collection.
- Retrieval checks (re-run on the rebuilt index, same hits and distances): 5 relevant, 2 partly, 1 miss (query 5 re-graded to partly); 48/48 hits matched filters. Both re-wordings are recorded, and one split-review hit (3 chunks) is shown. See `eval/retrieval_checks.md`.

**Orchestrator re-verification after review pass 1 (2026-09-16):** 122 tests OK (full suite); `uv run python -m src.index` in a fresh process skips in 1.5 s against fingerprint `53872:8094d0ca…`; the Pantene low-band demo query returns 6 Pantene 1–2★ hits; `.chroma` 244 MB and git-ignored; `eval/retrieval_checks.md` summary reads 5 relevant, 2 partly, 1 not. Not committed: AGENTS.md requires the maintainer to ask.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-16). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V, `VO` = its Other findings). Every finding is logged below, grouped by shared root cause; nothing routed to intent_gap or bad_spec, so no loopback.

| ID | Finding | Verdict | Route | Evidence |
|---|---|---|---|---|
| B1+E14 | Fingerprint hashes only review_ids, so brand/rating/product/normalization changes reuse a stale index | medium | patch | Confirmed in `src/clean.py:65`: `make_review_id` covers user, product, timestamp, raw text only. The Code Map prescribed this fingerprint; routed patch rather than bad_spec because the fix is one function with no new surface, and the supersession is recorded in Implementation Notes. |
| B3+E1+E2+E15+VO1 | `retrieve()` serves a half-built, other-model, or older-parquet index without checking | medium | patch | Confirmed: `_default_collection` only checks existence; `is_current` is never called at query time. An interrupted 70 s build leaves `fingerprint: "building"`, a reachable state. |
| B5+E4 | Rebuild deletes the working index before model load and chunking | medium | patch | Confirmed ordering in `build_index`. Reordering is a direct correction; a full build-then-swap is not required. |
| B6 | Rebuild disk growth not surfaced to the user | low | patch | Measured 244 → 345 MB after one rebuild; a log-line change is a direct correction. |
| B7 | Query 5 graded relevant against the doc's own scale | medium | patch | Write-up says no hit says *what* people like; the scale requires most hits to address the question. AGENTS.md: report failures honestly. |
| B8 | Query 1 says "1-star" but the low band includes 2★ | low | patch | Hit 4 is 2★. Doc clarification is a direct correction. |
| B9 | Re-wording experiments cited without results; no reproduce commands | low | patch | Claims in queries 1 and 3 have no recorded tables. Direct documentation fix. |
| B10 | Hand checks never show a split review | low | patch | No hit in the 8 tables has `n_chunks > 1`; one added check is a direct documentation addition. |
| V1 | Special-token budget inside `build_index` untested | medium | patch | Pre-verified: replacing the budget with `CHUNK_MAX_TOKENS` keeps 40 tests green. |
| V2+B13 | `_collapse` closest-chunk rule untested; CLI success path untested | medium | patch | Pre-verified: keep-last collapse passes all tests. CLI success is covered by the V4 test. |
| V3+B2 | Model/window-only setting changes untested for rebuild; no chunking-code version marker | medium | patch | Pre-verified: trimming `index_settings` passes. The version-marker half of B2 is rejected: it adds a manually bumped constant; the Implementation Notes already say to `--rebuild` after chunking-code changes. |
| V4 | Default (no-fakes) retrieve path untested | medium | patch | Pre-verified: dropping `reviews_by_id` passes, yet it would make every real call raise. |
| B4+E3+VO2 | lru-cached collection goes stale after an in-process rebuild | low | rejected | No caller builds and retrieves in one process today; story 7 would be the first. The fix adds public cache-reset surface. Recorded in Implementation Notes for story 7. |
| B11+E5+E6 | Window can overflow on no-whitespace runs or a mid-word end | low | rejected | Measured on the full corpus: all 2,057 split chunks fit in 256 tokens. The fix adds branches for a case not observed. |
| B12 | `content_token_budget` assumes 0 special tokens without the method | low | rejected | The only real tokenizer (MiniLM fast tokenizer) has the method; no caller passes another. |
| B14+E11+E12 | CLIs catch too few error types | low | rejected | Schema KeyError and Chroma errors fail loudly with a traceback and non-zero exit; not reachable from the clean parquet. |
| E7 | Null `review_text_normalized` crashes the tokenizer | false | rejected | Parquet has 0 null normalized texts; `src/clean.py` drops empty bodies. |
| E8 | NaN `product_name` stored as "nan" | false | rejected | Parquet has 0 null and 0 empty product names. |
| E9+E10 | Duplicate review_id crashes add or returns a Series | false | rejected | Parquet has 0 duplicate review_ids; `src/clean.py` drops them. |
| E13 | Float `k` reaches Chroma | low | rejected | The CLI parses `-k` as int; no caller passes a float. The fix is a guard. |

## Design Notes

**Deployment risk (story 7, not solved here):** `.chroma/` and the parquet are both git-ignored, and a full index is expected to be well over GitHub's 100 MB file limit. Streamlit Cloud will therefore start with neither. `build_index()` is importable so story 7 can choose between building at startup and indexing a subset; record the measured full-index size in Implementation Notes to inform that choice.

## Verification

**Commands:**
- `uv run python -m unittest -v` -- expected: 74 existing plus the new tests, all OK, with no network
- `uv run python -m src.index` -- expected: full build, count ≈ 53,872 reviews plus extra chunks, time printed; a second run skips
- `uv run python -m src.retrieve "what do 1-star Pantene reviewers complain about" --brand Pantene --band low` -- expected: 6 Pantene 1–2★ hits
- `du -sh .chroma` and `git status --short` -- expected: size recorded; `.chroma/` not listed
