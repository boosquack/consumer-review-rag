# Data schema and cleaning

## Source

Locked 2026-09-15: Amazon Reviews 2023, Hugging Face repo `McAuley-Lab/Amazon-Reviews-2023`, category `Beauty_and_Personal_Care` (Hou et al. 2024, arXiv:2403.03952). The `All_Beauty` subset was rejected as too thin for per-brand trends (13 / 70 / 34 matching products).

- Reviews are in `raw/review_categories/Beauty_and_Personal_Care.jsonl` (11 GB, 23,911,390 reviews); product metadata in `raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl` (2.8 GB, 1,028,914 products). Stream and filter both; never store either whole.
- Reviews join to metadata on `parent_asin`. The brand lives in the metadata `store` field.
- A product belongs to a brand when its `store` or `title` matches that brand's pattern in `src/config.py` (patterns are already case-insensitive via inline `(?i)`).
- Products matched by `title` but not `store` (40 / 15 / 71) can be third-party listings that merely name the brand. Deciding whether to keep them belongs to the cleaning story.
- Scraping is not used. It stays a last resort only if the locked source fails: small, respectful, half a day at most.

## Target schema

| Column | Type | Notes |
|---|---|---|
| `review_id` | str | Unique; generate one if the source lacks it |
| `brand` | str | Canonical brand name (e.g. "Head & Shoulders") |
| `product_name` | str | Raw product title |
| `rating` | int | 1–5 |
| `review_title` | str | May be empty |
| `review_text` | str | Body; original kept for display/citation |
| `review_date` | datetime | Parsed and coerced; unparseable values flagged |
| `verified` | bool | Only if the source has it; otherwise drop the column |

The normalized embedding text (stripped, whitespace collapsed) is stored separately from `review_text`.

## Cleaning rules

Document each step in the notebook.

- Drop rows with empty `review_text`.
- De-duplicate on `review_id` and on exact `review_text` (copy-paste spam).
- Normalize brand strings to the canonical set.
- Parse dates; keep undated reviews in an `undated` bucket instead of dropping them.
- Output: `data/processed/reviews_clean.parquet`.

## Quality bar before proceeding

- At least a few hundred reviews per brand; ideally 1,000+.
- Report total rows, rows per brand, date range, and percent of rows dropped with the reason for each drop. This report is itself a portfolio signal.
