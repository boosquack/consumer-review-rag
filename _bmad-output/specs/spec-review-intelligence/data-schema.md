# Data schema and cleaning

## Sourcing order

1. Search Kaggle and public sources for personal-care, haircare, or Amazon review datasets (Amazon Reviews Beauty / Personal Care subsets are the most likely).
2. Filter rows where the product title or brand field matches a chosen brand, using a case-insensitive substring match on brand names and common product-line names.
3. If no single dataset covers all brands with enough volume, drop to two brands or combine two datasets that share a schema.
4. Scrape only as a last resort, small and respectful, and for half a day at most.

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
