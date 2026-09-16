"""Project-wide constants: paths, dataset source, brand set, and model settings.

This module is the only place model names, paths, and constants are defined.
Everything in src/, notebooks/, and app/ imports from here.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / ".chroma"

# --- Dataset source ----------------------------------------------------------
# Amazon Reviews 2023 (Hou et al. 2024, arXiv:2403.03952).
DATASET_HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CATEGORY = "Beauty_and_Personal_Care"
DATASET_REVIEWS_FILE = f"raw/review_categories/{DATASET_CATEGORY}.jsonl"
DATASET_META_FILE = f"raw/meta_categories/meta_{DATASET_CATEGORY}.jsonl"
DATASET_RESOLVE_BASE = f"https://huggingface.co/datasets/{DATASET_HF_REPO}/resolve/main/"
DATASET_REVIEWS_URL = DATASET_RESOLVE_BASE + DATASET_REVIEWS_FILE
DATASET_META_URL = DATASET_RESOLVE_BASE + DATASET_META_FILE
# Content-Length measured 2026-09-16; used only as a progress-bar total, so a
# drifting upstream size costs a wrong ETA, never a wrong result.
DATASET_REVIEWS_BYTES = 11_021_458_876
DATASET_META_BYTES = 2_835_194_976

# --- Pipeline artifacts ---------------------------------------------------------
# Everything under data/raw/ is git-ignored and rebuilt by data/download_data.py.
RAW_BRAND_MAP_JSON = DATA_RAW_DIR / "brand_map.json"
RAW_FILTERED_JSONL = DATA_RAW_DIR / "reviews_filtered.jsonl"
RAW_MANIFEST_JSON = DATA_RAW_DIR / "reviews_filtered.manifest.json"
CLEAN_PARQUET = DATA_PROCESSED_DIR / "reviews_clean.parquet"
# The quality report is a deliverable in its own right, so it is persisted next
# to the parquet rather than only printed; stories 3 and 6 read these numbers.
CLEAN_REPORT_JSON = DATA_PROCESSED_DIR / "quality_report.json"

# --- Cleaning rules -------------------------------------------------------------
# Source timestamps are epoch milliseconds (UTC). Anything outside these years is
# treated as unparseable: the row is kept, but lands in the undated bucket.
REVIEW_YEAR_MIN = 2000
REVIEW_YEAR_MAX = 2026
UNDATED_BUCKET = "undated"
# Normalized-text dedup applies only at or above this length, so that genuine
# short praise ("Love it!") survives in a corpus that is ~2/3 five-star.
DEDUP_MIN_TEXT_CHARS = 50
# Digest size for the derived review_id; 8 bytes -> 16 hex characters.
REVIEW_ID_DIGEST_BYTES = 8

# --- Brand set -----------------------------------------------------------------
# Canonical brand name -> regex matched against the product metadata `store`
# field or the product `title`. The inline (?i) makes each pattern
# case-insensitive without callers passing re.IGNORECASE; groups are
# non-capturing so pandas str.contains does not warn.
BRANDS = {
    "Head & Shoulders": r"(?i)head\s*(?:&|and|n'?)\s*shoulders",
    "Pantene": r"(?i)pantene",
    "Herbal Essences": r"(?i)herbal\s+essence",
}

# --- Generation (Groq) ---------------------------------------------------------
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.1
