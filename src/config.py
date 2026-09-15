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
