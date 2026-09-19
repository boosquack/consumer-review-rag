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

# --- Rating bands ------------------------------------------------------------------
# Single source of truth for star-rating bands, shared by EDA and retrieval filters.
RATING_BANDS = {
    "low": (1, 2),
    "mid": (3,),
    "high": (4, 5),
}

# --- EDA -------------------------------------------------------------------------
NOTEBOOK_EDA = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"
# Brand-months below this count are drawn de-emphasized and kept out of the
# average-rating trend line, so a 3-review month cannot swing the trend.
EDA_MIN_MONTH_REVIEWS = 30
# The corpus snapshot ends 2023-09-01, so this year is labelled partial.
EDA_PARTIAL_YEAR = 2023
# Standard VADER compound cutoffs (Hutto & Gilbert 2014): >= POS is positive,
# <= NEG is negative, anything between (including 0.0) is neutral.
VADER_POS = 0.05
VADER_NEG = -0.05
# Rating bands for top terms; 3 stars sits in neither band. Both point at
# RATING_BANDS so EDA and retrieval filters can never disagree.
EDA_LOW_RATINGS = RATING_BANDS["low"]
EDA_HIGH_RATINGS = RATING_BANDS["high"]
EDA_TOP_TERMS = 15
# Added to the English stopword list so top terms show themes rather than the
# brand or product category every review shares.
EDA_EXTRA_STOPWORDS = (
    # Brand-name tokens.
    "head", "shoulders", "pantene", "pro", "herbal", "essence", "essences",
    # Generic product tokens every haircare review shares.
    "shampoo", "shampoos", "conditioner", "conditioners", "product", "products",
    "hair", "use", "used", "using",
    # Filler and generic praise that tops every cell and hides the themes
    # (measured: without these, "like", "just", "good" led all six cells).
    "like", "just", "really", "did", "does", "good", "great", "love", "time",
    "buy", "bought", "make", "makes", "feel", "get", "got", "try", "tried",
    "year", "years", "im", "dont",
    # Contraction fragments left by word tokenization, and <br> markup in the source.
    "don", "didn", "doesn", "isn", "wasn", "won", "ve", "ll", "br",
)
# Fixed categorical order (validated default palette: blue, orange, aqua), so a
# brand keeps its colour in every chart and in the app.
EDA_BRAND_COLORS = {
    "Head & Shoulders": "#2a78d6",
    "Pantene": "#eb6834",
    "Herbal Essences": "#1baf7a",
}
# Sentiment bands: diverging poles with a neutral gray midpoint.
EDA_SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#b5b4ae", "positive": "#2a78d6"}

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

# --- Indexing and retrieval ---------------------------------------------------------
# Local embeddings, no API key. Vectors are L2-normalized and the collection uses
# cosine space, so a hit's distance is 1 - cosine similarity.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 128
CHROMA_COLLECTION = "reviews"
# The model's window, special tokens included ([CLS] and [SEP] take 2). A review
# is split only when its normalized text does not fit (928 reviews, 1.72%).
CHUNK_MAX_TOKENS = 256
# Tokens shared by consecutive chunks, so a sentence cut at a boundary is still
# seen whole by one of them.
CHUNK_OVERLAP_TOKENS = 32
# Reviews returned per query (rag-design.md: start at 6, tune within 5-8).
RETRIEVAL_K = 6
# Chunks fetched per requested review before collapsing to one hit per review.
RETRIEVAL_OVERFETCH = 3

# --- Generation (Groq) ---------------------------------------------------------
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.1
# gpt-oss-120b is a reasoning model: its reasoning tokens count against the
# free-tier 8K tokens/min cap, so effort stays low and completions are capped.
GROQ_REASONING_EFFORT = "low"
GROQ_MAX_COMPLETION_TOKENS = 1024
# Retries the SDK makes itself (exponential backoff, honours retry-after) before
# generation reports status "rate_limited" or "error".
GROQ_MAX_RETRIES = 3
# Seconds per request attempt, so retries cannot block an app user for minutes.
GROQ_TIMEOUT_S = 30
# The fixed sentence the model must reply with when the reviews do not cover the
# question; generation detects refusals by this sentence.
REFUSAL_TEXT = "The reviews do not cover this."

# --- Evaluation (story 6) ---------------------------------------------------------
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_GOLD_JSON = EVAL_DIR / "gold_questions.json"
EVAL_ANSWERS_JSON = EVAL_DIR / "answers.json"
# A gold pattern must match at least this many reviews within the question's
# filters, so a hit is not luck against a handful of matching reviews.
EVAL_MIN_PATTERN_MATCHES = 10
EVAL_QUESTION_TYPES = ("factual", "comparative", "filtered", "broad", "unanswerable")
# Characters of review_text saved per source in answers.json.
EVAL_SNIPPET_CHARS = 300
# Groq free tier caps gpt-oss-120b at 8K tokens/min (reasoning tokens included).
# The runner budgets below that over a sliding 60 s window, using measured usage.
EVAL_TOKENS_PER_MIN = 7000
EVAL_WINDOW_S = 60.0
# Token cost assumed for a call before any usage has been measured.
EVAL_TOKEN_ESTIMATE = 3000
# Wait before the single retry of a question that came back rate_limited.
EVAL_RATE_LIMIT_WAIT_S = 60.0
# Agent-graded faithfulness of the saved answers (pass/fail with a note per id),
# tied to the answers.json run it grades.
EVAL_FAITHFULNESS_JSON = EVAL_DIR / "faithfulness.json"
# Human faithfulness grades in the same schema as EVAL_FAITHFULNESS_JSON, written by
# the maintainer and kept beside the agent's grades for comparison.
EVAL_FAITHFULNESS_HUMAN_JSON = EVAL_DIR / "faithfulness_human.json"
# Blind grading sheet (no agent grades shown); local working file, git-ignored.
EVAL_GRADING_SHEET = EVAL_DIR / "grading" / "grading_sheet.md"
