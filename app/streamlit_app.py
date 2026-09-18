"""Streamlit demo: cited RAG answers plus EDA highlights over P&G haircare reviews (CAP-6).

Run with: uv run streamlit run app/streamlit_app.py

This module is UI only. Retrieval, generation, and chart logic all live in
``src/`` and are imported unchanged: ``src.retrieve.retrieve`` (via
``src.generate.answer``), ``src.generate.answer``, and ``src.eda``'s loaders
and plot helpers. No client or retriever is injected into ``answer()``, so
``src/retrieve.py``'s module-level ``lru_cache(maxsize=1)`` loaders build the
embedding model, the Chroma collection, and the reviews frame once per
container and reuse them across reruns.

``GROQ_API_KEY`` is read the same way ``src.generate.answer`` already reads it
locally (a git-ignored ``.env``). When deployed, Streamlit Secrets holds the
key instead of a ``.env`` file, so this module copies
``st.secrets["GROQ_API_KEY"]`` into ``os.environ`` once, before the first
``answer()`` call, and only when the environment does not already have it. The
key is never displayed or logged.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Streamlit puts this script's own directory (app/) on sys.path, not the repo
# root, so "src" is not importable without this (same fix notebooks/01_eda.ipynb
# uses for the same reason).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config, eda, index
from src.generate import (
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_NO_REVIEWS,
    STATUS_RATE_LIMITED,
    STATUS_REFUSED,
    answer,
)

ALL_BRANDS = "All brands"
ALL_BANDS = "All ratings"
BAND_LABELS = {"low": "low (1-2★)", "mid": "mid (3★)", "high": "high (4-5★)"}


# --- Secrets and index setup -------------------------------------------------------------


def _load_groq_secret() -> None:
    """Copy GROQ_API_KEY from Streamlit Secrets into the environment, once.

    Only runs when the environment does not already have the key (local .env
    already sets it via python-dotenv inside answer()). Streamlit Secrets
    raises when no secrets.toml/section exists at all (the local dev case, and
    a deployed app that has not set the secret yet); that is not an error
    here, just "no secret available".
    """
    if os.environ.get(config.GROQ_API_KEY_ENV, "").strip():
        return
    try:
        secret_value = st.secrets.get(config.GROQ_API_KEY_ENV)
    except Exception:
        return
    if secret_value:
        os.environ[config.GROQ_API_KEY_ENV] = str(secret_value)


@st.cache_resource(show_spinner="Preparing the review index (first load only; can take several minutes)...")
def _ensure_index() -> dict:
    """Build the Chroma index once per running app if it is missing or stale.

    ``build_index`` checks the corpus fingerprint itself and skips embedding
    when the on-disk index already matches, so calling it here is cheap after
    the first successful build.
    """
    return index.build_index(rebuild=False, log=lambda _message: None)


# --- Rendering ---------------------------------------------------------------------------


def _render_source(number: int, hit: dict, cited_numbers: set[int]) -> None:
    date = (hit["review_date"] or "")[:10] or "undated"
    badge = " ✓ cited" if number in cited_numbers else ""
    st.markdown(f"**[{number}] {hit['brand']} — {hit['rating']}★ — {date}{badge}**")
    st.caption(hit["product_name"] or "(no product name)")
    st.write(hit["review_text"])
    st.divider()


def _render_answer(result) -> None:
    if result.status == STATUS_ANSWERED:
        st.markdown(result.text)
        if result.uncited:
            st.caption("Note: this answer cites no review.")
    elif result.status == STATUS_REFUSED:
        st.info(result.text)
    elif result.status == STATUS_NO_REVIEWS:
        st.warning(result.text)
    elif result.status == STATUS_RATE_LIMITED:
        st.error(result.text)
    elif result.status == STATUS_ERROR:
        st.error(result.text)

    if result.sources:
        cited_ids = {hit["review_id"] for hit in result.citations}
        cited_numbers = {
            number for number, hit in enumerate(result.sources, start=1) if hit["review_id"] in cited_ids
        }
        with st.expander(f"Sources ({len(result.sources)} reviews)"):
            for number, hit in enumerate(result.sources, start=1):
                _render_source(number, hit, cited_numbers)


def _render_eda() -> None:
    st.header("EDA highlights")
    try:
        clean = eda.load_clean()
        report = eda.load_report()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        eda.plot_rating_distribution(ax, eda.rating_distribution(clean))
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "All three brands skew heavily five-star (about two-thirds of reviews) with a "
            "roughly 8-9% one-star tail, so star mix alone barely separates the brands."
        )
    with col2:
        fig2, ax2 = plt.subplots()
        eda.plot_monthly_volume(ax2, eda.monthly_stats(clean))
        st.pyplot(fig2)
        plt.close(fig2)
        date_min = report["date_range"]["min"][:10]
        date_max = report["date_range"]["max"][:10]
        st.caption(
            f"Review volume ({date_min} to {date_max}) is uneven over time, with single-month "
            "bursts (e.g. Herbal Essences, March 2015) that may reflect a launch or review "
            "campaign this data cannot confirm."
        )


# --- App ---------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Consumer Review Intelligence", page_icon="\U0001f9f4", layout="wide")
    st.title("Consumer Review Intelligence for P&G Haircare Brands")
    st.caption(
        "Cited, grounded answers over public Amazon reviews of Head & Shoulders, Pantene, "
        "and Herbal Essences. Answers use only the retrieved reviews and cite them by number; "
        "see Limitations in the README before treating any answer as authoritative."
    )

    try:
        _ensure_index()
    except Exception as error:
        # A cold container can fail here in ways build_index() itself never
        # names (HF model download, chromadb, disk): show it, don't crash.
        st.error(str(error))
        st.stop()

    with st.form("query_form"):
        question = st.text_input(
            "Ask a question about these reviews",
            placeholder="what do 1-star Pantene reviewers complain about",
        )
        col1, col2 = st.columns(2)
        with col1:
            brand_choice = st.selectbox("Brand", [ALL_BRANDS] + list(config.BRANDS))
        with col2:
            band_choice = st.selectbox(
                "Rating band", [ALL_BANDS] + list(config.RATING_BANDS), format_func=lambda v: BAND_LABELS.get(v, v)
            )
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            brand = None if brand_choice == ALL_BRANDS else brand_choice
            rating_band = None if band_choice == ALL_BANDS else band_choice
            _load_groq_secret()
            with st.spinner("Retrieving reviews and asking the model..."):
                try:
                    result = answer(question, brand=brand, rating_band=rating_band)
                except (FileNotFoundError, ValueError, RuntimeError) as error:
                    st.error(str(error))
                else:
                    _render_answer(result)

    st.divider()
    _render_eda()


if __name__ == "__main__":
    main()
