"""Aggregations and small plot helpers behind the EDA notebook (story 3) and the app.

Used by ``notebooks/01_eda.ipynb`` and, later, ``app/streamlit_app.py``; neither
reimplements these. Input is the clean frame written by ``src/clean.py``.

Sentiment is VADER, a lexicon heuristic, scored on the original ``review_text``.
Every number describes the sampled Amazon reviews, not the customer population.
Monthly series use the ``date_bucket`` column, so undated rows leave the time
series but stay in the sentiment, rating, and term analyses.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

SENTIMENT_LABELS = ("negative", "neutral", "positive")
BANDS = ("low", "high")

_MISSING_HINT = (
    "{path} not found. Build it first with:\n"
    "  uv run python data/download_data.py\n"
    "  uv run python -m src.clean"
)


# --- Loading -----------------------------------------------------------------------


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Read the clean parquet, raising with the two pipeline commands when it is missing."""
    path = Path(path or config.CLEAN_PARQUET)
    if not path.exists():
        raise FileNotFoundError(_MISSING_HINT.format(path=path))
    return pd.read_parquet(path)


def load_report(path: Path | None = None) -> dict[str, Any]:
    """Read the persisted quality report, with the same hint when it is missing."""
    path = Path(path or config.CLEAN_REPORT_JSON)
    if not path.exists():
        raise FileNotFoundError(_MISSING_HINT.format(path=path))
    return json.loads(path.read_text())


def brand_order(frame: pd.DataFrame) -> list[str]:
    """Return the brands present, in the fixed config order so colours never shift."""
    present = set(frame["brand"])
    return [brand for brand in config.BRANDS if brand in present]


# --- Rating ----------------------------------------------------------------------


def rating_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the percent of each brand's reviews at each star rating (brand x 1-5)."""
    counts = pd.crosstab(frame["brand"], frame["rating"])
    counts = counts.reindex(index=brand_order(frame), columns=range(1, 6), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).mul(100)


def rating_band(rating: int) -> str | None:
    """Return "low" for 1-2 stars, "high" for 4-5, and None for 3 (excluded)."""
    if rating in config.EDA_LOW_RATINGS:
        return "low"
    if rating in config.EDA_HIGH_RATINGS:
        return "high"
    return None


# --- Time series -------------------------------------------------------------------


def undated_count(frame: pd.DataFrame) -> int:
    """Return how many rows sit in the undated bucket."""
    return int((frame["date_bucket"] == config.UNDATED_BUCKET).sum())


def monthly_stats(frame: pd.DataFrame, min_reviews: int | None = None) -> pd.DataFrame:
    """Per brand-month review count and mean rating, with a small-sample flag.

    Columns: brand, month (first-of-month Timestamp), n_reviews, mean_rating,
    small_sample, and trend_rating (mean_rating, or NaN for small-sample months so a
    line drawn through it skips them). Undated rows are excluded.
    """
    threshold = config.EDA_MIN_MONTH_REVIEWS if min_reviews is None else min_reviews
    dated = frame[frame["date_bucket"] != config.UNDATED_BUCKET]
    columns = ["brand", "month", "n_reviews", "mean_rating", "small_sample", "trend_rating"]
    if dated.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        dated.groupby(["brand", "date_bucket"])["rating"]
        .agg(n_reviews="size", mean_rating="mean")
        .reset_index()
    )
    grouped["month"] = pd.to_datetime(grouped["date_bucket"], format="%Y-%m")
    grouped["small_sample"] = grouped["n_reviews"] < threshold
    grouped["trend_rating"] = grouped["mean_rating"].where(~grouped["small_sample"])
    return grouped.sort_values(["brand", "month"]).reset_index(drop=True)[columns]


def complete_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Reindex each brand onto a continuous monthly range, filling absent months with 0 reviews.

    Without this, a volume line would bridge a gap as if reviews were steady.
    """
    parts = []
    for brand, rows in monthly.groupby("brand", sort=False):
        months = pd.date_range(rows["month"].min(), rows["month"].max(), freq="MS")
        filled = rows.set_index("month").reindex(months)
        filled["brand"] = brand
        filled["n_reviews"] = filled["n_reviews"].fillna(0).astype(int)
        filled["small_sample"] = filled["small_sample"].fillna(True).astype(bool)
        parts.append(filled.rename_axis("month").reset_index())
    if not parts:
        return monthly
    return pd.concat(parts, ignore_index=True)[monthly.columns]


def yearly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per brand-year review count and mean rating over dated rows, with a partial-year flag."""
    dated = frame[frame["date_bucket"] != config.UNDATED_BUCKET]
    years = dated["date_bucket"].str[:4].astype(int).rename("year")
    table = dated.groupby([dated["brand"], years])["rating"].agg(
        n_reviews="size", mean_rating="mean"
    ).reset_index()
    table["partial_year"] = table["year"] == config.EDA_PARTIAL_YEAR
    return table


def peak_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return each brand's highest-volume month and its review count."""
    if monthly.empty:
        return monthly[["brand", "month", "n_reviews"]]
    rows = monthly.loc[monthly.groupby("brand")["n_reviews"].idxmax(), ["brand", "month", "n_reviews"]]
    return rows.reset_index(drop=True)


def trend_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Count, mean, std, min, and max of monthly mean rating over months with n >= threshold."""
    return monthly.groupby("brand")["trend_rating"].agg(["count", "mean", "std", "min", "max"])


def small_sample_share(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per brand: observed months, percent of them small-sample, and percent of reviews they hold."""
    grouped = monthly.groupby("brand", sort=False)
    small_reviews = monthly["n_reviews"].where(monthly["small_sample"], 0)
    return pd.DataFrame(
        {
            "months": grouped.size(),
            "pct_months_small": grouped["small_sample"].mean().mul(100),
            "pct_reviews_in_small": small_reviews.groupby(monthly["brand"], sort=False).sum()
            .div(grouped["n_reviews"].sum())
            .mul(100),
        }
    )


# --- Sentiment ---------------------------------------------------------------------


def sentiment_label(compound: float) -> str:
    """Band a VADER compound score with the standard cutoffs; 0.0 is neutral."""
    if compound >= config.VADER_POS:
        return "positive"
    if compound <= config.VADER_NEG:
        return "negative"
    return "neutral"


def add_sentiment(frame: pd.DataFrame, analyzer: Any = None) -> pd.DataFrame:
    """Return a copy with VADER ``sentiment_compound`` and ``sentiment_label`` columns.

    Scores the original ``review_text``, because VADER reads casing, punctuation,
    and emphasis. Empty or non-string text scores 0.0 and is labelled neutral.
    """
    if analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    scored = frame.copy()
    scored["sentiment_compound"] = [
        analyzer.polarity_scores(text if isinstance(text, str) else "")["compound"]
        for text in scored["review_text"]
    ]
    scored["sentiment_label"] = scored["sentiment_compound"].map(sentiment_label)
    return scored


def sentiment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per brand: review count, mean and median compound, and percent in each label."""
    order = brand_order(scored)
    stats = scored.groupby("brand")["sentiment_compound"].agg(
        n_reviews="size", mean_compound="mean", median_compound="median"
    )
    shares = (
        pd.crosstab(scored["brand"], scored["sentiment_label"], normalize="index")
        .reindex(columns=SENTIMENT_LABELS, fill_value=0)
        .mul(100)
        .add_prefix("pct_")
    )
    return stats.join(shares).reindex(order)


def compound_by_brand_rating(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean VADER compound per brand (rows) and star rating (columns)."""
    table = scored.groupby(["brand", "rating"])["sentiment_compound"].mean().unstack()
    return table.reindex(brand_order(scored))


def sentiment_rating_agreement(scored: pd.DataFrame) -> pd.DataFrame:
    """Percent of each star rating's reviews in each VADER label (rows sum to 100)."""
    table = pd.crosstab(scored["rating"], scored["sentiment_label"], normalize="index")
    ratings = sorted(set(scored["rating"]))
    return table.reindex(index=ratings, columns=SENTIMENT_LABELS, fill_value=0).mul(100)


def agreement_rate(scored: pd.DataFrame) -> float:
    """Percent of non-3-star reviews whose label matches the rating band.

    Low (1-2 stars) agrees with negative, high (4-5) with positive; neutral never
    agrees. Returns NaN when no review has a banded rating.
    """
    bands = scored["rating"].map(rating_band)
    banded = bands.notna()
    if not banded.any():
        return float("nan")
    expected = bands[banded].map({"low": "negative", "high": "positive"})
    return float((scored.loc[banded, "sentiment_label"] == expected).mean() * 100)


# --- Top terms ---------------------------------------------------------------------


def top_terms_by_band(
    frame: pd.DataFrame, n: int | None = None
) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """TF-IDF top terms per (brand, band), where each cell's reviews form one document.

    Keys cover every configured brand and both bands; a cell with no reviews maps
    to an empty list. Stopwords are English plus ``config.EDA_EXTRA_STOPWORDS``.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    n = config.EDA_TOP_TERMS if n is None else n
    keys = [(brand, band) for brand in config.BRANDS for band in BANDS]
    result: dict[tuple[str, str], list[tuple[str, float]]] = {key: [] for key in keys}

    bands = frame["rating"].map(rating_band)
    documents = {}
    for brand, band in keys:
        texts = frame.loc[(frame["brand"] == brand) & (bands == band), "review_text"]
        if len(texts):
            documents[(brand, band)] = " ".join(texts.astype(str))
    if not documents:
        return result

    vectorizer = TfidfVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS.union(config.EDA_EXTRA_STOPWORDS)),
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",  # Unicode letters, so "champú" survives
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(documents.values())
    except ValueError:  # every document was stopwords only
        return result
    vocabulary = vectorizer.get_feature_names_out()
    for row, key in enumerate(documents):
        weights = matrix[row].toarray().ravel()
        top = weights.argsort()[::-1][:n]
        result[key] = [(str(vocabulary[i]), float(weights[i])) for i in top if weights[i] > 0]
    return result


def term_mention_share(frame: pd.DataFrame, words: list[str]) -> pd.DataFrame:
    """Percent of low-band and high-band reviews that mention each word.

    Matches the whole word with an optional trailing "s" (so "bottle" counts
    "bottles"), in any case. 3-star reviews are in neither band; a band with no
    reviews gives NaN.

    Sizes the TF-IDF terms, whose weights say which words stand out, not how common they are.
    """
    import re

    bands = frame["rating"].map(rating_band)
    rows = []
    for word in words:
        mentions = frame["review_text"].astype(str).str.contains(
            rf"\b{re.escape(word)}s?\b", case=False, regex=True
        )
        row = {"term": word}
        for band in BANDS:
            in_band = bands == band
            row[f"pct_{band}"] = mentions[in_band].mean() * 100 if in_band.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["term", "pct_low", "pct_high"]).set_index("term")


# --- Plot helpers ------------------------------------------------------------------
# Each takes a matplotlib Axes and returns it, so the notebook and app share styling.


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_rating_distribution(ax: Any, distribution: pd.DataFrame) -> Any:
    """Grouped bars: percent of each brand's reviews per star rating."""
    brands = list(distribution.index)
    width = 0.8 / max(len(brands), 1)
    for i, brand in enumerate(brands):
        xs = [r + (i - (len(brands) - 1) / 2) * width for r in distribution.columns]
        ax.bar(xs, distribution.loc[brand], width=width * 0.92, label=brand,
               color=config.EDA_BRAND_COLORS.get(brand))
    ax.set_xticks(list(distribution.columns))
    ax.set_xticklabels([f"{r}★" for r in distribution.columns])
    ax.set_ylabel("% of brand's reviews")
    ax.legend(frameon=False)
    _style(ax)
    return ax


def plot_monthly_volume(ax: Any, monthly: pd.DataFrame) -> Any:
    """Line per brand: reviews per month, with the partial final year shaded."""
    filled = complete_months(monthly)
    for brand in config.BRANDS:
        rows = filled[filled["brand"] == brand]
        if rows.empty:
            continue
        ax.plot(rows["month"], rows["n_reviews"], linewidth=1.5, label=brand,
                color=config.EDA_BRAND_COLORS.get(brand))
    _shade_partial_year(ax)
    ax.set_ylabel("Reviews per month")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    return ax


def plot_monthly_rating(ax: Any, monthly: pd.DataFrame, brand: str) -> Any:
    """One brand's mean rating per month: small-sample months as faint markers, the rest as a line."""
    rows = monthly[monthly["brand"] == brand]
    color = config.EDA_BRAND_COLORS.get(brand)
    small = rows[rows["small_sample"]]
    ax.scatter(small["month"], small["mean_rating"], s=10, color=color, alpha=0.2,
               label=f"n < {config.EDA_MIN_MONTH_REVIEWS} (excluded from line)")
    ax.plot(rows["month"], rows["trend_rating"], linewidth=1.5, marker="o", markersize=2.5,
            color=color, label=f"n ≥ {config.EDA_MIN_MONTH_REVIEWS}")
    _shade_partial_year(ax)
    ax.set_ylim(0.8, 5.2)
    ax.set_title(brand, loc="left")
    ax.set_ylabel("Mean rating")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    _style(ax)
    return ax


def _shade_partial_year(ax: Any) -> None:
    start = pd.Timestamp(f"{config.EDA_PARTIAL_YEAR}-01-01")
    end = pd.Timestamp(f"{config.EDA_PARTIAL_YEAR}-12-31")
    ax.axvspan(start, end, color="#f0efec", zorder=0)
    ax.text(start, 1, f" {config.EDA_PARTIAL_YEAR} partial", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="#52514e")


def plot_sentiment_distribution(ax: Any, scored: pd.DataFrame) -> Any:
    """Overlaid step histograms of VADER compound per brand, cutoffs marked."""
    bins = [i / 20 for i in range(-20, 21)]
    for brand in brand_order(scored):
        values = scored.loc[scored["brand"] == brand, "sentiment_compound"]
        ax.hist(values, bins=bins, histtype="step", linewidth=1.5, density=True,
                label=brand, color=config.EDA_BRAND_COLORS.get(brand))
    for cutoff in (config.VADER_NEG, config.VADER_POS):
        ax.axvline(cutoff, color="#52514e", linewidth=0.8, linestyle="--")
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    return ax


def plot_sentiment_shares(ax: Any, summary: pd.DataFrame) -> Any:
    """Horizontal 100% stacked bars of negative / neutral / positive per brand."""
    left = pd.Series(0.0, index=summary.index)
    for label in SENTIMENT_LABELS:
        values = summary[f"pct_{label}"]
        ax.barh(summary.index, values, left=left, label=label, edgecolor="white",
                linewidth=2, color=config.EDA_SENTIMENT_COLORS[label])
        left = left + values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of brand's reviews")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_agreement(ax: Any, agreement: pd.DataFrame) -> Any:
    """Heatmap of star rating (rows) by VADER label (columns), annotated with percents."""
    ax.imshow(agreement.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(agreement.shape[0]):
        for j in range(agreement.shape[1]):
            value = agreement.iat[i, j]
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center",
                    color="white" if value > 55 else "#0b0b0b")
    ax.set_xticks(range(agreement.shape[1]))
    ax.set_xticklabels(agreement.columns)
    ax.set_yticks(range(agreement.shape[0]))
    ax.set_yticklabels([f"{r}★" for r in agreement.index])
    ax.set_xlabel("VADER label")
    ax.set_ylabel("Star rating")
    return ax


def plot_top_terms(ax: Any, terms: list[tuple[str, float]], title: str, color: str | None = None) -> Any:
    """Horizontal bars of one cell's top terms, highest at the top; notes an empty cell."""
    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if not terms:
        ax.text(0.5, 0.5, "no terms", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
    words = [term for term, _ in terms][::-1]
    weights = [weight for _, weight in terms][::-1]
    ax.barh(words, weights, color=color)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("TF-IDF weight", fontsize=8)
    return ax
