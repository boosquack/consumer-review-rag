"""I/O matrix tests for src.eda (story 3).

No parquet: every fixture is an inline DataFrame shaped like the clean output.

Run with: uv run python -m unittest -v
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import config, eda


def frame(rows):
    """A clean-schema frame from (brand, rating, date_bucket, review_text) tuples."""
    records = []
    for brand, rating, bucket, text in rows:
        moment = None if bucket == config.UNDATED_BUCKET else pd.Timestamp(f"{bucket}-15", tz="UTC")
        records.append(
            {
                "brand": brand,
                "rating": rating,
                "review_text": text,
                "review_date": moment,
                "date_bucket": bucket,
            }
        )
    return pd.DataFrame(records)


class LoadTest(unittest.TestCase):
    def test_missing_parquet_raises_with_both_pipeline_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "reviews_clean.parquet"
            with self.assertRaises(FileNotFoundError) as caught:
                eda.load_clean(missing)
        message = str(caught.exception)
        self.assertIn("uv run python data/download_data.py", message)
        self.assertIn("uv run python -m src.clean", message)

    def test_missing_report_raises_with_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                eda.load_report(Path(tmp) / "quality_report.json")
        self.assertIn("uv run python -m src.clean", str(caught.exception))


class MonthlyStatsTest(unittest.TestCase):
    def test_small_sample_month_is_flagged_and_left_out_of_trend(self):
        big = [("Pantene", 5, "2020-01", "ok")] * 30
        small = [("Pantene", 1, "2020-02", "bad")] * 3
        monthly = eda.monthly_stats(frame(big + small))

        jan = monthly[monthly["month"] == pd.Timestamp("2020-01-01")].iloc[0]
        feb = monthly[monthly["month"] == pd.Timestamp("2020-02-01")].iloc[0]
        self.assertFalse(jan["small_sample"])
        self.assertEqual(jan["trend_rating"], 5.0)
        self.assertTrue(feb["small_sample"])
        self.assertEqual(feb["n_reviews"], 3)
        self.assertEqual(feb["mean_rating"], 1.0)  # still available to plot de-emphasized
        self.assertTrue(pd.isna(feb["trend_rating"]))

    def test_threshold_boundary_uses_config_value(self):
        at = [("Pantene", 4, "2021-05", "x")] * config.EDA_MIN_MONTH_REVIEWS
        below = [("Pantene", 4, "2021-06", "x")] * (config.EDA_MIN_MONTH_REVIEWS - 1)
        monthly = eda.monthly_stats(frame(at + below)).set_index("month")
        self.assertFalse(monthly.loc[pd.Timestamp("2021-05-01"), "small_sample"])
        self.assertTrue(monthly.loc[pd.Timestamp("2021-06-01"), "small_sample"])

    def test_undated_rows_leave_time_series_but_stay_in_other_analyses(self):
        rows = [
            ("Pantene", 1, "2020-01", "Terrible, it broke me out."),
            ("Pantene", 5, config.UNDATED_BUCKET, "Wonderful scent and softness."),
        ]
        data = frame(rows)

        monthly = eda.monthly_stats(data)
        self.assertEqual(int(monthly["n_reviews"].sum()), 1)
        self.assertEqual(eda.undated_count(data), 1)
        self.assertEqual(int(eda.yearly_summary(data)["n_reviews"].sum()), 1)

        self.assertEqual(eda.rating_distribution(data).loc["Pantene", 5], 50.0)
        scored = eda.add_sentiment(data)
        self.assertEqual(len(scored), 2)
        self.assertEqual(eda.sentiment_summary(scored).loc["Pantene", "n_reviews"], 2)
        terms = eda.top_terms_by_band(data)
        self.assertIn("scent", [t for t, _ in terms[("Pantene", "high")]])

    def test_small_sample_share_counts_months_and_reviews(self):
        big = [("Pantene", 5, "2020-01", "ok")] * 30
        small = [("Pantene", 1, "2020-02", "bad")] * 10
        share = eda.small_sample_share(eda.monthly_stats(frame(big + small))).loc["Pantene"]
        self.assertEqual(share["months"], 2)
        self.assertAlmostEqual(share["pct_months_small"], 50.0)
        self.assertAlmostEqual(share["pct_reviews_in_small"], 25.0)

    def test_peak_months_picks_each_brands_highest_volume_month(self):
        rows = (
            [("Pantene", 5, "2020-01", "a")] * 2
            + [("Pantene", 5, "2020-02", "b")] * 5
            + [("Herbal Essences", 4, "2019-06", "c")] * 4
            + [("Herbal Essences", 4, "2019-07", "d")] * 1
        )
        peaks = eda.peak_months(eda.monthly_stats(frame(rows))).set_index("brand")
        self.assertEqual(peaks.loc["Pantene", "month"], pd.Timestamp("2020-02-01"))
        self.assertEqual(peaks.loc["Pantene", "n_reviews"], 5)
        self.assertEqual(peaks.loc["Herbal Essences", "month"], pd.Timestamp("2019-06-01"))
        self.assertEqual(peaks.loc["Herbal Essences", "n_reviews"], 4)

    def test_trend_summary_counts_only_months_at_or_above_threshold(self):
        n = config.EDA_MIN_MONTH_REVIEWS
        rows = (
            [("Pantene", 5, "2020-01", "a")] * n
            + [("Pantene", 3, "2020-02", "b")] * n
            + [("Pantene", 1, "2020-03", "c")] * (n - 1)
        )
        summary = eda.trend_summary(eda.monthly_stats(frame(rows))).loc["Pantene"]
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["mean"], 4.0)
        self.assertAlmostEqual(summary["min"], 3.0)
        self.assertAlmostEqual(summary["max"], 5.0)

    def test_all_undated_gives_empty_monthly_frame(self):
        data = frame([("Pantene", 5, config.UNDATED_BUCKET, "fine")])
        self.assertTrue(eda.monthly_stats(data).empty)

    def test_complete_months_fills_gaps_with_zero(self):
        rows = [("Pantene", 5, "2020-01", "a"), ("Pantene", 5, "2020-03", "b")]
        filled = eda.complete_months(eda.monthly_stats(frame(rows)))
        self.assertEqual(list(filled["n_reviews"]), [1, 0, 1])


class SentimentTest(unittest.TestCase):
    def test_cutoffs_band_scores_and_zero_is_neutral(self):
        self.assertEqual(eda.sentiment_label(config.VADER_POS), "positive")
        self.assertEqual(eda.sentiment_label(config.VADER_NEG), "negative")
        self.assertEqual(eda.sentiment_label(0.0), "neutral")
        self.assertEqual(eda.sentiment_label(0.049), "neutral")

    def test_empty_or_odd_text_scores_zero_and_is_neutral(self):
        data = frame(
            [
                ("Pantene", 3, "2020-01", ""),
                ("Pantene", 3, "2020-01", "12345 !!!"),
                ("Pantene", 3, "2020-01", "ok"),
            ]
        )
        data.loc[2, "review_text"] = None
        scored = eda.add_sentiment(data)
        self.assertTrue((scored["sentiment_compound"] == 0.0).all())
        self.assertTrue((scored["sentiment_label"] == "neutral").all())

    def test_scores_original_text_not_normalized_copy(self):
        data = frame([("Pantene", 5, "2020-01", "I love it")])
        data["review_text_normalized"] = "i hate it"
        scored = eda.add_sentiment(data)
        self.assertEqual(scored.loc[0, "sentiment_label"], "positive")

    def test_sentiment_summary_shares_are_per_brand_and_sum_to_100(self):
        scored = pd.DataFrame(
            {
                "brand": ["Pantene"] * 4 + ["Herbal Essences"] * 2,
                "rating": [5, 5, 1, 3, 5, 1],
                "sentiment_compound": [0.9, 0.5, -0.5, 0.0, 0.8, -0.2],
                "sentiment_label": [
                    "positive", "positive", "negative", "neutral", "positive", "negative",
                ],
            }
        )
        summary = eda.sentiment_summary(scored)
        self.assertEqual(list(summary.index), ["Pantene", "Herbal Essences"])
        self.assertAlmostEqual(summary.loc["Pantene", "pct_positive"], 50.0)
        self.assertAlmostEqual(summary.loc["Pantene", "pct_negative"], 25.0)
        self.assertAlmostEqual(summary.loc["Pantene", "pct_neutral"], 25.0)
        self.assertAlmostEqual(summary.loc["Herbal Essences", "pct_positive"], 50.0)
        self.assertAlmostEqual(summary.loc["Herbal Essences", "pct_negative"], 50.0)
        self.assertAlmostEqual(summary.loc["Herbal Essences", "pct_neutral"], 0.0)
        shares = summary[["pct_negative", "pct_neutral", "pct_positive"]].sum(axis=1)
        for total in shares:
            self.assertAlmostEqual(total, 100.0)
        self.assertAlmostEqual(summary.loc["Pantene", "mean_compound"], 0.225)

    def test_agreement_table_rows_sum_to_100_and_rate_ignores_three_stars(self):
        scored = pd.DataFrame(
            {
                "rating": [1, 1, 5, 5, 3],
                "sentiment_label": ["negative", "positive", "positive", "neutral", "negative"],
            }
        )
        table = eda.sentiment_rating_agreement(scored)
        self.assertEqual(list(table.columns), list(eda.SENTIMENT_LABELS))
        for total in table.sum(axis=1):
            self.assertAlmostEqual(total, 100.0)
        self.assertAlmostEqual(table.loc[1, "positive"], 50.0)
        self.assertAlmostEqual(eda.agreement_rate(scored), 50.0)


class TopTermsTest(unittest.TestCase):
    def test_empty_band_returns_empty_list_without_crash(self):
        rows = [
            ("Pantene", 5, "2020-01", "silky soft shine"),
            ("Pantene", 3, "2020-01", "greasy residue"),  # 3 stars: in no band
        ]
        terms = eda.top_terms_by_band(frame(rows))
        self.assertEqual(terms[("Pantene", "low")], [])
        self.assertEqual(terms[("Head & Shoulders", "high")], [])
        high = [t for t, _ in terms[("Pantene", "high")]]
        self.assertIn("silky", high)
        self.assertNotIn("greasy", high)
        self.assertEqual(set(terms), {(b, band) for b in config.BRANDS for band in eda.BANDS})

    def test_brand_and_generic_product_tokens_are_stopped(self):
        rows = [("Pantene", 1, "2020-01", "Pantene shampoo made my hair itchy and flaky")]
        low = [t for t, _ in eda.top_terms_by_band(frame(rows))[("Pantene", "low")]]
        self.assertIn("itchy", low)
        for token in ("pantene", "shampoo", "hair", "and"):
            self.assertNotIn(token, low)

    def test_top_terms_are_heaviest_first_and_truncated_to_n(self):
        rows = [
            ("Pantene", 5, "2020-01", "silky silky silky silky shine shine shine glossy"),
            ("Pantene", 1, "2020-01", "flaky"),
        ]
        high = eda.top_terms_by_band(frame(rows), n=2)[("Pantene", "high")]
        self.assertEqual(len(high), 2)
        self.assertEqual([t for t, _ in high], ["silky", "shine"])
        weights = [w for _, w in high]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_accented_words_are_kept_as_terms(self):
        rows = [("Pantene", 5, "2020-01", "champú suave")]
        high = [t for t, _ in eda.top_terms_by_band(frame(rows))[("Pantene", "high")]]
        self.assertIn("champú", high)

    def test_term_mention_share_whole_word_plural_case_and_bands(self):
        rows = [
            ("Pantene", 1, "2020-01", "The Bottles leaked"),   # plural, capitalised
            ("Pantene", 2, "2020-01", "bottleneck at checkout"),  # not a whole word
            ("Pantene", 3, "2020-01", "bottle bottle"),  # 3 stars: in neither band
            ("Pantene", 3, "2020-01", "bottle"),
        ]
        share = eda.term_mention_share(frame(rows), ["bottle"]).loc["bottle"]
        self.assertAlmostEqual(share["pct_low"], 50.0)
        self.assertTrue(pd.isna(share["pct_high"]))

    def test_all_stopword_text_returns_empty_lists(self):
        rows = [("Pantene", 5, "2020-01", "the shampoo and the hair")]
        terms = eda.top_terms_by_band(frame(rows))
        self.assertTrue(all(value == [] for value in terms.values()))

    def test_no_rows_returns_empty_lists(self):
        terms = eda.top_terms_by_band(frame([]).reindex(columns=["brand", "rating", "review_text"]))
        self.assertTrue(all(value == [] for value in terms.values()))


if __name__ == "__main__":
    unittest.main()
