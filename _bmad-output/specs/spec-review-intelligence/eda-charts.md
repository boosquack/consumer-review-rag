# EDA charts (minimum set)

Every chart gets a 1–2 sentence takeaway. A chart without an insight is decoration.

1. Rating distribution per brand (grouped bars or small multiples).
2. Review volume over time per brand (line chart).
3. Average rating over time per brand. Annotate small-sample months so they don't distort the trend.
4. VADER sentiment on `review_text`: distribution and mean per brand.
5. Agreement between sentiment and star rating, shown and discussed as a sanity check.
6. Top terms in 1–2 star versus 4–5 star reviews per brand. Use frequency or TF-IDF; add topic modeling only if time allows.

Keep a running "insights" markdown cell that feeds the README's EDA highlights.
