# PLAN.md

**Project:** Consumer Review Intelligence for P&G Personal-Care Brands
**Repo:** `consumer-review-rag`
**Author:** Kent Joshua P. Alvarez (Booswaa)
**Built for:** P&G IT Internship application, AI Engineering track
**Timeframe:** One week (Week of Sep 15, 2026)

---

## 1. Context and Purpose

This is a portfolio project built specifically for a P&G AI Engineering Intern application. It is deliberately grounded in P&G's own brands rather than a generic dataset, so that a reviewer reads it as "built for us" instead of "repurposed from a thesis."

It does two things on top of one shared corpus of consumer product reviews:

1. **Exploratory Data Analysis (the Data Analyst layer).** Understand the review data: rating distributions, review volume over time, sentiment trends, and the recurring themes that drive positive and negative reviews for each brand. This maps to P&G's Consumer & Market Knowledge function and to the internship's general qualification about deriving business insight from data.
2. **Retrieval-Augmented Generation (the AI Engineering layer).** A grounded question-answering and summarization system over the same reviews. Every answer traces back to specific source reviews with citations, never a fluent-sounding guess. This maps directly to the JD line about developing "AI agents / AI-powered digital business solutions."

The intended narrative for the application: *a system that turns raw consumer review data into both analyst-ready insight and a grounded, cited answering tool a brand team could actually query.*

### What this project is NOT
- Not connected to Naya or MamaBot in any way. It reuses skills, not code or data.
- Not a scraper-first project. It uses an existing public dataset to protect the timeline (see Section 5).
- Not a claim of production readiness. Honest limitations are a first-class deliverable (Section 9).

---

## 2. Goals and Success Criteria

The project is "done" when all of the following are true:

- [ ] A cleaned, documented review dataset covering 2 to 3 P&G haircare brands is committed (or a reproducible download + clean script is committed if the raw file is too large for the repo).
- [ ] An EDA notebook runs top to bottom with no errors and produces at least the visualizations listed in Section 7, Phase 2, each with a one to two sentence written insight.
- [ ] A RAG pipeline answers natural-language questions over the reviews and returns citations (brand, rating, and a source snippet) for every answer.
- [ ] A hand-built evaluation set of 15 to 20 questions exists, with retrieval and faithfulness scored and the results written up honestly.
- [ ] A Streamlit app is deployed live on Streamlit Community Cloud with a working public URL.
- [ ] A README ties the whole thing to the P&G application, with an architecture diagram and clear run instructions.
- [ ] A LIMITATIONS section names where retrieval fails, where the model hallucinates, and what the system cannot answer.

### Non-goals for this week (candidate stretch work, Section 12)
- Multi-step agentic tool use / planning.
- Fine-tuning any model.
- A custom-trained sentiment model (use a pretrained one for EDA).

---

## 3. Recommended Scope Decisions (confirm or override before Day 1)

| Decision | Recommendation | Why |
|---|---|---|
| Brand set | **Locked (2026-09-15):** Head & Shoulders, Pantene, Herbal Essences (haircare) | Same category means clean like-for-like comparison. All three are P&G. Measured coverage clears the volume bar (Section 13). |
| Category fallback | Grooming (Gillette, Oral-B) or skincare (Olay) | Not needed: haircare coverage is strong. Aussie is a viable fourth haircare brand but exceeds the 2–3 brand scope. |
| Data source | **Locked (2026-09-15):** Amazon Reviews 2023 (`McAuley-Lab/Amazon-Reviews-2023` on Hugging Face), category `Beauty_and_Personal_Care`, filtered to the brand set | Public, no scraping. The `All_Beauty` subset is too thin for per-brand trends. See Section 13 for evidence and the dataset-terms caveat. |
| Generation model | **Locked (2026-09-15):** Groq `openai/gpt-oss-120b`, low temperature (set once in `src/config.py`) | Groq shut down its Llama chat models on 2026-08-16 ([deprecations](https://console.groq.com/docs/deprecations)), so the original "current Llama instruct model" plan no longer exists. Groq names it as the replacement for `llama-3.3-70b-versatile`; the expectation of strong grounding is the selection rationale, to be measured in evaluation (Phase 5), not a measured result. Trade-off: it emits reasoning tokens (121 on a trivial call during selection) that count against the free-tier 8K tokens/min cap ([rate limits](https://console.groq.com/docs/rate-limits)). Free tier. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run locally | Free, 384-dim, fast on Apple Silicon (CPU or MPS). No second API key. |
| Vector store | ChromaDB (persistent, local) | Fastest solo setup, persists to disk, trivial to load in the Streamlit app. |
| Deployment | Streamlit Community Cloud | Free hosting, GitHub-connected, secrets handling built in. |

---

## 4. Requirements and Environment

### Machine
MacBook (Apple Silicon assumed). Local embedding inference runs on CPU or MPS. No GPU required.

### Python
Target Python 3.11 (Streamlit Community Cloud supports it well; avoid 3.13 for now due to occasional wheel gaps in ML libs).

### `requirements.txt` (starting point)
```
# Data + EDA
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
scikit-learn>=1.4

# Sentiment for EDA
vaderSentiment>=3.3.2

# RAG
sentence-transformers>=3.0
chromadb>=0.5
groq>=0.11

# App + config
streamlit>=1.38
python-dotenv>=1.0
```
Pin exact versions with `pip freeze > requirements.txt` once the environment works, so Streamlit Cloud builds match local.

### Secrets and config
- `GROQ_API_KEY` lives in a local `.env` (git-ignored) and in Streamlit Cloud's Secrets manager for deployment. Never commit it.
- Add a `.env.example` with the key name and no value so the repo documents what is needed.
- The Groq model string is `openai/gpt-oss-120b` (confirmed 2026-09-15 with `uv run python -m src.groq_smoke`). It is set once as `GROQ_MODEL` in `src/config.py`, not scattered through the code. Model names change, so re-run the smoke test if calls start returning 404.

### `.gitignore` essentials
```
.env
__pycache__/
*.pyc
.chroma/            # or wherever the persistent vector store lives
data/raw/           # if the raw dataset is large; keep a download script instead
.ipynb_checkpoints/
```

---

## 5. Data Plan

### 5.1 Sourcing (Day 1, first thing)
1. Search Kaggle for existing personal-care / haircare / Amazon product review datasets. The Amazon Reviews family (Beauty / Personal Care subsets) is the most likely source of P&G-brand rows.
2. Filter to rows whose product title or brand field matches the chosen brands (case-insensitive substring match on brand names and common product-line names).
3. If no single dataset has all three brands with enough volume, either drop to two brands or combine two compatible datasets with a shared schema.
4. Only if public datasets genuinely fail: fall back to a small, respectful scrape, and time-box it to half a day maximum. Do not let scraping become the project.

### 5.2 Target schema (normalize whatever you find to this)
| Column | Type | Notes |
|---|---|---|
| `review_id` | str | Unique. Generate one if the source lacks it. |
| `brand` | str | One of the chosen brands, normalized (e.g. "Head & Shoulders"). |
| `product_name` | str | Raw product title. |
| `rating` | int | 1 to 5. |
| `review_title` | str | May be empty. |
| `review_text` | str | The body. This is what gets embedded. |
| `review_date` | datetime | For time-trend EDA. Parse and coerce; flag unparseable. |
| `verified` | bool | If available; otherwise drop. |

### 5.3 Minimum data quality bar before proceeding
- At least a few hundred reviews per brand, ideally 1,000+ per brand for meaningful trends.
- Record and report: total rows, rows per brand, date range, and percent of rows dropped during cleaning and why. This transparency is itself a portfolio signal.

### 5.4 Cleaning steps (document each in the notebook)
- Drop rows with empty `review_text`.
- De-duplicate on `review_id` and on exact `review_text` duplicates (copy-paste spam is common in review data).
- Normalize brand strings to the canonical set.
- Parse dates; keep an `undated` bucket rather than silently dropping.
- Basic text normalization for embedding input (strip, collapse whitespace); keep the original text intact for display and citation.
- Save the cleaned result to `data/processed/reviews_clean.parquet`.

---

## 6. Repository Structure

```
consumer-review-rag/
├── README.md
├── PLAN.md                      # this file
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                     # git-ignored; download script provided
│   ├── processed/
│   │   └── reviews_clean.parquet
│   └── download_data.py         # reproducible fetch + filter to brands
├── notebooks/
│   └── 01_eda.ipynb             # the Data Analyst deliverable
├── src/
│   ├── config.py                # model names, paths, constants
│   ├── clean.py                 # cleaning pipeline (importable, tested by notebook)
│   ├── index.py                 # chunk + embed + write to Chroma
│   ├── retrieve.py              # query -> top-k review chunks
│   ├── generate.py              # retrieved context + question -> cited answer
│   └── evaluate.py              # runs the gold set, prints metrics
├── eval/
│   ├── gold_questions.json      # 15-20 hand-written Q/A + expected sources
│   └── results.md               # scored results + commentary
└── app/
    └── streamlit_app.py         # the deployed live demo
```

Keep logic in `src/` importable so both the notebook and the app call the same functions. Do not duplicate the retrieval logic in the Streamlit file.

---

## 7. Timeline: Mapped to the AI Engineering Lifecycle

Six working phases across the week, with a buffer. Day 1 is today (Tue Sep 15). Each phase names its concrete nitty-gritty so you can implement straight from it.

### Phase 0 (Day 1, morning): Problem framing and setup
- Finalize the brand set against actual dataset availability (do not commit to brands before you have seen the data).
- Create the repo, virtual environment, `requirements.txt`, `.gitignore`, `.env.example`, and folder skeleton.
- Confirm the Groq model string and that a trivial Groq call works end to end.
- Write the one-paragraph problem statement into the README so the framing is locked before you build.
- **Exit check:** repo scaffolded, a "hello world" Groq call returns, dataset located.

### Phase 1 (Day 1, afternoon): Data collection and cleaning
- Implement `download_data.py` and `clean.py` to produce `reviews_clean.parquet`.
- Print the data quality report (rows per brand, date range, drop counts).
- **Nitty gritty:** decide the de-dup rule now; review data is full of near-duplicates. Decide how you handle reviews with no date. Keep original text separate from normalized text.
- **Exit check:** clean parquet exists, quality report captured in the notebook.

### Phase 2 (Day 2): Exploratory Data Analysis
The Data Analyst deliverable. Produce at minimum:
- Rating distribution per brand (grouped bar or small multiples).
- Review volume over time per brand (line chart).
- Average rating over time per brand (watch for small-sample months; annotate them).
- Sentiment via VADER on `review_text`, distribution and mean per brand, and sentiment-versus-star-rating agreement (a nice sanity check to show and discuss).
- Top terms / themes in 1 to 2 star reviews vs 4 to 5 star reviews per brand (start with simple frequency or TF-IDF; only reach for topic modeling if time allows).
- **Nitty gritty:** every chart gets a one to two sentence written takeaway. A chart without an insight is decoration. Watch for months with too few reviews distorting trend lines. Keep a running "insights" markdown cell that becomes README material.
- **Exit check:** `01_eda.ipynb` runs top to bottom clean, insights written.

### Phase 3 (Day 3): Indexing and retrieval
- `index.py`: chunk reviews, embed with `all-MiniLM-L6-v2`, write to persistent ChromaDB with metadata (`brand`, `rating`, `product_name`, `review_date`, `review_id`).
- **Chunking decision:** most single reviews are short enough to embed whole. Only split very long reviews. Store the star rating and brand as metadata so retrieval can filter (for example, "what do 1-star Pantene reviewers complain about" should be filterable, not just semantic).
- `retrieve.py`: query -> top-k chunks, with optional metadata filters (brand, rating band).
- **Nitty gritty:** test retrieval in isolation before touching generation. Eyeball 5 to 10 queries and confirm the returned reviews are actually relevant. Decide k (start with 5 to 8). Persist the index so the app does not re-embed on every launch.
- **Exit check:** retrieval returns sensible reviews for hand-tested queries.

### Phase 4 (Day 4): Grounded generation with citations
- `generate.py`: take the question + retrieved chunks, prompt the Groq model to answer using only the provided reviews and to cite them.
- **Prompt design nitty gritty:** instruct the model to answer only from the supplied context; to say "the reviews do not cover this" when they do not; and to reference reviews by an index that maps back to real source rows. Return the source reviews alongside the answer so the UI can show them. This is the anti-hallucination core of the project; treat the prompt as a real artifact and iterate on it.
- Handle the empty-retrieval case explicitly (no relevant reviews found).
- Handle Groq rate limits (you know the 30 requests/min ceiling from MamaBot): add simple backoff.
- **Exit check:** end-to-end question in, cited answer out, with visible sources.

### Phase 5 (Day 5): Evaluation
The step that separates a real project from a demo. Protect this phase even if the week gets tight.
- Hand-write 15 to 20 questions in `gold_questions.json`. Mix them: factual ("what do reviewers say about the scent of X"), comparative ("which brand has more complaints about residue"), and at least a few unanswerable ones to test refusal.
- For each, note the expected supporting review(s) or theme.
- `evaluate.py` measures:
  - **Retrieval quality:** for questions with a known relevant review, does it appear in top-k? Report hit rate.
  - **Answer faithfulness:** manually or with a checklist, does the answer stay grounded in retrieved reviews, no invented claims? Report as a scored pass/fail per question with notes.
  - **Refusal behavior:** does it correctly decline on the unanswerable ones?
- Write `eval/results.md` with the numbers and honest commentary on failures.
- **Exit check:** metrics computed, results written up including what failed.

### Phase 6 (Day 6): Deployment, documentation, buffer
- `streamlit_app.py`: a query box, optional brand/rating filters, the cited answer, expandable source reviews, and one or two of the strongest EDA charts embedded so the app shows both layers.
- Deploy to Streamlit Community Cloud, wire `GROQ_API_KEY` into Secrets, confirm the public URL works on a fresh browser.
- Finish the README: problem framing, architecture diagram, run instructions, EDA highlights, evaluation summary, limitations, and the explicit tie to the P&G application.
- **Exit check:** live URL works, README complete, everything committed.

### If a day slips
Priority order to protect: **Evaluation (Phase 5) and honest limitations > deployment > extra EDA charts > extra brands.** A scoped thing done well beats a broad thing done rough.

---

## 8. RAG Design Notes (consolidated for implementation)

- **Embeddings:** `all-MiniLM-L6-v2`, local. Cache the model load. Batch-embed during indexing.
- **Vector store:** ChromaDB persistent client pointed at a local directory. Include rich metadata for filtering.
- **Retrieval:** semantic top-k with optional metadata pre-filter (brand, rating band). Start k=6, tune during Phase 3.
- **Generation:** Groq instruct model. System prompt enforces grounding, citation, and refusal. Temperature low (grounded answering, not creative writing).
- **Citations:** pass retrieved reviews to the model with stable indices; render the real source review (brand, rating, snippet) in the UI next to the answer.
- **Failure modes to handle in code:** empty retrieval, rate limit, model ignoring the grounding instruction (catch in eval).

---

## 9. Limitations to Document Honestly (a required deliverable)

Write these into the README rather than hiding them:
- The reviews are a public dataset, not P&G's internal data, and may be dated or incomplete per brand.
- Retrieval can miss relevant reviews or surface loosely related ones; report the measured hit rate rather than claiming it "works."
- The model can still drift from the source despite grounding; report where it did in evaluation.
- Sentiment via VADER is a lexicon heuristic, not a trained classifier; it is directionally useful, not authoritative.
- Findings describe the sampled reviews, not the true customer population; no statistical generalization is claimed.
- Amazon Reviews 2023 publishes no terms of use for the data itself (checked on the Hugging Face card, project site, and MIT-licensed code repo). The project cites Hou et al. 2024 (arXiv:2403.03952) and does not redistribute the raw files.

This section is a feature. It demonstrates the same evaluation honesty you already showed catching data leakage in a prior model, and technical interviewers trust it more than a project that claims to be flawless.

---

## 10. README Outline (build during Phase 6)

1. One-line description and the live demo link.
2. Why this project (the P&G framing, in two or three sentences).
3. Architecture diagram (data -> clean -> EDA / index -> retrieve -> generate -> app).
4. What is in it: the EDA layer and the RAG layer, one paragraph each.
5. Key EDA insights (three or four bullets with a chart or two).
6. Evaluation summary (the honest numbers).
7. Limitations (from Section 9).
8. How to run locally (env, install, download data, launch app).
9. Tech stack.

---

## 11. Definition of Done Checklist

- [ ] Repo public on GitHub, clean commit history, sensible README.
- [ ] `01_eda.ipynb` runs clean with written insights.
- [ ] RAG pipeline returns cited answers; empty/rate-limit cases handled.
- [ ] `gold_questions.json` + scored `results.md` committed.
- [ ] Streamlit app live with a working public URL.
- [ ] Limitations documented.
- [ ] A short paragraph drafted that maps this project to the P&G JD language, ready to paste into the application.

---

## 12. Stretch Goals (only after Definition of Done, or as a follow-up build)

- **Agentic layer:** let the system decide when to filter by brand vs rating, or run a compare-brands tool, as multi-step tool use. This directly matches the JD's "AI agents" wording and is the natural next project if RAG lands early.
- **Lightweight topic modeling** on complaint themes for richer EDA.
- **A small automated faithfulness check** using a second LLM call as a grader.

---

## 13. Open Questions to Resolve on Day 1

All three were resolved on 2026-09-15.

### 13.1 Which exact dataset, and does it clear the volume bar?

**Answer:** Amazon Reviews 2023 (Hou et al. 2024, arXiv:2403.03952), Hugging Face repo `McAuley-Lab/Amazon-Reviews-2023`, category `Beauty_and_Personal_Care`. It clears the bar in Section 5.3 by roughly 10x to 30x per brand (see 13.2).

- **Files:** `raw/review_categories/Beauty_and_Personal_Care.jsonl` (11,021,458,876 bytes, 23,911,390 reviews) and `raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl` (2,835,194,976 bytes, 1,028,914 products). Reviews join to metadata on `parent_asin`; the brand lives in the metadata `store` field.
- **Rejected:** the `All_Beauty` subset. It has only 13 / 70 / 34 matching products (1,601 / 1,700 / 663 ratings) for Head & Shoulders / Pantene / Herbal Essences, which is too thin for per-brand trends.
- **Consequence for Phase 1:** the files are too large to store whole, so `data/download_data.py` should stream and filter to the brand set rather than download everything into `data/raw/`.
- **Dataset terms caveat:** neither the Hugging Face dataset card, the project site, nor the MIT-licensed code repository states terms of use for the data itself. The project cites Hou et al. 2024 and lists this under limitations (Section 9); it does not redistribute the raw files.

### 13.2 Final brand set confirmed against that data

**Answer:** Head & Shoulders, Pantene, Herbal Essences (haircare). The fallback categories are not needed.

Measured on 2026-09-15 with a full streaming pass over both files (nothing stored). A product belongs to a brand when its `store` or `title` matches the case-insensitive pattern in `src/config.py` (`(?i)head\s*(?:&|and|n'?)\s*shoulders`, `(?i)pantene`, `(?i)herbal\s+essence`). A "written review" has non-empty `text`. The star, verified-purchase, and date-range columns count written reviews only; dates are UTC.

| Brand | Products (store match) | Reviews in file | Written reviews | 1★ | 2★ | 3★ | 4★ | 5★ | Verified purchase (written) | Date range (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|
| Head & Shoulders | 402 (362) | 9,875 | **9,868** | 821 | 404 | 609 | 1,353 | 6,681 | 7,544 | 2005-06-28 to 2023-08-31 |
| Pantene | 1,426 (1,411) | 29,692 | **29,670** | 2,548 | 1,250 | 1,854 | 3,478 | 20,540 | 23,667 | 2005-03-04 to 2023-09-01 |
| Herbal Essences | 683 (612) | 16,339 | **16,325** | 1,413 | 815 | 1,210 | 2,121 | 10,766 | 11,693 | 2005-01-05 to 2023-08-31 |
| **Total** | 2,511 | 55,906 | **55,863** | 4,782 | 2,469 | 3,673 | 6,952 | 37,987 | 42,904 | |

Notes on the evidence:

- These are pre-cleaning counts. Cleaning (dedup, empty-text drops) will lower them; the Phase 1 quality report gives the final numbers.
- The metadata `rating_number` field sums to 227,681 / 270,991 / 171,707. That is Amazon's total rating count, which includes star-only ratings and reviews the dataset does not contain, so it overstates the available written reviews by roughly 9x to 23x per brand (12x overall). The written-review column is the number that matters.
- One product matched two brand patterns and was assigned to the brand named in its `store` field. That is why Head & Shoulders shows 402 products here versus 403 in an earlier store-or-title count.
- Title matches can include non-brand listings (for example a third-party product that names a brand in its title). Products matched only by title, not `store`, are 40 / 15 / 71. Story 2 should decide whether to keep title-only matches.
- Ratings skew heavily positive (66% to 69% five-star per brand), but each brand still has at least 1,200 written 1–2★ reviews for the complaint analysis.
- An independent single-stream pass with a separate script (which counted the one multi-brand product under both brands) agreed: Pantene 29,670 and Herbal Essences 16,325 written reviews exactly, Head & Shoulders 9,872 (+4, the multi-brand product).
- Aussie (522 products / 185,453 ratings in metadata; 16,618 written reviews in that independent pass) is a viable fourth haircare brand but exceeds the 2–3 brand scope.

### 13.3 Current Groq model string confirmed and set in `config.py`

**Answer:** `openai/gpt-oss-120b`, set once as `GROQ_MODEL` in `src/config.py` with `GROQ_TEMPERATURE = 0.1`.

- **Why Llama was dropped:** Groq's [deprecations page](https://console.groq.com/docs/deprecations) lists its Llama chat models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) as shut down on 2026-08-16, and this key's model list contains no Llama chat model. A call to `llama-3.3-70b-versatile` now returns HTTP 404 ("does not exist or you do not have access to it"). The maintainer chose `openai/gpt-oss-120b`, which Groq names as the replacement; stronger grounding is the expectation behind the choice and is measured in Phase 5, not assumed.
- **Accepted trade-off:** it is a reasoning model. Its reasoning tokens count against the free-tier 8K tokens/min cap: 121 on a trivial call during model selection, and 53 of 66 completion tokens in the smoke test below. Phase 4 should budget for this alongside the 30 requests/min limit (developer-plan limits for all three candidates: 30 requests/min, 1K requests/day, 8K tokens/min, 200K tokens/day; [rate limits](https://console.groq.com/docs/rate-limits)).
- **Smoke test (2026-09-15):** `uv run python -m src.groq_smoke` returned HTTP 200, reply `'hello from Groq'`, finish reason `stop`, exit 0. With `GROQ_API_KEY` blank it makes no call and exits 1.
