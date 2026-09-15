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
| Brand set | Head & Shoulders, Pantene, Herbal Essences (haircare) | Same category means clean like-for-like comparison. All three are P&G. Swap any if the dataset is thin. |
| Category fallback | Grooming (Gillette, Oral-B) or skincare (Olay) | Use only if haircare review coverage is weak in the chosen dataset. |
| Data source | Existing Kaggle / public reviews dataset, filtered to P&G brands | Scraping a live retailer risks ToS friction and can eat a full day. See Section 5. |
| Generation model | Groq, current Llama instruct model | You already know Groq's behavior and rate limits from MamaBot. Free tier. |
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
- Confirm the current Groq model string before coding (model names change); set it once in a config constant, not scattered through the code.

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

- Which exact dataset, and does it clear the minimum volume bar in Section 5.3 for the chosen brands?
- Final brand set confirmed against that data.
- Current Groq model string confirmed and set in `config.py`.
