---
id: SPEC-review-intelligence
companions:
  - stack.md
  - data-schema.md
  - eda-charts.md
  - rag-design.md
  - evaluation.md
  - documentation.md
  - architecture-diagrams.md
  - delivery-phases.md
sources:
  - ../../../PLAN.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Consumer Review Intelligence for P&G Haircare Brands

## Why

An opportunity and a deadline. Kent Joshua P. Alvarez is applying for P&G's AI Engineering Intern role and needs a portfolio piece that reads as "built for P&G," not a reused generic dataset. One corpus of public consumer reviews for P&G brands supports two layers. The EDA layer delivers analyst-ready insight, matching P&G Consumer & Market Knowledge and the JD's line about deriving business insight from data. The grounded RAG layer gives a brand team a cited answering tool, matching the JD's "AI agents / AI-powered digital business solutions." Evaluation honesty is part of the pitch.

## Capabilities

- **CAP-1**
  - **intent:** A reproducible pipeline produces a cleaned review corpus for 2–3 P&G brands in the target schema, with a data quality report.
  - **success:** Running the download and clean scripts yields `data/processed/reviews_clean.parquet`. The report shows total rows, rows per brand, date range, and percent dropped with reasons (see `data-schema.md`).
- **CAP-2**
  - **intent:** An analyst can see rating, volume, sentiment, and theme patterns per brand, each with a written insight.
  - **success:** `notebooks/01_eda.ipynb` runs top to bottom with no errors. It produces every chart in `eda-charts.md`, each with a 1–2 sentence takeaway.
- **CAP-3**
  - **intent:** A natural-language question retrieves the most relevant reviews, optionally filtered by brand and rating band.
  - **success:** 5–10 hand-tested queries return relevant reviews, filters restrict the results, and the persisted index loads without re-embedding.
- **CAP-4**
  - **intent:** Answers use only the retrieved reviews, cite every source (brand, rating, snippet), and decline when the reviews don't cover the question.
  - **success:** Every answer's citations map to real review rows. An unanswerable question returns a refusal. Empty retrieval and Groq rate limits are handled without a crash.
- **CAP-5**
  - **intent:** System quality is measured against a hand-written gold question set.
  - **success:** `src/evaluate.py` prints retrieval hit rate, faithfulness pass/fail, and refusal results for 15–20 questions. `eval/results.md` records the numbers and commentary on failures (see `evaluation.md`).
- **CAP-6**
  - **intent:** Anyone can use a public live demo that shows both the RAG and EDA layers.
  - **success:** The Streamlit Community Cloud URL works in a fresh browser. It has a query box, brand and rating filters, a cited answer, expandable sources, and 1–2 EDA charts.
- **CAP-7**
  - **intent:** A reviewer understands the project, its results, and its limits, and how it ties to the P&G application.
  - **success:** The README follows the outline in `documentation.md`, including the architecture diagram, eval numbers, and limitations. A paragraph mapping the project to P&G JD language is drafted.

## Constraints

- One-week timebox (week of 2026-09-15). If a day slips, protect evaluation and limitations first, then deployment, then extra EDA charts, then extra brands.
- P&G brands only, preferably from one category. Finalize the brand set only after inspecting what the dataset actually has.
- Use an existing public dataset. Scrape only if public data fails, and time-box scraping to half a day at most.
- Zero-cost stack: Groq free tier (30 requests/min, so backoff is required), local embeddings with no second API key, and Streamlit Community Cloud hosting.
- Python 3.11. Not 3.13, because of ML wheel gaps and Streamlit Cloud support.
- Never commit `GROQ_API_KEY`. It lives in a git-ignored `.env` locally and in Streamlit Secrets for deployment; `.env.example` documents the key name.
- Pipeline logic lives in importable modules under `src/`. The notebook and the app call the same functions, and the app never duplicates retrieval logic.
- The Groq model string is defined once, in `src/config.py`.
- Keep original review text intact for display and citation. Normalized text is stored separately and used for embedding.
- Limitations are a required deliverable. Report measured numbers and never claim the system works without evidence.
- No code or data from Naya or MamaBot.

## Non-goals

- Multi-step agentic tool use or planning.
- Fine-tuning any model.
- A custom-trained sentiment model (use pretrained VADER).
- Claims of production readiness.
- Scraper-first data sourcing.
- Statistical generalization from sampled reviews to the customer population.
- Stretch work, only after the Definition of Done: an agentic brand/rating router with a compare-brands tool, lightweight topic modeling, and an LLM-grader faithfulness check.

## Success signal

- A reviewer opens the public URL and asks something like "what do 1-star Pantene reviewers complain about?" They get an answer that cites real reviews by brand, rating, and snippet. The repo also shows an EDA notebook that runs clean and an `eval/results.md` with measured hit rate, faithfulness, and refusal numbers, failures included.

## Assumptions

- The working brand set is Head & Shoulders, Pantene, and Herbal Essences. If haircare coverage is thin, fall back to grooming (Gillette, Oral-B) or skincare (Olay).
- The Amazon Reviews Beauty/Personal Care subsets are the most likely source of P&G-brand rows.
- The machine is an Apple Silicon MacBook; embeddings run on CPU or MPS with no GPU.

## Open Questions

- Which dataset will be used, and does it clear the volume bar (a few hundred reviews per brand, ideally 1,000+)?
- What is the final brand set once the data has been inspected?
- What is the current Groq Llama instruct model string?
- Beyond exact-text dedup, what rule handles near-duplicate reviews? Decide in the data story.
- Above what length is a review split into chunks?
