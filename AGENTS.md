<!-- bmad:context -->
<!-- Verified 2026-09-15 against c9f890c. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## consumer-review-rag

One-week P&G AI Engineering internship portfolio project: EDA plus a cited RAG app over public consumer reviews of P&G haircare brands. Python 3.11, sentence-transformers, ChromaDB, Groq, Streamlit Community Cloud. The narrative plan is `PLAN.md`; the build contract is `_bmad-output/specs/spec-review-intelligence/`.

## Policy

- Never commit or push unless the maintainer asks in the session; leave changes uncommitted for review.
- Never hardcode or log `GROQ_API_KEY`; read it from `.env` locally or Streamlit Secrets when deployed, and add new key names to `.env.example`.
- Never hand-edit `SPEC.md`, its companions, `stories.yaml`, or `.memlog.md` in the spec folder; update them by running `bmad-spec`.
- When a decision changes (dataset, brand set, Groq model string), update `PLAN.md` and run `bmad-spec` so the spec follows.
- Keep raw datasets in git-ignored `data/raw/`; make them reproducible through `data/download_data.py` (TODO: created in story 2).
- Report measured evaluation numbers, failures included; never claim retrieval or faithfulness works without them.

## Where things are

- Story order, checkpoints, and dev notes: `_bmad-output/specs/spec-review-intelligence/stories.yaml`; implement one story per `bmad-build` run.
- Working on `src/clean.py` or `data/`: read `data-schema.md` in the spec folder first.
- Working on `src/index.py`, `src/retrieve.py`, or `src/generate.py`: read `rag-design.md` first.
- Working on `eval/` or `src/evaluate.py`: read `evaluation.md` first.

## Running and verifying

- Run Python with `uv run python …`; bare `python3` is the system 3.9.6, not the project's 3.11 `.venv`.
- Install packages with `uv pip install`; `.venv` has no `pip`.
- TODO (stories 3 and 7, verify on first refresh): launch the notebook with `uv run jupyter lab` and the app with `uv run streamlit run app/streamlit_app.py`.

## Conventions that differ from defaults

- Put cleaning, indexing, retrieval, and generation logic in `src/`; `notebooks/` and `app/` import it and never reimplement it.
- Define model names, paths, and constants only in `src/config.py`.
- Keep the original `review_text` untouched for display and citations; embed a separate normalized copy.
- Keep undated reviews in an `undated` bucket instead of dropping them.

<!-- /bmad:context -->
