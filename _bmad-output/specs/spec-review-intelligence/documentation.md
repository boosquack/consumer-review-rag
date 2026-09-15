# Documentation deliverables

## README outline (in order)

1. One-line description and live demo link.
2. Why this project: the P&G framing in 2–3 sentences.
3. Architecture diagram (see `architecture-diagrams.md`).
4. What's in it: one paragraph on the EDA layer, one on the RAG layer.
5. Key EDA insights: 3–4 bullets with one or two charts.
6. Evaluation summary with the honest numbers.
7. Limitations (below).
8. How to run locally: env, install, download data, launch app.
9. Tech stack.

Write the problem statement paragraph into the README during setup, before building, to lock the framing.

## Limitations (required, stated in the README)

- The reviews are a public dataset, not P&G internal data, and may be dated or incomplete per brand.
- Retrieval can miss relevant reviews or surface loosely related ones; report the measured hit rate.
- The model can drift from the sources despite grounding; report where it did during evaluation.
- VADER is a lexicon heuristic, not a trained classifier: directionally useful, not authoritative.
- Findings describe the sampled reviews, not the true customer population.

## Application paragraph

A short paragraph mapping the project to P&G JD language ("AI agents / AI-powered digital business solutions", deriving business insight from data), ready to paste into the application.

## Definition of Done

- Repo public on GitHub with a clean commit history and a sensible README.
- `01_eda.ipynb` runs clean with written insights.
- The RAG pipeline returns cited answers and handles empty retrieval and rate limits.
- `gold_questions.json` and a scored `results.md` are committed.
- The Streamlit app is live at a working public URL.
- Limitations are documented.
- The application paragraph is drafted.
