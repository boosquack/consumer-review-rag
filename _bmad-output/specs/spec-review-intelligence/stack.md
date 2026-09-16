# Stack

| Concern | Choice | Reason |
|---|---|---|
| Generation | Groq `openai/gpt-oss-120b`, temperature 0.1 | Free tier, known rate limits (30 req/min). Groq shut down its Llama chat models on 2026-08-16 and names this as the replacement. Reasoning model: its reasoning tokens count against the 8K tokens/min cap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local | Free, 384-dim, fast on Apple Silicon CPU/MPS, no second key |
| Vector store | ChromaDB persistent client, local directory (`.chroma/`) | Fastest solo setup, persists to disk, loads in Streamlit |
| Sentiment (EDA) | `vaderSentiment` | Pretrained lexicon heuristic; no model training |
| Data / EDA | pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter | TF-IDF / term frequency, charts |
| App | Streamlit, deployed on Streamlit Community Cloud | Free, GitHub-connected, built-in secrets |
| Config | python-dotenv; `src/config.py` holds model names, paths, constants | Single source for model string |
| Runtime | Python 3.11 | Streamlit Cloud support; avoid 3.13 |

## Dependencies

`requirements.txt` starts from the unpinned minimums (pandas>=2.2, numpy>=1.26, matplotlib>=3.8, seaborn>=0.13, jupyter>=1.0, scikit-learn>=1.4, vaderSentiment>=3.3.2, sentence-transformers>=3.0, chromadb>=0.5, groq>=0.11, streamlit>=1.38, python-dotenv>=1.0). Once the environment works, pin exact versions by freezing, so Streamlit Cloud builds match local.

## Secrets

- `GROQ_API_KEY`: local `.env` (git-ignored) and Streamlit Cloud Secrets. `.env.example` lists the name with no value.

## Git-ignored

`.env`, `__pycache__/`, `*.pyc`, `.chroma/`, `data/raw/`, `.ipynb_checkpoints/`.

## Repository layout

```
consumer-review-rag/
├── README.md, PLAN.md, LICENSE, requirements.txt, .env.example, .gitignore
├── data/
│   ├── raw/                     # git-ignored
│   ├── processed/reviews_clean.parquet
│   └── download_data.py         # reproducible fetch + brand filter
├── notebooks/01_eda.ipynb       # Data Analyst deliverable
├── src/
│   ├── config.py                # model names, paths, constants
│   ├── clean.py                 # cleaning pipeline (importable)
│   ├── index.py                 # chunk + embed + write to Chroma
│   ├── retrieve.py              # query -> top-k review chunks
│   ├── generate.py              # context + question -> cited answer
│   └── evaluate.py              # runs gold set, prints metrics
├── eval/gold_questions.json, eval/results.md
└── app/streamlit_app.py
```
