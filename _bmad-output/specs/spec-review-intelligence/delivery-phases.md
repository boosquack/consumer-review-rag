# Delivery phases and exit checks

| Phase | When | Scope | Exit check |
|---|---|---|---|
| 0 Framing & setup | Day 1 AM | Locate dataset and finalize brands against it; scaffold repo, venv, requirements, `.gitignore`, `.env.example`; confirm Groq model string with a trivial call; write problem statement into README | Repo scaffolded, Groq hello-world returns, dataset located |
| 1 Data | Day 1 PM | `download_data.py`, `clean.py` → parquet; quality report; decide de-dup and undated handling | Clean parquet exists, quality report captured in notebook |
| 2 EDA | Day 2 | Charts in `eda-charts.md` with takeaways; insights cell | `01_eda.ipynb` runs clean, insights written |
| 3 Index & retrieval | Day 3 | `index.py`, `retrieve.py` per `rag-design.md`; tune k | Retrieval returns sensible reviews on hand-tested queries |
| 4 Generation | Day 4 | `generate.py`, grounding/citation prompt, empty-retrieval and rate-limit handling | Question in → cited answer out with visible sources |
| 5 Evaluation | Day 5 | Gold set, `evaluate.py`, `results.md` per `evaluation.md` | Metrics computed, write-up includes failures |
| 6 Deploy & docs | Day 6 | Streamlit app, Cloud deploy with Secrets, README per `documentation.md` | Live URL works, README complete, everything committed |

Day 7 is buffer. Slip priority: evaluation and limitations > deployment > extra EDA charts > extra brands.
