# consumer-review-rag

Exploratory analysis and a cited question-answering app over public Amazon reviews of three P&G haircare brands (Head & Shoulders, Pantene, and Herbal Essences), drawn from the Amazon Reviews 2023 dataset (Hou et al. 2024, arXiv:2403.03952).

## Why this project

P&G brand teams need to know what shoppers praise and complain about, with answers they can check. This project builds two layers on one corpus of public Amazon reviews: an EDA layer that turns ratings, volume, sentiment, and recurring themes into analyst-ready insight per brand, and a retrieval-augmented generation (RAG) layer that answers natural-language questions using only the retrieved reviews. Every answer cites its sources by brand, rating, and snippet, and the system declines when the reviews do not cover the question.
