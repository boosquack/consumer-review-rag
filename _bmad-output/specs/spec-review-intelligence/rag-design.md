# RAG design

## Indexing

- Embed each review whole; split only very long reviews (threshold is an open question).
- Batch-embed with `all-MiniLM-L6-v2`; cache the model load.
- Write to the persistent ChromaDB store with metadata `brand`, `rating`, `product_name`, `review_date`, `review_id`.
- Persist the index so the app never re-embeds on launch.

## Retrieval

- Semantic top-k with an optional metadata pre-filter on brand and rating band. For example, "what do 1-star Pantene reviewers complain about" must be filterable, not purely semantic.
- Start at k=6 and tune within 5–8.
- Test retrieval on its own before building generation: eyeball 5–10 queries for relevance.

## Generation

- Pass the question plus retrieved reviews to Groq, with stable indices that map back to real source rows.
- The system prompt enforces three rules: answer only from the supplied reviews; cite reviews by index; say "the reviews do not cover this" when they don't.
- Use a low temperature.
- Return the source reviews with the answer so the UI can show brand, rating, and snippet next to it.
- Treat the prompt as a real artifact and iterate on it.

## Failure modes handled in code

- Empty retrieval: an explicit "no relevant reviews found" path, with no model call on empty context.
- Groq rate limit (30 req/min): simple backoff.
- The model ignoring grounding instructions: caught and reported in evaluation.
