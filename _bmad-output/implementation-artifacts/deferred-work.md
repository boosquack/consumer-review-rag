- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/1-dataset-discovery-brand-lock-groq-smoke-test.md`
  summary: Run `bmad-spec` so SPEC.md (Open Questions 1–3, Assumptions) and stack.md (Generation row) follow PLAN.md's locked dataset, brand set, and `openai/gpt-oss-120b` model instead of "current Llama instruct model".
  evidence: Story 1 locked these decisions in PLAN.md §3 and §13. `SPEC.md` and `stack.md:5` still describe a Llama model, which Groq shut down on 2026-08-16. AGENTS.md forbids hand-editing the spec, so stories 2–7 read a stale model decision until `bmad-spec` runs.
  status: resolved 2026-09-16 by a `bmad-spec` update run. SPEC.md now carries the locked brand set, corpus, and `openai/gpt-oss-120b` as constraints; stack.md:5 names the model; data-schema.md's sourcing section was re-derived to the locked source. Only the near-duplicate rule and the chunk-split threshold remain open, owned by stories 2 and 4.

- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/2-download-cleaning-pipeline-quality-report.md`
  summary: Run `bmad-spec` so `data-schema.md` records the cleaning rules story 2 actually implemented: brand match on `store` only (not `store` or `title`), and dedup on casefolded, whitespace-collapsed text gated at >=50 characters (not exact `review_text`).
  evidence: `data-schema.md` still says a product belongs to a brand when its `store` **or** `title` matches, and that dedup is on exact `review_text`. The schema doc explicitly delegated the title-only decision to the cleaning story ("Deciding whether to keep them belongs to the cleaning story"), and the maintainer's frozen decision settled it store-only, dropping the 40/15/71 title-only products. AGENTS.md forbids hand-editing the spec, so stories 3-7 read superseded cleaning rules until `bmad-spec` runs.

- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/2-download-cleaning-pipeline-quality-report.md`
  summary: Drop the stale "TODO: created in story 2" marker for `data/download_data.py` in AGENTS.md and record the two-command pipeline (`uv run python data/download_data.py`, then `uv run python -m src.clean`).
  evidence: AGENTS.md line 14 still marks the file as not yet created; it exists and ran a full pass. Deferred rather than patched because the fix edits an agent-context file, which this workflow routes to defer by rule.

- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/2-download-cleaning-pipeline-quality-report.md`
  summary: Bound the line buffer in `stream_lines` so a newline-free body cannot accumulate the whole 11 GB source in memory.
  evidence: `data/download_data.py` builds `remainder + chunk` per chunk and only splits on `\n`. A 200 response carrying a newline-free body (CDN interstitial, truncated gzip) would grow that buffer without limit, defeating the frozen "never store a source file whole" constraint. Not reproduced against the real source, which is well-formed JSONL; a max-line guard would settle it.

- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/2-download-cleaning-pipeline-quality-report.md`
  summary: Split the `undated` bucket by cause (missing timestamp vs out-of-range) so a units regression is visible instead of silent.
  evidence: `parse_timestamp_ms` maps null, 0, non-numeric, and out-of-bounds years all to None with no per-cause counter. The measured run reports `undated: 0`, so the bucket has no real-data coverage at all. If an upstream subset ever ships seconds instead of milliseconds, every row would land in 1970, fall outside the year bounds, and surface as a silent undated spike rather than a reported drop reason.

- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/2-download-cleaning-pipeline-quality-report.md`
  summary: Give the download pass resume support: separate `--refresh-meta` / `--refresh-reviews`, and an HTTP Range restart from the `.part` byte offset.
  evidence: `--refresh` currently re-streams all 2.8 GB of metadata even when only the reviews pass failed, and there is no resume. A real interruption at 51,645 matches forced a full restart this session. `StreamInterrupted` already carries `bytes_read`, so the offset is available.

- source_spec: `_bmad-output/specs/spec-review-intelligence/stories/2-download-cleaning-pipeline-quality-report.md`
  summary: Consider a test pinning the git-ignore rules for the two generated artifacts, so an edit to `.gitignore` cannot quietly make the corpus committable.
  evidence: `git check-ignore -v` resolves both `data/processed/reviews_clean.parquet` and `data/processed/quality_report.json` to `.gitignore:17`, and `git ls-files data/` returns only the two `.gitkeep` files plus the tracked script, but nothing in the suite asserts it; the fourth acceptance criterion is verified only by hand.
