---
title: 'Download and cleaning pipeline with data quality report'
type: 'feature'
created: '2026-09-16'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '3b6caf6d6dec9777d947493e6aaebcc65e01e2f0'
context:
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/data-schema.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The repo has no data. CAP-1 needs a reproducible cleaned corpus, and every later story reads `data/processed/reviews_clean.parquet`, which does not exist. The source is 11.0 GB of reviews plus 2.8 GB of product metadata on Hugging Face — far too large to store in the repo.

**Approach:** Two streaming passes, no new dependency. A metadata pass maps `parent_asin` → canonical brand; a reviews pass filters to those products into a git-ignored JSONL under `data/raw/`. `src/clean.py` turns that into the parquet in the target schema and emits the quality report.

## Boundaries & Constraints

**Always:** Stream and filter; never store a source file whole. Take paths, brand patterns and dataset constants from `src/config.py`, and add new constants there. Keep `review_text` byte-identical to the source `text` for display and citation; put the normalized embedding copy in its own column. Keep undated reviews in an `undated` bucket. Report measured counts with a reason per drop. Leave changes uncommitted.

**Never:** Commit any data. Add a dependency — `requests`, `huggingface-hub`, `tqdm`, `pandas` and `pyarrow` are installed, so `datasets` is not needed. Write EDA, notebook, indexing, retrieval or generation code (stories 3–5). Hand-edit `SPEC.md`, its companions, `stories.yaml` or `.memlog.md`.

**Decisions (maintainer, 2026-09-16):**
- **Dedup:** drop repeat `review_id`; drop duplicate normalized `review_text` (casefolded, whitespace-collapsed) only when that normalized text is ≥50 characters. Shorter reviews are exempt, so genuine short praise survives in a corpus that is 66–69% 5★.
- **Brand match:** assign brand from the metadata `store` field only, dropping the 40 / 15 / 71 title-only products, because third-party listings that merely name a brand would pollute attribution in every later story. A product matching two patterns takes the brand named in `store`.
- **Run scope:** execute the full pass this session (~25–35 min at a measured 9.4 MiB/s) so the story ends with the real parquet and a measured report.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | Both source files reachable | Filtered JSONL in `data/raw/`, parquet in `data/processed/`, report with per-brand counts | N/A |
| Cached raw | Filtered JSONL already present | Reuse it, skip the network pass; `--refresh` re-fetches | N/A |
| Bounded run | `--limit N` | Same pipeline over N matches; report labels the output a sample | N/A |
| Stream interrupted | Connection drops mid-pass | Exit non-zero naming bytes read; partial output stays `.part`, never promoted | Caught → stderr |
| Unknown product | `parent_asin` absent from the brand map | Skipped, counted `no_brand_match` | N/A |
| Empty body | `text` empty or whitespace | Dropped, counted `empty_text` | N/A |
| Missing date | `timestamp` null, 0, or outside 2000–2026 | `review_date` null, `date_bucket="undated"`, row kept | N/A |
| Duplicate | Repeat `review_id`, or normalized text ≥50 chars seen before | Dropped, counted per rule | N/A |

</frozen-after-approval>

## Code Map

- `src/config.py` -- Has `PROJECT_ROOT`, `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `BRANDS` (inline `(?i)`, no capture groups), `DATASET_*`. Add: resolve-URL base, `RAW_FILTERED_JSONL`, `CLEAN_PARQUET`, date bounds, dedup length. Do not touch the brand patterns or Groq constants.
- `data/download_data.py` -- New. Two `requests` streaming passes with `tqdm` against known byte sizes. Run `uv run python data/download_data.py` from the repo root (it imports `from src import config`; `src/` is a PEP 420 namespace package with no `__init__.py`).
- `src/clean.py` -- New. Importable functions over an iterable of raw records, plus an entry point that writes the parquet and returns the report. Stories 3+ import this; nothing reimplements it.
- `src/groq_smoke.py` -- Conventions only, do not change: docstring naming the run command, `main() -> int`, `sys.exit(main())`, errors to stderr.
- `tests/test_groq_smoke.py` -- Conventions: stdlib `unittest`, no network, mocks. New `tests/test_clean.py` follows it.
- `.gitignore`, `requirements.txt` -- No change; `data/raw/*` is already ignored. Installed `pandas` is 3.0.5 against `>=2.2` declared, with `infer_string=True`, so string columns land as `str` dtype.

**Verified source fields (2026-09-16, ranged read):** reviews carry `rating` (float), `title`, `text`, `parent_asin`, `asin`, `user_id`, `timestamp` (epoch **ms**, UTC), `helpful_vote`, `verified_purchase` — and no review id. Metadata carries `store`, `title`, `parent_asin`.

## Tasks & Acceptance

**Execution:**
- [x] `src/config.py` -- Add the paths, URL base, date bounds and dedup length -- AGENTS.md: constants live only here.
- [x] `data/download_data.py` -- Metadata pass → brand map; reviews pass → `.part` → final JSONL; `--limit`, `--refresh`, progress, cached skip -- streams instead of storing 13.8 GB.
- [x] `src/clean.py` -- Target schema, deterministic `review_id`, ms timestamps with the `undated` bucket, empty-text drop, the approved dedup, the separate normalized column, parquet, report -- CAP-1.
- [x] `tests/test_clean.py` -- Cover every matrix row with inline fixtures and no network -- the matrix requires covering tests.
- [x] Run the full pipeline and capture the measured report -- the exit check needs real numbers.

**Acceptance Criteria:**
- Given the parquet, when loaded, then it carries the `data-schema.md` target columns, `review_id` is unique, `rating` is an int in 1–5, and every `brand` is one of the three canonical names.
- Given identical input, when `src/clean.py` runs twice, then the `review_id` values are identical.
- Given the report, when read, then it shows total rows, rows per brand, date range, undated count, and percent dropped with a reason per drop class.
- Given `git status`, when inspected, then no file under `data/` is tracked beyond the two `.gitkeep` files.

## Implementation Notes

- **Files:** `src/config.py` (constants added), `data/download_data.py`, `src/clean.py`, `tests/test_clean.py` (all new), `.gitignore` (one rule). No dependency added; no data committed.
- **Measured full run (2026-09-16):** metadata pass matched **2,385 products** on `store` (cached to `data/raw/brand_map.json`). Reviews pass scanned **23,911,390** reviews in ~15 min at ~13 MB/s and kept **54,741** into a 32 MB JSONL. Story 1's pre-cleaning estimate was 55,863; the 1,122-row gap is expected, since story 1 counted written reviews per brand pattern while this pass joins on `parent_asin` through the store-only brand map.
- **Measured quality report:** 54,741 in → **53,872 out**, 869 dropped (**1.59%**) — `empty_text` 43 (0.08%), `duplicate_review_id` 456 (0.83%), `duplicate_text` 370 (0.68%), `no_brand_match` 0, `invalid_rating` 0. Per brand: Head & Shoulders **9,363**, Pantene **28,977**, Herbal Essences **15,532** — all well past the 1,000-per-brand quality bar. Date range 2005-01-05 .. 2023-09-01 UTC, **undated 0**. Parquet is 14.5 MB.
- **`no_brand_match` is structurally 0** because `download_data.py` only writes rows whose `parent_asin` is in the brand map. The check stays in `clean.py` as a guard for hand-fed or future inputs, and the report prints the measured 0 rather than hiding the class.
- **Undated 0** means the `undated` bucket is untested against real data; only the inline fixtures exercise it. Every source `timestamp` parsed inside 2000–2026.
- **Corpus skew:** 68.1% five-star (1★ 8.5%, 2★ 4.4%, 3★ 6.5%, 4★ 12.4%), mean rating 4.22–4.29 across the three brands. This is the skew the ≥50-char dedup exemption was chosen to protect, and stories 3 and 6 need to account for it.
- **Deviation from the Code Map:** it said `.gitignore` needed no change, but only `data/raw/*` was ignored, so the new `data/processed/reviews_clean.parquet` showed up as untracked and one `git add -A` from being committed. Added `data/processed/*` with a `!.gitkeep` negation, matching the raw rule. This serves the frozen "Never commit any data" constraint.
- **Interrupted-stream path was exercised for real:** an earlier attempt died at 51,645 matches; the `.part` was never promoted and the rerun started clean, as the matrix requires.
- **`.gitignore` also carries `_bmad/render/`,** which is *not* this story's change: the render cache and its ignore line predate this session and ride along in the uncommitted tree, so they appear in the story's diff. Flagged by review finding B11 and recorded here rather than silently absorbed.

**Review pass 1 patches (2026-09-16).** Applied by the orchestrator; no implementation subagent existed to re-engage. 38 findings triaged, nothing routed to intent_gap or bad_spec, so no loopback and `review_loop_iteration` stays 0.
- `requirements.txt` -- declared `requests`, `tqdm`, `pyarrow`. These were already installed, so this declares reality rather than adding a dependency; without it a fresh clone or the story 7 deploy fails on import.
- `src/clean.py` -- malformed source values now cost one counted drop instead of aborting the pass: non-finite ratings, non-string bodies, unhashable brands, non-object JSON lines, and undecodable bytes. Three of these were reproduced as live crashes before the fix.
- `src/clean.py` -- the report is persisted to `data/processed/quality_report.json` (new `config.CLEAN_REPORT_JSON`); the sample label is read from the manifest only for the configured raw input; a failed re-clean warns that an existing parquet is stale rather than deleting it.
- `data/download_data.py` -- the cache honours scope (a bounded sample no longer satisfies a full run); a body that ends early is an interruption, not a success; a corrupt brand map rebuilds instead of tracebacking and is written `.part`-then-promote; a zero-match pass exits 1 without promoting an empty corpus; unmapped reviews are counted and reported.
- `tests/test_clean.py` -- 30 to 52 tests. New coverage for `clean.main()`'s exit codes, manifest-derived sample labelling, cached brand-map reuse, the empty-brand-map guard, the malformed-record guards, cache scope, short reads, zero matches, and the multi-brand tiebreak. Two pre-existing tests changed: the cached-raw test now writes a manifest (it had asserted a zero-byte JSONL was a valid cache), and a loose `assertNotIn("cached", ...)` was tightened after it matched the unrelated brand-map line.
- **Re-verified after patching:** 52 tests OK, and the full corpus re-cleaned to byte-identical numbers — 54,741 in, 53,872 out, 1.59% dropped, same per-brand counts and date range. The guards changed no measured result.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-16). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V, `VO` = its Other findings). 38 findings, no loopback: nothing routed to intent_gap or bad_spec.

| ID | Finding | Verdict | Route | Evidence |
|---|---|---|---|---|
| B1+E18 | `requirements.txt` declares none of `requests`, `tqdm`, `pyarrow` | medium | patch | Confirmed by reading the file. The Code Map's "no change" reasoning held for this `.venv` only; a fresh clone or the story 7 Streamlit deploy would fail on import, and `pandas>=2.2` does not pull `pyarrow` transitively. Added all three. |
| B3+B14+E14+E15 | Cache check is existence-only, so a `--limit` sample is served as a full corpus | medium | patch | Confirmed: `main()` returned on `exists() and not refresh` without reading the manifest. Added `_cache_covers()`; a full cache still satisfies a bounded request, never the reverse. The old test asserted a zero-byte JSONL was a valid cache and now writes a manifest instead. |
| E1 | NaN/infinity rating crashes the run | medium | patch | Reproduced: `ValueError: cannot convert float NaN to integer`. One malformed row aborted a 54,741-row pass. Now screened by `math.isfinite` and counted `invalid_rating`. |
| E2 | Non-string `text` crashes the run | medium | patch | Reproduced: `TypeError: expected string or bytes-like object, got 'int'`. Now type-checked and counted `empty_text`. |
| E3 | Unhashable `brand` (JSON list) crashes the run | medium | patch | Reproduced: `TypeError: unhashable type: 'list'` on the `in config.BRANDS` test. Now `isinstance(brand, str)` first, counted `no_brand_match`. |
| E4 | A valid-JSON non-object line crashes `clean_records` | medium | patch | `read_jsonl` yielded lists/scalars, so `record.get` raised `AttributeError`. Now only dicts are yielded. |
| E5 | A non-UTF-8 byte raises past the `JSONDecodeError` handler | low | patch | Real: `UnicodeDecodeError` is raised by the file iterator, not by `json.loads`. Direct correction: `errors="replace"`. |
| E9 | A body ending early without an exception is promoted as complete | medium | patch | Real: only `RequestException` was caught, so a short read looked like success and cached a truncated corpus. Now compares bytes read against the served `Content-Length` and raises `StreamInterrupted`. |
| E10+VO1 | Corrupt cached `brand_map.json` tracebacks on every later run; non-atomic write | medium | patch | Real: `main` caught only `StreamInterrupted`, so `JSONDecodeError` escaped and recurred until someone deleted the file. Now falls back to rebuilding, and the map is written `.part`-then-promote like the JSONL beside it. |
| E13 | A zero-match pass promotes an empty JSONL that then reads as a valid cache | medium | patch | Real: `part_path.replace(out_path)` ran unconditionally. Now raises `NoReviewsMatched`, unlinks the `.part`, and exits 1 naming the brand patterns. |
| E6 | A failed re-clean leaves the previous parquet in place, unmentioned | medium | patch | Real: `clean_file` skips the write on an empty frame. Deleting the old parquet would destroy the only good copy, so the patch is a loud stderr warning naming the stale file instead. |
| E7 | `clean_file(input_path=...)` labels arbitrary input from the global manifest | low | patch | Real: `_manifest_says_sample()` ran regardless of the input path. Now consulted only when the input is the configured raw JSONL. |
| B7 | The quality report is printed and never persisted | medium | patch | Real: nothing wrote it to disk, yet `data-schema.md` calls it "itself a portfolio signal" and stories 3 and 6 need the numbers. Re-deriving them costs a ~15-minute pass. Now written to `data/processed/quality_report.json` via a new `config.CLEAN_REPORT_JSON`. |
| B4 | `no_brand_match` is never counted where products are actually skipped | low | patch | Confirmed: `filter_reviews` dropped unmapped lines silently, so the matrix row's "counted" half was satisfied only by `clean.py`, where it is structurally 0. Added a skip counter, surfaced in the manifest and on stdout. |
| V1 | The manifest-derived sample label is never exercised end-to-end | medium | patch | Pre-verified: the layer mutated `manifest.get("sample")` to `get("is_sample")` and all 30 tests still passed, because every test supplied the flag itself. A 200-row sample would print "full corpus". Added `CleanFileLabelTest` (sample, full, missing, corrupt, explicit-input). |
| V2 | `clean.main()` and its exit-code contract never execute under test | medium | patch | Pre-verified by `trace`: lines 303-319 never ran; flipping the missing-input branch to `return 0` kept the suite green. Added `CleanMainTest` covering both failure exits, the stale-parquet warning, and the happy path. |
| V3 | Cached brand-map reuse — the point of caching the 2.8 GB pass — is untested | medium | patch | Pre-verified by `trace`: the cached branch never ran; disabling it kept the suite green, so the cache could silently stop working or serve a stale map. Added a test asserting only the reviews URL is fetched. |
| V4 | The empty-brand-map guard is untested, and the brand set is expected to change | medium | patch | Pre-verified: deleting the guard kept all 30 tests green; without it an empty map triggers a full 11 GB pass, promotes an empty corpus and exits 0. Added a test asserting exit 1 before the reviews URL is touched. |
| B13 | Several guard branches have no coverage | medium | patch | Confirmed against the suite. Covered by the new tests above plus the multi-brand tiebreak, which also pins the docstring's "first in `config.BRANDS` order" against the frozen "the brand named in `store`". |
| B15 | Dead rebinding in `filter_reviews`; `StreamInterrupted` drops `cause` | low | patch | Confirmed: the prescreen lookup at line 172 was unconditionally overwritten at line 179. Now a membership test; `cause` is stored alongside `url` and `bytes_read`. |
| B11 | `.gitignore` also carries `_bmad/render/`, undisclosed in the notes | low | patch | Confirmed, with a correction to the finding: that rule is not mine. The render cache and its ignore line predate this session's edits and ride along in the uncommitted tree. Disclosed in Implementation Notes. |
| B6+E11+VO2 | Byte-level `store` prescreen silently loses products before parsing | false | rejected | Refuted by arithmetic: story 1 independently measured 2,511 matching products; this pass mapped 2,385 on `store`, and the frozen decision drops 40+15+71=126 title-only products. 2,385+126=2,511 exactly, which cannot hold if the prescreen were dropping matches. JSON also does not escape `&`, so the Head & Shoulders example does not arise; story 1 already rejected the `&amp;` variant on measured evidence. |
| E12 | `parent_asin` prescreen misses escaped or >32-char values | false | rejected | ASINs are 10-character alphanumerics with no escapable characters; the 54,741 matched reviews all parsed. No mechanism shown for the bad outcome. |
| E19 | AC 4 says no file under `data/` is tracked beyond the two `.gitkeep`s, but `data/download_data.py` is tracked | low | rejected | The claim is literally true and worth recording: the criterion's wording conflicts with the spec's own Code Map, which places the script under `data/`. Rejected only because the fix edits this build's spec. Read as intended (no *data* files tracked), it holds: `git ls-files data/` returns the script plus the two `.gitkeep`s. Surfaced to the maintainer rather than reinterpreted silently. |
| E8 | `clean_file` called directly with a missing input raises a bare `FileNotFoundError` | low | rejected | Real but negligible: `main()` prints the actionable hint, and a traceback naming the missing path is not misleading. The fix adds a branch for a case no caller has yet. |
| E16 | Two metadata records sharing `parent_asin` with different brands: last wins silently | low | rejected | Deterministic for a fixed input file, so reruns do not flip; the fix adds conflict bookkeeping for a case never observed in 2,385 mapped products. |
| E17 | Disk-full during the reviews write gives a raw `OSError` traceback | low | rejected | Failing loudly is correct here: the traceback exits non-zero and the `.part` is never promoted, which is the documented contract. |
| B12 | `src/clean.py` has no argparse CLI, unlike its sibling script | low | rejected | `clean_file()` already takes input, output, and sample as parameters for programmatic callers, and `main()` is now covered by tests. Adding a CLI is new public surface for no demonstrated need. |
| B2 | AGENTS.md still marks `data/download_data.py` as a story 2 TODO | low | defer | Confirmed stale. Routed to defer by rule: the fix edits an agent-context file. |
| B10 | `data-schema.md` still describes store-or-title matching and exact-text dedup | medium | defer | Confirmed drift from what this story implemented. AGENTS.md forbids hand-editing the spec, so this needs a `bmad-spec` run; the story's Spec Change Log is empty because no non-frozen spec section was amended. |
| B5 | `stream_lines` accumulates `remainder` without bound | maybe-false | defer | The mechanism is real for a newline-free 200 body, but nothing shows the source can produce one; it is well-formed JSONL and `raise_for_status` covers error statuses. A max-line guard would settle it. |
| B8 | The `undated` bucket conflates missing and out-of-range timestamps | low | defer | Real and worth doing later: with `undated: 0` measured, a seconds-vs-milliseconds regression would surface as a silent spike rather than a counted reason. Deferred because it adds counters for a case the corpus does not exhibit. |
| B9 | No resume; `--refresh` re-streams 2.8 GB of metadata to retry the reviews pass | low | defer | Confirmed by this session's own interruption. Deferred: the fix adds flags and Range-request handling, which is new surface beyond this story. |
| VO3 | Nothing pins the git-ignore rules for the generated artifacts | low | defer | Verified by hand (`git check-ignore -v` resolves both to `.gitignore:17`), but unpinned by any test. The layer itself proposed defer. |

## Design Notes

**`review_id`:** the source has none, so derive a short BLAKE2b digest over `user_id` + `parent_asin` + `timestamp` + `text`. Stability across reruns is required because citations in stories 4–7 reference these ids, which rules out row numbers and UUIDs.

## Verification

**Commands:**
- `uv run python -m unittest -v` -- expected: the existing 7 tests plus the new clean tests, all OK
- `uv run python data/download_data.py --limit 200` -- expected: filtered JSONL written, progress shown, exit 0
- `uv run python -m src.clean` -- expected: parquet written, report printed with per-brand counts
- `git status --short -- data/` -- expected: no tracked data files

**Results (2026-09-16, measured):**
- `uv run python -m unittest -v` -- **30 tests, OK** before review; **52 tests, OK** after the review-pass-1 patches (7 from story 1, 45 for this story; no network, inline fixtures).
- `uv run python data/download_data.py` -- full pass rather than `--limit 200`, per the maintainer's run-scope decision. Exit 0; 23,911,390 scanned, 54,741 kept; `.part` promoted; manifest written with `sample: false`.
- `uv run python -m src.clean` -- exit 0; parquet written; report as recorded in Implementation Notes.
- Acceptance checks against the written parquet: all 8 `data-schema.md` target columns present plus `review_text_normalized` and `date_bucket`; `review_id` unique with 0 nulls; `rating` `int64` in 1–5; `brand` exactly the three canonical names; 0 empty `review_text`.
- Determinism: a second `clean_file()` over the same input produced an identical frame (`DataFrame.equals` True), so `review_id` is stable for citations in stories 4–7.
- `git status --short -- data/` -- only `data/download_data.py` (code). `git ls-files data/` -- the tracked script plus the two `.gitkeep` files; `git check-ignore -v` resolves both generated artifacts to `.gitignore:17`. See review finding E19 on this criterion's wording.
- Post-patch re-run of `uv run python -m src.clean` -- exit 0, numbers unchanged, and `data/processed/quality_report.json` now written alongside the parquet.
