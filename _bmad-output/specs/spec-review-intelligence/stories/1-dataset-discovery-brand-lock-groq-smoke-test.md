---
title: 'Dataset discovery, brand lock, and Groq smoke test'
type: 'chore'
created: '2026-09-15'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '639558abdfa3e6362ff2dfb090b9a1b0f6a569ae'
context:
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/data-schema.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/documentation.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** SPEC open questions 1–3 (dataset, brand set, Groq model string) are unresolved and the README has no problem statement, so the Phase 0 exit check fails and story 2 cannot start. The plan's "current Llama instruct model" no longer exists: Groq shut down its Llama chat models on 2026-08-16.

**Approach:** Lock Amazon Reviews 2023 `Beauty_and_Personal_Care` as the source and Head & Shoulders, Pantene, and Herbal Essences as the brand set, backed by measured per-brand written-review counts. Define the Groq model once in `src/config.py`, prove it with a smoke-test call, record the decisions in `PLAN.md`, and write the README problem statement.

## Boundaries & Constraints

**Always:** Read `GROQ_API_KEY` from `.env` via python-dotenv and never print or log it; print only presence and HTTP status. Define the model string only in `src/config.py`. Report measured counts and label estimates as estimates. Leave all changes uncommitted.

**Never:** Download raw data into the repo, or write `data/download_data.py` or `src/clean.py` (story 2). Hand-edit `SPEC.md`, its companions, `stories.yaml`, or `.memlog.md`. Recreate `requirements.txt`, `.gitignore`, `.env.example`, `LICENSE`, or `.venv`. Add dependencies. Add retry, backoff, or prompt logic (story 5).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | Valid key in `.env`, model live | Prints model id, reply text, finish reason; exit 0 | N/A |
| Missing key | `GROQ_API_KEY` unset or empty | No API call; message names the variable and `.env.example` | Exit 1 |
| API error | 401, 404 decommissioned model, 429 | Prints status code and Groq's error message, never the key | Exit 1 |

**Decisions:**
- `GROQ_MODEL = "openai/gpt-oss-120b"` (maintainer, 2026-09-15). It replaces the retired Llama plan because it is Groq's named successor with the strongest grounding. Accepted trade-off: reasoning tokens (121 for a trivial call) count against the 8K tokens/min free-tier cap. Record this and why Llama was dropped in PLAN.md.

</frozen-after-approval>

## Code Map

- `README.md` -- Currently the single line `# consumer-review-rag`. Add the one-line description and "Why this project" (documentation.md outline items 1–2). Omit the demo link until story 7.
- `PLAN.md` -- §3 table (Data source, Generation model rows), §4 Secrets bullet naming the model, §13 open questions: record answers and evidence here.
- `src/config.py` -- New. The only home for constants (AGENTS.md).
- `src/groq_smoke.py` -- New. Run as `uv run python -m src.groq_smoke`. Use the installed `groq` SDK (1.7.0).
- `.env.example`, `requirements.txt` -- Already list `GROQ_API_KEY`, `groq`, and `python-dotenv`. No change.
- `.venv` -- Python 3.11.16. `load_dotenv()` without a path fails under stdin/`-m` frame lookup, so pass the explicit project-root `.env` path.

## Tasks & Acceptance

**Execution:**
- [x] `src/config.py` -- Define `PROJECT_ROOT`, `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `CHROMA_DIR`, `GROQ_MODEL`, `GROQ_TEMPERATURE` (low), `BRANDS` (canonical name → case-insensitive match pattern, as in Design Notes), and the dataset source constants (Hugging Face repo and category). -- Single source that stories 2–7 import.
- [x] `src/groq_smoke.py` -- Make one chat completion with `config.GROQ_MODEL` and follow the I/O matrix. -- Phase 0 "hello world" exit check.
- [x] `PLAN.md` -- Record the dataset, brand set, and model decisions with the measured evidence table and the dataset-terms caveat. -- AGENTS.md: decision changes go to PLAN.md first.
- [x] `README.md` -- Write the description line and a 2–3 sentence P&G problem statement covering the EDA and RAG layers. -- Locks the framing before building.

**Acceptance Criteria:**
- Given a valid `.env`, when `uv run python -m src.groq_smoke` runs, then it prints `config.GROQ_MODEL` and a non-empty reply and exits 0.
- Given the finished change, when `src/`, `app/`, and `notebooks/` are searched for the model id, then it appears only in `src/config.py`.
- Given `PLAN.md` §13, when read, then each of the three questions has an answer, and the brand answer shows measured written-review counts per brand.
- Given `git status` and `git diff`, when inspected, then no data files, `.env`, or `gsk_` key strings are staged or changed.

## Implementation Notes

- **Files:** `src/config.py` and `src/groq_smoke.py` (new), `PLAN.md` (§3, §4, §13), `README.md`, `tests/test_groq_smoke.py` (new). A subagent implemented these.
- **Verification added:** the spec had no test task, but the matrix audit requires covering tests. `tests/test_groq_smoke.py` uses stdlib `unittest` with a mocked Groq client, so no new dependency. It covers the happy path, blank and whitespace keys, and 401/404/429 errors, and asserts the key never appears in output. Run with `uv run python -m unittest tests.test_groq_smoke -v`: 3 tests, OK.
- **README:** trimmed "Why this project" from 4 sentences to 3 to meet the task's 2–3 sentence limit.
- **Measured counts** come from a sharded streaming pass over both files; nothing was stored, and the throwaway script lives in the session scratchpad. Written reviews: Head & Shoulders 9,868, Pantene 29,670, Herbal Essences 16,325. One multi-brand product was assigned by its `store` field, so Head & Shoulders has 402 products versus 403 in Design Notes.
- **Known behavior:** the groq SDK's default `max_retries=2` retries a 429 before the script exits 1. That still satisfies the matrix; no retry logic was added.
- **Review pass 1 patches** were applied by the orchestrator because the implementation subagent could not be resumed:
  - `src/groq_smoke.py` now calls `with_raw_response` and prints the measured `raw.status_code`.
  - `src/config.py` brand patterns use inline `(?i)` and a non-capturing group.
  - `tests/__init__.py` was added. Plain `uv run python -m unittest -v` now discovers 7 tests, OK. They add unset key, empty/None reply, connection error, the asserted `.env` path and repo root, error-body parsing, and brand-pattern checks.
  - `PLAN.md` got 7 edits: §3 and §13.3 sources plus the grounding claim reframed as rationale, the §9 dataset-terms limitation, §13.2 column labels and patterns, and the independent cross-check plus the Aussie written-review count.
  - `README.md` now names Amazon Reviews 2023 (Hou et al. 2024).
  - Verification re-run: live smoke `HTTP status: 200`, exit 0; blank key, exit 1; the model id appears only at `src/config.py:36`; tracked diff `gsk_` count 0.
- **Deferred:** SPEC.md and stack.md Llama follow-up, in `_bmad-output/implementation-artifacts/deferred-work.md`.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-15). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V).

| ID | Finding | Verdict | Route | Evidence |
|---|---|---|---|---|
| B1 | `git diff \| grep gsk_` misses untracked files | low | rejected | Real gap, but the fix edits this spec's Verification. A manual untracked scan found only the spec's own text. |
| B2+E6 | `groq_smoke.py` prints hardcoded `HTTP status: 200` | low | patch | The literal prints on any successful return; the frozen Always rule requires a printed HTTP status. `with_raw_response` measures it. |
| B3 | No max-tokens or reasoning-effort constants | false | rejected | The smoke call uses about 136 tokens with no budget risk; generation parameters are story 5 (rag-design.md). |
| B4 | `finish_reason == "length"` still exits 0 | low | rejected | Unrealistic for a one-line prompt with the default max; the fix adds a branch. |
| B5+E11+V2+V4 | Untested branches: empty or None reply, truly unset key, `APIConnectionError` | medium | patch | Confirmed: `patch.dict` always sets the key; no test reaches `if not reply` or the connection handler. V mutation `if False:` still passed. |
| B6 | Tests import `httpx`, which is not in requirements | low | rejected | `groq` is built on httpx, so breakage is unlikely. Adding a dependency is forbidden by Never. |
| B7a+E7 | `BRANDS` regexes rely on callers passing IGNORECASE | medium | patch | The comment says case-insensitive, but the bare strings are case-sensitive. Story 2 would silently undercount. Fix: inline `(?i)`. |
| B7b+E8 | Capturing group in the Head & Shoulders pattern | low | patch | `(&\|and\|n'?)` makes pandas `str.contains` warn about match groups. Direct correction to `(?:...)`. |
| B7c+E9 | `&amp;` and `'n'` variants missed | maybe-false | rejected | Would need a grep of metadata for `&amp;`. The `store` field shows clean "Head & Shoulders" (362 products), so the effect would be low. |
| B7d | No product-line terms (Pro-V, bio:renew) per data-schema.md | maybe-false | rejected | Would need title-only listings that omit the brand name. Store-field matches dominate, so the effect would be low. Story 2 owns filter rules. |
| B8 | §13.2 counts not reproducible (script discarded) | low | rejected | The method is stated in §13.2 and an independent pass agreed (Pantene and Herbal Essences exact, Head & Shoulders +4 from the multi-brand product). Committing a script adds surface; story 2's `download_data.py` owns reproducibility. |
| B9 | Head & Shoulders `rating_number` 227,681 (PLAN) vs 229,198 (Design Notes) | false | rejected | The 1,517 gap is the one multi-brand product reassigned by `store` (403→402), which PLAN documents. PLAN is internally consistent. |
| B10 | §13.2 table leaves verified-purchase scope and timezone implicit | low | patch | Script counts `verified` only for written reviews and dates in UTC; the table says neither. Direct doc correction. |
| B11 | §13.1 says the dataset caveat is "listed under limitations", but §9 lacks it | low | patch | Confirmed: §9 is unchanged. Add the bullet to §9. |
| B12 | Shutdown, rate-limit and "strongest grounding" claims unsourced | low | patch | Sources exist: console.groq.com/docs/deprecations and /docs/rate-limits. "Strongest grounding" is the selection rationale, not measured; story 6 measures it. |
| B13 | SPEC.md and stack.md still say Llama | medium | defer | Stories 2–7 read a stale model decision until `bmad-spec` runs. AGENTS.md forbids hand-editing the spec. |
| B14 | README never names or cites the dataset | low | patch | The description says only "public Amazon reviews". Direct correction: name Amazon Reviews 2023 (Hou et al. 2024). |
| B15a | Code Map note on `load_dotenv` under `-m` misleading | low | rejected | The fix edits this spec. |
| B15b | Verification grep may hit `__pycache__` binaries | false | rejected | The run after the smoke test printed only `src/config.py:34`. The fix would also edit this spec. |
| E1 | Empty `choices` causes IndexError | low | rejected | Groq returns choices on 2xx; the fix adds a guard. |
| E2 | Other `GroqError` subclasses uncaught | low | rejected | Rare (response validation). A traceback still exits non-zero; the fix adds a branch. |
| E3 | Default 60 s timeout with 2 retries can stall | low | rejected | Manual smoke run; the fix adds a parameter. |
| E4 | Stale shell `GROQ_API_KEY` shadows `.env` | low | rejected | Real python-dotenv precedence, but `override=True` would break the deploy precedence where Secrets and env must win. Rare. |
| E5 | `python src/groq_smoke.py` fails with ModuleNotFoundError | false | rejected | `-m` is the documented invocation (docstring, spec); failing loudly on an unsupported path is correct. |
| E10 | Multi-brand tie-break not codified | low | rejected | One product in 2,511; the codified rule is story 2's filter decision; the fix adds a constant. |
| V1 | Default `python -m unittest` discovers 0 tests | medium | patch | Confirmed: discovery printed `Ran 0 tests`, because `tests/` has no `__init__.py`. |
| V3 | `.env` path and PROJECT_ROOT never asserted | medium | patch | `load_dotenv` is mocked unasserted; V mutation of PROJECT_ROOT and bare `load_dotenv()` still passed. |
| V5 | API-error test can't detect body parsing being skipped | low | patch | Same text in exception and body, so the `str(exc)` fallback also passes. Direct test correction. |
| V6 | Model-id AC grep excludes `tests/` | false | rejected | The `llama-3.3-70b-versatile` string in tests is fixture data, not the configured model id. The AC scopes `src`, `app`, `notebooks`. |

## Design Notes

**Evidence (2026-09-15).** Metadata `store` holds the brand. Counts below are store-or-title matches using the data-schema rule. `rating_number` includes star-only ratings, so it overstates written reviews. Written-review counts come from a streaming pass over the 11 GB reviews file and must be copied into PLAN.md.

| Brand | All_Beauty products / ratings | Beauty_and_Personal_Care products / ratings |
|---|---|---|
| Head & Shoulders | 13 / 1,601 | 403 / 229,198 |
| Pantene | 70 / 1,700 | 1,426 / 270,991 |
| Herbal Essences | 34 / 663 | 683 / 171,707 |

All_Beauty is too thin for per-brand trends. Aussie (522 products / 185,453 ratings) is a viable fourth haircare brand but exceeds the 2–3 brand scope.

**Match patterns:** `head\s*(&|and|n'?)\s*shoulders`, `pantene`, `herbal\s+essence`.

**Dataset terms:** Neither the Hugging Face card, the project site, nor the MIT-licensed code repo states terms for the data itself. Cite Hou et al. 2024 (arXiv:2403.03952) and list this as a limitation.

**Story 2 consequence:** The reviews file is 11 GB and the metadata 2.8 GB. `download_data.py` should stream and filter rather than store whole files.

**Follow-up for the maintainer:** SPEC.md and stack.md still say "Llama". After approval, run `bmad-spec` so the spec follows PLAN.md.

## Verification

**Commands:**
- `uv run python -m src.groq_smoke` -- expected: model id and reply printed, exit 0
- `GROQ_API_KEY= uv run python -m src.groq_smoke; echo $?` -- expected: missing-key message, `1`
- `grep -rn "gpt-oss\|qwen" src app notebooks` -- expected: only `src/config.py`
- `git diff | grep -c gsk_` -- expected: `0`
