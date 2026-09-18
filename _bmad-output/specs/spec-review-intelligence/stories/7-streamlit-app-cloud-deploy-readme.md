---
title: 'Streamlit app, Cloud deploy, and README'
type: 'feature'
created: '2026-09-18'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '34b6086cad3aeee1418c4c5065ef6294e3b97bd1'
context:
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/documentation.md'
  - '{project-root}/_bmad-output/specs/spec-review-intelligence/architecture-diagrams.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CAP-6 and CAP-7 have no implementation. There is no Streamlit app, nothing is deployed, and `README.md` is a two-paragraph stub missing the architecture diagram, EDA insights, eval numbers, limitations, run-locally steps, tech stack, and the P&G application paragraph.

**Approach:** Build `app/streamlit_app.py`, importing `src.retrieve`, `src.generate`, and `src.eda` unchanged, with a query box, brand/rating-band filters, a cited answer with expandable sources, and 1–2 EDA charts. Pin `requirements.txt` from the working `.venv`. Rewrite `README.md` to `documentation.md`'s outline. Deploy to Streamlit Community Cloud.

## Boundaries & Constraints

**Always:** Reuse `src.retrieve.retrieve`, `src.generate.answer`, and `src.eda`'s loaders/plot helpers as-is; the app never reimplements retrieval, generation, or chart logic. Read `GROQ_API_KEY` from `.env` locally (already handled by `answer()`) and from Streamlit Secrets when deployed — copy `st.secrets["GROQ_API_KEY"]` into `os.environ` before the first `answer()` call only when it is not already set, and never display or log the key. Report only measured numbers in the README (pull from `eval/results.md`, not restated by hand). Follow `documentation.md`'s outline order. Confirm with the maintainer before `git push` and before any Streamlit Cloud connection step — both are irreversible/external and the implementer cannot complete the Cloud dashboard OAuth flow itself, so it stops and hands those steps to the maintainer with exact instructions.

**Never:** Change `src/retrieve.py`, `src/generate.py`, `src/index.py`, `src/eda.py`, or the prompt. Add a second API key or paid service. Claim production readiness. Auto-push or auto-deploy without the maintainer's go-ahead at each step named above.

**Decisions (planning, 2026-09-18):**
- **Deployment data strategy (maintainer):** Option A. Commit `data/processed/reviews_clean.parquet` (remove it from `.gitignore`); `.chroma/` stays git-ignored. `app/streamlit_app.py` builds the index at startup by calling `src.index.build_index()` (or equivalent), relying on `src/retrieve.py`'s `lru_cache(maxsize=1)` loaders to build it once per container and reuse it across reruns. State plainly in the README limitations that cold start, every redeploy, and every wake-from-sleep costs several minutes (66 s was the measured local MPS build; Streamlit Cloud's CPU-only runtime will be slower).
- **Spec size (maintainer):** keep as one spec; app, deploy, and README stay one story as already scoped in the spec breakdown.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Answered query | Valid question, key present, hits found | Cited answer text + expandable source list render | N/A |
| Filtered, no hits | Brand+band combo matches no reviews | `NO_REVIEWS_TEXT` shown, no crash | N/A |
| Unanswerable question | Outside corpus coverage | Refusal sentence shown | N/A |
| Rate limited | Groq returns 429 | `RATE_LIMITED_TEXT` shown | N/A |
| Missing key | No `.env`, no Secrets | `MISSING_KEY_TEXT` guidance shown, no crash | N/A |
| Deployed, key in Secrets | `st.secrets["GROQ_API_KEY"]` set, no `.env` | App copies it into `os.environ` before the first `answer()` call | N/A |
| EDA panel | Parquet + quality report present | 1–2 charts render with a short takeaway caption | N/A |

</frozen-after-approval>

## Code Map

- `app/streamlit_app.py` — new. Query box; brand selectbox (`config.BRANDS` keys); rating-band selectbox (`config.RATING_BANDS` keys); calls `retrieve()`/`answer()` with no injected client/retriever so `src/retrieve.py`'s module-level `lru_cache(maxsize=1)` loaders (`_default_collection`, `_default_embedder`, `_default_reviews`) cache the heavy resources across Streamlit reruns for free. On startup, ensures the index exists (calls `src.index.build_index()` when `.chroma/` is absent or stale) before the first query.
- `src/generate.py:164` `answer()` — reused unchanged; the app sets `os.environ[config.GROQ_API_KEY_ENV]` from `st.secrets` before calling it, only if unset.
- `src/eda.py` — reused for chart data and plots: `load_clean()`, `load_report()`, `rating_distribution()`, `monthly_stats()`, `plot_rating_distribution()`, `plot_monthly_volume()`.
- `requirements.txt` — currently loose `>=` pins (invoke_dev_with note: pin exact versions from the working `.venv` before deploying).
- `README.md` — 2-paragraph stub today; rewrite to `documentation.md`'s 9-point outline, embedding the `architecture-diagrams.md` Mermaid diagram and `eval/results.md`'s measured numbers.
- `.gitignore` — un-ignore `data/processed/reviews_clean.parquet` (keep `data/processed/quality_report.json` ignored-or-not per its current rule); `.chroma/` stays ignored.

## Tasks & Acceptance

**Execution:**
- [x] `requirements.txt` — pin exact versions from `uv pip freeze` inside `.venv` — invoke_dev_with note; a loose resolve at deploy time can break the live app.
- [x] `.gitignore`, `data/processed/reviews_clean.parquet` — un-ignore and commit the parquet — unblocks a working cold start (Option A).
- [x] `app/streamlit_app.py` — build per Code Map — CAP-6.
- [x] Automated: `tests/test_streamlit_app.py` (headless `AppTest`) plus a real end-to-end query and a background `streamlit run` boot check exercise the answered/no-hits/refusal/rate-limited/missing-key/secrets/EDA paths — CAP-6 success bar.
- [x] `README.md` — rewrite per `documentation.md` outline, including the live URL placeholder until deploy completes — CAP-7.
- [ ] Stop for maintainer confirmation, then `git push`, then hand the maintainer exact steps to connect the repo at share.streamlit.io and set the `GROQ_API_KEY` secret — CAP-6 live-URL requirement.
- [ ] Once live, verify the URL and fill in the README's live-demo link — CAP-6 acceptance.

**Acceptance Criteria:**
- Given the local app, when a user submits "what do 1-star Pantene reviewers complain about" with Pantene/low filters, then a cited answer with expandable sources renders, matching `answer()`'s contract.
- Given `GROQ_API_KEY` is unset, when the app runs, then it shows `MISSING_KEY_TEXT` guidance instead of raising.
- Given the deployed URL, when opened in a fresh browser, then the query box, filters, a cited answer, and at least one EDA chart render.
- Given `README.md`, when read top to bottom, then every `documentation.md` outline item appears in order, including the live URL and the P&G application paragraph.

## Implementation Notes

- **Files:** `app/streamlit_app.py` (new), `requirements.txt` (pinned exact versions), `.gitignore` (un-ignored the parquet), `data/processed/reviews_clean.parquet` (newly committed), `README.md` (rewritten), `docs/images/rating_distribution.png`, `docs/images/monthly_volume.png` (new, exported from `src.eda`'s plot helpers for the README), `tests/test_streamlit_app.py` (new, 7 tests). `src/retrieve.py`, `src/generate.py`, `src/index.py`, `src/eda.py`, and the prompt were not changed.
- **Handoff:** the first implementation subagent completed requirements pinning, the `.gitignore`/parquet change, and `app/streamlit_app.py`, then stalled mid-verification (600 s watchdog, no crash reported) while manually exercising the no-hits/rate-limited/missing-key paths. The orchestrator continued directly from that state rather than restarting: reviewed the existing `app/streamlit_app.py` against `src/retrieve.py`/`src/generate.py`'s real signatures (matched), then added the automated coverage below.
- **Verification performed:** `uv run python -m unittest -v` — 210 tests OK (203 prior + 7 new in `tests/test_streamlit_app.py`, headless via `streamlit.testing.v1.AppTest` with `src.generate.answer`/`src.index.build_index` patched at their source modules so the app's own fresh script exec picks up the fakes — covers answered/no-hits/refusal/rate-limited/missing-key/Secrets-copy/EDA-panel, i.e. every I/O matrix row). A real end-to-end call (`answer("what do 1-star Pantene reviewers complain about", brand="Pantene", rating_band="low")`, live index, live Groq key) returned `status="answered"` with 6 sources and 5 citations. A real `uv run streamlit run app/streamlit_app.py` was booted in the background on a scratch port, returned HTTP 200 with no exceptions in its log, and was killed — confirming the earlier stall was not a real app defect.
- **Deviation from the spec's stated verification:** the I/O-matrix task asked for a manual browser click-through; a fresh browser session wasn't available to the agent, so this was substituted with the automated `AppTest` suite above (which drives the same widgets and assertions a manual click-through would) plus the real end-to-end call and boot check. A human should still open the app in a real browser once before or during deploy.
- **Not done (by design, per the spec's Boundaries):** `git push` and the Streamlit Community Cloud connection/secret step — both need the maintainer's explicit go-ahead and, for the Cloud dashboard OAuth flow, the maintainer's own hands.
- **Review pass 1 patches applied directly** (no subagent was resumable; orchestrator applied them per the fallback in step-03's rules): un-ignored and committed `data/processed/quality_report.json` alongside the parquet (`.gitignore`); broadened `_ensure_index()`'s except clause from `(FileNotFoundError, ValueError)` to `Exception` so any cold-start failure shows `st.error` instead of crashing; fixed the citation badge to match by `review_id` instead of dict-equality `list.index()`; added `plt.close(fig)`/`plt.close(fig2)` after each `st.pyplot()` call in `_render_eda` to stop a per-rerun figure leak; added 5 tests (`test_uncited_answer_shows_caption`, `test_answer_raising_shows_error`, `test_index_build_failure_shows_error_and_stops`, a badge-placement assertion in the answered-query test, and a `sources=hits`-correct `test_missing_key_shows_error`) and cleared `st.cache_resource` in `setUp` (needed once tests patched `build_index` differently across cases, since its cache is a process-global singleton). Re-verified after patching: 213/213 tests OK, a real end-to-end `answer()` call, and a real `streamlit run` boot — all still pass. One item (`AGENTS.md`'s stale story-7 TODO) routed to defer per the agent-context-file rule; two items (dependency version-jump risk, "live demo" forcing function) were rejected as `false`, disproven by evidence already in hand; two items (parquet/clean.py drift check, architecture-diagram simplification) were rejected as `low` with a non-trivial fix. Full reasoning for every finding is in the Review Triage Log above.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-18). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V, `VO` = its Other findings).

| ID | Finding | Verdict | Route | Evidence |
|---|---|---|---|---|
| B1+E2+E4 | `data/processed/quality_report.json` stays git-ignored (`.gitignore`'s `data/processed/*` still blocks it; only the parquet was un-ignored) while `_render_eda()` calls `eda.load_report()`, which reads exactly that file | high | patch | Confirmed independently: `git check-ignore -v data/processed/quality_report.json` → matched by `.gitignore:17`. On a fresh Cloud clone the EDA panel raises `FileNotFoundError` and shows `st.error(...)` instead of charts, breaking the story's own "at least one EDA chart render" acceptance criterion on the actual deploy target. |
| B3+E1+V3 | `_ensure_index()`'s `except (FileNotFoundError, ValueError)` is narrower than `build_index()`'s real failure modes on a cold container (HF model download errors, `OSError`, chromadb errors), and no test ever makes the patched `build_index` raise | medium | patch | Confirmed by reading `index.build_index()`: nothing restricts it to those two exception types. `tests/test_streamlit_app.py`'s `setUp()` only ever returns `{"status": "skipped"}`, never `side_effect=...`, so this branch never runs in the suite (verification-gap, pre-verified). |
| B4+V1 | `_render_source`'s `✓ cited` badge is computed via `result.sources.index(hit)` (dict-equality lookup) instead of matching on the unique `review_id`, and `test_answered_query_renders_text_and_sources` never asserts the badge lands on the right source | medium | patch | My own check: `retrieve()`'s `_collapse` keeps one hit per distinct `review_id`, so no two dicts in one `sources` list compare equal today — not currently reachable, but citation trust is this project's core claim. Verification-gap (pre-verified): flipping the badge condition or introducing an off-by-one would leave both existing assertions (`result.text` substring, expander label) green. |
| V2 | `if result.uncited: st.caption(...)` branch has no test setting `uncited=True` | low | patch | Pre-verified: `grep -n "uncited" tests/test_streamlit_app.py` returns no matches; the one ANSWERED-path test leaves `uncited` at its `False` default. |
| V4 | The `except (FileNotFoundError, ValueError, RuntimeError)` around the submit-time `answer()` call is never exercised with a raise (e.g. the stale-index `RuntimeError` `_with_review` can raise) | low | patch | Pre-verified; reviewer's own filed disposition leaned defer (narrower window, mirrors established pattern in `src/generate.py`/`src/index.py` `main()`), but the fix is exactly as trivial as #2-#3's (one more `side_effect=...` test), so bundled into the same patch for consistency. |
| VO1 | `test_missing_key_shows_error`'s fake `Answer` omits `sources=hits`, but `answer()` always sets `sources=hits` for `STATUS_ERROR`/missing-key | low | patch | Confirmed at `src/generate.py:190`: `return Answer(status=STATUS_ERROR, text=MISSING_KEY_TEXT, sources=hits)`. The test double doesn't exercise the real composite case (error text + a populated Sources expander). |
| E3 | `_render_eda()`'s two `plt.subplots()` figures are never `plt.close()`d, and the function reruns on every Streamlit script rerun (every widget interaction) | low | patch | Confirmed by reading `_render_eda`: no `plt.close` call after either `st.pyplot()`. Matplotlib's pyplot keeps every created figure registered until closed, so a long-lived container leaks 2 figures per rerun; a single interview demo session would never accumulate enough to matter, but the fix is a one-line direct correction. |
| B2 | `AGENTS.md`'s "TODO (stories 3 and 7, verify on first refresh)" line for the app launch command was not updated even though this diff verifies it | n/a | defer | Fix edits an agent-context file (`AGENTS.md`), which routes to defer regardless of severity per the fixed rule. |
| B5 | No compatibility note for the `requirements.txt` major-version jumps (pandas 2.2→3.0.5, numpy 1.26→2.4.6, chromadb 0.5→1.5.9, sentence-transformers 3.0→6.0.1, streamlit 1.38→1.64.0) | false | reject | Disproven by direct evidence already in hand: the full 210-test suite is green under these exact pins, a real end-to-end `answer()` call succeeded against the real Chroma/embedding stack, and a real `streamlit run` booted cleanly — all already exercising these libraries under the pinned versions. |
| B6 | Committing `reviews_clean.parquet` has no automated drift check against what `src/clean.py` would currently produce | low | reject | Real but unlikely to be met in this one-week solo portfolio project's everyday use, and a real fix (fingerprint/CI check) is more than a direct correction — rejected per the low-and-nontrivial-fix rule. |
| B7 | The README's architecture diagram doesn't show `generate.py` calling `retrieve()` internally, or an edge from `APP` back to `Q` | low | reject | The diagram is an unmodified, already-approved copy of `architecture-diagrams.md` (a SPEC companion), not new content authored for this diff; it is a simplification, not incorrect, and a real fix is a non-trivial redesign — rejected per the low-and-nontrivial-fix rule. |
| B8 | "Live demo: not yet deployed" has no forcing function tying it to the deploy step | false | reject | Disproven: the story's own unchecked Tasks & Acceptance item ("Once live, verify the URL and fill in the README's live-demo link") is exactly that forcing function. |

## Verification

**Commands:**
- `uv run streamlit run app/streamlit_app.py` -- expected: opens locally; manual click-through of the I/O matrix scenarios above
- `uv run python -m unittest -v` -- expected: existing suite stays green (no `src/` behavior changed)

**Manual checks (if no CLI):**
- Open the deployed URL in a fresh/incognito browser and run one query end-to-end.
