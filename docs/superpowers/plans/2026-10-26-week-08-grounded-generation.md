# Week 8: Grounded Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieval is half the job; the answer must stay inside the sources. Four pieces:

- **context packing:** merge overlapping passages, order them, and fit a token budget
- **citation verification:** a post-check that each cited passage supports its sentence (lexical by default, an LLM as an option)
- **refusal:** a reliable "not in the documents", with a calibrated score gate as an option
- **prompt caching:** where it can pay

Every change is measured on faithfulness, citation precision, refusal accuracy and correctness.

**Architecture:**

- `raggy/generate/pack.py` turns hits into **passages**: same-source spans that overlap or touch are merged, each source's passages are kept in document order, sources are ordered by their best score, and lowest-ranked passages are dropped until the prompt fits `RAGGY_CONTEXT_TOKENS`.
- `raggy/generate/verify.py` splits the answer into sentences, maps each `[n]` to passage n, and scores support. **Lexical** support is the share of the sentence's content words (lower-cased, stemmed, stop words removed) found in the passage. **LLM** support is one call per answer that returns JSON verdicts.
- Unsupported citations are removed from the shown text and reported. An answer with no supported citation left can be turned into the refusal (`RAGGY_CITATION_ACTION`).
- The LLM interface gains `complete(system, user, cache_prefix=None)` and per-call `usage` (input, output, cache read, cache write).

**Tech Stack:** No new dependencies. Facts checked against the claude-api skill (loaded 2026-10-26):

- The minimum cacheable prefix is **512 tokens** on Opus 5.5 / Sonnet 5.5 / Haiku 5.5. Shorter prefixes silently don't cache.
- Anthropic caching is explicit (`cache_control: {"type": "ephemeral"}` on a content block). OpenAI caches automatically above 1,024 tokens of shared prefix.
- Usage is read from `usage.cache_read_input_tokens` / `cache_creation_input_tokens` (Anthropic) and `usage.prompt_tokens_details.cached_tokens` (OpenAI).

Observation 28 applies: the Anthropic adapter is an Anthropic-only file within the approved multi-provider design.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 8 row)

**Branching:** from `dev/week-07` (a worktree) after week 7's tests pass, rebased onto main after weeks 6 and 7 are merged.

## Global Constraints

- `uv run pytest` passes offline (FakeLLM; usage faked).
- **Where caching can pay, and where it can't** (stated in the lesson with token counts):
  - **Contextual retrieval (week 4):** the same document prefix repeats for every chunk of that document, so it gets `cache_prefix`.
  - **Answer generation:** the only shared prefix is the ~120-token system prompt, below the 512 minimum, and the sources differ per question. No cache marker is added. Adding one would only pay the cache-write premium.
  - **Judge prompts:** too short; same verdict.
- Context packing, verification and gating settings go in the run config. They are generation-side only, so they're not in the index keys.
- **Decision rules, set before any run (observation 31):**
  - **Packing** (dedupe + order) is on by default if mean correctness over the three fictional suites doesn't drop and context tokens fall, or correctness rises.
  - **Lexical citation verification with action `strip`** is on by default if citation precision rises and correctness doesn't drop by more than 0.03 (one question in 33).
  - **The refusal gate** (skip the LLM below a top-score threshold) is on by default only if unanswerable refusal stays at 1.000 *and* answerable refusal doesn't rise, on a held-out half of the questions. The threshold is calibrated on the other half (observation 33: calibrating and measuring on the same 41 questions would overfit).
  - Held constant: retrieval (week 6 and 7 defaults), top_k, the answer model and the judge.
- Never move week-01..07 branches or tags. Cut `week-08` (branch and tag) from a **verified SHA** (observation 34), then push.

## Review Focus

1. **Passages from the same source that overlap or touch** merge into one, keeping the best score and the union span. Passages from different sources never merge. Test: `test_pack_merges_overlapping_same_source`.
2. **A single passage larger than the budget** is cut at a sentence boundary from the end, never dropped entirely when it's the only one. Test: `test_pack_oversized_single_passage`.
3. **Citations that aren't on any sentence, markers out of range, or `[1][2]` runs** are parsed exactly like week 1 (`parse_citations`), and the verifier never invents a citation. Test: `test_verify_handles_runs_and_out_of_range`.
4. **The LLM verifier returning junk** keeps the answer unchanged, counts the failure and reports it. Test: `test_llm_verifier_garbage_keeps_answer`.
5. **The refusal gate with the hybrid/RRF and rerank score scales.** Scores mean different things per mode, so the gate's threshold is per mode and calibrated per mode. Without a calibration for the active mode, the gate is off. Test: `test_gate_off_without_calibration_for_mode`.

---

### Task 1: Context packing

**Files:** Create `raggy/generate/pack.py`. Modify `raggy/generate/prompt.py` (prompts are built from passages; citations still map `[n]` to a hit-like object with `source` and `chunk_id`), `raggy/generate/answer.py`, `raggy/config.py` (`context_tokens` 3000, `pack` on | off). Test `tests/week08/test_pack.py`.

- [ ] Tests: `test_pack_merges_overlapping_same_source`, `test_pack_keeps_document_order_within_source`, `test_pack_orders_sources_by_best_score`, `test_pack_fits_budget_dropping_lowest`, `test_pack_oversized_single_passage`, `test_pack_off_is_week7_prompt` (byte-identical prompt). FAIL → implement → all pass. Commit.

### Task 2: Citation verification

**Files:** Create `raggy/generate/verify.py`. Modify `raggy/generate/answer.py` (`Answer` gains `unsupported: list[int]` and `verification: str`), `raggy/config.py` (`citation_check` off | lexical | llm, `citation_threshold` 0.5, `citation_action` report | strip | refuse), `raggy/cli.py` (prints "⚠ citation [2] not supported by its source" lines). Test `tests/week08/test_verify.py`.

- [ ] Tests: `test_lexical_supported`, `test_lexical_unsupported`, `test_verify_handles_runs_and_out_of_range`, `test_action_strip_removes_marker_only`, `test_action_refuse_when_none_supported`, `test_llm_verifier_parses_verdicts`, `test_llm_verifier_garbage_keeps_answer`. FAIL → implement → all pass. Commit.

### Task 3: The refusal gate, usage and caching

**Files:** Modify `raggy/generate/llm.py` (protocol: `complete(system, user, cache_prefix=None)`, `usage` dict), `raggy/generate/anthropic_llm.py` (with a `cache_prefix`, the user content is two text blocks and the first carries `cache_control: {"type": "ephemeral"}`; usage accumulates the cache fields), `raggy/generate/openai_llm.py` (prefix + rest concatenated; reads `cached_tokens`), the Ollama adapter (concatenates), `raggy/chunking/contextual.py` (passes the document as `cache_prefix`), `raggy/generate/answer.py` (gate), `raggy/evals/runner.py` (records LLM usage per question and in totals), and `evals/calibrate_gate.py` (picks a threshold per mode on a split). Test `tests/week08/test_gate_and_usage.py`.

- [ ] Tests: `test_anthropic_cache_prefix_block_shape` (fake client records the request), `test_openai_cache_prefix_concatenates`, `test_usage_accumulates`, `test_contextual_passes_document_as_prefix`, `test_gate_refuses_below_threshold_without_llm_call`, `test_gate_off_without_calibration_for_mode`. FAIL → implement → all pass. Commit.

### Task 4: Measure, decide, lesson

- [ ] Answer and judge runs (gpt-5.5 answers, gpt-5.4 judge, as in earlier weeks) on fictional, fictional-web and manual for: the week 7 default; + packing; + packing + lexical verify (report / strip); + the LLM verifier; + the gate (calibrated on odd-numbered questions, measured on even). Record correctness, faithfulness, citation precision, refusal metrics, context tokens and LLM usage (cache reads) for contextual retrieval on the manual.
- [ ] Apply the rules, set the defaults, and ledger the rulings. Commit baselines to `evals/baselines/week08/`.
- [ ] Write `lessons/week08-grounded-generation.md` and `docs/grounded-generation.md` (packing, lost-in-the-middle, citation verification lexical vs LLM, refusal calibration, when caching pays: the 512-token minimum and prefix stability). Update the README, COURSE_PLAN and the docs hub. Commit.
- [ ] Final review → rebase onto main → cut `week-08` from a verified SHA → push.
