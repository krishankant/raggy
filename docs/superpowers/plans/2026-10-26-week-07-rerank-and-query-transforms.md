# Week 7: Reranking and Query Transformation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieve wide and cheap, then rank narrow and precise. Add a cross-encoder reranker (local by default, an LLM as the hosted option) over the retrieved candidates, and three LLM query transformations as switches: rewrite, multi-query and HyDE. Every stage is timed against a budget. Each technique stays on by default only if a rule set before the runs says it earns its accuracy against its latency and cost.

**Architecture:** Search becomes stages: `transform → retrieve (vector / keyword / hybrid, week 6) → rerank → parent expansion (week 4)`. Each stage is timed into `Raggy.last_timings`, and the eval reports p50/p95 per stage against `Config.budgets`. Rerankers implement `Reranker.rerank(question, hits) -> hits`. Query transforms call the configured LLM through the existing `LLM` interface, with versioned prompts. Their outputs are cached in SQLite (like week 4's context cache), so eval reruns don't pay twice. A failed transform falls back to the original question and is counted.

**Tech Stack:** No new dependencies. The local reranker is fastembed's `TextCrossEncoder`; the default is `Xenova/ms-marco-MiniLM-L-6-v2` (0.08 GB, Apache-2.0), and the larger option is `BAAI/bge-reranker-base` (1.04 GB, MIT). Both were listed by fastembed 0.9.0 before planning. The hosted option is `LLMReranker` (listwise, through the Claude or OpenAI adapters), so there's no extra provider SDK.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 7 row; section 9 "heavy local models: small default, configurable larger")

**Branching:** from `dev/week-06` in a third worktree. It's rebased onto main after weeks 5 and 6 are merged. No heavy runs until weeks 5 and 6 have finished measuring.

## Global Constraints

- `uv run pytest` passes offline. A fake reranker and a fake LLM are used in tests; no model downloads.
- Reranker, transform, their models, prompt versions and candidate counts go in the run config. They're search-side only, so they are not in the eval cache key or the registry key.
- Transform calls are cached in `index_dir/transform-cache.sqlite` (added to `CACHE_FILES`). The cache key is model + prompt version + question.
- **Decision rules, set before any run (observation 31):**
  - **Reranking** is on by default if it raises mean recall@4 over fictional, fictional-web and manual, *and* raises nDCG@10 on SciFact by at least 0.01 without lowering FiQA's, *and* its p95 latency is within its budget (500 ms on this M2) at the default 20 candidates.
  - **A query transform** is on by default only if it raises mean recall@4 over the three fictional suites by at least 0.03 (one question in 33) *and* raises SciFact nDCG@10 by at least 0.01, because each costs an LLM call per question (cost is priority 3) and adds about a second (speed is priority 2).
  - Held constant in all comparisons: the index, the chunks, top_k and the context size.
- (Observation 33) The data can tell them apart: the paraphrase-tagged fictional questions (12) are where a rewrite should help, and SciFact's claims are phrased unlike the abstracts, which is where HyDE is reported to help.
- Never move the week-01..06 branches or tags. Cut `week-07` (branch and tag) and push at the end.

## Review Focus

1. **A reranker failure** (a model download error, or an LLM reranker returning junk) falls back to the retrieval order. It's counted and printed, and the run continues. Test: `test_llm_reranker_garbage_keeps_order`.
2. **The LLM reranker returns ids that weren't in the candidates, or duplicates.** Unknown ids are ignored, duplicates are kept once, and missing candidates are appended in their original order. Test: `test_llm_reranker_partial_permutation`.
3. **Multi-query variants that duplicate the original** are deduped before retrieval, so they don't double-count in RRF. Test: `test_multi_query_dedupes_variants`.
4. **HyDE with keyword search.** The keyword side uses the original question, never the hypothetical answer (an invented document would add invented keywords). Test: `test_hyde_keyword_leg_uses_original`.
5. **Pairs longer than the cross-encoder's 512 tokens** are truncated by the model, so the reranker scores the context line plus the chunk, in that order, with the most informative text first. Test: `test_cross_encoder_input_order` (the chunk text is the start of the passage).

---

### Task 1: Rerankers

**Files:** Create `raggy/retrieve/rerank.py`. Modify `raggy/pipeline.py`, `raggy/config.py`, `raggy/cli.py` (`--rerank`). Test `tests/week07/test_rerank.py`.

**Interfaces (Produces):**
- A `Reranker` protocol: `name: str` and `rerank(question: str, hits: list[Hit]) -> list[Hit]` (reordered, `score` set to the reranker's score).
- `CrossEncoderReranker(model)` uses a lazy fastembed `TextCrossEncoder`. Passages are `source_text(hit)` (context, then text), and `batch_size` is 16.
- `LLMReranker(llm)` uses the prompt "Rank these passages by how well they answer the question. Return JSON {"ranking": [ids]}" with `RERANK_VERSION = "rerank-v1"`. It parses with week 2's `parse_json_object` and falls back as in Review Focus 1–2.
- `make_reranker(config, llm_factory)`.
- `Config`: `rerank` (`RAGGY_RERANK`: none | local | llm, default none until measured), `rerank_model` (`RAGGY_RERANK_MODEL`, default `Xenova/ms-marco-MiniLM-L-6-v2`), and `rerank_candidates` (`RAGGY_RERANK_CANDIDATES`, 20).
- `Raggy.search` retrieves `max(rerank_candidates, want)` children, reranks them and keeps the top `want`, then expands parents.

- [ ] Step 1: Tests: `test_rerank_reorders_by_score` (fake cross-encoder), `test_cross_encoder_input_order`, `test_llm_reranker_orders_by_json`, `test_llm_reranker_partial_permutation`, `test_llm_reranker_garbage_keeps_order`, `test_rerank_before_parent_expansion`, `test_rerank_flag_cli`.
- [ ] Step 2: FAIL → implement → all pass. Commit.

### Task 2: Query transformations

**Files:** Create `raggy/retrieve/transforms.py`. Modify `raggy/pipeline.py`, `raggy/config.py`, `raggy/cli.py` (`--transform`), `raggy/evals/runner.py` (`CACHE_FILES`). Test `tests/week07/test_transforms.py`.

**Interfaces (Produces):**
- `TRANSFORM_VERSION = "qt-v1"`. Three prompts, each producing plain text:
  - **rewrite:** a search query of keywords and likely document phrasing
  - **multi:** 3 alternative phrasings, one per line
  - **hyde:** a short passage that would answer the question, in the documents' style
- `TransformCache(path)` (SQLite).
- `transform(kind, question, llm, cache) -> TransformResult(queries: list[str], hyde_passage: str | None, cached: bool, failed: bool)`.
- `Config.query_transform` (`RAGGY_QUERY_TRANSFORM`: none | rewrite | multi | hyde).
- `Raggy.search` per kind:
  - **rewrite:** both sides use the rewritten query.
  - **multi:** retrieve with the original plus each variant (deduped, case-insensitive), then fuse with RRF.
  - **hyde:** the vector side embeds the hypothetical passage (`embed_documents`, passage-side) and the keyword side uses the original question.

- [ ] Step 1: Tests: `test_rewrite_used_for_both_legs`, `test_multi_query_dedupes_variants`, `test_multi_query_fuses`, `test_hyde_vector_leg_embeds_passage`, `test_hyde_keyword_leg_uses_original`, `test_transform_cached`, `test_transform_failure_falls_back`.
- [ ] Step 2: FAIL → implement → all pass. Commit.

### Task 3: Stage timings and budgets

**Files:** Modify `raggy/pipeline.py` (`last_timings: dict[str, float]`, keys `transform`, `retrieve`, `rerank`), `raggy/config.py` (`budgets`, default `{"transform": 2000, "retrieve": 100, "rerank": 500, "generate": 5000}` ms, `RAGGY_BUDGETS="rerank=300,…"`), `raggy/evals/runner.py` (per-stage p50/p95, `over_budget` list), `raggy/evals/report.py` (prints the stage table with ✓/✗ against the budget), and `raggy/cli.py` (`ask --timings`). Test `tests/week07/test_timings.py`.

- [ ] Step 1: Tests: `test_timings_recorded_per_stage`, `test_report_flags_over_budget`, `test_budgets_parse_env`, `test_ask_timings_flag`.
- [ ] Step 2: FAIL → implement → all pass. Commit.

### Task 4: Measure, decide, lesson

- [ ] Step 1: Retrieval-only runs (week 6's default retrieval as the base) for: rerank none / local MiniLM-L-6 / local bge-reranker-base / LLM; and transform none / rewrite / multi / hyde. Run on fictional, fictional-web, manual and SciFact, with FiQA for rerank only (648 questions × an LLM call per transform is not worth it). Record per-stage latency, and LLM calls and cost per technique.
- [ ] Step 2: Apply the rules, set the defaults, ledger the rulings. Run answer evals for the final default against week 6's default on the three fictional suites. Commit baselines to `evals/baselines/week07/`.
- [ ] Step 3: Write `lessons/week07-rerank-and-query-transforms.md` and `docs/reranking.md` (bi-encoder vs cross-encoder, why rerank only the top N, listwise LLM rerankers, rewrite / multi-query / HyDE, the latency-budget table). Update the README, COURSE_PLAN and the docs hub. Commit.
- [ ] Step 4: Final review → rebase onto main after week 6 → cut `week-07` → push.
