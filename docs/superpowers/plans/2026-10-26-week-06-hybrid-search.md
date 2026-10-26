# Week 6: Hybrid Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vectors miss exact names, codes and rare words; keywords miss paraphrases. Add a BM25 full-text index beside the vectors in the same LanceDB table, a keyword-only search mode, and hybrid search that fuses the two rankings with reciprocal rank fusion (RRF). Week 5's filters apply to both sides. Measure vector-only, keyword-only and hybrid on every suite, and choose the default by a rule written down before the runs.

**Architecture:** LanceStore gains a `fts_text` column holding the same text that is embedded (context header + chunk), and a native LanceDB FTS index on it (lower-casing, English stemming, stop words removed). `LanceStore.keyword_search` sends the question as a `MatchQuery`, so user text is never parsed as query syntax. `raggy/retrieve/fusion.py: rrf()` is a 15-line pure function. `Raggy.search` takes N candidates from each side (default 50), fuses them, keeps the top k, then does week 4's parent expansion. NumpyStore has no full-text index; keyword and hybrid modes on it fail with a clear message.

**Tech Stack:** No new dependencies (LanceDB 0.40's native FTS). Verified before planning: the `create_index(field, config=FTS(...))` API; codes such as `BW-CH-1142` match through the tokenizer; stemming ("refund" finds "Refunds"); `where` prefilters work on FTS; rows added after the index was built are still found; questions containing `?`, quotes, `AND`, `(` are handled safely by `MatchQuery`.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 6 row)

**Branching:** week 5's code is complete but its measurements and lesson are still running, so this work branches from `dev/week-05` and is rebased onto main once week 5 is merged. The `week-05` snapshot must not contain week 6 code. No heavy eval runs until week 5's FiQA measurements are done, so its throughput numbers stay clean.

## Global Constraints

- `uv run pytest` passes offline.
- The retrieval mode, RRF k, the candidate count and `FTS_VERSION` go into the eval cache key and the run config. FTS needs no settings-key change for `raggy ingest`, because `finish()` creates or optimizes the FTS index on every ingest, so existing indexes gain it on the next run.
- **Decision rule, set before any run (observation 31).** Hybrid becomes the default if (a) mean recall@4 over fictional, fictional-web and manual is higher than vector-only, and (b) nDCG@10 on SciFact and on FiQA is not lower than vector-only by more than 0.005, and (c) p95 retrieval latency on FiQA stays under 100 ms. Otherwise vector-only stays the default. Held constant: chunks, embedder, top-k, and so context size. Latency is reported with the hardware named.
- (Observation 33) The data can tell the modes apart: fictional has 5 `exact-term` questions (part codes, a name), and SciFact and FiQA are standard BM25-vs-dense benchmarks with published BM25 numbers to check against (BEIR: SciFact 0.665, FiQA 0.236 nDCG@10).
- Never move week-01..05 branches or tags. Cut `week-06` (branch and tag) at the end and push.

## Review Focus

1. **A question that is all stop words or punctuation** ("what is it?", "???") returns no keyword hits, not an error. Hybrid then falls back to the vector ranking. Test: `test_keyword_empty_query`.
2. **A chunk found by both sides counts once**, with both contributions summed. Ties break deterministically (by best rank, then id). Tests: `test_rrf_sums_and_dedupes`, `test_rrf_tie_break_stable`.
3. **Filters apply to both sides before fusion.** "Top 4 PDFs" must not let a markdown keyword hit through. Test: `test_hybrid_respects_filters`.
4. **An index built before week 6** (no `fts_text` column) is detected, and a clear error says to re-ingest, rather than a LanceDB schema error. Test: `test_old_lance_index_without_fts_text`.
5. **Parent expansion after fusion** keeps each parent once, and `top_k` counts parents. Test: `test_hybrid_with_parents`.

---

### Task 1: Full-text index and keyword search in LanceStore

**Files:** Modify `raggy/store/lance_store.py`, `raggy/store/base.py` (`keyword_search` in the protocol; NumpyStore raises `KeywordSearchUnavailable`), `raggy/pipeline.py` (`embed_text` reused for `fts_text`). Test `tests/week06/test_keyword.py`.

**Interfaces (Produces):**
- `FTS_VERSION = 1`. Rows gain `fts_text = embed_text(chunk)`.
- `LanceStore.finish()` creates `FTS(stem=True, remove_stop_words=True, lower_case=True, ascii_folding=True, with_position=False)` on `fts_text` if it's missing and the table has rows; otherwise it calls `optimize()`.
- `LanceStore.keyword_search(question, k, where=None) -> list[Hit]` uses `MatchQuery(question, "fts_text")` with `query_type="fts"`, and the score is the BM25 `_score`. An empty or unknown query returns `[]`.
- `class KeywordSearchUnavailable(Exception)`: raised by NumpyStore, and by LanceStore for an index without `fts_text`, with the message "this index was built before week 6: run `raggy ingest` again" (RAGGY_STORE=lance needed for keyword search).

- [ ] Step 1: Tests: `test_keyword_finds_exact_code`, `test_keyword_stems`, `test_keyword_empty_query`, `test_keyword_respects_filters`, `test_keyword_finds_rows_added_after_index`, `test_keyword_on_numpy_store_raises`, `test_old_lance_index_without_fts_text`.
- [ ] Step 2: FAIL → implement → `uv run pytest -q` all pass. Commit.

### Task 2: Reciprocal rank fusion and retrieval modes

**Files:** Create `raggy/retrieve/__init__.py` and `raggy/retrieve/fusion.py`. Modify `raggy/pipeline.py`, `raggy/config.py`, `raggy/cli.py` (`--mode` on search, ask and eval), `raggy/evals/runner.py`. Test `tests/week06/test_fusion.py`, `tests/week06/test_hybrid.py`.

**Interfaces (Produces):**
- `rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]`. The score of chunk c is Σ 1/(k + rank_i(c)), with ranks starting at 1. A chunk is identified by `chunk.id`. Ties break by best single rank, then by id. The returned Hit carries the fused score.
- `Config.retrieval: str = "vector"` (`RAGGY_RETRIEVAL`: vector | keyword | hybrid), `rrf_k: int = 60` (`RAGGY_RRF_K`) and `fusion_candidates: int = 50` (`RAGGY_FUSION_CANDIDATES`). The default is changed in Task 3 only if the rule passes.
- `Raggy.search(question, k, filters, exact, mode=None)`. Candidates are `max(k * 4, fusion_candidates)` from each side when fusing. Week 5's filters go to both sides, and parent expansion happens after fusion.
- The run config records `retrieval`, `rrf_k` and `fusion_candidates`, and they're in the eval cache key with `FTS_VERSION`.

- [ ] Step 1: Tests: `test_rrf_sums_and_dedupes` (hand-computed: rankings [a,b,c] and [c,a] with k=60 give a = 1/61+1/62, c = 1/63+1/61, b = 1/62), `test_rrf_tie_break_stable`, `test_rrf_single_ranking_preserves_order`, `test_hybrid_finds_code_vector_misses` (FakeEmbedder with a code-heavy corpus), `test_hybrid_respects_filters`, `test_hybrid_with_parents`, `test_mode_flag_cli`, `test_eval_key_has_retrieval_settings`.
- [ ] Step 2: FAIL → implement → all pass. Commit.

### Task 3: Measure, decide, lesson

- [ ] Step 1 (after week 5's runs): Retrieval-only runs for vector, keyword and hybrid on fictional, fictional-web, manual, SciFact and FiQA (embeddings come from the cache). Record recall@k, nDCG, MRR and p50/p95 latency, plus a per-tag breakdown for `exact-term` and `paraphrase`. Check keyword-only SciFact and FiQA nDCG@10 against BEIR's published BM25 (0.665, 0.236) as a ruler check.
- [ ] Step 2: Apply the rule, set the default, ledger the ruling with the numbers. Commit baselines to `evals/baselines/week06/`.
- [ ] Step 3: Write `lessons/week06-hybrid-search.md` and `docs/hybrid-search.md` (BM25 worked by hand: tf saturation k1, length normalisation b, idf; RRF worked by hand; why ranks and not scores are fused). Update the README, COURSE_PLAN and the docs hub. Commit.
- [ ] Step 4: Final review → after week 5 is on main: rebase onto main, fast-forward, cut `week-06` (branch and tag), push.
