# Week 4: Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chunk boundaries decide what can be found. Add token-sized recursive and structure-aware (Markdown) chunkers, contextual chunk headers (title › section path), parent/child retrieval (search small, answer with the bigger parent), and optional LLM-written chunk context ("contextual retrieval", a paid switch with a cache). Compare them all by numbers on fictional, fictional-web and SciFact, then make the winner the default.

**Architecture:** Every chunker returns `Chunk`s whose `text == doc.text[start:end]`, so week 2's span-based gold keeps working unchanged. `Chunk` gains `context: str = ""` (the header or LLM context: embedded, and shown to the answer model, but not part of the span) and `parent_start/parent_end/parent_text` (optional). The text sent to the embedder is built by one function, `embed_text(chunk)`. Search can expand child hits to their parents (deduplicated), and the eval scores what the model actually sees. It also reports the context size, so a bigger context can't pass for better retrieval.

**Tech Stack:** No new dependencies. Token counts come from the embedding model's own tokenizer (fastembed exposes `model.tokenizer`, a HuggingFace `tokenizers.Tokenizer`, max 512 tokens for bge-small). `FakeEmbedder` counts words, so tests stay offline.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 4 row)

## Global Constraints

- `uv run pytest` passes offline, with no keys and no downloads.
- Weeks 1–3 behaviour stays available: `RAGGY_CHUNKER=fixed` with `chunk_size`/`chunk_overlap` in characters reproduces week 3 exactly.
- Every chunk setting is in the registry's `pipeline_key` and in the eval cache key.
- No embedded text may exceed the embedder's token limit (512 for bge-small). Config validation rejects sizes that can't fit.
- Contextual retrieval costs money: it's off by default, its calls are cached by `(sha256(doc text), chunk start, chunk end, model, CONTEXT_VERSION)` in `index_dir/context-cache.sqlite`, and the CLI prints the number of LLM calls before and after.
- The default chunker is chosen from the measurements: the best mean recall@4 over fictional and fictional-web, provided SciFact nDCG@10 drops by no more than 0.01 against week 3. The lesson shows the table the choice came from.
- Week-01..03 branches and tags are never moved. Cut `week-04` (branch and tag) at the end, then push (approved by the owner).

## Review Focus

1. **Offsets.** For every chunker, on every eval corpus and on random generated Markdown, `chunk.text == doc.text[c.start:c.end]`, chunks are in order, and together they cover every non-whitespace character. Test: `test_offsets_exact_on_all_corpora`, `test_offsets_property_random_markdown`.
2. **A block bigger than the chunk size** (a 3,000-token table, a long code block, a PDF wall of text with no blank lines, one 5,000-character "word") is split down the separator ladder, then by tokens, then by characters. No chunk ever goes over the limit, and the loop always ends. Tests: `test_oversized_table_splits_by_rows`, `test_wall_of_text_splits`, `test_giant_token_hard_split`.
3. **Chunk settings change** → re-index in `raggy ingest` and a rebuild in `raggy eval`. Tests: `test_chunker_change_reindexes`, `test_eval_key_has_chunk_settings`.
4. **Degenerate docs** (empty, whitespace only, headings only, one character) produce no empty chunks and no crash. Test: `test_degenerate_documents`.
5. **Contextual retrieval failure mid-ingest** (LLM error on one chunk). That chunk falls back to its header context, the failure is counted and printed, nothing goes into the cache, and the run continues. Test: `test_llm_context_failure_falls_back_and_is_reported`.
6. **Parent expansion.** Two children of one parent give one hit. `top_k` counts parents, and the eval records context tokens. Test: `test_parent_expansion_dedupes`.

---

### Task 1: Token counting, the recursive and structure chunkers

**Files:** Create `raggy/chunking/tokens.py`, `raggy/chunking/recursive.py`, `raggy/chunking/structure.py`, `raggy/chunking/__init__.py` (`chunk_document`). Modify `raggy/types.py`, `raggy/embed/base.py`, `raggy/embed/fake.py`, `raggy/embed/fastembed_embedder.py`, `raggy/config.py`, `.env.example`. Test `tests/week04/test_chunkers.py`, `test_config.py`.

**Interfaces (Produces):**
- `Embedder.count_tokens(text) -> int` and `Embedder.max_tokens: int`. FastEmbed uses the tokenizer, excluding special tokens, with max 512. Fake uses the `\w+|[^\w\s]` token count, with max 512.
- `Chunk` gains `context: str = ""`, `parent_start: int | None = None`, `parent_end: int | None = None`, `parent_text: str | None = None`. Old `chunks.jsonl` files (without these keys) still load.
- `ChunkSettings(chunker="structure", tokens=256, overlap=32, chars=800, char_overlap=100, headers=True, parent_tokens=0)` is frozen, with `.key() -> dict`.
- `chunk_recursive(doc, count, size, overlap) -> list[Chunk]`. The separator ladder is `["\n\n", "\n", ". ", " ", ""]`. It splits at the highest separator that leaves every piece ≤ size, then greedily merges neighbours up to size, carrying trailing pieces up to `overlap` tokens into the next chunk. Offsets always come from the original text.
- `chunk_structure(doc, count, size, overlap) -> list[Chunk]` parses Markdown blocks (ATX headings, fenced code, pipe tables, list runs, paragraphs) and tracks the heading path. A new heading at level ≤ the current section's starts a new chunk. Blocks are packed whole up to size. An oversized table is split by rows, with the header row kept in each piece's `context`. Oversized code is split by lines. Any other oversized block goes to `chunk_recursive`. Each chunk's `context` is `"<title> › <h1> › <h2>"`, and the title comes from `doc.metadata["title"]`.
- `chunk_document(doc, settings, count) -> list[Chunk]` dispatches by `settings.chunker`, clears `context` when `headers=False`, and adds parents when `parent_tokens > 0` (Task 2).
- `Config` gains `chunker` (`RAGGY_CHUNKER`: fixed|recursive|structure), `chunk_tokens` (`RAGGY_CHUNK_TOKENS`), `chunk_overlap_tokens` (`RAGGY_CHUNK_OVERLAP_TOKENS`), `chunk_headers` (`RAGGY_CHUNK_HEADERS`), `parent_tokens` (`RAGGY_PARENT_TOKENS`, 0 = off) and `chunk_context` (`RAGGY_CHUNK_CONTEXT`: none|llm). `Config.chunk_settings()`. Validation: overlap < tokens; tokens + 64 (header budget) ≤ 512; parent_tokens is 0 or > tokens.

- [ ] Step 1: Tests: the offsets property tests (Review Focus 1), oversized cases (2), degenerate docs (4), `test_structure_breaks_at_headings`, `test_structure_context_path`, `test_table_split_keeps_header_in_context`, `test_code_block_never_split_mid_line`, `test_recursive_respects_size_and_overlap`, `test_old_chunks_jsonl_loads`, and config validation.
- [ ] Step 2: FAIL → implement → `uv run pytest -q` all pass. Commit `feat(week4): token-sized recursive and structure-aware chunkers with section paths`.

### Task 2: Wire it in: embedded text, headers, parents, the prompt, the keys

**Files:** Modify `raggy/pipeline.py`, `raggy/generate/prompt.py`, `raggy/evals/judge.py`, `raggy/evals/runner.py`. Test `tests/week04/test_pipeline_chunking.py`.

**Interfaces (Produces):**
- `embed_text(chunk) -> str` returns `f"{chunk.context}\n\n{chunk.text}"` when there is a context, else `chunk.text`. `ingest_documents` embeds `embed_text(c)`, and refuses (with a `ValueError` naming the chunk) any embed text over `embedder.max_tokens`.
- Parents: in `chunk_document`, when `parent_tokens > 0`, the doc is chunked once at `parent_tokens` (same chunker, no overlap) and once at `tokens`. Each child gets the span and text of the parent that contains its start.
- `Raggy.search(question, k)` fetches `k * 4` children when parents are on, replaces each with a parent `Chunk` (`id=f"{source}#p{parent_start}"`, the parent span and text, the child's context), keeps the first of each parent, and returns `k`.
- The prompt shows `[n] (source: name › section path)` followed by the text. The judge's grounding prompt shows the same text the model saw.
- `pipeline_key` and the eval cache key include `config.chunk_settings().key()` and `chunk_context`. The run config records them.
- Run metrics gain `context_tokens_p50` and `context_tokens_p95`: the tokens in the top_k texts given to the model, per question.

- [ ] Step 1: Tests: `test_headers_are_embedded_not_in_span`, `test_parent_expansion_dedupes`, `test_top_k_counts_parents`, `test_prompt_shows_section_path`, `test_embed_text_over_limit_raises`, `test_chunker_change_reindexes`, `test_eval_key_has_chunk_settings`, `test_run_records_context_tokens`, `test_fixed_chunker_reproduces_week3` (same chunks as `chunk_fixed`).
- [ ] Step 2: FAIL → implement → all pass. Commit `feat(week4): contextual headers, parent/child retrieval, chunk settings in keys`.

### Task 3: Contextual retrieval (LLM-written chunk context, cached)

**Files:** Create `raggy/chunking/contextual.py`. Modify `raggy/pipeline.py`, `raggy/cli.py`. Test `tests/week04/test_contextual.py`.

**Interfaces (Produces):**
- `CONTEXT_VERSION = "ctx-v1"` and `CONTEXT_SYSTEM`: write 1–2 sentences situating the chunk within the document for search; output only the context.
- `ContextCache(path)` is SQLite with `get(key)`, `put(key, text)` and `commit()`.
- `add_llm_context(doc, chunks, llm, cache) -> ContextStats(calls, cached, failed)`. The user prompt is `<document>…</document>` followed by `<chunk>…</chunk>`, with the document first so a provider's prompt cache can reuse it across that document's chunks. The generated text is placed before the header context: `"{llm text}\n{header}"`. On `LLMError`, the chunk keeps its header and the error is counted.
- `ingest_documents` calls it when `config.chunk_context == "llm"`, using `self.llm`. `IngestReport` gains `context_calls`, `context_cached` and `context_failed`. The CLI prints `contextual retrieval: N LLM calls (M cached, F failed)`.

- [ ] Step 1: Tests: `test_context_prepended_and_cached` (second ingest makes 0 calls), `test_llm_context_failure_falls_back_and_is_reported`, `test_cache_key_includes_model_and_version`, `test_document_first_in_prompt`.
- [ ] Step 2: FAIL → implement → all pass. Commit `feat(week4): optional contextual retrieval with a persistent cache`.

### Task 4: Measure, choose the default, lesson

- [ ] Step 1: Retrieval-only runs on fictional, fictional-web and SciFact for: fixed 800/100 chars (week 3); recursive 256/32; structure 256/32 without headers; structure 256/32 with headers; structure with headers and parent 1024; structure with headers at 128 and 512 tokens. Contextual retrieval on fictional and fictional-web only (SciFact would be 13k paid calls), with its call count and cost.
- [ ] Step 2: Choose the default by the Global Constraints rule, set it in `Config`, and ledger the ruling with the numbers. Answer and judge runs (same models as the week 3 baselines) for the week 3 config and the new default on fictional and fictional-web. Commit the baselines to `evals/baselines/week04-*.json`.
- [ ] Step 3: Write `lessons/week04-chunking.md` and `docs/chunking.md`, and update the README, COURSE_PLAN, the docs hub and the week 3 "next" link. Commit `docs(week4): baselines, lesson, chunking guide`.
- [ ] Step 4: Final whole-branch review → fast-forward main → cut the `week-04` branch and tag → push main, the branch and the tag.
