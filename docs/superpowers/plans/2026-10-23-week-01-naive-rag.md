# Week 1: Naive RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working end-to-end RAG in about 600 lines. `raggy ingest <folder|file>` indexes `.txt`/`.md`, and `raggy ask "<q>"` answers from them with numbered citations. A lesson and a concept guide explain every line.

**Architecture:** The pipeline is load → fixed-size chunks → local embeddings → numpy matrix on disk → cosine top-k → prompt with numbered sources → LLM → answer with parsed citations. Every stage sits behind a tiny interface (Embedder, LLM, store), so later weeks can swap one stage without touching the others.

**Tech Stack:** Python 3.11+, uv, numpy, fastembed (ONNX, no PyTorch), anthropic SDK, openai SDK (also reaches Ollama through its OpenAI-compatible endpoint), python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md`

## Global Constraints

- No RAG/agent frameworks (no LangChain, LlamaIndex, Haystack).
- `uv run pytest` passes offline with no API keys and no model downloads. Tests use `FakeEmbedder` and `FakeLLM`.
- API keys only from `.env` (loaded by python-dotenv), never the shell rc file.
- Default answer model: Ollama `qwen3.5:4b` (local, free). Claude default model: `claude-opus-5-5` with `output_config={"effort": "medium"}` set explicitly, `fallbacks="default"` with beta `server-side-fallback-2026-07-01`, and a check for `stop_reason == "refusal"`. OpenAI default model: from `OPENAI_MODEL`. All three are overridable with `RAGGY_LLM_MODEL`.
- Default embedder: fastembed `BAAI/bge-small-en-v1.5` (384-dim). The owner's machine is an M2 with 8 GB, so defaults stay small.
- Every code sample and command output in the lesson is copied from a real run. Library versions are named in the lesson.
- At the end, branch `week-01` and tag `week-01` are cut from the week's final commit. They are never moved afterwards.

## Review Focus

1. **`ask` before any `ingest`.** It must print "No index found — run `raggy ingest <path>` first" and exit 1, not raise a traceback. Test: `test_ask_without_index_errors_cleanly`.
2. **Re-ingesting the same file.** It must replace that source's chunks, not duplicate them. Test: `test_reingest_replaces_source`.
3. **Index built with one embedding model, queried with another.** It must raise a clear mismatch error that names both models. Test: `test_store_rejects_embedder_mismatch`.
4. **Folder with no supported files, or a non-UTF-8 file.** It must report "0 documents" or skip the file with a warning, not crash. Tests: `test_load_empty_folder`, `test_load_skips_undecodable_file`.
5. **LLM unreachable** (Ollama not running, no key). It must give a one-line actionable message naming the provider and how to fix it. Test: `test_cli_reports_llm_error` using a FakeLLM that raises `LLMError`.

---

### Task 1: Project skeleton, types, loader, chunker

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `raggy/__init__.py`, `raggy/types.py`, `raggy/ingest/__init__.py`, `raggy/ingest/loaders.py`, `raggy/chunking/__init__.py`, `raggy/chunking/fixed.py`
- Test: `tests/week01/test_loaders.py`, `tests/week01/test_chunking.py`

**Interfaces (Produces):**
- `@dataclass Document(source: str, text: str, metadata: dict = {})`. `source` is the absolute path, or later a URL.
- `@dataclass Chunk(id: str, source: str, text: str, index: int, start: int, end: int)`. `id = f"{source}#{index}"`. `start`/`end` are character offsets into the document text.
- `@dataclass Hit(chunk: Chunk, score: float)`
- `@dataclass Citation(number: int, source: str, chunk_id: str, snippet: str)`
- `@dataclass Answer(question: str, text: str, citations: list[Citation], hits: list[Hit])`
- `load_path(path: str | Path) -> list[Document]` handles a file or a recursive folder, `.txt`/`.md`/`.markdown` only, sorted by path, skipping hidden directories. Undecodable files are skipped with a `warnings.warn`.
- `SUPPORTED_SUFFIXES: frozenset[str]`
- `chunk_fixed(doc: Document, size: int = 800, overlap: int = 100) -> list[Chunk]` uses character windows with step `size - overlap`. A trailing window shorter than `overlap` is dropped. Whitespace-only chunks are dropped. `overlap >= size` raises `ValueError`.

- [ ] Step 1: Write `pyproject.toml` (hatchling build, `[project.scripts] raggy = "raggy.cli:main"`, deps as in Tech Stack, dev group pytest, `pythonpath=["."]`). Run `uv sync`. Record the resolved versions of fastembed/numpy/anthropic/openai.
- [ ] Step 2: Write the tests: `test_load_single_file`, `test_load_folder_recursive_sorted`, `test_load_ignores_unsupported_and_hidden`, `test_load_empty_folder` (returns `[]`), `test_load_skips_undecodable_file` (writes `b"\xff\xfe\x00bad"` to `bad.md` and asserts a warning and no exception), `test_load_missing_path_raises` (`FileNotFoundError`). `test_chunk_covers_text` (joining the chunks with overlap removed reproduces the text), `test_chunk_ids_and_offsets` (`doc.text[c.start:c.end] == c.text`), `test_chunk_short_doc_single_chunk`, `test_chunk_bad_overlap_raises`.
- [ ] Step 3: Run `uv run pytest tests/week01 -q`. Expect failures (ImportError).
- [ ] Step 4: Implement types, loader and chunker.
- [ ] Step 5: Run the tests again. Expect PASS. Commit `feat(week1): types, folder loader, fixed-size chunker`.

### Task 2: Embedders and the numpy vector store

**Files:**
- Create: `raggy/embed/__init__.py`, `raggy/embed/base.py`, `raggy/embed/fake.py`, `raggy/embed/fastembed_embedder.py`, `raggy/store/__init__.py`, `raggy/store/numpy_store.py`
- Test: `tests/week01/test_store.py`, `tests/week01/test_fake_embedder.py`

**Interfaces:**
- Consumes: `Chunk`, `Hit` from Task 1.
- Produces:
  - `class Embedder(Protocol): name: str; dim: int; def embed_documents(self, texts: list[str]) -> np.ndarray; def embed_query(self, text: str) -> np.ndarray`. Rows are L2-normalised float32.
  - `FakeEmbedder(dim: int = 64)` is deterministic. It hashes lowercase word tokens into buckets, then normalises (bag of words), so word overlap means similarity. `name = "fake-bow-64"`.
  - `FastEmbedEmbedder(model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64)` wraps `fastembed.TextEmbedding`. It uses the library's query/passage methods if the installed version exposes them (verify against the installed release), and normalises the output.
  - `class NumpyStore(path: Path)` with `add(chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None` (replaces existing chunks with the same `source` first), `remove_source(source: str) -> int`, `search(query_vec: np.ndarray, k: int, embedder_name: str) -> list[Hit]` (cosine = dot product of normalised vectors, `np.argpartition` then sort, best first), `sources() -> dict[str, int]`, `save() -> None`, `@classmethod load(path) -> NumpyStore` (raises `IndexNotFoundError` if missing), `__len__`.
  - On-disk layout: `vectors.npy`, `chunks.jsonl`, `meta.json` (`{"embedder": name, "dim": d, "version": 1}`).
  - `class EmbedderMismatchError(Exception)` and `class IndexNotFoundError(Exception)`, both in `raggy/store/numpy_store.py`.

- [ ] Step 1: Write the tests: `test_fake_embedder_normalised_and_deterministic`, `test_fake_embedder_overlap_scores_higher`, `test_store_search_returns_best_first` (3 chunks, query equals one chunk's text, that chunk ranks first with score ≈ 1.0), `test_store_k_larger_than_size`, `test_store_roundtrip_save_load`, `test_reingest_replaces_source`, `test_store_rejects_embedder_mismatch` (message contains both names), `test_load_missing_raises_index_not_found`.
- [ ] Step 2: Run the tests. Expect FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Run the tests. Expect PASS. Then a live smoke check (not a test): `uv run python -c "from raggy.embed.fastembed_embedder import FastEmbedEmbedder as E; e=E(); print(e.dim, e.embed_query('hi').shape)"` must print `384 (384,)`. Record the real output.
- [ ] Step 5: Commit `feat(week1): embedders and numpy vector store`.

### Task 3: LLM adapters, prompt, answer with citations

**Files:**
- Create: `raggy/generate/__init__.py`, `raggy/generate/llm.py`, `raggy/generate/anthropic_llm.py`, `raggy/generate/openai_llm.py`, `raggy/generate/fake.py`, `raggy/generate/prompt.py`, `raggy/generate/answer.py`
- Test: `tests/week01/test_prompt.py`, `tests/week01/test_answer.py`, `tests/week01/test_llm_factory.py`

**Interfaces:**
- Consumes: `Hit`, `Citation`, `Answer`.
- Produces:
  - `class LLM(Protocol): name: str; def complete(self, system: str, user: str) -> str`
  - `class LLMError(Exception)`: every adapter wraps SDK and connection errors into it, with a fix hint.
  - `AnthropicLLM(model="claude-opus-5-5", effort="medium", max_tokens=16000)` uses `client.beta.messages.create(..., output_config={"effort": effort}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")`. It raises `LLMError` on `stop_reason == "refusal"` and joins the `text` blocks.
  - `OpenAILLM(model: str, base_url: str | None = None, api_key: str | None = None)` uses Chat Completions. Ollama is `OpenAILLM(model, base_url="http://localhost:11434/v1", api_key="ollama")`.
  - `FakeLLM(reply: str | Callable[[str, str], str])` records `.calls`.
  - `make_llm(provider: str, model: str | None) -> LLM` takes provider `ollama|anthropic|openai`. An unknown provider raises `ValueError`.
  - `SYSTEM_PROMPT: str` says: answer only from the sources; cite each claim like `[1]`; if the sources don't contain the answer, say exactly `I couldn't find this in the documents.`
  - `build_user_prompt(question: str, hits: list[Hit]) -> str` lists numbered sources `[n] (source: <file name>)\n<text>`, then the question.
  - `parse_citations(text: str, hits: list[Hit]) -> list[Citation]` uses the unique `[n]` markers in order of first appearance. Out-of-range numbers are ignored. The snippet is the first 160 characters of the chunk.
  - `generate_answer(question: str, hits: list[Hit], llm: LLM) -> Answer`. With empty `hits`, it returns the not-found sentence without calling the LLM.

- [ ] Step 1: Write the tests: `test_prompt_numbers_sources_in_rank_order`, `test_parse_citations_dedup_order_and_range`, `test_generate_answer_uses_llm_and_parses` (FakeLLM returns `"Paris [2] and [1] [9]"` → citations 2 then 1), `test_generate_answer_no_hits_skips_llm`, `test_make_llm_providers` (checks types only; the clients are constructed lazily, so there are no network calls or keys), `test_make_llm_unknown_raises`.
- [ ] Step 2: Run the tests. Expect FAIL.
- [ ] Step 3: Implement. Adapters create SDK clients lazily on the first `complete`, so constructing one needs no key.
- [ ] Step 4: Run the tests. Expect PASS. Live smoke check against Ollama `qwen3.5:4b`: `complete("Be brief.", "Say hi")` returns non-empty text. Run the Claude check only if a key or profile is available, and report which providers were actually exercised.
- [ ] Step 5: Commit `feat(week1): LLM adapters and grounded answer with citations`.

### Task 4: Pipeline, config, CLI, sample corpus

**Files:**
- Create: `raggy/config.py`, `raggy/pipeline.py`, `raggy/cli.py`, `samples/week01/*.md` (5–6 short fictional docs about an invented company, so answers can't come from model memory)
- Test: `tests/week01/test_pipeline.py`, `tests/week01/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass Config(index_dir: Path, embed_model: str, llm_provider: str, llm_model: str | None, top_k: int = 4, chunk_size: int = 800, chunk_overlap: int = 100)` and `Config.from_env() -> Config`. It reads `.env`, then `RAGGY_INDEX_DIR` (default `.raggy`), `RAGGY_EMBED_MODEL`, `RAGGY_LLM` (default `ollama`), `RAGGY_LLM_MODEL`, `RAGGY_TOP_K`.
  - `class Raggy(config: Config, embedder: Embedder | None = None, llm: LLM | None = None)` with `ingest(paths: list[str]) -> IngestReport(documents: int, chunks: int, skipped: list[str])`, `search(question: str, k: int | None = None) -> list[Hit]`, `ask(question: str, k: int | None = None) -> Answer`, `sources() -> dict[str, int]`. Embedder and LLM are injected in tests and built from config otherwise.
  - CLI `main(argv: list[str] | None = None) -> int` with subcommands `ingest PATH...`, `search QUERY [-k]`, `ask QUESTION [-k] [--show-context]`, `sources`. `ask` prints the answer, then `Sources:` lines `[n] <file name> — "<snippet>…"`. `IndexNotFoundError` and `LLMError` print one line to stderr and return 1.

- [ ] Step 1: Write the tests (with `tmp_path`, FakeEmbedder and FakeLLM): `test_pipeline_ingest_then_search_finds_relevant`, `test_pipeline_ask_returns_citations`, `test_pipeline_ingest_report_counts`, `test_ask_without_index_errors_cleanly` (CLI, exit 1, message contains `raggy ingest`), `test_cli_reports_llm_error`, `test_cli_sources_lists_files`.
- [ ] Step 2: Run the tests. Expect FAIL.
- [ ] Step 3: Implement. The CLI gets fakes through a module-level factory hook `_build_raggy(config)` that tests monkeypatch.
- [ ] Step 4: Run the full suite. Expect all PASS. Then a live end-to-end run on `samples/week01` with the real embedder and Ollama: `uv run raggy ingest samples/week01`, `uv run raggy search "..."`, `uv run raggy ask "..."` with one answerable and one unanswerable question. Save the real outputs for the lesson.
- [ ] Step 5: Commit `feat(week1): pipeline, CLI and sample corpus`.

### Task 5: Lesson, concept guide, README, course plan, week branch

**Files:**
- Create: `lessons/week01-naive-rag.md`, `docs/embeddings-and-vector-search.md`, `docs/README.md`, `README.md`, `COURSE_PLAN.md`

**Content requirements:**
- `lessons/week01-naive-rag.md` uses harnessy's lesson shape: breadcrumb nav, time, "You're done when", 1. Read first (links), 2. What you're building (Mermaid flow of the pipeline), 3. the walkthrough, stage by stage, pointing at real files and functions with short excerpts copied from the code, 4. Run it (real outputs from Task 4), 5. Why each choice (chunk size and overlap, why normalise, why cosine equals dot product, why number the sources, why the not-found sentence, why a local model by default), 6. What breaks (the naive pipeline's known weaknesses, each tied to the week that fixes it), 7. Notes prompts. It names the library versions used.
- `docs/embeddings-and-vector-search.md` covers what an embedding is, cosine similarity with a tiny worked numeric example (computed and checked with numpy), brute force vs ANN (preview of week 5), and query vs passage embeddings.
- `README.md` covers what raggy is, quick start, CLI cheat sheet and links. `COURSE_PLAN.md` has the 12-week table from the spec (week 1 linked, later weeks marked "coming"), how to use week branches, and setup.

- [ ] Step 1: Write the docs. Check that every code excerpt matches the file (grep each quoted line).
- [ ] Step 2: `uv run pytest -q`. All PASS.
- [ ] Step 3: Commit `docs(week1): lesson, embeddings guide, README and course plan`.
- [ ] Step 4: `git branch week-01 && git tag week-01`.
