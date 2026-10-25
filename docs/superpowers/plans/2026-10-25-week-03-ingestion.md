# Week 3: Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `raggy ingest <file|folder|url>...` loads Markdown, text, HTML (main content only) and PDF text, from folders with ignore rules and from URLs. The text is normalised, duplicates are dropped, and each document carries metadata. A SQLite registry makes re-ingest incremental, so unchanged sources are skipped, changed ones are re-indexed and deleted ones are removed. A new `fictional-web` eval suite measures what cleaner text and dedupe are worth.

**Architecture:** Loaders are chosen by file suffix or HTTP content type from one registry (`raggy/ingest/loaders/`). Every loader returns a `Document` whose text has been normalised by `normalize_text`, with metadata `{name, title, type, kind, modified}`. `raggy/ingest/registry.py` stores one row per source: content hash, size and mtime, the pipeline settings key, the MinHash signature, `duplicate_of`, the chunk count and the timestamps. `Raggy.ingest` stats each file, reads only those whose size or mtime changed, compares hashes, and sends only new or changed documents to the existing `ingest_documents` batch path. It saves the vector store first and commits the registry second, so a crash between the two re-embeds rather than losing chunks.

**Tech Stack:** Adds `trafilatura==2.3.x` (Apache-2.0, HTML main-content extraction to Markdown), `pypdf>=6.21` (BSD, PDF text), `pathspec>=1.1` (MPL-2.0, gitignore-style patterns) and `httpx>=0.28` (BSD; already present through the SDKs, now declared because raggy imports it). SQLite, unicodedata and hashlib come from the standard library.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 4 `raggy ingest` / `sources` / `remove`; section 7 week 3 row)

## Global Constraints

- `uv run pytest` passes offline, with no keys and no network. URL tests use `httpx.MockTransport`. PDF tests build their PDF with pypdf in the test.
- No RAG framework. Each new dependency is imported by raggy code (no dependency without a consumer).
- Week-01 and week-02 branches and tags are never moved. Cut `week-03` (branch and tag) from main at the end.
- Behaviour from weeks 1–2 is kept: `load_path`, `IngestReport.documents/chunks/skipped`, `ingest_documents` and `raggy eval` for the existing suites.
- Every number in the lesson comes from a real run that the lesson shows.
- Nothing raggy does ever deletes or modifies a user's source files. "Remove" only ever removes from the index and registry.

## Review Focus

1. **A folder target that's missing** (typo, unmounted drive) must not wipe its sources from the index. It is reported as not found, with a hint to run `raggy remove`. Only files missing *inside a folder that was walked* count as deleted. Test: `test_missing_folder_target_keeps_its_sources`.
2. **Changing chunk size or HTML extraction mode** with unchanged files must re-index them, not skip them, because the registry's per-source `pipeline_key` differs. Test: `test_settings_change_reindexes_unchanged_files`.
3. **A crash between the store save and the registry commit** must leave a state that the next run repairs. Order: store save, then registry commit. Test: `test_registry_commit_after_store_save` (registry write raises → the next ingest re-indexes the file, not skips it).
4. **One broken file** (a corrupt PDF, an HTML page with no extractable text, a fetch error or 404) is counted as `failed` with its reason. The run continues, and a previously indexed copy of that source is kept. Tests: `test_corrupt_pdf_is_reported_and_run_continues`, `test_failed_refetch_keeps_old_chunks`.
5. **The canonical copy of a duplicate is deleted or changed.** Its duplicates must be re-examined in the same run, and the first surviving one gets indexed, so content never silently disappears from the index. Test: `test_duplicate_promoted_when_canonical_deleted`.
6. (Observation 29) The only deletes are `NumpyStore.remove_source` and registry rows. The registry file lives at `config.index_dir / "registry.sqlite"`, and `index_dir` is already blank-safe. No step deletes files.

---

### Task 1: Normalisation, the loader registry, HTML and PDF, folder walk with ignore rules, metadata

**Files:**
- Create: `raggy/ingest/normalize.py`; `raggy/ingest/loaders/__init__.py`, `text.py`, `html.py`, `pdf.py`; `raggy/ingest/walk.py`
- Delete: `raggy/ingest/loaders.py` (it becomes the package; the `from raggy.ingest.loaders import load_path` import keeps working)
- Modify: `raggy/config.py`, `.env.example`, `raggy/evals/dataset.py` (table pipes), `pyproject.toml`
- Test: `tests/week03/test_normalize.py`, `test_loaders.py`, `test_walk.py`, `test_config.py`

**Interfaces (Produces):**
- `NORMALIZE_VERSION = 1`. `normalize_text(text: str) -> str` applies NFKC; drops soft hyphens, zero-width characters and the BOM; turns `\r\n`/`\r` into `\n`; collapses runs of spaces and tabs *after* each line's leading indentation (code indentation survives); strips trailing spaces; turns 3+ newlines into 2; strips the ends.
- `dehyphenate(text) -> str` joins `([a-z])-\n([a-z])`. Used by the PDF loader only.
- `@dataclass LoadOptions(html_extract: str = "main")`. `"main"` uses trafilatura Markdown output with tables; `"full"` is every visible text node (stdlib `HTMLParser`, skipping script/style/noscript/template) and is kept as the baseline that week 3 measures against.
- `class LoadError(Exception)`: the reason one source couldn't be loaded.
- `load_bytes(data: bytes, source: str, kind: str, options: LoadOptions | None = None) -> Document`. `kind` is a key of `LOADERS` (`"text"`, `"markdown"`, `"html"`, `"pdf"`). Raises `LoadError` for no extractable text, undecodable text or a broken PDF.
- `SUFFIXES: dict[str, str]`: `.txt→text`, `.md/.markdown→markdown`, `.html/.htm→html`, `.pdf→pdf`.
- `load_file(path: Path, options=None) -> Document` sets `source=str(path.resolve())`. Metadata: `name`, `title` (HTML `<title>` or trafilatura title; Markdown's first `# ` heading; PDF `/Title`; else the file stem), `type` (the kind), `kind: "file"`, `modified` (ISO-8601 UTC from mtime).
- `load_path(path, options=None) -> list[Document]` keeps week 1's behaviour: a missing path raises `FileNotFoundError`; an undecodable file warns `UserWarning` naming the file and is skipped. It now walks with `iter_files`.
- `iter_files(root: Path) -> list[Path]`: sorted, supported suffixes only. It prunes hidden files and folders, `node_modules/` and `__pycache__/`, and patterns from `root/.raggyignore` and `root/.gitignore` (gitwildmatch via pathspec). Ignored folders are not descended into.
- `Config` gains `html_extract: str = "main"` (`RAGGY_HTML_EXTRACT`, anything other than main/full raises `ConfigError`) and `dedupe: bool = True` (`RAGGY_DEDUPE`, accepts `1/0/true/false/yes/no`, else `ConfigError`). `Config.load_options() -> LoadOptions`.
- `resolve_gold` treats `|` tokens in a quote as optional separators, so a table-row quote matches with or without pipes.

- [ ] Step 1: Write the tests.
  - Normalisation: `test_normalize_collapses_inner_space_keeps_indent`, `test_normalize_nfkc_and_invisible_chars` ("ﬁ"→"fi", `­`/`​` gone), `test_normalize_newlines`, `test_normalize_idempotent`, `test_dehyphenate_only_lowercase_wraps`.
  - Loaders: `test_html_main_drops_boilerplate` (nav/footer/cookie text absent, body present, table kept as `| a | b |`), `test_html_full_keeps_everything_but_scripts`, `test_html_title_metadata`, `test_markdown_title_from_heading`, `test_pdf_text_and_title` (build a PDF with pypdf in the test), `test_corrupt_pdf_raises_load_error`, `test_empty_html_raises_load_error`, `test_load_file_metadata_fields`.
  - Walk: `test_walk_respects_raggyignore`, `test_walk_respects_gitignore_dirs`, `test_walk_skips_node_modules_and_hidden`, `test_walk_includes_html_and_pdf`.
  - Config: `test_bad_html_extract_raises`, `test_dedupe_flag_parsing`.
  - Dataset: `test_resolve_gold_ignores_table_pipes` (in `tests/week03/test_loaders.py`).
- [ ] Step 2: `uv run pytest tests/week03 -q` → FAIL. `uv add "trafilatura>=2.3.1" "pypdf>=6.21" "pathspec>=1.1" "httpx>=0.28"`. Implement. `uv run pytest -q` → all pass, including weeks 1–2. Commit `feat(week3): loader registry (text, markdown, html, pdf), normalisation, ignore rules, metadata`.

### Task 2: URLs

**Files:** Create `raggy/ingest/fetch.py`. Test `tests/week03/test_fetch.py`.

**Interfaces (Produces):**
- `is_url(s: str) -> bool` (http/https only).
- `class FetchError(LoadError)`.
- `fetch_url(url: str, options=None, client: httpx.Client | None = None, max_bytes: int = 20_000_000) -> Document`. GET with redirects followed, a 20 s timeout and the user agent `raggy/<version>`. The response's content type picks the loader: `text/html` and `application/xhtml+xml` → html; `application/pdf` → pdf; `text/plain` → text; `text/markdown` → markdown. When the content type is missing or generic (`application/octet-stream`), the URL path suffix decides. Anything else raises `FetchError("unsupported content type …")`. HTTP ≥ 400 raises `FetchError` with the status. A body over `max_bytes` (streamed, checked while reading) raises. `source` is the URL as given. Metadata adds `kind: "url"`, `final_url`, and `modified` from `Last-Modified` when present.

- [ ] Step 1: Tests with `httpx.MockTransport`: `test_fetch_html_main_content`, `test_fetch_pdf_by_content_type`, `test_fetch_follows_redirect_records_final_url`, `test_fetch_404_raises_with_status`, `test_fetch_unsupported_type`, `test_fetch_too_large`, `test_fetch_suffix_fallback_for_octet_stream`, `test_is_url`.
- [ ] Step 2: FAIL → implement → all pass. Commit `feat(week3): fetch URLs (html, pdf, text) with size and type limits`.

### Task 3: Duplicates

**Files:** Create `raggy/ingest/dedupe.py`. Test `tests/week03/test_dedupe.py`.

**Interfaces (Produces):**
- `content_hash(text: str) -> str`: sha256 of the normalised text.
- `minhash(text: str) -> np.ndarray | None`: 128 uint64 values over 5-word shingles of lower-cased `\w+` words. Hash: blake2b-64 per shingle, then `(a*x + b) mod (2**61 - 1)` with fixed seeded `a, b`. Returns `None` below `MIN_WORDS = 50` (too short to judge, so exact matching only).
- `similarity(a, b) -> float` = the fraction of equal positions.
- `class DuplicateIndex`: `add(source, hash, sig)` and `find(hash, sig) -> str | None`. An exact hash match comes first. Otherwise LSH with 16 bands × 8 rows proposes candidates, and the first candidate with `similarity ≥ NEAR_DUP = 0.9` is returned.
- `drop_duplicates(docs: list[Document]) -> tuple[list[Document], dict[str, str]]`. It processes docs in `(len(source), source)` order, so the shorter path is the one kept. It returns the kept docs in their original order, plus `{duplicate: kept}`.

- [ ] Step 1: Tests: `test_exact_duplicate_detected`, `test_near_duplicate_detected` (one changed sentence in ~200 words), `test_different_docs_not_duplicates` (two docs sharing a 60-word footer but different bodies), `test_short_docs_exact_only`, `test_drop_duplicates_keeps_shorter_path`, `test_minhash_deterministic`.
- [ ] Step 2: FAIL → implement → pass. Commit `feat(week3): exact and near-duplicate detection (MinHash + LSH)`.

### Task 4: Registry, incremental ingest, CLI

**Files:** Create `raggy/ingest/registry.py`. Modify `raggy/pipeline.py`, `raggy/cli.py`, `raggy/store/numpy_store.py` (mismatch message). Test `tests/week03/test_registry.py`, `test_incremental.py`, `test_cli_ingest.py`. Update week-1 tests only where the printed ingest summary changed.

**Interfaces (Produces):**
- `@dataclass Record(source, kind, content_hash, size, mtime_ns, pipeline_key, signature: bytes | None, duplicate_of: str | None, title, doc_type, modified, chunks, ingested_at)`.
- `Registry(path)` (SQLite, table `documents`, `source` primary key, `PRAGMA user_version=1`): `get(source)`, `all()`, `under(folder: str)` (file records whose source starts with `folder + os.sep`), `upsert(rec)`, `delete(source)`, `duplicates_of(source)`, `commit()`. Writes stay pending until `commit()`.
- `pipeline_key(config, embedder_name) -> str`: JSON of embedder, chunk_size, chunk_overlap, html_extract, normalize version and dedupe.
- `IngestReport` gains `added, updated, unchanged, removed, duplicates: int` and `failed: list[tuple[str, str]]`. It keeps `documents`, `chunks` and `skipped` (not found).
- `Raggy.ingest(paths: list[str], full: bool = False, progress=None) -> IngestReport`:
  1. Expand the targets. A URL becomes a fetch. A folder is walked with `iter_files`, and its registered files not seen in the walk are removed. A file is loaded. A missing path goes to `skipped` and removes nothing.
  2. A file whose record matches size, mtime_ns and pipeline_key is `unchanged` without being read (`full=True` turns this shortcut off). Otherwise it is loaded and hashed, and an equal hash with an equal key counts as `unchanged` (the stat is updated).
  3. Re-queue `duplicates_of` every removed or changed canonical.
  4. Dedupe the to-index documents against a `DuplicateIndex` built from the registry's non-duplicate records (when `config.dedupe`). A duplicate gets `duplicate_of` set and its old chunks removed from the store.
  5. `ingest_documents` (no save inside; see the next point), then `store.save()`, then `registry.commit()`.
- `ingest_documents(..., save=True)` gains a `save` flag. The registry path saves the store itself.
- `Raggy.remove(target: str) -> int` removes a source, or every source under a folder or URL prefix, from the store and the registry. `Raggy.sources()` returns the registry records, falling back to the store's counts for a pre-week-3 index.
- CLI: `raggy ingest PATHS... [--full]` prints `Indexed into .raggy: 3 added, 1 updated, 40 unchanged, 1 removed, 2 duplicates, 1 failed (N chunks embedded)`, then one stderr line per failure and per not-found path. `raggy sources` prints `chunks  type  title  source` (duplicates shown as `dup of <name>`). `raggy remove TARGET` prints `Removed N sources`. Citation and hit lines show a URL whole and a file by its name.

- [ ] Step 1: Tests.
  - Registry: roundtrip, uncommitted writes discarded on reopen, `under` doesn't match a sibling prefix (`/a/doc` vs `/a/docs2`).
  - Incremental: `test_second_ingest_skips_unchanged_without_reading` (monkeypatch `load_file` to fail on the second call), `test_changed_file_reindexed`, `test_deleted_file_removed`, `test_newly_ignored_file_removed`, `test_missing_folder_target_keeps_its_sources`, `test_settings_change_reindexes_unchanged_files`, `test_registry_commit_after_store_save`, `test_corrupt_pdf_is_reported_and_run_continues`, `test_failed_refetch_keeps_old_chunks`, `test_duplicate_recorded_not_embedded`, `test_duplicate_promoted_when_canonical_deleted`, `test_full_flag_rehashes`, `test_url_ingest_and_unchanged_refetch`, `test_remove_folder_prefix`.
  - CLI: `test_cli_ingest_summary_line`, `test_cli_sources_shows_titles`, `test_cli_remove`.
- [ ] Step 2: FAIL → implement → `uv run pytest -q` all pass. Commit `feat(week3): document registry with incremental re-ingest, dedupe, raggy remove`.

### Task 5: Measure it — the `fictional-web` suite, baselines, lesson

**Files:**
- Create: `evals/fictional-web/make_pages.py` (deterministic, standard library only: the fictional Markdown docs → HTML pages in a shared site template with nav, cookie banner, a "popular questions" sidebar and a footer, plus two duplicates: `help/shipping.html`, the same body under a different template, and `archive/returns-and-refunds-2025.html`, a near-copy with one older sentence); `evals/fictional-web/pages/**` (generated, committed); `evals/baselines/week03-*.json`; `lessons/week03-ingestion.md`; `docs/ingestion.md`
- Modify: `raggy/evals/suites.py` (`Suite.documents` takes `LoadOptions`; adds the `fictional-web` suite, which reuses `evals/fictional/questions.jsonl` with gold `x.md` → `x.html`), `raggy/evals/runner.py` (passes `config.load_options()`; drops duplicates when `config.dedupe`; adds `html_extract`/`dedupe` to the index cache key and to the run config), week-2 `conftest.py` (`documents=lambda options=None: …`), README, COURSE_PLAN, `docs/README.md`, the week 2 lesson's "next" link
- Test: `tests/week03/test_web_suite.py`: `test_pages_match_generator` (regenerate in memory, compare bytes); `test_web_gold_resolves_in_both_modes`; `test_cache_key_changes_with_extract_mode`; `test_eval_dedupe_removes_duplicate_pages`.

- [ ] Step 1: Tests → FAIL → implement → pass. Commit `feat(week3): fictional-web eval suite; ingest settings in the eval cache key`.
- [ ] Step 2: Real runs, retrieval-only (free): fictional-web with `full`/no dedupe, `full`+dedupe, `main`/no dedupe and `main`+dedupe; fictional (normalisation delta against week 1); SciFact. Then one answer-plus-judge run per extraction mode on fictional-web, if a hosted key works (otherwise say so in the lesson). Save the chosen runs to `evals/baselines/week03-*.json`.
- [ ] Step 3: Lesson and guide, with numbers copied from those runs, plus a live demo of a real URL ingest and an incremental re-run (output copied from the terminal). Commit `docs(week3): baselines, lesson, ingestion guide`.
- [ ] Step 4: Final whole-branch review. Then fast-forward main, cut the `week-03` branch and tag, and ask before pushing.
