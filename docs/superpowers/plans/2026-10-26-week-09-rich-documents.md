# Week 9: Rich Documents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real documents aren't plain text. PDFs keep their headings and tables (as Markdown) and gain page numbers in citations, scanned pages are OCR'd, and DOCX, PPTX (with slide numbers), XLSX and CSV (row-aware tables) are loaded. All of it is measured on a new `rich` suite whose answers sit in tables, slides, spreadsheets and a scanned page.

**Architecture:**

- **PDFs.** The PDF loader moves to pdfplumber in `layout` mode, which works page by page:
  - ruled tables are detected and written as Markdown pipe tables;
  - the remaining text lines go in reading order;
  - lines whose font size is 1.2× the page's median or more become `#` / `##` / `###` headings, ranked by size;
  - a page with almost no text (fewer than 20 characters) is rendered at 200 dpi and OCR'd with rapidocr.

  `RAGGY_PDF=text` keeps week 3's pypdf loader as the baseline.
- **Office files and CSV.** New loaders follow the same pattern. Each returns Markdown, so week 4's structure chunker handles headings and splits tables by row with the header carried along.
- **Locations.** Every loader records `Document.metadata["pages"]`, a list of `(label, start_offset)` with labels like `p. 3`, `slide 4` or `sheet Prices`. Ingest stamps each chunk with the label covering its start (`chunk.meta["page"]`). The prompt, the citations and the CLI then show `fleet-prices.pdf, p. 2`.

**Tech Stack:** All versions checked on PyPI 2026-10-26.

| Library | Version | Licence | Role |
|---|---|---|---|
| pdfplumber | 0.11.10 | MIT, on pdfminer.six 20260107 | PDF layout and tables |
| rapidocr | 3.10.0 | Apache-2.0 | OCR. Models are bundled in the wheel, so it works offline. Verified: it read a rendered 3-line notice exactly, in 1.4 s per page on the M2 |
| python-docx | 1.2.0 | MIT | DOCX |
| python-pptx | 1.0.2 | MIT | PPTX |
| openpyxl | 3.1.5 | MIT | XLSX |
| reportlab | 5.1.0 | BSD | dev group only, to generate the eval PDFs |

Rejected:

- **PyMuPDF / pymupdf4llm:** AGPL.
- **docling and unstructured:** they pull in PyTorch-sized models, and this 8 GB machine is already swapping (week 7).
- **Borderless tables:** pdfplumber's line strategy finds ruled tables only. Its text strategy over-detects tables in body text. The lesson names this as a known limit.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 9 row)

**Branching:** from `dev/week-08` (code complete) in a worktree; rebased onto main after weeks 7 and 8 are merged. No heavy runs while weeks 7 and 8 are measuring.

## Global Constraints

- `uv run pytest` passes offline. OCR has one real test (bundled models, one page). Every other test uses a fake OCR.
- Every new dependency is imported by raggy code. `pypdf` stays, as the `RAGGY_PDF=text` baseline.
- `LoadOptions` gains `pdf` (layout | text) and `ocr` (on | off). Both go into the settings key, so a change re-indexes (the week 3 mechanism).
- LanceDB tables gain a `page` column. An existing table without it gets the column through schema evolution (`add_columns`, nulls), never by dropping the table (observation 29: no deletes to fix a schema).
- **Decision rule, set before the runs (observation 31):**
  - `pdf=layout` and `ocr=on` become the defaults if `rich` recall@4 rises against `pdf=text`, *and* the three fictional suites (Markdown and HTML, unaffected by PDF parsing) stay identical.
  - Ingest time per PDF page is reported. OCR stays on only if a scanned page costs under 5 s.
- (Observation 33) The data can tell them apart: the `rich` questions are tagged by where the answer lives (`pdf-table`, `pdf-text`, `scanned`, `slide`, `sheet`, `csv`, `docx-table`), and the text baseline is expected to fail on `scanned` (no text layer) and to blur tables.
- Never move week-01..08 branches or tags. Cut `week-09` from a **verified SHA** (observation 34) and push.

## Review Focus

1. **A PDF where pdfplumber raises** (corrupt or encrypted) gives a `LoadError` with the reason, like week 3, and the run continues. Test: `test_corrupt_pdf_layout_raises_load_error`.
2. **A table cell containing `|` or a newline** must not break the Markdown table. Cells are escaped (`\|`) and newlines become spaces. Test: `test_table_cells_escaped`.
3. **Spreadsheets with empty rows, merged cells or formulas.** Empty rows are skipped, formulas give their cached values (`data_only=True`), and a sheet of 10,000 rows doesn't load into one string the chunker can't split. Rows go out as table lines, so the week 4 table split keeps the header with each piece. Test: `test_big_sheet_splits_with_header`.
4. **Page labels after packing and merging.** A merged passage keeps the first page label and shows a range (`p. 2–3`) when it spans pages. Test: `test_passage_spanning_pages_label`.
5. **The schema change.** Opening a week 5–8 Lance table adds the `page` column without losing rows. Test: `test_old_table_gains_page_column`.

---

### Task 1: PDF layout, tables and OCR

**Files:** Create `raggy/ingest/loaders/pdf_layout.py` and `raggy/ingest/ocr.py`. Modify `raggy/ingest/loaders/__init__.py` (`LoadOptions.pdf`, `.ocr`; the dispatch for kind `pdf`), `raggy/config.py` (`RAGGY_PDF`, `RAGGY_OCR`), `pyproject.toml`. Test `tests/week09/test_pdf_layout.py` (PDFs built with reportlab in the test, from the dev group).

- [ ] Tests: `test_layout_headings_by_font_size`, `test_layout_ruled_table_as_markdown` (gold row `| 5-9 | 4,700 credits | 4 weeks |`), `test_table_text_not_duplicated_in_body`, `test_pages_metadata_offsets`, `test_scanned_page_ocr` (real rapidocr, one page), `test_ocr_off_leaves_scanned_page_empty_and_warns`, `test_corrupt_pdf_layout_raises_load_error`, `test_table_cells_escaped`, `test_text_mode_is_week3`. FAIL → implement → all pass. Commit.

### Task 2: DOCX, PPTX, XLSX and CSV

**Files:** Create `raggy/ingest/loaders/office.py` (docx, pptx, xlsx) and `raggy/ingest/loaders/csv_loader.py`. Add `.docx/.pptx/.xlsx/.csv` to `SUFFIXES`. Test `tests/week09/test_office.py` (files built with the same libraries in the test).

- [ ] Tests: `test_docx_headings_and_table`, `test_pptx_slides_titles_notes_and_labels`, `test_xlsx_sheets_as_tables_with_labels`, `test_xlsx_formulas_give_values`, `test_csv_sniffs_delimiter`, `test_big_sheet_splits_with_header`, `test_walk_includes_office`. FAIL → implement → all pass. Commit.

### Task 3: Locations in chunks, prompts and citations

**Files:** Modify `raggy/pipeline.py` (stamp `chunk.meta["page"]`), `raggy/store/lance_store.py` (a `page` column; `add_columns` for old tables), `raggy/generate/pack.py` (page ranges for merged passages), `raggy/generate/prompt.py` (`(source: x.pdf, p. 2)`), `raggy/types.py` (`Citation.location`), `raggy/cli.py`. Test `tests/week09/test_locations.py`.

- [ ] Tests: `test_chunk_gets_page_label`, `test_citation_shows_location`, `test_prompt_shows_location`, `test_passage_spanning_pages_label`, `test_old_table_gains_page_column`, `test_numpy_store_keeps_page_in_meta`. FAIL → implement → all pass. Commit.

### Task 4: The `rich` suite, measurements, lesson

- [ ] `evals/rich/make_docs.py` (deterministic, uses the dev group) writes:
  - `fleet-prices.pdf`: 3 pages, ruled tables, headings
  - `workshop-notice-scan.pdf`: one image-only page
  - `staff-rota.docx`: a table
  - `ebb-launch.pptx`: 6 slides with notes
  - `parts-stock.xlsx`: 2 sheets, about 300 rows
  - `deliveries.csv`

  Plus `questions.jsonl`: about 24 answerable questions tagged by location, and 4 unanswerable ones. A test checks that every gold quote resolves under `pdf=layout, ocr=on`.
- [ ] Runs: `rich` retrieval-only and answers under `pdf=text, ocr=off` against `pdf=layout, ocr=on`; the fictional suites once, to confirm they're unchanged; ingest seconds per PDF page and per OCR page.
- [ ] Apply the rule, set the defaults, ledger. Write the lesson `lessons/week09-rich-documents.md` and the guide `docs/rich-documents.md` (what a PDF really contains, how table and heading detection work, OCR's costs, why slides and sheets need locations). Update the README, COURSE_PLAN and the docs hub. Commit, then do the final review, cut `week-09` from a verified SHA, and push.
