# Rich documents: what's really inside a PDF, a slide deck and a spreadsheet

**[← Docs hub](README.md)** · **[Week 9 lesson](../lessons/week09-rich-documents.md)**

This guide sits beside week 9. It explains what a PDF actually stores, how raggy rebuilds headings and tables from it, when a page needs OCR and what that costs, and why slides and sheets need *locations* as much as pages do.

---

## 1. A PDF is a drawing, not a document

A Word file stores paragraphs, headings and tables. A PDF stores instructions for drawing a page: "put glyph `4` in Helvetica 10 pt at x = 72, y = 540", "draw a line from (72, 520) to (540, 520)". There is no paragraph and no table in the file, only characters at positions and lines between points.

So every PDF text extractor *reconstructs* the text, and the reconstruction decides what your chunks look like. Week 3's loader (pypdf) reads the characters in drawing order and joins them. On the fleet price list, a four-column price table comes out as:

```text
Bikes ordered
Price per bike
Lead time
1-4
4,950 credits
3 weeks
5-9
4,700 credits
...
```

Each cell sits on its own line. The words are all there, so keyword and embedding search still find the chunk. But the row structure is gone: nothing says that "4,700 credits" belongs to "5-9" and not to "1-4". A model reading this has to guess the column count, and a chunk boundary in the middle of the table makes the guess impossible.

## 2. Rebuilding the page

pdfplumber exposes every character with its position (`x0`, `top`, `x1`, `bottom`), its font and its size, plus every drawn line and rectangle. That's enough to rebuild three things a reader sees.

**Headings, from font size.** Body text is the most common size on a document, so raggy takes the *median* character size as body size. A line whose largest character is at least 1.2× that is a heading. The distinct heading sizes are ranked, biggest first, into `#`, `##` and `###` (at most three levels). On the fleet price list, body text is 10 pt, section titles are 14 pt (`##`) and the document title is 18 pt (`#`). Week 4's structure chunker then splits on those headings, exactly as it does for Markdown.

The ratio is a heuristic. A document that uses bold body-size text for headings gets none detected, and a document with large pull quotes gets false ones. Both are cheap failures: the text is still there, just with less structure.

**Tables, from drawn lines.** `page.find_tables()` looks for cells enclosed by ruling lines, the grid you see in most generated reports. raggy writes each table as a Markdown pipe table and removes its characters from the body text, so they don't appear twice:

```text
| Bikes ordered | Price per bike | Lead time |
|---|---|---|
| 1-4 | 4,950 credits | 3 weeks |
| 5-9 | 4,700 credits | 4 weeks |
```

Every format in week 9 ends up in this one shape: PDF tables, Word tables, slide tables, sheets and CSV files (`loaders/tables.py`, 24 lines). That's why a large table chunks well: week 4's chunker splits between rows and carries the header row into each piece.

**Borderless tables are the known gap.** Without drawn lines, a table and two columns of prose look the same: words aligned at a few x positions. pdfplumber has a text-based strategy that guesses, but it also "finds" tables in ordinary paragraphs. raggy doesn't guess. A borderless table comes out as text lines, like week 3's output.

**Reading order.** Text lines and tables are sorted by their `top` position, so a table stays between the paragraph above it and the one below. Multi-column layouts (two newspaper columns side by side) need column detection on top of this, and raggy doesn't do it.

## 3. Scans: when there's no text at all

A scanned page is a single image with no text layer. Every text extractor returns nothing for it, and pypdf returns an empty string without complaint. In week 3, a scanned page was silently missing from the index. On the `rich` suite that's the whole workshop notice: three questions whose answers simply aren't in any chunk.

raggy decides a page is a scan when its text layer has fewer than 20 characters. Such a page is rendered at 200 dpi and read by OCR. raggy uses RapidOCR: PaddleOCR's detection and recognition models in ONNX format. They're bundled in the wheel, run on the CPU and need no network.

OCR has three costs:

| Cost | Measured on this laptop (Apple M2, 8 GB) |
| --- | --- |
| Time | 0.54 s per scanned page warm; 1.21 s for the first page, which loads the models |
| Memory | Model load of a few hundred MB, only once a scan is seen |
| Accuracy | Clean printed text: exact on the test scan. Photos, handwriting, skew and low resolution: misreads, especially of digits |

Pages with a text layer never pay any of this: the check costs nothing, and the 19 ms a page of layout extraction is unchanged. That's why OCR can be on by default. A misread is a risk only for scans, and without OCR those pages contribute nothing at all.

With OCR off (`RAGGY_OCR=off`), a scanned page produces a warning that names the switch, and a PDF that is *only* scans fails to load with the same hint. A silent empty page is the one outcome raggy avoids.

## 4. Locations: pages, slides and sheets

A citation that says `[2] fleet-prices.pdf` sends the reader through every page. `[2] fleet-prices.pdf, p. 2` sends them to one. Week 9 records where each chunk came from:

- **Loaders return labelled parts.** The PDF loader returns `[("p. 1", text), ("p. 2", text), ...]`, PowerPoint returns `("slide 4", ...)`, Excel returns `("sheet Stock", ...)`. Formats with no fixed pages (Markdown, HTML, Word, CSV) return plain text.
- **Ingest keeps the offsets.** The parts are joined into one document, and `metadata["pages"]` records where each label starts. Chunking still sees one continuous text, so a section that runs from page 1 to page 2 isn't cut at the page break.
- **Each chunk gets a label range.** `page_label(pages, start, end)` finds the labels covering the chunk's first and last character. The chunk's metadata gets `page` (`"p. 1"`) and, if it crosses a boundary, `page_end` (`"p. 2"`). The prompt and the citation print `p. 1–2`.

Slides need this more than PDFs do. A deck's pages *are* its structure: "the launch slide", "the pricing slide". A sheet name is often the only thing that says what a table is ("Stock" against "Suppliers").

**Speaker notes are content.** In a pitch deck, the slide says "Range: 90 km" and the notes say "about 70 km with 80 kg in the box": the qualification lives in the notes. raggy appends each slide's notes after its text as `Speaker notes: ...`, in the same location.

**Spreadsheets store formulas.** A cell can hold `=B2*C2`. openpyxl with `data_only=True` reads the value Excel last computed and saved. A file written by a script that never ran Excel has no cached values, so formula cells come out empty. Opening and saving the file in Excel or LibreOffice fills them.

## 5. What retrieval metrics can't see

On the `rich` suite, text mode's retrieval loss is the scanned page and nothing else. The table questions retrieve as well as in layout mode, because the gold quote "5-9 | 4,700 credits | 4 weeks" matches the text-mode chunk too: the eval treats `|` and line breaks alike, and the right chunk is found either way.

Table structure should matter at the *answer* step, when a model has to tell which number belongs to which row. Week 9 measured it, and on these tables it didn't: gpt-5.5 answered every table question correctly from text mode's one-cell-per-line output too. It counts the columns from the header lines and reads the rows back. Small, clean tables with a strong model are the easy case. Structure still pays where that inference breaks: wide tables, empty cells (a missing cell shifts every later value into the wrong column), a table split across chunks without its header, and smaller local models. The Markdown form makes the row explicit, so nothing has to be inferred.

## Further reading

- [pdfplumber](https://github.com/jsvine/pdfplumber): the README's "Extracting tables" section explains the lines and text strategies, and its visual debugger shows what the line strategy sees.
- [RapidOCR](https://github.com/RapidAI/RapidOCR): the ONNX packaging of PaddleOCR's models used here.
- [PDF 2.0, ISO 32000-2](https://pdfa.org/resource/iso-32000-2/), section 9: how text is positioned and drawn. Reading it once explains why "extract text from PDF" is a reconstruction, never a read.
