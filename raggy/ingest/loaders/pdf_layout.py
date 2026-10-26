"""PDF with layout: headings, ruled tables as Markdown, page labels, OCR for scans.

A PDF stores positioned glyphs, not paragraphs or tables. pdfplumber gives each
character with its position and font size, which is enough to rebuild what a
reader sees:

- headings: a line whose font is 1.2x the page's typical size or bigger,
  ranked by size into #, ##, ###
- tables: cells bounded by drawn lines (find_tables), written as Markdown
  rows, so week 4's chunker splits them by row and keeps the header
- reading order: text lines and tables sorted by their position on the page
- a page with almost no text is a scan: render it and OCR it

Borderless tables are a known gap: without drawn lines, telling a table from
two columns of prose is a guess this loader doesn't make.
"""

from __future__ import annotations

import io
import statistics
import warnings

from raggy.ingest import loaders

HEADING_RATIO = 1.2
MIN_PAGE_CHARS = 20  # fewer than this and the page is treated as a scan


def _cell(value) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|")


def _markdown(rows: list[list]) -> str:
    rows = [[_cell(c) for c in r] for r in rows if any(c not in (None, "") for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _inside(obj, box) -> bool:
    x0, top, x1, bottom = box
    return obj["x0"] >= x0 - 1 and obj["x1"] <= x1 + 1 and obj["top"] >= top - 1 and obj["bottom"] <= bottom + 1


def _page_text(page, heading_sizes: list[float]) -> str:
    tables = page.find_tables()
    boxes = [t.bbox for t in tables]
    body = page.filter(lambda o: o.get("object_type") != "char" or not any(_inside(o, b) for b in boxes))
    items = []  # (top, text)
    for line in body.extract_text_lines():
        size = max(c["size"] for c in line["chars"])
        text = line["text"].strip()
        if not text:
            continue
        if size in heading_sizes:
            text = "#" * (heading_sizes.index(size) + 1) + " " + text
            items.append((line["top"], "\n" + text + "\n"))
        else:
            items.append((line["top"], text))
    for t in tables:
        md = _markdown(t.extract())
        if md:
            items.append((t.bbox[1], "\n" + md + "\n"))
    return "\n".join(text for _, text in sorted(items, key=lambda x: x[0]))


def load(data: bytes, options) -> tuple[list[tuple[str, str]], str | None]:
    import pdfplumber
    from pdfminer.psexceptions import PSException
    from pdfplumber.utils.exceptions import PdfminerException

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            title = (pdf.metadata or {}).get("Title")
            sizes = [round(c["size"], 1) for p in pdf.pages for c in p.chars]
            body = statistics.median(sizes) if sizes else 0
            # Bigger fonts than body text are headings, biggest first; at most three levels.
            heading_sizes = sorted({s for s in sizes if body and s >= body * HEADING_RATIO}, reverse=True)[:3]
            pages, ocr_skipped = [], 0
            for n, page in enumerate(pdf.pages, start=1):
                text = _page_text(page, [s for s in heading_sizes]) if page.chars else ""
                if len(text.strip()) < MIN_PAGE_CHARS:
                    if options.ocr:
                        from raggy.ingest.ocr import ocr_image

                        text = ocr_image(page.to_image(resolution=200).original)
                    else:
                        ocr_skipped += 1
                pages.append((f"p. {n}", text))
    except (PSException, PdfminerException, ValueError, KeyError, TypeError) as e:
        raise loaders.LoadError(f"not a readable PDF ({e})") from e
    if not any(t.strip() for _, t in pages):
        hint = " (looks scanned: set RAGGY_OCR=on to read it)" if ocr_skipped else ""
        raise loaders.LoadError(f"no text found{hint}")
    if ocr_skipped:
        warnings.warn(f"{ocr_skipped} page(s) look scanned and were skipped: set RAGGY_OCR=on", UserWarning,
                      stacklevel=2)
    return pages, (str(title).strip() or None) if title else None
