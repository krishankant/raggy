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
- raster figures on a text page (week 10): cropped, rendered and read by the
  image reader, placed by their position like tables

Borderless tables are a known gap: without drawn lines, telling a table from
two columns of prose is a guess this loader doesn't make.
"""

from __future__ import annotations

import io
import statistics
import warnings

from raggy.ingest import loaders
from raggy.ingest.images import FigureReader
from raggy.ingest.loaders.tables import markdown_table as _markdown

HEADING_RATIO = 1.2
MIN_PAGE_CHARS = 20  # fewer than this and the page is treated as a scan


def _inside(obj, box) -> bool:
    x0, top, x1, bottom = box
    return obj["x0"] >= x0 - 1 and obj["x1"] <= x1 + 1 and obj["top"] >= top - 1 and obj["bottom"] <= bottom + 1


def _page_items(page, heading_sizes: list[float]) -> list[tuple[float, str]]:
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
    return items


def _join(items: list[tuple[float, str]]) -> str:
    return "\n".join(text for _, text in sorted(items, key=lambda x: x[0]))


def _figures(page, n: int, figures) -> list[tuple[float, str]]:
    """Each raster image on the page, rendered as it appears (masks, colour spaces applied)."""
    items = []
    for k, im in enumerate(page.images, start=1):
        box = (max(im["x0"], 0), max(im["top"], 0), min(im["x1"], page.width), min(im["bottom"], page.height))
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue  # off the page
        label = f"{figures.name} p. {n}, figure {k}"
        try:
            rendered = page.crop(box).to_image(resolution=150).original
            buf = io.BytesIO()
            rendered.save(buf, "PNG")
        except (OSError, ValueError, RuntimeError) as e:
            warnings.warn(f"{figures.name}: {label} skipped (can't render: {e})", UserWarning, stacklevel=2)
            continue
        block = figures.read(buf.getvalue(), label, f"p. {n}")
        if block:
            items.append((box[1], "\n" + block + "\n"))
    return items


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
            figures = FigureReader(options)
            for n, page in enumerate(pdf.pages, start=1):
                items = _page_items(page, heading_sizes) if page.chars else []
                text = _join(items)  # the scan test counts real text only, never figure blocks
                if len(text.strip()) >= MIN_PAGE_CHARS and figures.on:
                    text = _join(items + _figures(page, n, figures))
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
