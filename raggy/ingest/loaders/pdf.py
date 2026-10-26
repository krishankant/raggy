"""PDF: the text layer only, page by page. Layout, tables and OCR arrive in week 9."""

from __future__ import annotations

import io

from raggy.ingest import loaders
from raggy.ingest.normalize import dehyphenate


def load(data: bytes, options) -> tuple[str, str | None]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        title = (reader.metadata or {}).get("/Title") if reader.metadata else None
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as e:
        raise loaders.LoadError(f"not a readable PDF ({e})") from e
    if not any(p.strip() for p in pages):
        raise loaders.LoadError("no text found (scanned pages need OCR: RAGGY_PDF=layout RAGGY_OCR=on)")
    return [(f"p. {n}", dehyphenate(p.strip())) for n, p in enumerate(pages, start=1)], \
        (str(title).strip() or None) if title else None
