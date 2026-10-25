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
    text = dehyphenate("\n\n".join(p.strip() for p in pages))
    if not text.strip():
        raise loaders.LoadError("no text found (scanned pages need OCR, which arrives in week 9)")
    return text, (str(title).strip() or None) if title else None
