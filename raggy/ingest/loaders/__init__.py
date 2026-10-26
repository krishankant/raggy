"""Turn bytes into Documents, choosing a loader by file type.

    LOADERS:  kind ─► function(bytes, options) ─► (text, title or None)
    SUFFIXES: ".md" ─► "markdown", ".pdf" ─► "pdf", …

A new format is one function plus one line in each table (week 9 adds
DOCX, PPTX and spreadsheets this way). Every loader's text goes through
``normalize_text``, so the rest of the pipeline sees one kind of text.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from raggy.ingest.normalize import normalize_text
from raggy.types import Document


class LoadError(Exception):
    """One source couldn't be turned into text. The message says which and why."""


@dataclass(frozen=True)
class LoadOptions:
    html_extract: str = "main"  # "main": article text only; "full": every visible word (the baseline)
    pdf: str = "layout"  # "layout": headings, tables, page labels (week 9 default); "text": week 3's pypdf layer
    ocr: bool = True  # OCR pages with no text layer (scans): ~0.5 s a page, only on pages without text
    images: str = "off"  # off | ocr | vision: how pictures become text (week 10)
    # Bound by Raggy to a cached reader. Not settings: they never change what a key means.
    read_image: Callable[[bytes, str], str | None] | None = field(default=None, compare=False, hash=False)
    on_image: Callable[[bytes, str, str | None], None] | None = field(default=None, compare=False, hash=False)
    name: str = field(default="", compare=False, hash=False)  # the file's name, set by load_bytes for labels
    keep_images: bool = field(default=False, compare=False, hash=False)  # put the pictures read in metadata["images"]


from raggy.ingest.loaders import csv_loader, html, image, office, pdf, pdf_layout, text  # noqa: E402  (they import LoadError above)


def _pdf(data: bytes, options: LoadOptions):
    return (pdf_layout if options.pdf == "layout" else pdf).load(data, options)

LOADERS: dict[str, Callable[[bytes, LoadOptions], tuple[str, str | None]]] = {
    "text": text.load,
    "markdown": text.load_markdown,
    "html": html.load,
    "pdf": _pdf,
    "docx": office.load_docx,
    "pptx": office.load_pptx,
    "xlsx": office.load_xlsx,
    "csv": csv_loader.load,
    "image": image.load,
}

SUFFIXES: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
}


def load_bytes(data: bytes, source: str, kind: str, options: LoadOptions | None = None) -> Document:
    from dataclasses import replace

    name = source.rstrip("/").rsplit("/", 1)[-1] or source
    opts = replace(options or LoadOptions(), name=name)
    kept: list[tuple[bytes, str, str | None]] = []
    if opts.keep_images:  # for CLIP image search: (bytes, label, page) of every picture that was read
        outer = opts.on_image

        def keep(data: bytes, label: str, page: str | None) -> None:
            kept.append((data, label, page))
            if outer is not None:
                outer(data, label, page)

        opts = replace(opts, on_image=keep)
    try:
        raw, title = LOADERS[kind](data, opts)
    except LoadError as e:
        raise LoadError(f"{source}: {e}") from e
    pages = None
    if isinstance(raw, list):  # [(label, text)]: normalise page by page so the offsets stay exact
        parts, pages, pos = [], [], 0
        for label, page_text in raw:
            t = normalize_text(page_text)
            if not t:
                continue
            pages.append((label, pos))
            parts.append(t)
            pos += len(t) + 2
        body = "\n\n".join(parts)
    else:
        body = normalize_text(raw)
    if not body:
        raise LoadError(f"{source}: no text found")
    fallback = Path(name).stem if Path(name).suffix.lower() in SUFFIXES else name  # "2005.11401" stays whole
    meta = {"name": name, "title": title or fallback, "type": kind}
    if pages:
        meta["pages"] = pages  # (label, start offset): "p. 3", "slide 4", "sheet Prices"
    if opts.keep_images:
        meta["images"] = kept
    return Document(source=source, text=body, metadata=meta)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_file(path: Path, options: LoadOptions | None = None) -> Document:
    path = Path(path)
    kind = SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise LoadError(f"{path}: unsupported file type {path.suffix!r}")
    doc = load_bytes(path.read_bytes(), str(path.resolve()), kind, options)
    doc.metadata.update(name=path.name, kind="file", modified=_iso(path.stat().st_mtime))
    return doc


def load_path(path: str | Path, options: LoadOptions | None = None) -> list[Document]:
    """Load one file, or every supported file under a folder (sorted; ignore rules apply)."""
    from raggy.ingest.walk import iter_files

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    docs = []
    for f in [path] if path.is_file() else iter_files(path):
        try:
            docs.append(load_file(f, options))
        except LoadError as e:
            warnings.warn(f"skipping {e}", UserWarning, stacklevel=2)
    return docs
