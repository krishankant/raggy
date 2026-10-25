"""Turn bytes into Documents, choosing a loader by file type.

    LOADERS:  kind ─► function(bytes, options) ─► (text, title or None)
    SUFFIXES: ".md" ─► "markdown", ".pdf" ─► "pdf", …

A new format is one function plus one line in each table (week 9 adds
DOCX, PPTX and spreadsheets this way). Every loader's text goes through
``normalize_text``, so the rest of the pipeline sees one kind of text.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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


from raggy.ingest.loaders import html, pdf, text  # noqa: E402  (they import LoadError above)

LOADERS: dict[str, Callable[[bytes, LoadOptions], tuple[str, str | None]]] = {
    "text": text.load,
    "markdown": text.load_markdown,
    "html": html.load,
    "pdf": pdf.load,
}

SUFFIXES: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


def load_bytes(data: bytes, source: str, kind: str, options: LoadOptions | None = None) -> Document:
    try:
        raw, title = LOADERS[kind](data, options or LoadOptions())
    except LoadError as e:
        raise LoadError(f"{source}: {e}") from e
    body = normalize_text(raw)
    if not body:
        raise LoadError(f"{source}: no text found")
    name = source.rstrip("/").rsplit("/", 1)[-1] or source
    return Document(source=source, text=body,
                    metadata={"name": name, "title": title or Path(name).stem, "type": kind})


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
