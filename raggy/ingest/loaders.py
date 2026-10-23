"""Turn a path on disk into Documents.

Week 1 reads plain text only. Week 3 adds URLs, HTML and a registry; week 9
adds PDFs and Office files. Each new type plugs in by file suffix.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from raggy.types import Document

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown"})


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _read(path: Path) -> Document | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        warnings.warn(f"skipping {path}: not valid UTF-8 text", UserWarning, stacklevel=3)
        return None
    return Document(source=str(path.resolve()), text=text, metadata={"name": path.name})


def load_path(path: str | Path) -> list[Document]:
    """Load one file, or every supported file under a folder (sorted, hidden skipped)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_SUFFIXES
            and not _is_hidden(p, path)
        )
    docs = [_read(f) for f in files]
    return [d for d in docs if d is not None]
