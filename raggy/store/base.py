"""What every vector store provides, plus the errors and filters they share.

    numpy  week 1's exact, in-memory store: every search scans every vector
    lance  LanceDB on disk: incremental writes, SQL filters, an optional ANN index
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from raggy.types import Chunk, Hit

if TYPE_CHECKING:
    from raggy.config import Config


class IndexNotFoundError(Exception):
    pass


class EmbedderMismatchError(Exception):
    pass


def mismatch(stored: str, given: str, path) -> EmbedderMismatchError:
    return EmbedderMismatchError(
        f"index was built with '{stored}' but this run uses '{given}'. Vectors from different "
        f"models can't be compared: delete the index folder {path} and ingest again, "
        f"or set RAGGY_EMBED_MODEL={stored}.")


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"  # SQL string literal: double any single quote


@dataclass(frozen=True)
class Filters:
    """Restrict a search to some documents. Unset fields don't filter."""

    doc_type: str | None = None  # "markdown", "html", "pdf", "text"
    source_prefix: str | None = None  # a folder path or URL prefix
    modified_after: str | None = None  # ISO date or time; documents without a date are excluded

    def __bool__(self) -> bool:
        return any((self.doc_type, self.source_prefix, self.modified_after))

    def sql(self) -> str:
        """A LanceDB `where` clause. Values are quoted literals, never pasted in raw."""
        parts = []
        if self.doc_type:
            parts.append(f"doc_type = {_quote(self.doc_type)}")
        if self.source_prefix:
            parts.append(f"starts_with(source, {_quote(self.source_prefix)})")  # literal: no LIKE wildcards
        if self.modified_after:
            parts.append(f"modified >= {_quote(self.modified_after)}")
        return " AND ".join(parts)

    def matches(self, chunk: Chunk) -> bool:
        m = chunk.meta or {}
        if self.doc_type and m.get("type") != self.doc_type:
            return False
        if self.source_prefix and not chunk.source.startswith(self.source_prefix):
            return False
        if self.modified_after and not (m.get("modified") and m["modified"] >= self.modified_after):
            return False
        return True


class VectorStore(Protocol):
    embedder: str | None

    def __len__(self) -> int: ...

    def add(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        """Add chunks, first removing every existing chunk from the same sources."""

    def remove_source(self, source: str) -> int: ...

    def search(self, query_vec: np.ndarray, k: int, embedder_name: str,
               where: Filters | None = None, exact: bool = True) -> list[Hit]: ...

    def sources(self) -> dict[str, int]: ...

    def save(self) -> None: ...


def make_store(config: "Config", create: bool = True) -> VectorStore:
    """Open the configured store. ``create=False`` raises IndexNotFoundError if there's none."""
    if config.store == "lance":
        from raggy.store.lance_store import LanceStore

        path = Path(config.index_dir) / "lance"
        if not create and not (path / "meta.json").exists():
            raise IndexNotFoundError(f"no index at {path}")
        return LanceStore(path, nprobes=config.ann_nprobes, refine=config.ann_refine)
    from raggy.store.numpy_store import NumpyStore

    try:
        return NumpyStore.load(config.index_dir)
    except IndexNotFoundError:
        if not create:
            raise
        return NumpyStore(config.index_dir)
