"""What raggy has ingested, and in what state: one SQLite row per source.

The vector store answers "which chunks are similar to this query?". The
registry answers "have I seen this file before, and has it changed?". That
is what makes a second `raggy ingest` of 5,000 files take a second instead
of an hour: a file whose size and modification time match its row is
skipped without being read, and one whose content hash matches is skipped
without being embedded.

Writes stay in an open transaction until ``commit()``. The pipeline commits
only after the vector store is saved, so a crash in between leaves rows
that still describe the old state, and the next run redoes the work.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import astuple, dataclass, fields
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Record:
    source: str  # absolute path or URL
    kind: str  # "file" | "url"
    content_hash: str  # sha256 of the normalised text
    size: int | None  # file size in bytes (files only)
    mtime_ns: int | None  # file modification time (files only)
    pipeline_key: str  # the settings this source was indexed with
    signature: bytes | None  # MinHash signature, for near-duplicate checks
    duplicate_of: str | None  # set when this is a copy of another source (not embedded)
    title: str | None
    doc_type: str | None  # loader kind: markdown, html, pdf, text
    modified: str | None  # ISO time from the file or the server
    chunks: int
    ingested_at: str


_COLUMNS = [f.name for f in fields(Record)]


class Registry:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (source TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, size INTEGER, mtime_ns INTEGER, pipeline_key TEXT NOT NULL, "
            "signature BLOB, duplicate_of TEXT, title TEXT, doc_type TEXT, modified TEXT, "
            "chunks INTEGER NOT NULL, ingested_at TEXT NOT NULL)")
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _rows(self, where: str = "", args: tuple = ()) -> list[Record]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM documents {where} ORDER BY source"
        return [Record(*row) for row in self.conn.execute(sql, args)]

    def get(self, source: str) -> Record | None:
        rows = self._rows("WHERE source = ?", (source,))
        return rows[0] if rows else None

    def all(self) -> list[Record]:
        return self._rows()

    def under(self, folder: str) -> list[Record]:
        """File records inside ``folder`` (not a sibling that shares its prefix)."""
        prefix = folder.rstrip(os.sep) + os.sep
        return [r for r in self._rows("WHERE kind = 'file'") if r.source.startswith(prefix)]

    def duplicates_of(self, source: str) -> list[Record]:
        return self._rows("WHERE duplicate_of = ?", (source,))

    def upsert(self, rec: Record) -> None:
        marks = ", ".join("?" for _ in _COLUMNS)
        self.conn.execute(f"INSERT OR REPLACE INTO documents ({', '.join(_COLUMNS)}) VALUES ({marks})",
                          astuple(rec))

    def delete(self, source: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE source = ?", (source,))

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()  # without commit(): pending writes are rolled back
