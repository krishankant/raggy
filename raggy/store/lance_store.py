"""LanceDB: an embedded vector database that lives in a folder, like SQLite.

Why move off week 1's numpy matrix? Not search speed. At tens of thousands of
chunks, exact numpy search takes a few milliseconds. Week 5 measures it. The
reasons are the other things a real index needs:

- Writes are incremental. NumpyStore rewrites every vector and every chunk on
  each save; Lance appends a new version and deletes rows in place, and a
  crash mid-write leaves the previous version intact.
- Filters run inside the search (`where doc_type = 'pdf'`), not after it, so
  "top 4 PDFs" really returns 4 PDFs.
- It can build an approximate (ANN) index when a corpus outgrows exact search,
  and week 6 adds a full-text index to the same table.

Scores: with distance_type "dot", Lance returns `_distance = 1 - dot`.
Rows are unit length, so `1 - _distance` is the cosine similarity NumpyStore gives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raggy.store.base import EmbedderMismatchError, Filters, _quote, mismatch
from raggy.types import Chunk, Hit

TABLE = "chunks"
ANN_MIN_ROWS = 1000  # an IVF index trains k-means on the vectors; below this there's nothing to cluster
META = ("type", "title", "modified")  # Document metadata copied onto every row, for filters


class LanceStore:
    def __init__(self, path: str | Path, nprobes: int = 20, refine: int = 4):
        import lancedb

        self.path = Path(path)
        self.nprobes, self.refine = nprobes, refine
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.path))
        self._tbl = self._db.open_table(TABLE) if TABLE in self._db.list_tables().tables else None
        meta = self.path / "meta.json"
        self.embedder: str | None = json.loads(meta.read_text())["embedder"] if meta.exists() else None

    def __len__(self) -> int:
        return self._tbl.count_rows() if self._tbl is not None else 0

    def _check(self, embedder_name: str) -> None:
        if self.embedder is not None and self.embedder != embedder_name and len(self):
            raise mismatch(self.embedder, embedder_name, self.path)

    def _rows(self, chunks: list[Chunk], vectors: np.ndarray):
        import pyarrow as pa

        vectors = np.asarray(vectors, dtype=np.float32)
        cols = {
            "id": [c.id for c in chunks], "source": [c.source for c in chunks], "text": [c.text for c in chunks],
            "idx": [c.index for c in chunks], "start": [c.start for c in chunks], "end": [c.end for c in chunks],
            "context": [c.context for c in chunks],
            "parent_start": pa.array([c.parent_start for c in chunks], pa.int64()),
            "parent_end": pa.array([c.parent_end for c in chunks], pa.int64()),
            "parent_text": pa.array([c.parent_text for c in chunks], pa.string()),
        }
        for key in META:
            cols["doc_type" if key == "type" else key] = pa.array([(c.meta or {}).get(key) for c in chunks], pa.string())
        cols["vector"] = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), vectors.shape[1])
        return pa.table(cols)

    def add(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        if not chunks:
            return
        self._check(embedder_name)
        table = self._rows(chunks, vectors)
        if self._tbl is None:
            self._tbl = self._db.create_table(TABLE, table)
        else:
            sources = sorted({c.source for c in chunks})
            for i in range(0, len(sources), 200):  # keep each IN (...) list a sane size
                self._tbl.delete(f"source IN ({', '.join(_quote(s) for s in sources[i:i + 200])})")
            self._tbl.add(table)
        if self.embedder != embedder_name:
            self.embedder = embedder_name
            (self.path / "meta.json").write_text(json.dumps({"embedder": embedder_name, "version": 1}))

    def remove_source(self, source: str) -> int:
        if self._tbl is None:
            return 0
        n = self._tbl.count_rows(f"source = {_quote(source)}")
        if n:
            self._tbl.delete(f"source = {_quote(source)}")
        return n

    def search(self, query_vec: np.ndarray, k: int, embedder_name: str,
               where: Filters | None = None, exact: bool = True) -> list[Hit]:
        self._check(embedder_name)
        if self._tbl is None or k < 1:
            return []
        q = self._tbl.search(np.asarray(query_vec, dtype=np.float32)).distance_type("dot").limit(k)
        if where:
            q = q.where(where.sql(), prefilter=True)
        q = q.bypass_vector_index() if exact else q.nprobes(self.nprobes).refine_factor(self.refine)
        return [Hit(chunk=self._chunk(r), score=1.0 - float(r["_distance"])) for r in q.to_list()]

    @staticmethod
    def _chunk(r: dict) -> Chunk:
        meta = {k: r["doc_type" if k == "type" else k] for k in META
                if r.get("doc_type" if k == "type" else k) is not None}
        return Chunk(id=r["id"], source=r["source"], text=r["text"], index=r["idx"], start=r["start"],
                     end=r["end"], context=r["context"], parent_start=r["parent_start"],
                     parent_end=r["parent_end"], parent_text=r["parent_text"], meta=meta)

    def sources(self) -> dict[str, int]:
        if self._tbl is None:
            return {}
        counts = self._tbl.to_arrow().column("source").value_counts().to_pylist()
        return {c["values"]: c["counts"] for c in counts}

    def has_index(self) -> bool:
        return self._tbl is not None and any("vector" in i.columns for i in self._tbl.list_indices())

    def finish(self, build_ann: bool) -> None:
        """Build the ANN index once the table is big enough, or fold new rows into it.

        Rows added after the index was built are still found: Lance scans them
        exactly and merges them with the index results. ``optimize`` moves
        them into the index so that scan stays small."""
        if self._tbl is None:
            return
        if self.has_index():
            self._tbl.optimize()
        elif build_ann and len(self) >= ANN_MIN_ROWS:
            from lancedb.index import IvfHnswSq

            self._tbl.create_index("vector", config=IvfHnswSq(distance_type="dot"))

    def save(self) -> None:
        """Nothing to do: every Lance write is already committed as a new version."""
