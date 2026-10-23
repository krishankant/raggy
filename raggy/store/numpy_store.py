"""The simplest possible vector store: one numpy matrix, searched exhaustively.

Every search multiplies the query by every stored vector. That's exact and,
up to tens of thousands of chunks, fast enough. Week 5 replaces it with a
real vector database once the corpus grows past that.

On disk (inside ``path``):
    vectors.npy   (n, dim) float32, unit-length rows
    chunks.jsonl  one Chunk per line, same order as the rows
    meta.json     {"embedder": ..., "dim": ..., "version": 1}
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from raggy.types import Chunk, Hit


class IndexNotFoundError(Exception):
    pass


class EmbedderMismatchError(Exception):
    pass


class NumpyStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.embedder: str | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def _check(self, embedder_name: str) -> None:
        if self.embedder is not None and self.embedder != embedder_name and self.chunks:
            raise EmbedderMismatchError(
                f"index was built with '{self.embedder}' but this run uses "
                f"'{embedder_name}'. Vectors from different models can't be compared: "
                f"re-ingest, or set RAGGY_EMBED_MODEL={self.embedder}."
            )

    def add(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        """Add chunks, first removing any existing chunks from the same sources."""
        self._check(embedder_name)
        for source in {c.source for c in chunks}:
            self.remove_source(source)
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.embedder = embedder_name

    def remove_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if self.vectors is not None else None
        return removed

    def search(self, query_vec: np.ndarray, k: int, embedder_name: str) -> list[Hit]:
        self._check(embedder_name)
        if not self.chunks or k < 1:
            return []
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)  # cosine: rows are unit length
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]  # the k best, unordered: O(n)
        top = top[np.argsort(-scores[top])]  # then sort only those k
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        dim = 0 if self.vectors is None else int(self.vectors.shape[1])
        vectors = self.vectors if self.vectors is not None else np.zeros((0, dim), np.float32)
        np.save(self.path / "vectors.npy", vectors)
        with open(self.path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        meta = {"embedder": self.embedder, "dim": dim, "version": 1}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NumpyStore":
        path = Path(path)
        if not (path / "meta.json").exists():
            raise IndexNotFoundError(f"no index at {path}")
        store = cls(path)
        store.embedder = json.loads((path / "meta.json").read_text())["embedder"]
        vectors = np.load(path / "vectors.npy")
        store.vectors = vectors if len(vectors) else None
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            store.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return store
