"""Never embed the same text twice.

Embedding is the slow part of ingest: about 10 texts a second for bge-small on
a laptop CPU. Most re-runs embed text that was embedded before: a changed
setting that doesn't change chunk text (parent size, ANN, the store), an eval
rebuild, a re-ingest after one file changed. The cache maps
sha256(model name + text) to its vector, so any text a model has embedded
once is free forever after, whatever else changed.

The model name is part of the key, so vectors from different models can never
be mixed up. Queries aren't cached: each is new and only takes milliseconds.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from raggy.embed.base import Embedder

CACHE_NAME = "embed-cache.sqlite"


class EmbeddingCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self.conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):  # SQLite caps the number of ? parameters
            part = keys[i:i + 500]
            rows = self.conn.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO vectors (key, dim, vec) VALUES (?, ?, ?)",
                              [(k, len(v), np.asarray(v, np.float32).tobytes()) for k, v in items.items()])

    def commit(self) -> None:
        self.conn.commit()


class CachedEmbedder:
    """An Embedder that asks the cache first and embeds only what it has never seen."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache):
        self.inner, self.cache = inner, cache
        self.hits = self.misses = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def max_tokens(self) -> int:
        return self.inner.max_tokens

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\0{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        known = self.cache.get_many(list(dict.fromkeys(keys)))
        todo = {k: t for k, t in zip(keys, texts) if k not in known}  # each distinct text once
        if todo:
            vectors = self.inner.embed_documents(list(todo.values()))
            fresh = dict(zip(todo, vectors))
            self.cache.put_many(fresh)
            known.update(fresh)
        self.misses += len(todo)
        self.hits += len(texts) - len(todo)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([known[k] for k in keys]).astype(np.float32)
