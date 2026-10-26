"""A tiny, deterministic stand-in for a real embedding model, used by tests.

It hashes each word into one of ``dim`` buckets (a "bag of words"), so texts
that share words point in similar directions. It knows nothing about meaning:
"car" and "automobile" are unrelated to it. That is exactly the gap a real
embedding model closes.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from raggy.chunking.tokens import simple_count
from raggy.embed.base import normalize

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"fake-bow-{dim}"
        self.max_tokens = 510

    def count_tokens(self, text: str) -> int:
        return simple_count(text)

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            v[bucket] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(self._vector(text))
