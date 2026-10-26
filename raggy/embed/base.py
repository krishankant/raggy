"""What every embedder must provide. The rest of raggy only sees this."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str  # stored in the index, so a different model can't be mixed in
    dim: int
    max_tokens: int  # longer inputs are silently truncated by the model, so never send one

    def count_tokens(self, text: str) -> int:
        """Tokens this model sees in ``text`` (without its special start/end markers)."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of unit-length rows."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return one unit-length (dim,) float32 vector for a search query."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
