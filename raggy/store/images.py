"""The image index: one CLIP vector per picture, beside the chunk table (week 10)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TABLE = "images"


@dataclass
class ImageHit:
    source: str  # the file the picture is in (or is)
    label: str  # "spec.pdf p. 2, figure 1", "chart.png"
    page: str | None
    score: float  # cosine similarity


class ImageStore:
    def __init__(self, path: str | Path, dim: int):
        import lancedb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(path))
        self.dim = dim
        self._tbl = self._db.open_table(TABLE) if TABLE in self._db.list_tables().tables else None

    @property
    def exists(self) -> bool:
        return self._tbl is not None

    def _quote(self, s: str) -> str:
        return "'" + s.replace("'", "''") + "'"

    def remove_source(self, source: str) -> None:
        if self._tbl is not None:
            self._tbl.delete(f"source = {self._quote(source)}")

    def replace(self, source: str, items: list[tuple[bytes, str, str | None]], vectors: np.ndarray) -> None:
        """This source's pictures, replacing any it had before."""
        import pyarrow as pa

        self.remove_source(source)
        if not items:
            return
        table = pa.table({
            "source": [source] * len(items),
            "label": [label for _, label, _ in items],
            "page": [page for _, _, page in items],
            "sha256": [hashlib.sha256(data).hexdigest() for data, _, _ in items],
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.astype(np.float32).ravel()), self.dim),
        })
        if self._tbl is None:
            self._tbl = self._db.create_table(TABLE, table)
        else:
            self._tbl.add(table)

    def search(self, vector: np.ndarray, k: int) -> list[ImageHit]:
        if self._tbl is None:
            return []
        rows = self._tbl.search(vector.astype(np.float32)).metric("cosine").limit(k).to_list()
        return [ImageHit(r["source"], r["label"], r["page"], 1.0 - float(r["_distance"])) for r in rows]
