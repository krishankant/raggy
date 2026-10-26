"""Find documents that are copies, or nearly copies, of one already indexed.

Duplicates cost twice: they're embedded for nothing, and at query time two
copies of one paragraph take two of the model's four context slots.

- Exact copies: equal SHA-256 of the normalised text.
- Near copies (an archived version, a print view, a mirror with one changed
  line): MinHash. Split the text into overlapping 5-word "shingles". The
  fraction of shingles two documents share is their Jaccard similarity.
  Comparing every pair is O(n²), so each document gets a 128-number
  signature whose positions agree with probability equal to the Jaccard
  similarity. Locality-sensitive hashing (LSH) then only compares documents
  that agree on at least one band of 8 positions.

    P(candidate) = 1 - (1 - s^8)^16   ≈ 0.98 at s=0.9, ≈ 0.01 at s=0.5
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from raggy.types import Document

NUM_PERM = 128
BANDS, ROWS = 16, 8  # BANDS * ROWS == NUM_PERM
SHINGLE = 5
MIN_WORDS = 50  # shorter texts share too few shingles to judge; exact match only
NEAR_DUP = 0.9  # estimated Jaccard similarity at or above which we call it a copy

# Each of the 128 "permutations" is h(x) = (a*x + b) mod p. The product must wrap
# around p many times or every h orders shingles almost the same way, and the
# 128 positions stop being independent (the estimate's error grows ~5x).
# With p = 2^31 - 1 and a, x < p, a*x < 2^62: it wraps ~2^31 times and fits uint64.
_PRIME = np.uint64((1 << 31) - 1)
_rng = np.random.default_rng(20261025)  # fixed seed: signatures must match across runs
_A = _rng.integers(1, int(_PRIME), NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, int(_PRIME), NUM_PERM, dtype=np.uint64)
_WORD = re.compile(r"\w+")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def minhash(text: str) -> np.ndarray | None:
    words = _WORD.findall(text.lower())
    if len(words) < MIN_WORDS:
        return None
    shingles = {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}
    x = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % int(_PRIME)
                     for s in shingles), dtype=np.uint64, count=len(shingles))
    return ((np.outer(_A, x) + _B[:, None]) % _PRIME).min(axis=1)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


class DuplicateIndex:
    def __init__(self) -> None:
        self._by_hash: dict[str, str] = {}
        self._sigs: dict[str, np.ndarray] = {}
        self._bands: dict[tuple[int, bytes], list[str]] = {}

    def add(self, source: str, digest: str, sig: np.ndarray | None) -> None:
        self._by_hash.setdefault(digest, source)
        if sig is not None:
            self._sigs[source] = sig
            for b in range(BANDS):
                self._bands.setdefault((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), []).append(source)

    def find(self, digest: str, sig: np.ndarray | None) -> str | None:
        """The source this document duplicates, or None."""
        if digest in self._by_hash:
            return self._by_hash[digest]
        if sig is None:
            return None
        seen = set()
        for b in range(BANDS):
            for cand in self._bands.get((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), ()):
                if cand not in seen:
                    seen.add(cand)
                    if similarity(sig, self._sigs[cand]) >= NEAR_DUP:
                        return cand
        return None


def drop_duplicates(docs: list[Document]) -> tuple[list[Document], dict[str, str]]:
    """Keep one document per group of copies: the one with the shortest source
    path (usually the canonical one, not ``archive/2025/...``). Returns the kept
    documents in their original order and ``{duplicate: kept}``."""
    idx, dups = DuplicateIndex(), {}
    for d in sorted(docs, key=lambda d: (len(d.source), d.source)):
        digest, sig = content_hash(d.text), minhash(d.text)
        original = idx.find(digest, sig)
        if original is None:
            idx.add(d.source, digest, sig)
        else:
            dups[d.source] = original
    return [d for d in docs if d.source not in dups], dups
