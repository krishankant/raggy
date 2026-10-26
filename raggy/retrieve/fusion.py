"""Reciprocal rank fusion: merge rankings by position, not by score.

BM25 scores (0 to ~20, unbounded) and cosine similarities (about 0.5 to 0.9)
live on different scales, and either scale shifts from query to query. Adding
them would let one side drown the other. RRF ignores the scores and uses each
chunk's *rank* in each list:

    score(c) = Σ over rankings  1 / (k + rank(c))        (rank starts at 1)

A chunk ranked first by both sides beats one ranked first by only one side.
k (60 in the original paper, Cormack et al. 2009) flattens the curve, so
being 1st instead of 5th matters, but not overwhelmingly.
"""

from __future__ import annotations

from raggy.types import Hit


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            cid = h.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best[cid] = min(best.get(cid, rank), rank)
            hits.setdefault(cid, h)
    order = sorted(scores, key=lambda c: (-scores[c], best[c], c))  # ties: best single rank, then id
    return [Hit(chunk=hits[c].chunk, score=scores[c]) for c in order]
