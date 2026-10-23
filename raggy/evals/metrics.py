"""Retrieval metrics, written out so you can read them.

Each retrieved item is described by the set of gold items it covers. A gold
item counts once: a second chunk covering the same evidence adds nothing,
because the reader (or the LLM) learns nothing new from it.
"""

from __future__ import annotations

import math

from raggy.types import Hit


def retrieval_metrics(hit_gold: list[set[int]], n_gold: int, ks: tuple[int, ...]) -> dict[str, float]:
    """recall@k and nDCG@k for each k, plus MRR over the whole list.

    hit_gold[i] is the set of gold indices the item at rank i+1 covers.
    """
    metrics: dict[str, float] = {}
    for k in ks:
        seen: set[int] = set()
        dcg = 0.0
        for rank, golds in enumerate(hit_gold[:k], start=1):
            new = golds - seen
            if new:
                dcg += 1.0 / math.log2(rank + 1)  # gain 1, discounted by position
                seen |= new
        ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_gold, k) + 1))
        metrics[f"recall@{k}"] = len(seen) / n_gold if n_gold else 0.0
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    first = next((rank for rank, g in enumerate(hit_gold, start=1) if g), None)
    metrics["mrr"] = 1.0 / first if first else 0.0
    return metrics


def collapse_to_documents(hits: list[Hit]) -> list[Hit]:
    """Keep each source's best hit: turns a chunk ranking into a document ranking."""
    seen: set[str] = set()
    out = []
    for h in hits:
        if h.chunk.source not in seen:
            seen.add(h.chunk.source)
            out.append(h)
    return out


def mean(xs) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(xs, p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
