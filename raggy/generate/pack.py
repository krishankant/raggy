"""Context packing: turn the retrieved hits into the passages the model reads.

Retrieval hands back up to k chunks; three things are worth doing before
they become the prompt:

1. Merge. Two hits from one document that overlap or touch (overlapping
   chunks, a child and its neighbour) are one passage, not two: duplicates
   waste the context and split one citation into two.
2. Order. Sources by their best score; within one source, document order,
   so a passage reads the way it was written.
3. Budget. Keep the best passages that fit RAGGY_CONTEXT_TOKENS. A lone
   passage bigger than the budget is cut at a sentence end, never dropped.
"""

from __future__ import annotations

import re

from raggy.chunking.tokens import Counter
from raggy.generate.prompt import source_text
from raggy.types import Chunk, Hit

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _merge(a: Hit, b: Hit) -> Hit:
    """b starts at or before a's end (same source): one passage covering both."""
    ca, cb = a.chunk, b.chunk
    text = ca.text + cb.text[ca.end - cb.start:] if cb.end > ca.end else ca.text
    end = max(ca.end, cb.end)
    chunk = Chunk(id=f"{ca.source}#{ca.start}-{end}", source=ca.source, text=text, index=ca.index,
                  start=ca.start, end=end, context=ca.context, meta=ca.meta)
    return Hit(chunk=chunk, score=max(a.score, b.score))


def _cut(hit: Hit, count: Counter, budget: int) -> Hit:
    c = hit.chunk
    cuts = [m.start() for m in _SENTENCE_END.finditer(c.text)] + [len(c.text)]
    keep = next((i for i in reversed(cuts) if count(c.text[:i]) <= budget), cuts[0])
    text = c.text[:keep]
    return Hit(Chunk(id=c.id, source=c.source, text=text, index=c.index, start=c.start, end=c.start + len(text),
                     context=c.context, meta=c.meta), hit.score)


def pack(hits: list[Hit], count: Counter, budget: int) -> list[Hit]:
    by_source: dict[str, list[Hit]] = {}
    for hit in hits:
        by_source.setdefault(hit.chunk.source, []).append(hit)
    passages: list[Hit] = []
    for group in by_source.values():
        group.sort(key=lambda x: x.chunk.start)
        merged = [group[0]]
        for hit in group[1:]:
            if hit.chunk.start <= merged[-1].chunk.end:  # overlaps or touches
                merged[-1] = _merge(merged[-1], hit)
            else:
                merged.append(hit)
        passages += merged
    if not passages:
        return []
    # Fit the budget: keep the best-scoring passages that fit, whole.
    kept, used = [], 0
    for p in sorted(passages, key=lambda x: -x.score):
        n = count(source_text(p))
        if used + n <= budget:
            kept.append(p)
            used += n
    if not kept:  # even the best passage alone is too big: cut it rather than send nothing
        best = max(passages, key=lambda x: x.score)
        kept = [_cut(best, count, budget)]
    best_of = {}
    for p in kept:
        best_of[p.chunk.source] = max(best_of.get(p.chunk.source, p.score), p.score)
    return sorted(kept, key=lambda p: (-best_of[p.chunk.source], p.chunk.source, p.chunk.start))
