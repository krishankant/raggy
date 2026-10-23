"""The simplest chunker: fixed-size character windows that overlap.

It ignores sentences, headings and code blocks entirely. That is the point:
week 4 measures how much smarter chunking gains over this baseline.
"""

from __future__ import annotations

from raggy.types import Chunk, Document


def chunk_fixed(doc: Document, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split ``doc`` into windows of ``size`` characters, each sharing
    ``overlap`` characters with the previous one."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        text = doc.text[start:end]
        if text.strip():
            index = len(chunks)
            chunks.append(
                Chunk(id=f"{doc.source}#{index}", source=doc.source, text=text,
                      index=index, start=start, end=end)
            )
        if end == len(doc.text):
            break
        start += step
    return chunks
