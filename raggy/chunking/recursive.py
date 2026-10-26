"""Recursive splitting: cut at the biggest natural boundary that makes pieces fit.

Try paragraphs first, then lines, then sentences, then words, and only as a
last resort raw characters. Then merge neighbouring pieces back together up
to the size limit, carrying a few trailing pieces into the next chunk as
overlap. Every piece is a span of the original text, so chunk offsets stay exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from raggy.chunking.tokens import Counter
from raggy.types import Chunk, Document

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Unit:
    """A span that fits on its own. Chunks are runs of units."""

    start: int
    end: int
    tokens: int
    context: str = ""
    brk: bool = False  # start a new chunk at this unit, whatever the size


def _cut(text: str, lo: int, hi: int, sep: str) -> list[tuple[int, int]]:
    """Cut [lo, hi) after each occurrence of sep (the separator stays on the left)."""
    parts, s = [], lo
    while True:
        i = text.find(sep, s, hi)
        if i < 0:
            break
        parts.append((s, i + len(sep)))
        s = i + len(sep)
    if s < hi:
        parts.append((s, hi))
    return parts


def _hard_split(text: str, lo: int, hi: int, count: Counter, size: int) -> list[tuple[int, int]]:
    out, s = [], lo
    while s < hi:
        a, b = s + 1, hi  # the largest end whose span still fits (at least one character)
        while a < b:
            mid = (a + b + 1) // 2
            if count(text[s:mid]) <= size:
                a = mid
            else:
                b = mid - 1
        out.append((s, a))
        s = a
    return out


def split_span(text: str, lo: int, hi: int, count: Counter, size: int,
               separators: list[str] = SEPARATORS) -> list[tuple[int, int]]:
    """Pieces covering [lo, hi), each within ``size`` tokens."""
    if count(text[lo:hi]) <= size:
        return [(lo, hi)]
    for i, sep in enumerate(separators):
        if sep == "":
            return _hard_split(text, lo, hi, count, size)
        if text.find(sep, lo, hi) < 0:
            continue
        out = []
        for s, e in _cut(text, lo, hi, sep):
            out += split_span(text, s, e, count, size, separators[i + 1:])
        return out
    return _hard_split(text, lo, hi, count, size)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def make_units(text: str, spans: list[tuple[int, int]], count: Counter, context: str = "") -> list[Unit]:
    units = []
    for s, e in spans:
        s, e = _trim(text, s, e)
        if s < e:
            units.append(Unit(s, e, count(text[s:e]), context))
    return units


def pack(doc: Document, units: list[Unit], count: Counter, size: int, overlap: int) -> list[Chunk]:
    """Greedily merge units into chunks of at most ``size`` tokens.

    When a chunk closes because it is full (not at a forced break), trailing
    units worth at most ``overlap`` tokens start the next chunk too.
    """
    text = doc.text
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, u in enumerate(units):
        if cur and (u.brk or count(text[units[cur[0]].start:u.end]) > size):
            groups.append(cur)
            carry: list[int] = []
            if not u.brk and overlap:
                total = 0
                for j in reversed(cur[1:]):  # never carry the whole previous chunk
                    total += units[j].tokens
                    if total > overlap:
                        break
                    carry.insert(0, j)
                while carry and count(text[units[carry[0]].start:u.end]) > size:
                    carry.pop(0)
            cur = carry
        cur.append(i)
    if cur:
        groups.append(cur)
    chunks = []
    for g in groups:
        s, e = units[g[0]].start, units[g[-1]].end
        idx = len(chunks)
        chunks.append(Chunk(id=f"{doc.source}#{idx}", source=doc.source, text=text[s:e], index=idx,
                            start=s, end=e, context=units[g[0]].context))
    return chunks


def sentence_spans(text: str, lo: int, hi: int, count: Counter, size: int) -> list[tuple[int, int]]:
    """Paragraphs, then lines, then sentences: always cut that far, so chunks and
    overlap are made of whole sentences. A sentence over ``size`` is split further."""
    out = []
    for a, b in _cut(text, lo, hi, "\n\n"):
        for c, d in _cut(text, a, b, "\n"):
            for e, f in _cut(text, c, d, ". "):
                out += split_span(text, e, f, count, size, [" ", ""])
    return out


def chunk_recursive(doc: Document, count: Counter, size: int = 256, overlap: int = 32) -> list[Chunk]:
    text = doc.text
    if not text.strip():
        return []
    spans = sentence_spans(text, 0, len(text), count, size)
    title = doc.metadata.get("title", "")
    return pack(doc, make_units(text, spans, count, title), count, size, overlap)
