"""Chunking: turn a Document into the pieces that get embedded and retrieved.

    fixed      week 1's character windows (the baseline)
    recursive  token-sized, cut at paragraphs → lines → sentences → words
    structure  Markdown-aware: whole blocks, section paths, tables and code kept intact

``chunk_document`` picks one from ``ChunkSettings``, adds or clears the
contextual header, and adds parents when parent/child retrieval is on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from raggy.chunking.fixed import chunk_fixed
from raggy.chunking.recursive import chunk_recursive
from raggy.chunking.structure import chunk_structure
from raggy.chunking.tokens import Counter
from raggy.types import Chunk, Document

CHUNKERS = ("fixed", "recursive", "structure")


@dataclass(frozen=True)
class ChunkSettings:
    chunker: str = "structure"
    tokens: int = 256  # recursive / structure: chunk size in model tokens
    overlap: int = 32  # tokens shared with the previous chunk
    chars: int = 800  # fixed: chunk size in characters (week 1)
    char_overlap: int = 100
    headers: bool = True  # embed "title › section" in front of each chunk
    parent_tokens: int = 512  # > 0: search these chunks, answer with the parent that holds them

    def key(self) -> dict:
        return asdict(self)


def _base(doc: Document, s: ChunkSettings, count: Counter, size: int, overlap: int) -> list[Chunk]:
    if s.chunker == "fixed":
        return chunk_fixed(doc, s.chars, s.char_overlap)
    if s.chunker == "recursive":
        return chunk_recursive(doc, count, size, overlap)
    if s.chunker == "structure":
        return chunk_structure(doc, count, size, overlap)
    raise ValueError(f"unknown chunker {s.chunker!r}")


def chunk_document(doc: Document, settings: ChunkSettings, count: Counter) -> list[Chunk]:
    chunks = _base(doc, settings, count, settings.tokens, settings.overlap)
    if not settings.headers:
        for c in chunks:
            c.context = ""
    if settings.parent_tokens and settings.chunker != "fixed":
        _add_parents(doc, chunks, count, settings.parent_tokens)
    return chunks


def _add_parents(doc: Document, chunks: list[Chunk], count: Counter, size: int) -> None:
    """Group consecutive chunks into parents of up to ``size`` tokens.

    Parents are built from the children, not chunked separately, so every child
    lies wholly inside its parent: a parent never cuts a child in half. Groups
    ignore section boundaries; adding the neighbouring section a small,
    section-bound child cut off is what a parent is for."""
    groups: list[list[Chunk]] = []
    for c in chunks:
        if groups and count(doc.text[groups[-1][0].start:c.end]) <= size:
            groups[-1].append(c)
        else:
            groups.append([c])
    for g in groups:
        s, e = g[0].start, max(c.end for c in g)
        for c in g:
            c.parent_start, c.parent_end, c.parent_text = s, e, doc.text[s:e]
