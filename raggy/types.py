"""The data that flows through raggy. Every stage takes and returns these."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """One loaded source: a file today, a web page or a PDF page later."""

    source: str  # absolute path (or URL from week 3)
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document small enough to embed and to quote."""

    id: str  # f"{source}#{index}" — stable, so re-ingest can replace it
    source: str
    text: str
    index: int  # position of the chunk inside its document
    start: int  # character offset into Document.text
    end: int


@dataclass
class Hit:
    """A chunk returned by search, with its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A source the answer actually referred to, as [number]."""

    number: int
    source: str
    chunk_id: str
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    hits: list[Hit]
