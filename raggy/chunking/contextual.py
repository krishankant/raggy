"""Contextual retrieval: have an LLM explain where each chunk sits in its document.

A chunk like "Paid within 14 days." means little alone: paid by whom, for
what? Anthropic's "contextual retrieval" (2024) asks a model, for every
chunk, to write one or two sentences situating it in the whole document, and
embeds that context with the chunk. It works, and it costs one LLM call per
chunk, so here it's off by default, its results are cached on disk (by
document hash, chunk span, model and prompt version), and every run reports
how many calls it made.

The prompt puts the whole document first and the chunk last. Providers that
cache prompt prefixes (OpenAI does it automatically above 1,024 tokens) can
then reuse the document part across all of that document's chunks.
"""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from raggy.chunking.tokens import Counter
from raggy.generate.llm import LLM, LLMError
from raggy.types import Chunk, Document

CONTEXT_VERSION = "ctx-v1"  # bump when the prompt changes: cached contexts are then ignored
CACHE_NAME = "context-cache.sqlite"
MAX_DOC_CHARS = 60_000  # longer documents are cut, to bound the cost per call
WORKERS = 8

CONTEXT_SYSTEM = """You write short context for chunks of a document, to improve search retrieval.
Given the whole document and one chunk from it, write one or two sentences that situate the
chunk within the document: what the document is, and what this part of it covers. Name the
product, policy or entity the chunk refers to if the chunk itself doesn't say it.
Answer with only the context sentences, nothing else."""


@dataclass
class ContextStats:
    calls: int = 0
    cached: int = 0
    failed: int = 0


class ContextCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS context (key TEXT PRIMARY KEY, text TEXT NOT NULL)")
        self.conn.commit()

    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT text FROM context WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO context (key, text) VALUES (?, ?)", (key, text))

    def commit(self) -> None:
        self.conn.commit()


def cache_key(doc: Document, chunk: Chunk, model: str) -> str:
    doc_hash = hashlib.sha256(doc.text.encode()).hexdigest()
    return hashlib.sha256(f"{doc_hash}:{chunk.start}:{chunk.end}:{model}:{CONTEXT_VERSION}".encode()).hexdigest()


def _prompt(doc: Document, chunk: Chunk) -> str:
    return f"<document>\n{doc.text[:MAX_DOC_CHARS]}\n</document>\n\n<chunk>\n{chunk.text}\n</chunk>"


def _fit(context: str, chunk: Chunk, count: Counter | None, max_tokens: int | None) -> str:
    """Drop words from the end of the context until context + chunk fits the embedder."""
    if count is None or max_tokens is None:
        return context
    words = context.split(" ")
    while words and count(f"{' '.join(words)}\n\n{chunk.text}") > max_tokens:
        words = words[:-max(1, len(words) // 8)]
    return " ".join(words)


def add_llm_context(doc: Document, chunks: list[Chunk], llm: LLM, cache: ContextCache,
                    count: Counter | None = None, max_tokens: int | None = None) -> ContextStats:
    """Put an LLM-written context in front of each chunk's header context (in place)."""
    stats = ContextStats()
    keys = [cache_key(doc, c, llm.name) for c in chunks]
    texts: dict[int, str] = {}
    todo = []
    for i, key in enumerate(keys):
        hit = cache.get(key)
        if hit is None:
            todo.append(i)
        else:
            texts[i] = hit
            stats.cached += 1

    def call(i: int) -> tuple[int, str | None]:
        try:
            return i, " ".join(llm.complete(CONTEXT_SYSTEM, _prompt(doc, chunks[i])).split())
        except LLMError:
            return i, None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, text in pool.map(call, todo):
            stats.calls += 1
            if text:
                texts[i] = text
                cache.put(keys[i], text)  # only successes are cached, so failures are retried next time
            else:
                stats.failed += 1  # this chunk keeps its header-only context
    for i, text in texts.items():
        c = chunks[i]
        header = c.context
        room = _fit(f"{text}\n{header}" if header else text, c, count, max_tokens)
        c.context = room if room else header
    return stats
