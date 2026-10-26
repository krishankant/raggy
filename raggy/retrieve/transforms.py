"""Query transformations: change the question before retrieving with it.

A question is phrased the way a person asks; the answer is phrased the way a
document states it. Three standard tricks bridge the gap with one LLM call:

- rewrite: turn the question into a search query ("refunds" "days" "return")
- multi:   ask it three ways, retrieve with each, fuse the rankings (RRF)
- hyde:    write a *hypothetical answer* and search with that (Gao et al., 2022),
           because an answer looks more like the documents than a question does

Each costs a call per question (money, and about a second), so each is a
switch measured by week 7's eval. Results are cached by model, prompt version
and question, and a failed call falls back to the original question.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from raggy.generate.llm import LLM, LLMError

TRANSFORM_VERSION = "qt-v1"
CACHE_NAME = "transform-cache.sqlite"
KINDS = ("rewrite", "multi", "hyde")

SYSTEMS = {
    "rewrite": ("Rewrite the user's question as a short search query for a document collection: the key terms "
                "and the words a document answering it would likely use. Reply with the query only."),
    "multi": ("Write 3 alternative phrasings of the user's question that a search engine might match better: "
              "different words, same meaning. Reply with one phrasing per line, nothing else."),
    "hyde": ("Write a short passage (2-4 sentences) that would answer the user's question, in the plain, factual "
             "style of a help page or manual. It's only used to search for the real document, so invent "
             "plausible details if you must. Reply with the passage only."),
}


@dataclass
class TransformResult:
    queries: list[str]  # what to search with (keyword side always gets queries[0] for hyde)
    hyde_passage: str | None = None
    cached: bool = False
    failed: bool = False
    raw: str = field(default="", repr=False)


class TransformCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS transforms (key TEXT PRIMARY KEY, text TEXT NOT NULL)")
        self.conn.commit()

    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT text FROM transforms WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO transforms (key, text) VALUES (?, ?)", (key, text))

    def commit(self) -> None:
        self.conn.commit()


def _key(kind: str, question: str, model: str) -> str:
    return hashlib.sha256(f"{TRANSFORM_VERSION}\0{kind}\0{model}\0{question}".encode()).hexdigest()


def _result(kind: str, question: str, text: str) -> TransformResult:
    if kind == "rewrite":
        return TransformResult(queries=[" ".join(text.split())], raw=text)
    if kind == "multi":
        lines = [question] + [ln.strip(" -*0123456789.\t") for ln in text.splitlines()]
        seen, queries = set(), []
        for q in lines:  # dedupe, case-insensitively: a repeated question would count twice in RRF
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
        return TransformResult(queries=queries, raw=text)
    return TransformResult(queries=[question], hyde_passage=text.strip(), raw=text)


def transform(kind: str, question: str, llm: LLM, cache: TransformCache) -> TransformResult:
    key = _key(kind, question, llm.name)
    hit = cache.get(key)
    if hit is not None:
        r = _result(kind, question, hit)
        r.cached = True
        return r
    try:
        text = llm.complete(SYSTEMS[kind], question)
    except LLMError:
        return TransformResult(queries=[question], failed=True)
    if not text.strip():
        return TransformResult(queries=[question], failed=True)
    cache.put(key, text)
    return _result(kind, question, text)
