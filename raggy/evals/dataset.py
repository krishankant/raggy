"""Eval questions and their gold evidence.

Gold evidence is stored as a verbatim quote from the corpus, not as a chunk
id. Chunk ids change whenever chunking changes (week 4 changes it a lot);
a quote does not. At eval time each quote is located as a character span,
and a retrieved chunk counts as relevant when its span overlaps a gold span.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from raggy.types import Chunk, Document


@dataclass
class GoldSpan:
    source: str  # a document's source, or just its file name
    quote: str | None = None  # None: the whole document is relevant


@dataclass
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    gold: list[GoldSpan]
    reference: str | None = None  # a correct answer, for the judge
    tags: list[str] = field(default_factory=list)


@dataclass
class Located:
    source: str
    start: int
    end: int


class GoldNotFoundError(ValueError):
    """A gold quote isn't in the corpus. Fix the dataset; don't score around it."""


def load_questions(path: str | Path) -> list[EvalQuestion]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(EvalQuestion(
            id=row["id"],
            question=row["question"],
            answerable=row["answerable"],
            gold=[GoldSpan(g["source"], g.get("quote")) for g in row.get("gold", [])],
            reference=row.get("reference"),
            tags=row.get("tags", []),
        ))
    return questions


def _find_doc(source: str, docs_by_source: dict[str, Document], docs_by_name: dict[str, Document]):
    return docs_by_source.get(source) or docs_by_name.get(source)


def resolve_gold(questions: list[EvalQuestion], docs: list[Document]) -> dict[str, list[Located]]:
    """Turn every gold quote into a character span. Raises on the first miss."""
    by_source = {d.source: d for d in docs}
    by_name = {Path(d.source).name: d for d in docs}
    resolved: dict[str, list[Located]] = {}
    for q in questions:
        spans = []
        for g in q.gold:
            doc = _find_doc(g.source, by_source, by_name)
            if doc is None:
                raise GoldNotFoundError(f"question {q.id}: gold source {g.source!r} is not in the corpus")
            if g.quote is None:
                spans.append(Located(doc.source, 0, len(doc.text)))
                continue
            # Match the quote's words with any whitespace between them, so a
            # line break in the document doesn't make a correct quote fail.
            pattern = r"\s+".join(re.escape(w) for w in g.quote.split())
            m = re.search(pattern, doc.text)
            if m is None:
                raise GoldNotFoundError(
                    f"question {q.id}: quote not found in {g.source!r}: {g.quote!r}")
            spans.append(Located(doc.source, m.start(), m.end()))
        resolved[q.id] = spans
    return resolved


def chunk_gold(chunk: Chunk, gold: list[Located]) -> set[int]:
    """Indices of the gold spans this chunk overlaps."""
    return {
        i for i, g in enumerate(gold)
        if g.source == chunk.source and chunk.start < g.end and g.start < chunk.end
    }
