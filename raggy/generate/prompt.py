"""The prompt is where retrieval meets generation.

Three rules do most of the work: answer only from the numbered sources, cite
them as [n], and use one fixed sentence when the answer isn't there. The
fixed sentence lets code (and week 2's evals) recognise a refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

from raggy.types import Citation, Hit

NOT_FOUND = "I couldn't find this in the documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- After each claim, cite the source it came from, like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
- Be concise: a few sentences unless the question asks for more."""

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SNIPPET_CHARS = 160


def source_text(hit: Hit) -> str:
    """A retrieved chunk as the model reads it: its context line (section path), then its text."""
    c = hit.chunk
    return f"{c.context}\n{c.text.strip()}" if c.context else c.text.strip()


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [
        f"[{n}] (source: {Path(h.chunk.source).name})\n{source_text(h)}"
        for n, h in enumerate(hits, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n"


def parse_citations(text: str, hits: list[Hit]) -> list[Citation]:
    """Collect the [n] markers in the answer, in order of first use."""
    seen: list[int] = []
    for group in _CITE.findall(text):
        for part in group.split(","):
            n = int(part)
            if 1 <= n <= len(hits) and n not in seen:
                seen.append(n)
    citations = []
    for n in seen:
        chunk = hits[n - 1].chunk
        snippet = " ".join(chunk.text.split())[:SNIPPET_CHARS]
        citations.append(Citation(number=n, source=chunk.source, chunk_id=chunk.id, snippet=snippet))
    return citations
