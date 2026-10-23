"""Generation: turn retrieved chunks plus a question into a cited answer."""

from __future__ import annotations

from raggy.generate.llm import LLM
from raggy.generate.prompt import NOT_FOUND, SYSTEM_PROMPT, build_user_prompt, parse_citations
from raggy.types import Answer, Hit


def generate_answer(question: str, hits: list[Hit], llm: LLM) -> Answer:
    if not hits:  # nothing retrieved: don't pay for a model call to say so
        return Answer(question=question, text=NOT_FOUND, citations=[], hits=[])
    text = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
    return Answer(question=question, text=text, citations=parse_citations(text, hits), hits=hits)
