"""Generation: turn retrieved chunks plus a question into a cited answer."""

from __future__ import annotations

from raggy.generate.llm import LLM
from raggy.generate.prompt import NOT_FOUND, SYSTEM_PROMPT, build_user_prompt, parse_citations
from raggy.types import Answer, Hit


def generate_answer(question: str, hits: list[Hit], llm: LLM, check: str = "off", threshold: float = 0.5,
                    action: str = "report", check_llm: LLM | None = None, on_token=None) -> Answer:
    """``check`` verifies each citation (off | lexical | llm); ``action`` decides what an
    unsupported one does to the text (report | strip | refuse). See generate/verify.py."""
    if not hits:  # nothing retrieved: don't pay for a model call to say so
        return Answer(question=question, text=NOT_FOUND, citations=[], hits=[])
    if on_token is not None and hasattr(llm, "stream"):  # week 12: show the answer as it's written
        pieces = []
        for piece in llm.stream(SYSTEM_PROMPT, build_user_prompt(question, hits)):
            pieces.append(piece)
            on_token(piece)
        text = "".join(pieces).strip()
    else:
        text = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
    if check == "off":
        return Answer(question=question, text=text, citations=parse_citations(text, hits), hits=hits)
    from raggy.generate.verify import apply_action, verify

    v = verify(text, hits, method=check, threshold=threshold, llm=check_llm or llm)
    text = apply_action(text, v, action)
    return Answer(question=question, text=text, citations=parse_citations(text, hits), hits=hits,
                  unsupported=v.unsupported, verification="failed" if v.failed else check)
