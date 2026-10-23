"""An LLM as judge: grade answers the way a careful reader would.

Two questions are asked separately, because they need different evidence:

- correctness: does the answer say what the reference answer says?
  (needs the reference, not the sources)
- grounding: is every claim backed by the sources, and does each [n]
  point at a source that supports its sentence? (needs the sources, not
  the reference: an answer can be true and still not come from the docs)

The prompts are versioned. Changing a word in them can move the scores, so
a run records JUDGE_VERSION and runs with different versions aren't compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from raggy.evals.dataset import EvalQuestion
from raggy.generate.llm import LLM, LLMError
from raggy.generate.prompt import NOT_FOUND
from raggy.types import Answer

JUDGE_VERSION = "judge-v1"

CORRECTNESS_SYSTEM = """You grade answers to questions about a company's documents.
You get a question, a reference answer written by an expert, and a candidate answer.

Grade the candidate against the reference only:
- "correct": it contains every key fact of the reference, and nothing that contradicts it. Different wording is fine. Extra detail is fine if it doesn't contradict the reference.
- "partial": it is right as far as it goes but misses a key fact of the reference.
- "incorrect": it contradicts the reference, gives a wrong key fact, or says it can't answer.

Ignore citation markers like [1]. Reply with JSON only:
{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}"""

GROUNDING_SYSTEM = """You check whether an answer is supported by its numbered sources.
Use only the sources. Do not use outside knowledge, even if a claim is true in the real world.

1. "faithful": true if every factual claim in the answer is stated in, or directly follows from, the sources. false if any claim is not.
2. "citations": for each citation number [n] used in the answer, "supported" is true if source n backs the sentence where [n] appears.

Reply with JSON only:
{"faithful": true | false, "citations": [{"n": 1, "supported": true | false}], "reason": "<one sentence>"}"""

_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass
class Verdict:
    correctness: float | None = None  # 1, 0.5 or 0
    faithful: bool | None = None
    citation_precision: float | None = None  # supported citations / citations used
    reason: str = ""
    error: str | None = None


def is_refusal(text: str) -> bool:
    """Did the model use the fixed not-found sentence (anywhere in its reply)?"""
    norm = lambda s: s.replace("’", "'").lower().rstrip(".")
    return norm(NOT_FOUND) in norm(text)


def parse_json_object(text: str) -> dict:
    """The first balanced {...} in the text that parses, so code fences and chatter don't matter."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth, in_str, escaped = 0, False, False
        for end in range(start, len(text)):
            ch = text[end]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    raise ValueError(f"no JSON object in judge reply: {text[:120]!r}")


class Judge:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = llm.name

    def judge(self, q: EvalQuestion, answer: Answer) -> Verdict:
        v = Verdict()
        errors, reasons = [], []
        try:
            reply = parse_json_object(self.llm.complete(CORRECTNESS_SYSTEM, _correctness_prompt(q, answer)))
            if reply.get("verdict") not in _SCORES:
                raise ValueError(f"unknown verdict {reply.get('verdict')!r}")
            v.correctness = _SCORES[reply["verdict"]]
            reasons.append(str(reply.get("reason", "")))
        except (LLMError, ValueError) as e:
            errors.append(f"correctness: {e}")
        try:
            reply = parse_json_object(self.llm.complete(GROUNDING_SYSTEM, _grounding_prompt(q, answer)))
            if not isinstance(reply.get("faithful"), bool):
                raise ValueError("missing 'faithful'")
            v.faithful = reply["faithful"]
            supported = {int(c["n"]) for c in reply.get("citations", []) if c.get("supported") is True}
            used = [c.number for c in answer.citations]
            v.citation_precision = sum(n in supported for n in used) / len(used) if used else None
            reasons.append(str(reply.get("reason", "")))
        except (LLMError, ValueError, KeyError, TypeError) as e:
            errors.append(f"grounding: {e}")
        v.reason = " | ".join(r for r in reasons if r)
        v.error = "; ".join(errors) or None
        return v


def _correctness_prompt(q: EvalQuestion, answer: Answer) -> str:
    return f"Question: {q.question}\n\nReference answer: {q.reference}\n\nCandidate answer: {answer.text}\n"


def _grounding_prompt(q: EvalQuestion, answer: Answer) -> str:
    sources = "\n\n".join(f"[{n}]\n{h.chunk.text.strip()}" for n, h in enumerate(answer.hits, start=1))
    return f"Sources:\n\n{sources}\n\nQuestion: {q.question}\n\nAnswer to check: {answer.text}\n"
