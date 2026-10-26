"""Rerankers: retrieve wide and cheap, then rank narrow and precise.

Retrieval compares the question with each chunk *separately*: the chunk was
turned into a vector long before the question arrived. A cross-encoder reads
the question and the passage *together* and outputs one relevance score, so
it can see that "refunds for a bike I rode" matches "unused means ridden less
than 10 km". It's far more accurate and far slower: one model run per pair,
nothing precomputed. So it only reranks the top few dozen candidates.

- local: a fastembed cross-encoder (default ms-marco-MiniLM-L-6-v2, 80 MB)
- llm:   the configured chat model ranks all candidates in one call (listwise)

A reranker that fails returns the retrieval order unchanged and counts the
failure: reranking is an improvement, never a reason to return nothing.
"""

from __future__ import annotations

from typing import Callable, Protocol

from raggy.evals.judge import parse_json_object
from raggy.generate.llm import LLM, LLMError
from raggy.generate.prompt import source_text
from raggy.types import Hit

RERANK_VERSION = "rerank-v1"
DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str
    failures: int

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]: ...


class CrossEncoderReranker:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name, self.name = model, f"local:{model}"
        self._model = None
        self.failures = 0

    def _load(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # imported here: tests never load it

        return TextCrossEncoder(model_name=self.model_name)

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        try:
            if self._model is None:
                self._model = self._load()
            # The passage is what the answer model will read: section path, then text.
            # Pairs past 512 tokens are cut at the end, so the most telling text comes first.
            scores = [float(s) for s in self._model.rerank(question, [source_text(h) for h in hits], batch_size=16)]
        except (OSError, RuntimeError, ValueError):
            self.failures += 1
            return hits
        order = sorted(range(len(hits)), key=lambda i: -scores[i])  # stable: ties keep retrieval order
        return [Hit(chunk=hits[i].chunk, score=scores[i]) for i in order]


LLM_SYSTEM = ("You rank passages by how well they answer a question. Judge relevance to the question only, "
              "not writing quality. Reply with JSON only.")


class LLMReranker:
    """Listwise: the model sees every candidate at once and returns an order."""

    def __init__(self, llm: LLM):
        self.llm, self.name = llm, f"llm:{llm.name}"
        self.failures = 0

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        passages = "\n\n".join(f"[{i}] {source_text(h)}" for i, h in enumerate(hits, start=1))
        user = (f"Question: {question}\n\nPassages:\n\n{passages}\n\n"
                'Return {"ranking": [passage numbers, most relevant first]}, with every passage number once.')
        try:
            ranking = parse_json_object(self.llm.complete(LLM_SYSTEM, user))["ranking"]
            if not isinstance(ranking, list):
                raise ValueError("ranking is not a list")
        except (LLMError, ValueError, KeyError, TypeError):
            self.failures += 1
            return hits
        order: list[int] = []
        for x in ranking:  # ignore numbers out of range, repeats and non-numbers
            if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= len(hits) and x - 1 not in order:
                order.append(x - 1)
        if not order:
            self.failures += 1
            return hits
        order += [i for i in range(len(hits)) if i not in order]  # anything it left out keeps its place
        return [Hit(chunk=hits[i].chunk, score=float(len(hits) - pos)) for pos, i in enumerate(order)]


def make_reranker(config, llm_factory: Callable[[], LLM], kind: str | None = None) -> Reranker | None:
    kind = kind or config.rerank
    if kind == "local":
        return CrossEncoderReranker(config.rerank_model)
    if kind == "llm":
        return LLMReranker(llm_factory())
    return None
