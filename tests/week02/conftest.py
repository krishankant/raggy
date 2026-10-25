import json
from pathlib import Path

import pytest

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.dataset import EvalQuestion, GoldSpan
from raggy.evals.suites import Suite
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.types import Document

FIXTURES = Path(__file__).parent / "fixtures"

DOCS = [
    Document("refunds.md", "Refunds are paid back within 14 days of the return arriving."),
    Document("cats.md", "The office cat is called Biscuit and sleeps on the printer."),
]
QUESTIONS = [
    EvalQuestion("a1", "when are refunds paid back", True, [GoldSpan("refunds.md", "within 14 days")],
                 "Within 14 days.", ["lookup"]),
    EvalQuestion("a2", "what is the office cat called", True, [GoldSpan("cats.md", "called Biscuit")],
                 "Biscuit.", ["lookup"]),
    EvalQuestion("u1", "do you sell skateboards", False, [], None, ["unanswerable"]),
]


def tiny_suite(fingerprint="v1", unit="chunk", answers=True) -> Suite:
    return Suite(name="tiny", documents=lambda options=None: list(DOCS), questions=lambda: list(QUESTIONS),
                 fingerprint=lambda: fingerprint, ks=(1, 2), unit=unit, answers=answers)


def make_raggy(index_dir: Path, reply="Refunds take 14 days [1].", chunk_size=800, error=None) -> Raggy:
    cfg = Config(index_dir=index_dir, embed_model="fake", llm_provider="fake", llm_model=None,
                 top_k=1, chunk_size=chunk_size, chunk_overlap=min(100, chunk_size // 4))
    return Raggy(cfg, embedder=FakeEmbedder(), llm=FakeLLM(reply, error=error))


def judge_llm():
    def reply(system, user):
        if "Reference answer" in user:
            return json.dumps({"verdict": "correct", "reason": "ok"})
        return json.dumps({"faithful": True, "citations": [{"n": 1, "supported": True}]})
    return FakeLLM(reply)


@pytest.fixture
def suite():
    return tiny_suite()
