import numpy as np

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.runner import CACHE_FILES
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.retrieve.transforms import TransformCache, transform
from raggy.types import Document


def scripted(rewrite="refund policy days", multi="money back time\nreturn refund days\nHow long do refunds take?",
             hyde="Refunds are paid within 14 days of the return arriving."):
    def reply(system, user):
        if "search query" in system:
            return rewrite
        if "alternative" in system:
            return multi
        return hyde
    return FakeLLM(reply)


def test_transform_kinds(tmp_path):
    cache = TransformCache(tmp_path / "t.sqlite")
    llm = scripted()
    r = transform("rewrite", "How long do refunds take?", llm, cache)
    assert r.queries == ["refund policy days"] and r.hyde_passage is None
    m = transform("multi", "How long do refunds take?", llm, cache)
    assert m.queries == ["How long do refunds take?", "money back time", "return refund days"]  # original first, deduped
    h = transform("hyde", "How long do refunds take?", llm, cache)
    assert h.queries == ["How long do refunds take?"] and h.hyde_passage.startswith("Refunds are paid")


def test_transform_cached(tmp_path):
    cache = TransformCache(tmp_path / "t.sqlite")
    llm = scripted()
    transform("rewrite", "q?", llm, cache)
    cache.commit()
    again = transform("rewrite", "q?", llm, TransformCache(tmp_path / "t.sqlite"))
    assert again.cached and len(llm.calls) == 1


def test_transform_failure_falls_back(tmp_path):
    r = transform("rewrite", "q?", FakeLLM(error=LLMError("down")), TransformCache(tmp_path / "t.sqlite"))
    assert r.failed and r.queries == ["q?"] and r.hyde_passage is None
    blank = transform("rewrite", "q?", FakeLLM("   "), TransformCache(tmp_path / "t2.sqlite"))
    assert blank.failed and blank.queries == ["q?"]


def test_transform_cache_in_cache_files():
    assert "transform-cache.sqlite" in CACHE_FILES


DOCS = [Document("/d/refunds.md", "Refunds are paid within 14 days of the return arriving.", {"type": "markdown"}),
        Document("/d/cat.md", "The office cat is called Biscuit.", {"type": "markdown"}),
        Document("/d/ship.md", "Delivery takes 3 to 5 working days.", {"type": "markdown"})]


class Spy(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.queries, self.passages = [], []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)

    def embed_documents(self, texts):
        self.passages.append(list(texts))
        return super().embed_documents(texts)


def make(tmp_path, llm=None, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0, "embed_cache": False} | cfg))
    emb = Spy()
    r = Raggy(c, embedder=emb, llm=llm or scripted())
    r.ingest_documents(DOCS)
    emb.queries.clear(), emb.passages.clear()
    return r, emb


def test_rewrite_used_for_both_legs(tmp_path, monkeypatch):
    r, emb = make(tmp_path, query_transform="rewrite", retrieval="hybrid")
    seen = []
    from raggy.store.lance_store import LanceStore
    orig = LanceStore.keyword_search
    monkeypatch.setattr(LanceStore, "keyword_search", lambda self, q, k, where=None: seen.append(q) or orig(self, q, k, where))
    r.search("How long do refunds take?", k=1)
    assert emb.queries == ["refund policy days"] and seen == ["refund policy days"]


def test_multi_query_dedupes_variants(tmp_path):
    r, emb = make(tmp_path, query_transform="multi")
    r.search("How long do refunds take?", k=1)
    assert emb.queries == ["How long do refunds take?", "money back time", "return refund days"]


def test_multi_query_fuses(tmp_path):
    r, _ = make(tmp_path, query_transform="multi")
    assert r.search("How long do refunds take?", k=1)[0].chunk.source == "/d/refunds.md"


def test_hyde_vector_leg_embeds_passage(tmp_path):
    r, emb = make(tmp_path, query_transform="hyde")
    r.search("How long until I get my money back?", k=1)
    assert emb.queries == [] and emb.passages == [["Refunds are paid within 14 days of the return arriving."]]


def test_hyde_keyword_leg_uses_original(tmp_path, monkeypatch):
    r, _ = make(tmp_path, query_transform="hyde", retrieval="hybrid")
    seen = []
    from raggy.store.lance_store import LanceStore
    orig = LanceStore.keyword_search
    monkeypatch.setattr(LanceStore, "keyword_search", lambda self, q, k, where=None: seen.append(q) or orig(self, q, k, where))
    r.search("How long until I get my money back?", k=1)
    assert seen == ["How long until I get my money back?"]


def test_transform_override_and_stats(tmp_path):
    r, emb = make(tmp_path)
    r.search("How long do refunds take?", k=1, transform="rewrite")
    assert emb.queries == ["refund policy days"]
    assert r.transform_stats == {"calls": 1, "cached": 0, "failed": 0}
