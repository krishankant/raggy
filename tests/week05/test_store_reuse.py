import raggy.pipeline as pipeline_mod
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.types import Document


def test_search_reuses_the_open_store_until_a_write(tmp_path, monkeypatch):
    opened = []
    real = pipeline_mod.make_store
    monkeypatch.setattr(pipeline_mod, "make_store", lambda config, create=True: opened.append(create) or real(config, create))
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               chunker="fixed", parent_tokens=0)
    r = Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM())
    r.ingest_documents([Document("/d/a.md", "refunds take 14 days")])
    opened.clear()
    for _ in range(5):
        r.search("refunds", k=1)
    assert len(opened) == 1  # opened once, not once per question
    r.ingest_documents([Document("/d/b.md", "the cat is called Biscuit")])
    assert r.search("Biscuit cat", k=1)[0].chunk.source == "/d/b.md"  # a write makes the next search reopen
    r.remove("/d/b.md")
    assert all(h.chunk.source != "/d/b.md" for h in r.search("Biscuit cat", k=2))
