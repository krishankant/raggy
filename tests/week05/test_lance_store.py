import numpy as np
import pytest

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.store.base import EmbedderMismatchError, Filters, IndexNotFoundError, make_store
from raggy.store.lance_store import LanceStore
from raggy.store.numpy_store import NumpyStore
from raggy.types import Chunk

from week04.test_pipeline_chunking import write

emb = FakeEmbedder()


def chunks(source, texts, meta=None):
    return [Chunk(f"{source}#{i}", source, t, i, i * 10, i * 10 + len(t), context="Doc › Sec",
                  parent_start=0, parent_end=99, parent_text="parent " + t, meta=dict(meta or {}))
            for i, t in enumerate(texts)]


def fill(store):
    a = chunks("/d/refunds.md", ["refunds are paid within 14 days", "returns need the box"],
               {"type": "markdown", "title": "Refunds", "modified": "2026-03-01T00:00:00Z"})
    b = chunks("/d/cat.pdf", ["the office cat is called Biscuit"],
               {"type": "pdf", "title": "Cat", "modified": "2025-01-01T00:00:00Z"})
    for cs in (a, b):
        store.add(cs, emb.embed_documents([c.text for c in cs]), embedder_name=emb.name)
    store.save()
    return a + b


def test_lance_add_search_roundtrip(tmp_path):
    lance, num = LanceStore(tmp_path / "l"), NumpyStore(tmp_path / "n")
    fill(lance), fill(num)
    q = emb.embed_query("when are refunds paid")
    lh, nh = lance.search(q, 3, emb.name), num.search(q, 3, emb.name)
    assert [h.chunk.id for h in lh] == [h.chunk.id for h in nh]
    assert np.allclose([h.score for h in lh], [h.score for h in nh], atol=1e-5)
    c = lh[0].chunk
    assert (c.text, c.context, c.parent_text, c.start) == ("refunds are paid within 14 days", "Doc › Sec",
                                                           "parent refunds are paid within 14 days", 0)
    assert c.meta == {"type": "markdown", "title": "Refunds", "modified": "2026-03-01T00:00:00Z"}
    assert len(lance) == 3 and lance.sources() == {"/d/refunds.md": 2, "/d/cat.pdf": 1}


def test_lance_reingest_replaces_rows(tmp_path):
    s = LanceStore(tmp_path / "l")
    fill(s)
    new = chunks("/d/refunds.md", ["refunds now take 30 days"])
    s.add(new, emb.embed_documents([c.text for c in new]), emb.name)
    again = LanceStore(tmp_path / "l")  # a second process sees the same rows
    assert again.sources() == {"/d/refunds.md": 1, "/d/cat.pdf": 1}


def test_lance_remove_source(tmp_path):
    s = LanceStore(tmp_path / "l")
    fill(s)
    assert s.remove_source("/d/cat.pdf") == 1 and s.remove_source("/d/nope") == 0
    assert s.sources() == {"/d/refunds.md": 2}


def test_lance_persists(tmp_path):
    fill(LanceStore(tmp_path / "l"))
    s = LanceStore(tmp_path / "l")
    assert len(s) == 3 and s.embedder == emb.name


def test_lance_embedder_mismatch(tmp_path):
    s = LanceStore(tmp_path / "l")
    fill(s)
    with pytest.raises(EmbedderMismatchError):
        s.search(emb.embed_query("x"), 3, "another-model")
    with pytest.raises(EmbedderMismatchError):
        s.add(chunks("/d/x.md", ["x"]), emb.embed_documents(["x"]), "another-model")


def test_missing_lance_index_raises(tmp_path):
    cfg = Config(index_dir=tmp_path / "nothing", embed_model="fake", llm_provider="fake", llm_model=None, store="lance")
    with pytest.raises(IndexNotFoundError):
        make_store(cfg, create=False)


def test_filter_values_are_escaped(tmp_path):
    s = LanceStore(tmp_path / "l")
    cs = chunks("/d/it's here.md", ["quoted path text"], {"type": "markdown"})
    s.add(cs, emb.embed_documents([c.text for c in cs]), emb.name)
    fill(s)
    q = emb.embed_query("text")
    hits = s.search(q, 5, emb.name, where=Filters(source_prefix="/d/it's"))
    assert [h.chunk.source for h in hits] == ["/d/it's here.md"]
    hits = s.search(q, 5, emb.name, where=Filters(doc_type="pdf' OR '1'='1"))
    assert hits == []
    assert s.remove_source("/d/it's here.md") == 1


@pytest.mark.parametrize("store", ["numpy", "lance"])
def test_filters_same_on_both_stores(tmp_path, store):
    s = make_store(Config(index_dir=tmp_path / store, embed_model="fake", llm_provider="fake",
                          llm_model=None, store=store))
    fill(s)
    q = emb.embed_query("refunds cat box")
    ids = lambda f: sorted(h.chunk.id for h in s.search(q, 5, emb.name, where=f))
    assert ids(Filters(doc_type="pdf")) == ["/d/cat.pdf#0"]
    assert ids(Filters(source_prefix="/d/ref")) == ["/d/refunds.md#0", "/d/refunds.md#1"]
    assert ids(Filters(modified_after="2026-01-01")) == ["/d/refunds.md#0", "/d/refunds.md#1"]
    assert ids(Filters(doc_type="pdf", modified_after="2026-01-01")) == []


def test_store_change_reindexes(tmp_path):
    write(tmp_path / "d" / "a.md", "# A\n\nalpha text here")

    def make(store):
        c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None, store=store)
        return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM())
    make("numpy").ingest([str(tmp_path / "d")])
    r = make("lance").ingest([str(tmp_path / "d")])
    assert r.updated == 1
    assert make("lance").search("alpha", k=1)[0].chunk.source.endswith("a.md")
    assert (tmp_path / "index" / "vectors.npy").exists()  # the old store's files are left alone


def test_pipeline_end_to_end_on_lance(tmp_path):
    write(tmp_path / "d" / "a.md", "# Refunds\n\nRefunds are paid within 14 days.")
    write(tmp_path / "d" / "b.md", "# Cat\n\nThe office cat is called Biscuit.")
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None, store="lance")
    r = Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("14 days [1]."))
    assert r.ingest([str(tmp_path / "d")]).added == 2
    assert r.ask("when are refunds paid").citations[0].source.endswith("a.md")
    assert set(map(lambda s: s.rsplit("/", 1)[1], r.sources())) == {"a.md", "b.md"}
    assert r.remove(str(tmp_path / "d" / "b.md")) == 1
    (tmp_path / "d" / "a.md").write_text("# Refunds\n\nRefunds now take 30 days.")
    assert r.ingest([str(tmp_path / "d")]).updated == 1
    assert "30 days" in r.search("refunds", k=1)[0].chunk.text


def test_unchanged_lance_run_does_not_load_embedder(tmp_path):
    write(tmp_path / "d" / "a.md", "# A\n\nalpha text")
    c = Config(index_dir=tmp_path / "index", embed_model="fake-bow-64", llm_provider="fake", llm_model=None, store="lance")
    Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM()).ingest([str(tmp_path / "d")])
    r = Raggy(c)
    assert r.ingest([str(tmp_path / "d")]).unchanged == 1 and r._embedder is None
