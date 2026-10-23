import numpy as np
import pytest

from raggy.embed.fake import FakeEmbedder
from raggy.store.numpy_store import EmbedderMismatchError, IndexNotFoundError, NumpyStore
from raggy.types import Chunk

E = FakeEmbedder()


def chunks_for(source: str, texts: list[str]) -> list[Chunk]:
    return [Chunk(id=f"{source}#{i}", source=source, text=t, index=i, start=0, end=len(t))
            for i, t in enumerate(texts)]


def add(store: NumpyStore, source: str, texts: list[str]) -> None:
    cs = chunks_for(source, texts)
    store.add(cs, E.embed_documents([c.text for c in cs]), embedder_name=E.name)


def test_store_search_returns_best_first(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["apples are red", "bananas are yellow", "grapes are purple"])
    hits = s.search(E.embed_query("bananas are yellow"), k=3, embedder_name=E.name)
    assert hits[0].chunk.text == "bananas are yellow"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_store_k_larger_than_size(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["one", "two"])
    assert len(s.search(E.embed_query("one"), k=10, embedder_name=E.name)) == 2


def test_store_search_empty_returns_nothing(tmp_path):
    assert NumpyStore(tmp_path).search(E.embed_query("x"), k=3, embedder_name=E.name) == []


def test_store_roundtrip_save_load(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["alpha text", "beta text"])
    s.save()
    t = NumpyStore.load(tmp_path)
    assert len(t) == 2
    assert t.sources() == {"a.md": 2}
    hits = t.search(E.embed_query("beta text"), k=1, embedder_name=E.name)
    assert hits[0].chunk.id == "a.md#1"


def test_reingest_replaces_source(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["old one", "old two", "old three"])
    add(s, "b.md", ["other"])
    add(s, "a.md", ["new one"])
    assert s.sources() == {"a.md": 1, "b.md": 1}
    assert len(s) == 2


def test_remove_source(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["x", "y"])
    add(s, "b.md", ["z"])
    assert s.remove_source("a.md") == 2
    assert s.sources() == {"b.md": 1}


def test_store_rejects_embedder_mismatch(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["text"])
    with pytest.raises(EmbedderMismatchError, match="fake-bow-64.*other-model|other-model.*fake-bow-64"):
        s.search(np.zeros(E.dim, dtype=np.float32), k=1, embedder_name="other-model")
    with pytest.raises(EmbedderMismatchError):
        s.add(chunks_for("b.md", ["t"]), E.embed_documents(["t"]), embedder_name="other-model")


def test_load_missing_raises_index_not_found(tmp_path):
    with pytest.raises(IndexNotFoundError):
        NumpyStore.load(tmp_path / "nothing-here")


def test_store_save_empty_then_add(tmp_path):
    NumpyStore(tmp_path).save()
    s = NumpyStore.load(tmp_path)
    add(s, "a.md", ["hello there"])
    assert len(s) == 1


def test_store_non_positive_k_returns_nothing(tmp_path):
    s = NumpyStore(tmp_path)
    add(s, "a.md", ["one", "two", "three"])
    assert s.search(E.embed_query("one"), k=0, embedder_name=E.name) == []
    assert s.search(E.embed_query("one"), k=-1, embedder_name=E.name) == []
