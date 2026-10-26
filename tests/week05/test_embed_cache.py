import numpy as np

from raggy.config import Config
from raggy.embed.cache import CachedEmbedder, EmbeddingCache
from raggy.embed.fake import FakeEmbedder
from raggy.evals.runner import prepare_index
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy

from week04.test_pipeline_chunking import tiny_suite, write


class Counting(FakeEmbedder):
    def __init__(self, name=None):
        super().__init__()
        if name:
            self.name = name
        self.seen = []

    def embed_documents(self, texts):
        self.seen.append(list(texts))
        return super().embed_documents(texts)


def test_cached_embedder_order_and_duplicates(tmp_path):
    inner = Counting()
    emb = CachedEmbedder(inner, EmbeddingCache(tmp_path / "c.sqlite"))
    first = emb.embed_documents(["a b", "c d", "a b"])
    assert inner.seen == [["a b", "c d"]]  # each distinct text embedded once
    assert np.allclose(first[0], first[2]) and np.allclose(first, FakeEmbedder().embed_documents(["a b", "c d", "a b"]))
    second = emb.embed_documents(["c d", "e f", "a b"])
    assert inner.seen[-1] == ["e f"]
    assert np.allclose(second, FakeEmbedder().embed_documents(["c d", "e f", "a b"]))
    assert (emb.hits, emb.misses) == (2 + 1, 2 + 1)  # 1st call: 1 dup hit; 2nd: 2 hits


def test_cache_survives_reopen(tmp_path):
    emb = CachedEmbedder(Counting(), EmbeddingCache(tmp_path / "c.sqlite"))
    emb.embed_documents(["hello world"])
    emb.cache.commit()
    inner = Counting()
    CachedEmbedder(inner, EmbeddingCache(tmp_path / "c.sqlite")).embed_documents(["hello world"])
    assert inner.seen == []


def test_cache_key_includes_model_name(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    CachedEmbedder(Counting("model-a"), cache).embed_documents(["same text"])
    other = Counting("model-b")
    CachedEmbedder(other, cache).embed_documents(["same text"])
    assert other.seen == [["same text"]]


def test_delegates_protocol(tmp_path):
    emb = CachedEmbedder(FakeEmbedder(), EmbeddingCache(tmp_path / "c.sqlite"))
    assert emb.name == "fake-bow-64" and emb.dim == 64 and emb.max_tokens == 510
    assert emb.count_tokens("a b") == 2 and emb.embed_query("a").shape == (64,)


def make(tmp_path, inner, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               store="numpy", **cfg)
    return Raggy(c, embedder=inner, llm=FakeLLM())


DOC = "# Help\n\n## Refunds\n\nRefunds are paid within 14 days.\n\n## Bikes\n\n" + " ".join(f"Bike {i} is red." for i in range(60))


def test_second_ingest_after_settings_change_embeds_only_new_text(tmp_path):
    write(tmp_path / "d" / "h.md", DOC)
    inner = Counting()
    r = make(tmp_path, inner).ingest([str(tmp_path / "d")])
    assert r.embedded > 0 and r.embed_cached == 0
    r = make(tmp_path, inner, parent_tokens=1024).ingest([str(tmp_path / "d")])  # same children
    assert r.updated == 1 and r.embedded == 0 and r.embed_cached > 0
    r = make(tmp_path, inner, chunk_headers=False).ingest([str(tmp_path / "d")])  # embedded text changes
    assert r.embedded > 0


def test_cache_off(tmp_path):
    write(tmp_path / "d" / "h.md", DOC)
    make(tmp_path, Counting(), embed_cache=False).ingest([str(tmp_path / "d")])
    assert not (tmp_path / "index" / "embed-cache.sqlite").exists()


def test_eval_rebuild_keeps_embed_cache(tmp_path):
    inner = Counting()
    prepare_index(tiny_suite(), make(tmp_path, inner))
    n = len(inner.seen)
    prepare_index(tiny_suite(), make(tmp_path, inner, parent_tokens=1024))  # rebuilt, same texts
    assert (tmp_path / "index" / "embed-cache.sqlite").exists() and len(inner.seen) == n


def test_fastembed_order_restored():
    from raggy.embed.fastembed_embedder import FastEmbedEmbedder

    class FakeModel:
        def passage_embed(self, texts, batch_size):
            self.order = list(texts)
            return (np.array([len(t), 1.0], dtype=np.float32) for t in texts)
    e = FastEmbedEmbedder.__new__(FastEmbedEmbedder)
    e._model, e.dim, e.batch_size = FakeModel(), 2, 16
    texts = ["ccc", "a", "bb"]
    out = e.embed_documents(texts)
    assert e._model.order == ["a", "bb", "ccc"]  # embedded shortest first
    assert [round(float(v[0] / v[1]), 3) for v in out] == [3.0, 1.0, 2.0]  # returned in input order
