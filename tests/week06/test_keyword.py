import json

import numpy as np
import pytest

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.store.base import Filters, KeywordSearchUnavailable
from raggy.store.lance_store import LanceStore
from raggy.store.numpy_store import NumpyStore
from raggy.types import Chunk

emb = FakeEmbedder()

TEXTS = {
    "/d/parts.md": ("markdown", "| BW-CH-1142 | 8-speed chain, 116 links | Heron Commuter | 24 credits |"),
    "/d/refunds.md": ("markdown", "Refunds are paid back within 14 days of the return arriving."),
    "/d/cat.pdf": ("pdf", "The office cat is called Biscuit and sleeps on the printer."),
    "/d/policy.pdf": ("pdf", "Our refund policy for PDFs: refunds are paid in credits."),
}


def store(tmp_path, finish=True):
    s = LanceStore(tmp_path / "l")
    for src, (kind, text) in TEXTS.items():
        c = [Chunk(f"{src}#0", src, text, 0, 0, len(text), context=src.rsplit("/", 1)[1], meta={"type": kind})]
        s.add(c, emb.embed_documents([text]), emb.name)
    if finish:
        s.finish(build_ann=False)
    return s


def test_keyword_finds_exact_code(tmp_path):
    hits = store(tmp_path).keyword_search("What does BW-CH-1142 fit?", 3)
    assert hits[0].chunk.source == "/d/parts.md" and hits[0].score > 0


def test_keyword_stems(tmp_path):
    assert {h.chunk.source for h in store(tmp_path).keyword_search("refund", 5)} == {"/d/refunds.md", "/d/policy.pdf"}


@pytest.mark.parametrize("q", ["what is it?", "???", "", "   "])
def test_keyword_empty_query(tmp_path, q):
    assert store(tmp_path).keyword_search(q, 5) == []


def test_keyword_respects_filters(tmp_path):
    hits = store(tmp_path).keyword_search("refunds paid", 5, where=Filters(doc_type="pdf"))
    assert [h.chunk.source for h in hits] == ["/d/policy.pdf"]


def test_keyword_finds_rows_added_after_index(tmp_path):
    s = store(tmp_path)
    text = "Narwhal tusks are teeth."
    s.add([Chunk("/d/new.md#0", "/d/new.md", text, 0, 0, len(text))], emb.embed_documents([text]), emb.name)
    assert s.keyword_search("narwhal", 3)[0].chunk.source == "/d/new.md"


def test_keyword_searches_the_context_header_too(tmp_path):
    hits = store(tmp_path).keyword_search("parts", 3)  # "parts.md" is only in the context
    assert hits and hits[0].chunk.source == "/d/parts.md"


def test_keyword_on_numpy_store_raises(tmp_path):
    with pytest.raises(KeywordSearchUnavailable, match="RAGGY_STORE=lance"):
        NumpyStore(tmp_path / "n").keyword_search("x", 3)


def test_old_lance_index_without_fts_text(tmp_path):
    import lancedb
    import pyarrow as pa
    db = lancedb.connect(str(tmp_path / "l"))
    db.create_table("chunks", pa.table({"id": ["a#0"], "source": ["a"], "text": ["hello"],
                                        "vector": pa.FixedSizeListArray.from_arrays(pa.array(np.ones(64, np.float32)), 64)}))
    (tmp_path / "l" / "meta.json").write_text(json.dumps({"embedder": emb.name, "version": 1}))
    with pytest.raises(KeywordSearchUnavailable, match="raggy ingest"):
        LanceStore(tmp_path / "l").keyword_search("hello", 3)


def test_ingest_builds_fts_index(tmp_path):
    from week04.test_pipeline_chunking import write
    write(tmp_path / "d" / "a.md", "# Parts\n\nThe BW-CH-1142 chain fits the Heron.")
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None)
    Raggy(c, embedder=emb, llm=FakeLLM()).ingest([str(tmp_path / "d")])
    s = LanceStore(tmp_path / "index" / "lance")
    assert s.has_fts_index() and s.keyword_search("BW-CH-1142", 1)
