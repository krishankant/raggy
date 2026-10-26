import json

import numpy as np

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.generate.pack import pack
from raggy.generate.prompt import build_user_prompt, parse_citations
from raggy.chunking.tokens import simple_count
from raggy.pipeline import Raggy, page_label
from raggy.store.lance_store import LanceStore
from raggy.types import Chunk, Document, Hit

DOC = Document("/d/fleet.pdf", "Page one text about bikes.\n\nPage two text about prices: 4,700 credits.",
               {"type": "pdf", "pages": [("p. 1", 0), ("p. 2", 28)]})


def make(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "chunk_size": 30, "chunk_overlap": 5, "parent_tokens": 0} | cfg))
    return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("4,700 credits [1]."))


def test_page_label():
    pages = [("p. 1", 0), ("p. 2", 28)]
    assert page_label(pages, 0, 10) == ("p. 1", None)
    assert page_label(pages, 20, 40) == ("p. 1", "p. 2")
    assert page_label(pages, 30, 40) == ("p. 2", None)
    assert page_label([], 0, 5) == (None, None)


def test_chunk_gets_page_label(tmp_path):
    r = make(tmp_path)
    r.ingest_documents([DOC])
    hits = r.search("prices credits", k=1)
    assert hits[0].chunk.meta["page"] == "p. 2"


def test_citation_shows_location(tmp_path):
    r = make(tmp_path)
    r.ingest_documents([DOC])
    a = r.ask("price?", k=1)
    assert a.citations[0].location in ("p. 2", "p. 1–2")  # 30-char chunks: the best one may straddle pages


def test_prompt_shows_location():
    h = Hit(Chunk("/d/f.pdf#1", "/d/f.pdf", "4,700 credits", 1, 28, 41, meta={"page": "p. 2"}), 0.9)
    assert "[1] (source: f.pdf, p. 2)" in build_user_prompt("q", [h])
    assert parse_citations("x [1]", [h])[0].location == "p. 2"


def test_passage_spanning_pages_label():
    a = Hit(Chunk("/d/f.pdf#0", "/d/f.pdf", "Page one text", 0, 0, 13, meta={"page": "p. 1"}), 0.9)
    b = Hit(Chunk("/d/f.pdf#1", "/d/f.pdf", "text about", 1, 10, 40, meta={"page": "p. 1", "page_end": "p. 2"}), 0.8)
    (m,) = pack([a, b], simple_count, 1000)
    assert m.chunk.meta["page"] == "p. 1" and m.chunk.meta["page_end"] == "p. 2"
    assert "(source: f.pdf, p. 1–2)" in build_user_prompt("q", [m])


def test_old_table_gains_page_column(tmp_path):
    import lancedb
    import pyarrow as pa
    emb = FakeEmbedder()
    db = lancedb.connect(str(tmp_path / "l"))
    old = {"id": ["a#0"], "source": ["a"], "text": ["old row"], "idx": [0], "start": [0], "end": [7], "context": [""],
           "parent_start": pa.array([None], pa.int64()), "parent_end": pa.array([None], pa.int64()),
           "parent_text": pa.array([None], pa.string()), "doc_type": ["markdown"], "title": ["A"],
           "modified": pa.array([None], pa.string()), "fts_text": ["old row"],
           "vector": pa.FixedSizeListArray.from_arrays(pa.array(emb.embed_documents(["old row"])[0]), 64)}
    db.create_table("chunks", pa.table(old))
    (tmp_path / "l" / "meta.json").write_text(json.dumps({"embedder": emb.name, "version": 1}))
    s = LanceStore(tmp_path / "l")
    c = Chunk("b#0", "b", "new row", 0, 0, 7, meta={"type": "pdf", "page": "p. 3"})
    s.add([c], emb.embed_documents(["new row"]), emb.name)
    assert s.sources() == {"a": 1, "b": 1}  # the old row survived
    hit = s.search(emb.embed_query("new row"), 1, emb.name)[0]
    assert hit.chunk.meta["page"] == "p. 3"


def test_numpy_store_keeps_page_in_meta(tmp_path):
    r = make(tmp_path, store="numpy")
    r.ingest_documents([DOC])
    assert r.search("prices credits", k=1)[0].chunk.meta["page"] == "p. 2"
