from pathlib import Path

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.store.numpy_store import NumpyStore
from raggy.types import Document


def make(tmp_path: Path, chunk_size=800, overlap=100) -> Raggy:
    cfg = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", chunker="fixed",
                 llm_model=None, chunk_size=chunk_size, chunk_overlap=overlap)
    return Raggy(cfg, embedder=FakeEmbedder(), llm=FakeLLM())


def test_ingest_documents_counts(tmp_path):
    r = make(tmp_path)
    report = r.ingest_documents([Document("a", "refunds take fourteen days"), Document("b", "the cat sleeps")])
    assert (report.documents, report.chunks) == (2, 2)


def test_ingest_documents_keeps_docs_whole_across_batches(tmp_path):
    r = make(tmp_path, chunk_size=10, overlap=0)
    long_doc = Document("a", "x" * 50)  # five 10-char chunks
    r.ingest_documents([long_doc, Document("b", "short")], batch_size=1)
    assert NumpyStore.load(tmp_path / "index").sources() == {"a": 5, "b": 1}


def test_reingest_documents_replaces(tmp_path):
    r = make(tmp_path)
    r.ingest_documents([Document("a", "first version")])
    r.ingest_documents([Document("a", "second version")])
    store = NumpyStore.load(tmp_path / "index")
    assert [c.text for c in store.chunks] == ["second version"]


def test_ingest_paths_still_works(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("Refunds are paid back within 14 days.")
    report = make(tmp_path).ingest([str(docs)])
    assert (report.documents, report.chunks, report.skipped) == (1, 1, [])


def test_ingest_documents_reports_progress(tmp_path):
    seen = []
    r = make(tmp_path, chunk_size=10, overlap=0)
    r.ingest_documents([Document("a", "x" * 30), Document("b", "y" * 20)], batch_size=3,
                       progress=lambda done, total: seen.append((done, total)))
    assert seen == [(3, 5), (5, 5)]  # chunks embedded so far, out of all chunks
