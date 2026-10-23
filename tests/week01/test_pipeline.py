from pathlib import Path

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy


def make(tmp_path: Path, reply="Refunds take 14 days [1].") -> Raggy:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text("Refunds are paid back within 14 days of the return arriving.")
    (docs / "cats.md").write_text("The office cat is called Biscuit and sleeps on the printer.")
    cfg = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None, top_k=1)
    return Raggy(cfg, embedder=FakeEmbedder(), llm=FakeLLM(reply))


def test_pipeline_ingest_report_counts(tmp_path):
    r = make(tmp_path)
    report = r.ingest([str(tmp_path / "docs")])
    assert report.documents == 2 and report.chunks == 2 and report.skipped == []


def test_pipeline_ingest_then_search_finds_relevant(tmp_path):
    r = make(tmp_path)
    r.ingest([str(tmp_path / "docs")])
    hits = r.search("what is the office cat called")
    assert Path(hits[0].chunk.source).name == "cats.md"


def test_pipeline_index_persists_between_instances(tmp_path):
    make(tmp_path).ingest([str(tmp_path / "docs")])
    cfg = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None)
    again = Raggy(cfg, embedder=FakeEmbedder(), llm=FakeLLM())
    assert len(again.sources()) == 2


def test_pipeline_ask_returns_citations(tmp_path):
    r = make(tmp_path)
    r.ingest([str(tmp_path / "docs")])
    ans = r.ask("when are refunds paid back")
    assert ans.citations and Path(ans.citations[0].source).name == "refunds.md"


def test_pipeline_ingest_missing_path_is_skipped(tmp_path):
    r = make(tmp_path)
    report = r.ingest([str(tmp_path / "docs"), str(tmp_path / "missing")])
    assert report.documents == 2 and report.skipped == [str(tmp_path / "missing")]
