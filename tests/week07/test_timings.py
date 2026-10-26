import pytest

from raggy import cli
from raggy.config import Config, ConfigError
from raggy.embed.fake import FakeEmbedder
from raggy.evals.report import format_run
from raggy.evals.runner import run_suite
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.types import Document

from week04.test_pipeline_chunking import tiny_suite
from week07.test_rerank import FakeCrossEncoder, cross
from week07.test_transforms import scripted

DOCS = [Document("/d/a.md", "Refunds are paid within 14 days.", {"type": "markdown"}),
        Document("/d/b.md", "The office cat is called Biscuit.", {"type": "markdown"})]


def make(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0} | cfg))
    return Raggy(c, embedder=FakeEmbedder(), llm=scripted())


def test_timings_recorded_per_stage(tmp_path):
    r = make(tmp_path, rerank="local", query_transform="rewrite")
    r._reranker = cross()
    r.ingest_documents(DOCS)
    r.search("refunds?", k=1)
    assert set(r.last_timings) == {"transform", "retrieve", "rerank"}
    assert all(v >= 0 for v in r.last_timings.values())
    plain = make(tmp_path / "p")
    plain.ingest_documents(DOCS)
    plain.search("refunds?", k=1)
    assert set(plain.last_timings) == {"retrieve"}


def test_budgets_parse_env(monkeypatch):
    monkeypatch.setattr("dotenv.find_dotenv", lambda **kw: "")
    monkeypatch.setenv("RAGGY_BUDGETS", "rerank=300, transform=1500")
    b = Config.from_env().budgets
    assert b["rerank"] == 300 and b["transform"] == 1500 and b["retrieve"] == 100 and b["generate"] == 5000
    monkeypatch.setenv("RAGGY_BUDGETS", "rerank=fast")
    with pytest.raises(ConfigError, match="RAGGY_BUDGETS"):
        Config.from_env()
    monkeypatch.setenv("RAGGY_BUDGETS", "warp=10")
    with pytest.raises(ConfigError, match="warp"):
        Config.from_env()


def test_report_flags_over_budget(tmp_path):
    r = make(tmp_path, budgets={"transform": 2000, "retrieve": 0, "rerank": 500, "generate": 5000})
    run = run_suite(tiny_suite(), r, None, answers=False)
    assert run["metrics"]["over_budget"] == ["retrieve"]
    assert run["config"]["budgets"]["retrieve"] == 0
    out = format_run(run)
    assert "Stage" in out and "retrieve" in out and "✗" in out


def test_ask_timings_flag(monkeypatch, tmp_path, capsys):
    r = make(tmp_path)
    r.ingest_documents(DOCS)
    monkeypatch.setattr(cli, "_build_raggy", lambda config: r)
    assert cli.main(["ask", "refunds?", "--timings"]) == 0
    out = capsys.readouterr().out
    assert "retrieve" in out and "generate" in out and " ms" in out


def test_compare_skips_list_metrics_and_names_over_budget(tmp_path):
    from raggy.evals.report import format_compare
    r = make(tmp_path, budgets={"transform": 2000, "retrieve": 0, "rerank": 500, "generate": 5000})
    slow = run_suite(tiny_suite(), r, None, answers=False)
    fast = run_suite(tiny_suite(), make(tmp_path / "f"), None, answers=False)
    out = format_compare(fast, slow)
    assert "new: over budget at p95: retrieve" in out and "over_budget " not in out
