import pytest

from raggy import cli
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "refunds.md").write_text("Refunds are paid back within 14 days.")
    return d


def use_fakes(monkeypatch, tmp_path, llm):
    def build(config):
        cfg = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None)
        return Raggy(cfg, embedder=FakeEmbedder(), llm=llm)
    monkeypatch.setattr(cli, "_build_raggy", build)


def test_ask_without_index_errors_cleanly(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, FakeLLM())
    assert cli.main(["ask", "anything"]) == 1
    err = capsys.readouterr().err
    assert "raggy ingest" in err and "Traceback" not in err


def test_cli_ingest_then_ask_prints_sources(monkeypatch, tmp_path, docs, capsys):
    use_fakes(monkeypatch, tmp_path, FakeLLM("14 days [1]."))
    assert cli.main(["ingest", str(docs)]) == 0
    assert cli.main(["ask", "refund time?"]) == 0
    out = capsys.readouterr().out
    assert "14 days [1]." in out and "Sources:" in out and "[1] refunds.md" in out


def test_cli_reports_llm_error(monkeypatch, tmp_path, docs, capsys):
    use_fakes(monkeypatch, tmp_path, FakeLLM(error=LLMError("ollama request failed: is Ollama running?")))
    cli.main(["ingest", str(docs)])
    assert cli.main(["ask", "refund time?"]) == 1
    err = capsys.readouterr().err
    assert "is Ollama running?" in err and "Traceback" not in err


def test_cli_sources_lists_files(monkeypatch, tmp_path, docs, capsys):
    use_fakes(monkeypatch, tmp_path, FakeLLM())
    cli.main(["ingest", str(docs)])
    assert cli.main(["sources"]) == 0
    assert "refunds.md" in capsys.readouterr().out


def test_cli_search_shows_scores(monkeypatch, tmp_path, docs, capsys):
    use_fakes(monkeypatch, tmp_path, FakeLLM())
    cli.main(["ingest", str(docs)])
    assert cli.main(["search", "refunds"]) == 0
    assert "refunds.md" in capsys.readouterr().out


@pytest.mark.parametrize("k", ["0", "-1", "two"])
def test_cli_rejects_bad_k(monkeypatch, tmp_path, k):
    use_fakes(monkeypatch, tmp_path, FakeLLM())
    with pytest.raises(SystemExit) as exc:
        cli.main(["search", "refunds", "-k", k])
    assert exc.value.code == 2
