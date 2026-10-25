from raggy import cli
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy

from .conftest import write


def use_fakes(monkeypatch, tmp_path):
    def build(config):
        cfg = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None)
        return Raggy(cfg, embedder=FakeEmbedder(), llm=FakeLLM())
    monkeypatch.setattr(cli, "_build_raggy", build)


def test_cli_ingest_summary_line(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path)
    write(tmp_path / "d" / "a.md", "# A\n\nalpha text")
    write(tmp_path / "d" / "bad.pdf", b"not a pdf")
    assert cli.main(["ingest", str(tmp_path / "d"), str(tmp_path / "nope")]) == 0
    out, err = capsys.readouterr()
    assert "1 added, 0 updated, 0 unchanged, 0 removed, 0 duplicates, 1 failed (1 chunks embedded)" in out
    assert "failed: " in err and "bad.pdf" in err and f"not found: {tmp_path / 'nope'}" in err


def test_cli_sources_shows_titles(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path)
    write(tmp_path / "d" / "a.md", "# Alpha title\n\nalpha text")
    write(tmp_path / "d" / "z" / "copy.md", "# Alpha title\n\nalpha text")
    cli.main(["ingest", str(tmp_path / "d")])
    capsys.readouterr()
    assert cli.main(["sources"]) == 0
    out = capsys.readouterr().out
    assert "markdown" in out and "Alpha title" in out and "dup of a.md" in out


def test_cli_remove(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path)
    write(tmp_path / "d" / "a.md", "alpha")
    cli.main(["ingest", str(tmp_path / "d")])
    assert cli.main(["remove", str(tmp_path / "d" / "a.md")]) == 0
    assert "Removed 1 source" in capsys.readouterr().out
    assert (tmp_path / "d" / "a.md").exists()
