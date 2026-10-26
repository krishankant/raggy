import os
import time

import pytest

from raggy import cli
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.store.base import Filters

from week04.test_pipeline_chunking import write
from week03.conftest import PAGE


def corpus(tmp_path):
    d = tmp_path / "d"
    write(d / "notes" / "refunds.md", "# Refunds\n\nRefunds about bikes are paid within 14 days.")
    write(d / "notes2" / "bikes.md", "# Bikes\n\nBikes about refunds: the Heron is a city bike.")
    write(d / "shop.html", PAGE)
    os.utime(d / "notes" / "refunds.md", (time.time(), 1_800_000_000))  # 2027
    os.utime(d / "notes2" / "bikes.md", (time.time(), 1_600_000_000))  # 2020
    return d


def make(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None, **cfg)
    return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("ok [1]."))


@pytest.mark.parametrize("store", ["lance", "numpy"])
def test_pipeline_filters(tmp_path, store):
    d = corpus(tmp_path)
    r = make(tmp_path, store=store)
    r.ingest([str(d)])
    names = lambda f: {h.chunk.source.rsplit("/", 1)[1] for h in r.search("refunds bikes parts", k=5, filters=f)}
    assert names(Filters(doc_type="html")) == {"shop.html"}
    assert names(Filters(source_prefix=str((d / "notes").resolve()) + os.sep)) == {"refunds.md"}
    assert names(Filters(modified_after="2026-01-01")) >= {"refunds.md"} and "bikes.md" not in names(Filters(modified_after="2026-01-01"))


def test_parent_expansion_after_filtering(tmp_path):
    d = corpus(tmp_path)
    r = make(tmp_path, parent_tokens=512, chunk_tokens=32, chunk_overlap_tokens=0)
    r.ingest([str(d)])
    hits = r.search("refunds", k=3, filters=Filters(doc_type="markdown"))
    assert hits and all(h.chunk.id.split("#")[1].startswith("p") for h in hits)
    assert all(h.chunk.meta.get("type") == "markdown" for h in hits)


def use_fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_raggy", lambda config: make(tmp_path))


def test_cli_search_type_flag(monkeypatch, tmp_path, capsys):
    d = corpus(tmp_path)
    use_fakes(monkeypatch, tmp_path)
    cli.main(["ingest", str(d)])
    capsys.readouterr()
    assert cli.main(["search", "refunds", "--type", "html", "-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "shop.html" in out and ".md" not in out


def test_cli_source_folder_does_not_match_sibling_prefix(monkeypatch, tmp_path, capsys):
    d = corpus(tmp_path)
    use_fakes(monkeypatch, tmp_path)
    cli.main(["ingest", str(d)])
    capsys.readouterr()
    assert cli.main(["search", "refunds bikes", "--source", str(d / "notes"), "-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "refunds.md" in out and "bikes.md" not in out


def test_cli_after_flag_and_ask(monkeypatch, tmp_path, capsys):
    d = corpus(tmp_path)
    use_fakes(monkeypatch, tmp_path)
    cli.main(["ingest", str(d)])
    capsys.readouterr()
    assert cli.main(["ask", "refunds?", "--after", "2026-01-01", "--show-context"]) == 0
    out = capsys.readouterr().out
    assert "refunds.md" in out and "bikes.md" not in out


def test_cli_rejects_bad_date(monkeypatch, tmp_path):
    use_fakes(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["search", "x", "--after", "last tuesday"])
    assert e.value.code == 2


def test_cli_filter_matching_nothing_says_so(monkeypatch, tmp_path, capsys):
    d = corpus(tmp_path)
    use_fakes(monkeypatch, tmp_path)
    cli.main(["ingest", str(d)])
    capsys.readouterr()
    assert cli.main(["search", "refunds", "--type", "pdf"]) == 0
    assert "no chunks match" in capsys.readouterr().err
