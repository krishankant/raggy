import dataclasses
import importlib.util
from pathlib import Path

import pytest

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.dataset import resolve_gold
from raggy.evals.runner import prepare_index, run_suite
from raggy.evals.suites import SUITES, fictional_web_suite
from raggy.generate.fake import FakeLLM
from raggy.ingest.loaders import LoadOptions
from raggy.pipeline import Raggy

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "evals" / "fictional-web"


def generator():
    spec = importlib.util.spec_from_file_location("make_pages", WEB / "make_pages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raggy_for(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None, **cfg)
    return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("x"))


def test_pages_match_generator():
    pages = generator().render_all()
    on_disk = {p.relative_to(WEB / "pages").as_posix(): p.read_text(encoding="utf-8")
               for p in (WEB / "pages").rglob("*.html")}
    assert on_disk == pages, "pages/ is stale: run `uv run python evals/fictional-web/make_pages.py`"
    assert len(pages) == 14


@pytest.mark.parametrize("mode", ["main", "full"])
def test_web_gold_resolves_in_both_modes(mode):
    suite = fictional_web_suite(WEB)
    docs = suite.documents(LoadOptions(html_extract=mode))
    gold = resolve_gold(suite.questions(), docs)  # raises on any quote extraction lost
    assert len(gold) == 41 and all(s.source.endswith(".html") for spans in gold.values() for s in spans)


def test_web_suite_registered():
    assert SUITES["fictional-web"]().name == "fictional-web"


def test_cache_key_changes_with_extract_mode(tmp_path):
    suite = fictional_web_suite(WEB)
    r = raggy_for(tmp_path)
    assert prepare_index(suite, r, suite.documents(r.config.load_options()))["cached"] is False
    assert prepare_index(suite, r, suite.documents(r.config.load_options()))["cached"] is True
    full = raggy_for(tmp_path, html_extract="full")
    assert prepare_index(suite, full, suite.documents(full.config.load_options()))["cached"] is False
    off = raggy_for(tmp_path, html_extract="full", dedupe=False)
    assert prepare_index(suite, off, suite.documents(off.config.load_options()))["cached"] is False


def test_eval_dedupe_removes_duplicate_pages(tmp_path):
    run = run_suite(fictional_web_suite(WEB), raggy_for(tmp_path), None, answers=False)
    assert run["ingest"]["documents"] == 12 and sorted(run["ingest"]["duplicates"]) == [
        "archive/returns-and-refunds-2025.html", "help/shipping-and-delivery.html"]
    assert run["config"]["html_extract"] == "main" and run["config"]["dedupe"] is True
    kept = run_suite(fictional_web_suite(WEB), raggy_for(tmp_path / "b", dedupe=False), None, answers=False)
    assert kept["ingest"]["documents"] == 14 and kept["ingest"]["duplicates"] == []
