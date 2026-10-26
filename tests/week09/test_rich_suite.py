from pathlib import Path

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.dataset import resolve_gold
from raggy.evals.runner import run_suite
from raggy.evals.suites import SUITES, rich_suite
from raggy.generate.fake import FakeLLM
from raggy.ingest.loaders import LoadOptions
from raggy.pipeline import Raggy

ROOT = Path(__file__).resolve().parents[2] / "evals" / "rich"


def test_rich_gold_resolves_with_layout_and_ocr():
    s = rich_suite(ROOT)
    qs = s.questions()
    assert len(qs) == 28 and sum(not q.answerable for q in qs) == 4 and "rich" in SUITES
    resolve_gold(qs, s.documents(LoadOptions(pdf="layout", ocr=True)))  # strict: raises on any miss


def test_rich_suite_counts_missing_gold_as_a_miss(tmp_path):
    s = rich_suite(ROOT)
    assert s.strict_gold is False
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None, pdf="text", ocr=False)
    run = run_suite(s, Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM()), None, answers=False)
    assert set(run["gold_missing"]) == {"r08", "r09", "r10"}  # the scanned page has no text layer
    by_id = {q["id"]: q for q in run["questions"]}
    assert by_id["r08"]["retrieval"]["recall@4"] == 0.0
    assert run["config"]["pdf"] == "text" and run["config"]["ocr"] is False
