import pytest

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.report import format_run
from raggy.evals.runner import prepare_index, run_suite
from raggy.evals.suites import Suite
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.types import Document
from raggy.evals.dataset import EvalQuestion, GoldSpan


def suite(root):
    docs = [Document(f"{root}/a.md", "Refunds are paid within 14 days of the return arriving."),
            Document(f"{root}/b.md", "The office cat is called Biscuit.")]
    qs = [EvalQuestion("q1", "when are refunds paid", True, [GoldSpan("a.md", "within 14 days")], "14 days")]
    return Suite(name="s", documents=lambda options=None: list(docs), questions=lambda: list(qs),
                 fingerprint=lambda: "same-bytes", ks=(1, 2), unit="chunk", answers=True)


def make(tmp_path, llm=None, **cfg):
    c = Config(index_dir=tmp_path / "ev", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0} | cfg))
    return Raggy(c, embedder=FakeEmbedder(), llm=llm or FakeLLM("ok [1]."))


def test_cache_key_changes_when_document_paths_change(tmp_path):
    assert prepare_index(suite("/old/checkout"), make(tmp_path))["cached"] is False
    assert prepare_index(suite("/old/checkout"), make(tmp_path))["cached"] is True
    # Same file names and bytes, different checkout: the stored sources would be stale paths.
    assert prepare_index(suite("/new/checkout"), make(tmp_path))["cached"] is False
    run = run_suite(suite("/new/checkout"), make(tmp_path), None, answers=False)
    assert run["metrics"]["recall@1"] == 1.0


def test_run_counts_and_reports_fallbacks(tmp_path):
    broken = FakeLLM(error=LLMError("openai request failed (RateLimitError)"))
    r = make(tmp_path, llm=broken, query_transform="rewrite", rerank="llm")
    run = run_suite(suite("/x"), r, None, answers=False)
    assert run["fallbacks"] == {"transform": 1, "rerank": 1}
    out = format_run(run)
    assert "⚠ 1 of 1 query transform calls failed" in out and "⚠ 1 of 1 rerank calls failed" in out


def test_anthropic_without_credentials_is_llm_error(monkeypatch):
    from raggy.generate.anthropic_llm import AnthropicLLM
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    llm = AnthropicLLM()

    class NoCreds:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    raise TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token"')
    llm._client = NoCreds()
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        llm.complete("s", "u")
