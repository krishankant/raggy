import json
from types import SimpleNamespace as NS

import pytest

from raggy.config import Config, ConfigError
from raggy.costs import cost, parse_prices
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.generate.openai_llm import OpenAILLM
from raggy.pipeline import Raggy
from raggy.types import Document


def make(tmp_path, llm=None, **cfg):
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0} | cfg))
    r = Raggy(c, embedder=FakeEmbedder(), llm=llm or FakeLLM("Refunds take 14 days [1]."))
    r.ingest_documents([Document("/d/refunds.md", "Refunds are paid within 14 days of the return arriving."),
                        Document("/d/cat.md", "The office cat is called Biscuit.")])
    return r


# --- streaming --------------------------------------------------------------

def test_fake_llm_streams_its_reply():
    assert "".join(FakeLLM("one two three").stream("s", "u")) == "one two three"


class StreamingOpenAI:
    def __init__(self):
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        chunks = [NS(choices=[NS(delta=NS(content=t))], usage=None) for t in ("Refunds ", "take ", "14 days [1].")]
        return iter(chunks + [NS(choices=[], usage=NS(prompt_tokens=50, completion_tokens=6, prompt_tokens_details=None))])


def test_openai_adapter_streams_and_counts_usage():
    llm = OpenAILLM("gpt-x")
    llm._client = StreamingOpenAI()
    assert list(llm.stream("sys", "q")) == ["Refunds ", "take ", "14 days [1]."]
    assert llm._client.requests[0]["stream"] is True
    assert (llm.usage["calls"], llm.usage["input"], llm.usage["output"]) == (1, 50, 6)


def test_ask_streams_tokens_then_parses_citations(tmp_path):
    r = make(tmp_path)
    got = []
    a = r.ask("how long do refunds take", on_token=got.append)
    assert "".join(got) == a.text == "Refunds take 14 days [1]." and len(got) > 1
    assert [c.number for c in a.citations] == [1]


# --- answer cache -----------------------------------------------------------

def test_answer_cache_repeats_cost_nothing(tmp_path):
    llm = FakeLLM("Refunds take 14 days [1].")
    r = make(tmp_path, llm=llm)
    first = r.ask("How long do refunds take?")
    again = r.ask("  how long do REFUNDS take?  ")  # same question, normalised
    assert len(llm.calls) == 1 and again.text == first.text and again.cached and not first.cached
    assert [c.number for c in again.citations] == [1]


def test_answer_cache_misses_when_passages_or_model_change(tmp_path):
    llm = FakeLLM("Refunds take 14 days [1].")
    r = make(tmp_path, llm=llm)
    r.ask("How long do refunds take?")
    r.ingest_documents([Document("/d/refunds.md", "Refunds are paid within 30 days of the return arriving.")])
    r.ask("How long do refunds take?")
    assert len(llm.calls) == 2  # the passage text changed: a different input
    other = FakeLLM("x [1].")
    other.name = "fake:other"
    r2 = Raggy(r.config, embedder=FakeEmbedder(), llm=other)
    r2.ask("How long do refunds take?")
    assert len(other.calls) == 1


def test_answer_cache_off(tmp_path):
    llm = FakeLLM("Refunds take 14 days [1].")
    r = make(tmp_path, llm=llm, answer_cache=False)
    r.ask("q refunds")
    r.ask("q refunds")
    assert len(llm.calls) == 2


# --- costs ------------------------------------------------------------------

def test_prices_and_cost():
    prices = parse_prices("openai:gpt-x=2/8/0.2, anthropic:claude-y=5/25")
    usage = {"calls": 1, "input": 1_000_000, "output": 100_000, "cache_read": 400_000, "cache_write": 0}
    # 600k uncached input at 2 + 400k cached at 0.2 + 100k output at 8, per million
    assert cost(usage, "openai:gpt-x", prices) == pytest.approx(1.2 + 0.08 + 0.8)
    assert cost(usage, "anthropic:claude-y", prices) == pytest.approx(3.0 + 2.0 + 2.5)  # no cache price: full rate
    assert cost(usage, "ollama:qwen", prices) is None  # unknown: say nothing rather than guess
    with pytest.raises(ConfigError, match="gpt-x=two"):
        parse_prices("openai:gpt-x=two/8")


# --- traces -----------------------------------------------------------------

def test_trace_records_each_ask(tmp_path):
    r = make(tmp_path, trace=True)
    r.ask("How long do refunds take?")
    r.ask("How long do refunds take?")
    files = list((tmp_path / "i" / "traces").glob("*.jsonl"))
    rows = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["question"] == "How long do refunds take?"
    # (the fake embedder's ranking is arbitrary: check what was recorded, not the order)
    assert {h["source"] for h in rows[0]["hits"]} <= {"/d/refunds.md", "/d/cat.md"} and "retrieve" in rows[0]["timings"]
    assert rows[0]["cached"] is False and rows[1]["cached"] is True
    assert rows[0]["answer"] == "Refunds take 14 days [1]." and rows[0]["citations"] == [1]


def test_trace_off_writes_nothing(tmp_path):
    r = make(tmp_path)
    r.ask("How long do refunds take?")
    assert not (tmp_path / "i" / "traces").exists()


# --- profiles ---------------------------------------------------------------

def test_profile_fills_defaults_but_explicit_settings_win(monkeypatch, tmp_path):
    for k in ("RAGGY_RETRIEVAL", "RAGGY_RERANK", "RAGGY_PACK", "RAGGY_IMAGES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAGGY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("RAGGY_PROFILE", "accurate")
    c = Config.from_env()
    assert (c.retrieval, c.rerank, c.images) == ("hybrid", "local", "ocr")
    monkeypatch.setenv("RAGGY_RERANK", "none")
    assert Config.from_env().rerank == "none"  # the user's own setting beats the profile
    monkeypatch.setenv("RAGGY_PROFILE", "turbo")
    with pytest.raises(ConfigError, match="local, fast, accurate"):
        Config.from_env()


# --- concurrent ingest ------------------------------------------------------

def test_concurrent_ingest_matches_serial(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(12):
        (docs / f"d{i:02d}.md").write_text(f"# Doc {i}\n\nFact number {i} is {i * 7}.")
    results = []
    for workers in (1, 4):
        c = Config(index_dir=tmp_path / f"i{workers}", embed_model="fake", llm_provider="fake", llm_model=None,
                   ingest_workers=workers)
        r = Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM())
        rep = r.ingest([str(docs)])
        results.append((rep.added, sorted(r.sources().items())))
    assert results[0] == results[1] and results[0][0] == 12


def test_trace_command_lists_latest(tmp_path, monkeypatch, capsys):
    from raggy import cli

    r = make(tmp_path, trace=True)
    r.ask("How long do refunds take?")
    monkeypatch.setattr(cli, "Raggy", lambda config: r, raising=False)
    monkeypatch.setenv("RAGGY_INDEX_DIR", str(tmp_path / "i"))
    assert cli.main(["trace", "-n", "1"]) == 0
    assert "How long do refunds take?" in capsys.readouterr().out
