import json

import pytest

from raggy.agent import AGENT_SYSTEM, parse_action, run_agent
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.types import Document

DOCS = [Document("/d/bike-models.md", "The Ebb E-Cargo has a box that seats two children. It weighs 38 kg."),
        Document("/d/service-centres.md", "Saltmarsh Cycles in Port Edda is the only centre that works on the Ebb motor."),
        Document("/d/cat.md", "The office cat is called Biscuit.")]


def scripted(*decisions, answer="Saltmarsh Cycles [2]."):
    """A fake model: agent decisions in order, then the final answer for the answer prompt."""
    queue = list(decisions)

    def reply(system, user):
        if system == AGENT_SYSTEM:
            return queue.pop(0) if queue else json.dumps({"action": "answer"})
        return answer
    return FakeLLM(reply)


def make(tmp_path, llm, **cfg):
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0, "top_k": 2} | cfg))
    r = Raggy(c, embedder=FakeEmbedder(), llm=llm)
    r.ingest_documents(DOCS)
    return r


def test_parse_action_tolerates_chatter_and_rejects_junk():
    assert parse_action('Sure! {"action": "search", "query": "Ebb motor repair"}') == ("search", "Ebb motor repair")
    assert parse_action('```json\n{"action": "answer"}\n```') == ("answer", None)
    assert parse_action('{"action": "not_found"}') == ("not_found", None)
    assert parse_action("I think we should search for it") is None
    assert parse_action('{"action": "search", "query": "  "}') is None  # a search needs a query


def test_agent_searches_again_then_answers(tmp_path):
    llm = scripted(json.dumps({"action": "search", "query": "which centre works on the Ebb motor"}))
    r = make(tmp_path, llm)
    result = run_agent(r, "Which centre can repair the motor of the bike that seats two children?")
    assert [s.query for s in result.steps] == [
        "Which centre can repair the motor of the bike that seats two children?",  # always searched first
        "which centre works on the Ebb motor"]
    sources = [h.chunk.source for h in result.passages]
    assert len(sources) == len(set(h.chunk.id for h in result.passages))  # no passage twice
    assert result.answer.text == "Saltmarsh Cycles [2]." and result.stopped == "answer"


def test_agent_stops_at_the_step_limit(tmp_path):
    llm = scripted(*[json.dumps({"action": "search", "query": f"more {i}"}) for i in range(10)])
    r = make(tmp_path, llm, agent_steps=3)
    result = run_agent(r, "q")
    assert len(result.steps) == 3 and result.stopped == "limit"
    assert result.answer.text == "Saltmarsh Cycles [2]."  # it still answers from what it gathered


def test_agent_repeated_query_stops(tmp_path):
    q = "Which centre repairs the Ebb motor?"
    llm = scripted(json.dumps({"action": "search", "query": q.lower()}))
    result = run_agent(make(tmp_path, llm), q)
    assert len(result.steps) == 1 and result.stopped == "repeat"  # searching the same thing again finds nothing new


def test_agent_not_found_refuses_without_answer_call(tmp_path):
    llm = scripted(json.dumps({"action": "not_found"}))
    result = run_agent(make(tmp_path, llm), "What is the CEO's shoe size?")
    assert result.answer.text.startswith("I couldn't find") and result.stopped == "not_found"
    assert not any(sys_ != AGENT_SYSTEM for sys_, _ in llm.calls)  # no answer call was paid for


def test_agent_unreadable_decision_answers_with_what_it_has(tmp_path):
    llm = scripted("let me think about this...")
    result = run_agent(make(tmp_path, llm), "q")
    assert result.failures == 1 and result.stopped == "unreadable" and result.answer.text == "Saltmarsh Cycles [2]."


def test_agent_decision_call_failure_is_counted(tmp_path):
    calls = []

    def reply(system, user):
        calls.append(system)
        if system == AGENT_SYSTEM:
            raise LLMError("down")
        return "ok [1]."
    result = run_agent(make(tmp_path, FakeLLM(reply)), "q")
    assert result.failures == 1 and result.stopped == "error" and result.answer.text == "ok [1]."


def test_agent_prompt_shows_found_passages_and_previous_searches(tmp_path):
    llm = scripted(json.dumps({"action": "search", "query": "Ebb motor centre"}))
    run_agent(make(tmp_path, llm), "first question")
    prompts = [u for s, u in llm.calls if s == AGENT_SYSTEM]
    assert "first question" in prompts[0] and "[1]" in prompts[0]
    assert "Ebb motor centre" in prompts[1]  # it sees what it already searched for


def test_agent_setting_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAGGY_AGENT", "on")
    monkeypatch.setenv("RAGGY_AGENT_STEPS", "5")
    monkeypatch.setenv("RAGGY_INDEX_DIR", str(tmp_path))
    c = Config.from_env()
    assert (c.agent, c.agent_steps) == (True, 5)
    with pytest.raises(Exception):
        monkeypatch.setenv("RAGGY_AGENT_STEPS", "0")
        Config.from_env()


def test_eval_records_agent_searches(tmp_path):
    from raggy.evals.dataset import EvalQuestion, GoldSpan
    from raggy.evals.runner import run_suite
    from raggy.evals.suites import Suite

    qs = [EvalQuestion("m1", "Which centre repairs the motor of the bike that seats two children?", True,
                       [GoldSpan("bike-models.md", "seats two children"), GoldSpan("service-centres.md", "works on the Ebb motor")],
                       "Saltmarsh Cycles.")]
    suite = Suite(name="mh", documents=lambda options=None: list(DOCS), questions=lambda: list(qs),
                  fingerprint=lambda: "x", ks=(1, 4), unit="chunk", answers=True)
    llm = scripted(json.dumps({"action": "search", "query": "Ebb motor centre"}))
    c = Config(index_dir=tmp_path / "ev", embed_model="fake", llm_provider="fake", llm_model=None,
               chunker="fixed", parent_tokens=0, top_k=1, agent=True)
    run = run_suite(suite, Raggy(c, embedder=FakeEmbedder(), llm=llm), None, answers=False)
    rec = run["questions"][0]
    assert rec["agent"]["searches"] == [qs[0].question, "Ebb motor centre"] and rec["agent"]["stopped"] in ("answer", "limit")
    assert "agent" in rec["latency_ms"]


def test_multihop_suite_gold_resolves_and_every_question_has_two_hops():
    from pathlib import Path

    from raggy.evals.dataset import resolve_gold
    from raggy.evals.suites import SUITES, multihop_suite

    root = Path(__file__).resolve().parents[2] / "evals"
    s = multihop_suite(root / "multihop", root / "fictional" / "docs")
    qs = s.questions()
    assert "multihop" in SUITES and len(qs) == 15 and sum(not q.answerable for q in qs) == 3
    assert all(len({g.source for g in q.gold}) == 2 for q in qs if q.answerable)  # two different documents
    resolve_gold(qs, s.documents())  # strict: every quote must be in the corpus
