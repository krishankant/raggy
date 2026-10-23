import pytest

from raggy.evals.dataset import EvalQuestion, GoldSpan
from raggy.evals.judge import (
    CORRECTNESS_SYSTEM, GROUNDING_SYSTEM, JUDGE_VERSION, Judge, is_refusal, parse_json_object,
)
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.types import Answer, Chunk, Citation, Hit

Q = EvalQuestion("f01", "How long do refunds take?", True, [GoldSpan("r.md", "14 days")], "Within 14 days.")


def answer(text="Refunds take 14 days [1]. Also free tea [2].", cites=(1, 2)):
    hits = [Hit(Chunk(f"r.md#{i}", "r.md", f"passage {i}", i, 0, 9), 0.9) for i in range(2)]
    citations = [Citation(n, "r.md", f"r.md#{n - 1}", "s") for n in cites]
    return Answer(Q.question, text, citations, hits)


def scripted(correct='{"verdict": "correct", "reason": "matches"}',
             grounding='{"faithful": false, "citations": [{"n": 1, "supported": true}, {"n": 2, "supported": false}]}'):
    def reply(system, user):
        return correct if system == CORRECTNESS_SYSTEM else grounding
    return FakeLLM(reply)


def test_judge_version_is_set():
    assert JUDGE_VERSION == "judge-v1"


@pytest.mark.parametrize("text,expected", [
    ("I couldn't find this in the documents.", True),
    ("The sources talk about bikes only. I couldn't find this in the documents.", True),
    ("I couldn’t find this in the documents.", True),
    ("i COULDN'T find this in the documents", True),
    ("Refunds take 14 days [1].", False),
])
def test_is_refusal_variants(text, expected):
    assert is_refusal(text) is expected


def test_parse_json_plain():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_in_fence():
    assert parse_json_object('Here you go:\n```json\n{"a": {"b": "x}"}}\n```') == {"a": {"b": "x}"}}


def test_parse_json_none_raises():
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_judge_scores_correct_and_citations():
    llm = scripted()
    v = Judge(llm).judge(Q, answer())
    assert (v.correctness, v.faithful, v.citation_precision, v.error) == (1.0, False, 0.5, None)
    assert len(llm.calls) == 2
    grounding_user = next(u for s, u in llm.calls if s == GROUNDING_SYSTEM)
    assert "[1]" in grounding_user and "passage 0" in grounding_user  # judge sees the sources


@pytest.mark.parametrize("verdict,score", [("partial", 0.5), ("incorrect", 0.0)])
def test_judge_verdict_scores(verdict, score):
    v = Judge(scripted(correct=f'{{"verdict": "{verdict}", "reason": "x"}}')).judge(Q, answer())
    assert v.correctness == score


def test_judge_missing_citation_counts_unsupported():
    v = Judge(scripted(grounding='{"faithful": true, "citations": [{"n": 1, "supported": true}]}')).judge(Q, answer())
    assert v.citation_precision == 0.5


def test_judge_no_citations_gives_none():
    v = Judge(scripted(grounding='{"faithful": true, "citations": []}')).judge(Q, answer("14 days.", cites=()))
    assert v.citation_precision is None and v.faithful is True


def test_judge_garbage_records_error():
    v = Judge(scripted(grounding="I think it is fine")).judge(Q, answer())
    assert v.correctness == 1.0  # the part that worked is kept
    assert v.faithful is None and v.citation_precision is None
    assert "grounding" in v.error


def test_judge_unknown_verdict_records_error():
    v = Judge(scripted(correct='{"verdict": "mostly", "reason": "x"}')).judge(Q, answer())
    assert v.correctness is None and "correctness" in v.error


def test_judge_llm_error_records_error():
    v = Judge(FakeLLM(error=LLMError("rate limited"))).judge(Q, answer())
    assert v.correctness is None and v.faithful is None
    assert "rate limited" in v.error


def test_parse_json_escaped_quote_with_brace():
    assert parse_json_object(r'{"reason": "says \"}\" here", "verdict": "correct"}')["verdict"] == "correct"
