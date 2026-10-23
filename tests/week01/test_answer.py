import pytest

from raggy.generate.answer import generate_answer
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError, make_llm
from raggy.generate.prompt import NOT_FOUND
from raggy.types import Chunk, Hit


def hits():
    return [Hit(Chunk(id=f"/d/a.md#{i}", source="/d/a.md", text=t, index=i, start=0, end=len(t)), 0.9)
            for i, t in enumerate(["Paris is in France.", "France uses the euro."])]


def test_generate_answer_uses_llm_and_parses():
    llm = FakeLLM("Paris [2] and [1] [9]")
    ans = generate_answer("Where is Paris?", hits(), llm)
    assert ans.text == "Paris [2] and [1] [9]"
    assert [c.number for c in ans.citations] == [2, 1]
    assert len(llm.calls) == 1 and "Where is Paris?" in llm.calls[0][1]


def test_generate_answer_no_hits_skips_llm():
    llm = FakeLLM("should not be called")
    ans = generate_answer("Anything?", [], llm)
    assert ans.text == NOT_FOUND and ans.citations == [] and llm.calls == []


def test_make_llm_providers():
    assert make_llm("ollama", None).name.startswith("ollama:")
    assert make_llm("anthropic", None).name == "anthropic:claude-opus-5-5"
    assert make_llm("openai", "gpt-x").name == "openai:gpt-x"


def test_make_llm_unknown_raises():
    with pytest.raises(ValueError):
        make_llm("bard", None)


def test_ollama_unreachable_raises_llm_error():
    llm = make_llm("ollama", None, ollama_url="http://127.0.0.1:9/v1")  # port 9: nothing listens
    with pytest.raises(LLMError, match="ollama"):
        llm.complete("s", "u")
