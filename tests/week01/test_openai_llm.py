from types import SimpleNamespace

from raggy.generate.llm import make_llm


class RecordingClient:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content=" hi ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_ollama_turns_thinking_off():
    llm = make_llm("ollama", None)
    llm._client = RecordingClient()
    assert llm.complete("s", "u") == "hi"
    assert llm._client.kwargs["reasoning_effort"] == "none"


def test_openai_leaves_reasoning_default():
    llm = make_llm("openai", "gpt-x")
    llm._client = RecordingClient()
    llm.complete("s", "u")
    assert "reasoning_effort" not in llm._client.kwargs
