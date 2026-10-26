"""One tiny interface for every answer model, and a factory to pick one.

RAG needs only "system prompt + user prompt in, text out". Keeping the
interface that small means Claude, OpenAI and a local Ollama model are
interchangeable — the same idea as harnessy's week 1, with much less surface.
"""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    name: str  # "provider:model", shown in output and later in traces
    usage: dict  # running totals: calls, input, output, cache_read, cache_write (tokens)

    def complete(self, system: str, user: str, cache_prefix: str | None = None,
                 images: list[bytes] | None = None) -> str:
        """``cache_prefix`` is text that comes before ``user`` and repeats across calls
        (week 8): providers that support prompt caching may cache it. ``images`` are PNG
        bytes shown to a vision model with the user text (week 10)."""
        ...


def new_usage() -> dict:
    return {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


class LLMError(Exception):
    """Any failure talking to a model, with a hint on how to fix it."""


DEFAULT_MODELS = {
    "ollama": "qwen3.5:4b",
    "anthropic": "claude-opus-5-5",
}
OLLAMA_URL = "http://localhost:11434/v1"


def make_llm(provider: str, model: str | None, ollama_url: str = OLLAMA_URL) -> LLM:
    provider = provider.lower()
    if provider == "ollama":
        from raggy.generate.openai_llm import OpenAILLM

        return OpenAILLM(model or DEFAULT_MODELS["ollama"], base_url=ollama_url,
                         api_key="ollama", provider="ollama")
    if provider == "anthropic":
        from raggy.generate.anthropic_llm import AnthropicLLM

        return AnthropicLLM(model or DEFAULT_MODELS["anthropic"])
    if provider == "openai":
        import os

        from raggy.generate.openai_llm import OpenAILLM

        model = model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError("set RAGGY_LLM_MODEL or OPENAI_MODEL to choose an OpenAI model")
        return OpenAILLM(model)
    raise ValueError(f"unknown LLM provider '{provider}' (use ollama, anthropic or openai)")
