"""OpenAI Chat Completions — which also covers Ollama, because Ollama serves
the same API at http://localhost:11434/v1."""

from __future__ import annotations

from raggy.generate.llm import LLMError, new_usage


class OpenAILLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 provider: str = "openai"):
        self.model = model
        self.provider = provider
        self.name = f"{provider}:{model}"
        self._base_url = base_url
        self._api_key = api_key
        self._client = None  # built on first use, so constructing needs no key
        self.usage = new_usage()

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                max_retries=0 if self.provider == "ollama" else 2,
                # a cold local model can take ~30 s just to load on a small laptop
                timeout=300 if self.provider == "ollama" else 120,
            )
        return self._client

    def _hint(self) -> str:
        if self.provider == "ollama":
            return f"is Ollama running (`ollama serve`) and is the model pulled (`ollama pull {self.model}`)?"
        return "check OPENAI_API_KEY in .env and the model name"

    def complete(self, system: str, user: str, cache_prefix: str | None = None) -> str:
        import openai

        extra = {}
        if self.provider == "ollama":
            # Small local "thinking" models (qwen3.5, gemma4…) can reason for
            # minutes before answering. Grounded RAG answers don't need it:
            # on an 8 GB M2 this took a reply from a timeout to under 2 s.
            extra["reasoning_effort"] = "none"
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          # OpenAI caches long shared prefixes (1,024+ tokens) automatically:
                          # putting the repeated part first is all it needs.
                          {"role": "user", "content": (cache_prefix or "") + user}],
                **extra,
            )
        except openai.OpenAIError as e:  # connection, auth, missing model, rate limit
            raise LLMError(f"{self.provider} request failed ({type(e).__name__}): {self._hint()}") from e
        u = getattr(response, "usage", None)
        self.usage["calls"] += 1
        if u is not None:
            self.usage["input"] += u.prompt_tokens or 0
            self.usage["output"] += u.completion_tokens or 0
            details = getattr(u, "prompt_tokens_details", None)
            self.usage["cache_read"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return (response.choices[0].message.content or "").strip()
