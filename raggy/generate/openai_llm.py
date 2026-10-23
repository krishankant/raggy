"""OpenAI Chat Completions — which also covers Ollama, because Ollama serves
the same API at http://localhost:11434/v1."""

from __future__ import annotations

from raggy.generate.llm import LLMError


class OpenAILLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 provider: str = "openai"):
        self.model = model
        self.provider = provider
        self.name = f"{provider}:{model}"
        self._base_url = base_url
        self._api_key = api_key
        self._client = None  # built on first use, so constructing needs no key

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                max_retries=0 if self.provider == "ollama" else 2,
            )
        return self._client

    def _hint(self) -> str:
        if self.provider == "ollama":
            return f"is Ollama running (`ollama serve`) and is the model pulled (`ollama pull {self.model}`)?"
        return "check OPENAI_API_KEY in .env and the model name"

    def complete(self, system: str, user: str) -> str:
        import openai

        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
        except openai.OpenAIError as e:  # connection, auth, missing model, rate limit
            raise LLMError(f"{self.provider} request failed ({type(e).__name__}): {self._hint()}") from e
        return (response.choices[0].message.content or "").strip()
