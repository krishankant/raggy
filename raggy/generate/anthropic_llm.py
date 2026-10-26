"""Claude through the official Anthropic SDK.

Uses the beta Messages endpoint only to opt into server-side refusal
fallbacks (``fallbacks="default"``): if the model declines, the API re-runs
the request on a fallback model inside the same call.
"""

from __future__ import annotations

from raggy.generate.llm import LLMError


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        self.model = model
        self.effort = effort  # thinking depth; Opus 5.5 defaults to medium, set explicitly
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant` login)
        return self._client

    def complete(self, system: str, user: str) -> str:
        import anthropic

        try:
            response = self._get_client().beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("anthropic: authentication failed — set ANTHROPIC_API_KEY in .env") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"anthropic: model '{self.model}' not found — check RAGGY_LLM_MODEL") from e
        except anthropic.RateLimitError as e:
            raise LLMError("anthropic: rate limited — wait a moment and retry") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"anthropic: API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("anthropic: could not reach the API — check your network") from e
        except TypeError as e:  # the SDK raises this when no API key or login is configured
            if "authentication" not in str(e):
                raise
            raise LLMError("anthropic: no credentials — set ANTHROPIC_API_KEY in .env") from e
        if response.stop_reason == "refusal":
            raise LLMError("anthropic: the model declined to answer this request")
        return "".join(b.text for b in response.content if b.type == "text").strip()
