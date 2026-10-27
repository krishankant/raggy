"""Claude through the official Anthropic SDK.

Uses the beta Messages endpoint only to opt into server-side refusal
fallbacks (``fallbacks="default"``): if the model declines, the API re-runs
the request on a fallback model inside the same call.
"""

from __future__ import annotations

from raggy.generate.llm import LLMError, new_usage


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        self.model = model
        self.effort = effort  # thinking depth; Opus 5.5 defaults to medium, set explicitly
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        self._client = None
        self.usage = new_usage()

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant` login)
        return self._client

    def stream(self, system: str, user: str):
        """The answer as it's written (week 12), through the SDK's streaming helper."""
        import anthropic

        try:
            with self._get_client().beta.messages.stream(
                    model=self.model, max_tokens=self.max_tokens, system=system,
                    messages=[{"role": "user", "content": user}], output_config={"effort": self.effort},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default") as s:
                yield from s.text_stream
                final = s.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMError("anthropic: authentication failed — set ANTHROPIC_API_KEY in .env") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"anthropic: API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("anthropic: could not reach the API — check your network") from e
        u = final.usage
        self.usage["calls"] += 1
        self.usage["input"] += u.input_tokens or 0
        self.usage["output"] += u.output_tokens or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    def complete(self, system: str, user: str, cache_prefix: str | None = None,
                 images: list[bytes] | None = None) -> str:
        import base64

        import anthropic

        content: str | list = user
        if cache_prefix:
            # The repeated part (e.g. a whole document for contextual retrieval) is its own
            # block marked for caching; later calls with the same prefix read it at ~0.1x price.
            # Prefixes under the model's minimum (512 tokens on Opus 5.5) silently aren't cached.
            content = [{"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                       {"type": "text", "text": user}]
        if images:  # images before the text that asks about them
            blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                   "data": base64.b64encode(im).decode()}} for im in images]
            content = blocks + (content if isinstance(content, list) else [{"type": "text", "text": content}])
        try:
            response = self._get_client().beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
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
        u = response.usage
        self.usage["calls"] += 1
        self.usage["input"] += u.input_tokens or 0
        self.usage["output"] += u.output_tokens or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        if response.stop_reason == "refusal":
            raise LLMError("anthropic: the model declined to answer this request")
        return "".join(b.text for b in response.content if b.type == "text").strip()
