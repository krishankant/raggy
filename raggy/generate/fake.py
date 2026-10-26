"""A scripted LLM for tests: returns a fixed reply and records every call."""

from __future__ import annotations

from typing import Callable


class FakeLLM:
    name = "fake:scripted"

    def __init__(self, reply: str | Callable[[str, str], str] = "ok", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.image_counts: list[int] = []  # images sent with each call that had any

    def complete(self, system: str, user: str, cache_prefix: str | None = None,
                 images: list[bytes] | None = None) -> str:
        user = (cache_prefix or "") + user
        if images:
            self.image_counts.append(len(images))
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply(system, user) if callable(self.reply) else self.reply
