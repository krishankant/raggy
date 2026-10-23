"""A scripted LLM for tests: returns a fixed reply and records every call."""

from __future__ import annotations

from typing import Callable


class FakeLLM:
    name = "fake:scripted"

    def __init__(self, reply: str | Callable[[str, str], str] = "ok", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply(system, user) if callable(self.reply) else self.reply
