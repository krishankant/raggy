"""Settings, read from the environment (and a .env file in the current folder)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from raggy.embed.fastembed_embedder import DEFAULT_MODEL as DEFAULT_EMBED_MODEL


class ConfigError(ValueError):
    """A setting in the environment or .env has a value raggy can't use."""


def _positive_int_env(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    if not raw.strip().isdigit() or int(raw) < 1:
        raise ConfigError(f"{name} must be a whole number of 1 or more, got {raw!r}")
    return int(raw)


def _default_judge() -> str:
    # Judging needs a strong model, so prefer a hosted one when a key exists;
    # otherwise fall back to the local model rather than refusing to run.
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"


@dataclass
class Config:
    index_dir: Path
    embed_model: str
    llm_provider: str
    llm_model: str | None
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 100
    judge_provider: str = "ollama"  # the model that grades answers in `raggy eval`
    judge_model: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))  # keys live in .env, never in your shell rc
        return cls(
            index_dir=Path(os.environ.get("RAGGY_INDEX_DIR", ".raggy")),
            embed_model=os.environ.get("RAGGY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            llm_provider=os.environ.get("RAGGY_LLM", "ollama"),
            llm_model=os.environ.get("RAGGY_LLM_MODEL") or None,
            top_k=_positive_int_env("RAGGY_TOP_K", "4"),
            judge_provider=os.environ.get("RAGGY_JUDGE") or _default_judge(),
            judge_model=os.environ.get("RAGGY_JUDGE_MODEL") or None,
        )
