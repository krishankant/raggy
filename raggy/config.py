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


def _choice_env(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = (os.environ.get(name) or default).strip().lower()
    if raw not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, got {raw!r}")
    return raw


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be 1 or 0 (true/false), got {raw!r}")


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
    html_extract: str = "main"  # "main": article text only; "full": every visible word
    dedupe: bool = True  # skip documents that duplicate one already indexed

    def load_options(self):
        from raggy.ingest.loaders import LoadOptions

        return LoadOptions(html_extract=self.html_extract)

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))  # keys live in .env, never in your shell rc
        return cls(
            index_dir=Path(os.environ.get("RAGGY_INDEX_DIR") or ".raggy"),  # blank means default, never "."
            embed_model=os.environ.get("RAGGY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            llm_provider=os.environ.get("RAGGY_LLM", "ollama"),
            llm_model=os.environ.get("RAGGY_LLM_MODEL") or None,
            top_k=_positive_int_env("RAGGY_TOP_K", "4"),
            judge_provider=os.environ.get("RAGGY_JUDGE") or _default_judge(),
            judge_model=os.environ.get("RAGGY_JUDGE_MODEL") or None,
            html_extract=_choice_env("RAGGY_HTML_EXTRACT", "main", ("main", "full")),
            dedupe=_bool_env("RAGGY_DEDUPE", True),
        )
