"""Settings, read from the environment (and a .env file in the current folder)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from raggy.embed.fastembed_embedder import DEFAULT_MODEL as DEFAULT_EMBED_MODEL


@dataclass
class Config:
    index_dir: Path
    embed_model: str
    llm_provider: str
    llm_model: str | None
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))  # keys live in .env, never in your shell rc
        return cls(
            index_dir=Path(os.environ.get("RAGGY_INDEX_DIR", ".raggy")),
            embed_model=os.environ.get("RAGGY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            llm_provider=os.environ.get("RAGGY_LLM", "ollama"),
            llm_model=os.environ.get("RAGGY_LLM_MODEL") or None,
            top_k=int(os.environ.get("RAGGY_TOP_K", "4")),
        )
