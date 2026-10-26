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


MAX_EMBED_TOKENS = 512
HEADER_BUDGET = 64  # tokens kept free for the contextual header in front of each chunk


def _nonneg_int_env(name: str, default: str) -> int:
    raw = os.environ.get(name) or default
    if not raw.strip().isdigit():
        raise ConfigError(f"{name} must be a whole number of 0 or more, got {raw!r}")
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
    chunker: str = "structure"  # fixed | recursive | structure (week 4 measured structure best)
    chunk_tokens: int = 256  # recursive / structure chunk size, in the embedding model's tokens
    chunk_overlap_tokens: int = 32
    chunk_headers: bool = True  # embed "title › section" with each chunk
    parent_tokens: int = 512  # > 0: retrieve small chunks, answer with their parent of this size
    chunk_context: str = "none"  # none | llm (contextual retrieval: one LLM call per chunk, cached)
    embed_cache: bool = True  # reuse vectors for text this model has embedded before
    store: str = "lance"  # lance (on disk, filters, optional ANN) | numpy (week 1's in-memory matrix)
    ann: str = "off"  # off: exact search | on: approximate (IVF-HNSW) index, Lance only
    ann_nprobes: int = 20  # ANN: how many IVF partitions each search visits
    ann_refine: int = 4  # ANN: re-rank this many times k candidates with exact scores
    retrieval: str = "vector"  # vector | keyword (BM25) | hybrid (both, fused with RRF)
    rrf_k: int = 60  # reciprocal rank fusion constant
    fusion_candidates: int = 50  # hybrid: candidates taken from each side before fusing

    def chunk_settings(self):
        from raggy.chunking import ChunkSettings

        return ChunkSettings(chunker=self.chunker, tokens=self.chunk_tokens, overlap=self.chunk_overlap_tokens,
                             chars=self.chunk_size, char_overlap=self.chunk_overlap,
                             headers=self.chunk_headers, parent_tokens=self.parent_tokens)

    def load_options(self):
        from raggy.ingest.loaders import LoadOptions

        return LoadOptions(html_extract=self.html_extract)

    def __post_init__(self):
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ConfigError(f"RAGGY_CHUNK_OVERLAP_TOKENS ({self.chunk_overlap_tokens}) must be smaller than "
                              f"RAGGY_CHUNK_TOKENS ({self.chunk_tokens})")
        if self.chunk_tokens + HEADER_BUDGET > MAX_EMBED_TOKENS:
            raise ConfigError(f"RAGGY_CHUNK_TOKENS ({self.chunk_tokens}) plus {HEADER_BUDGET} tokens for the "
                              f"header must fit the embedding model's 512-token limit")
        if self.parent_tokens and self.parent_tokens <= self.chunk_tokens:
            raise ConfigError(f"RAGGY_PARENT_TOKENS ({self.parent_tokens}) must be 0 (off) or larger than "
                              f"RAGGY_CHUNK_TOKENS ({self.chunk_tokens})")

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
            chunker=_choice_env("RAGGY_CHUNKER", "structure", ("fixed", "recursive", "structure")),
            chunk_tokens=_positive_int_env("RAGGY_CHUNK_TOKENS", "256"),
            chunk_overlap_tokens=_nonneg_int_env("RAGGY_CHUNK_OVERLAP_TOKENS", "32"),
            chunk_headers=_bool_env("RAGGY_CHUNK_HEADERS", True),
            parent_tokens=_nonneg_int_env("RAGGY_PARENT_TOKENS", "512"),
            chunk_context=_choice_env("RAGGY_CHUNK_CONTEXT", "none", ("none", "llm")),
            embed_cache=_bool_env("RAGGY_EMBED_CACHE", True),
            store=_choice_env("RAGGY_STORE", "lance", ("numpy", "lance")),
            ann=_choice_env("RAGGY_ANN", "off", ("off", "on")),
            ann_nprobes=_positive_int_env("RAGGY_ANN_NPROBES", "20"),
            ann_refine=_positive_int_env("RAGGY_ANN_REFINE", "4"),
            retrieval=_choice_env("RAGGY_RETRIEVAL", "vector", ("vector", "keyword", "hybrid")),
            rrf_k=_positive_int_env("RAGGY_RRF_K", "60"),
            fusion_candidates=_positive_int_env("RAGGY_FUSION_CANDIDATES", "50"),
        )
