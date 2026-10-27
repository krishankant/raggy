"""Settings, read from the environment (and a .env file in the current folder)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from raggy.embed.fastembed_embedder import DEFAULT_MODEL as DEFAULT_EMBED_MODEL


class ConfigError(ValueError):
    """A setting in the environment or .env has a value raggy can't use."""


def _positive_int_env(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    if not raw.strip().isdigit() or int(raw) < 1:
        raise ConfigError(f"{name} must be a whole number of 1 or more, got {raw!r}")
    return int(raw)


DEFAULT_BUDGETS = {"transform": 2000, "retrieve": 100, "rerank": 500, "generate": 5000}  # ms, on an M2 laptop


def _gate_env() -> dict:
    """RAGGY_REFUSAL_GATE="vector=0.62,hybrid=0.031": per retrieval mode, the top score below
    which raggy answers "not in the documents" without calling the model. Calibrate it with
    evals/calibrate_gate.py; scores mean different things per mode, so each needs its own."""
    gate = {}
    for part in filter(None, (p.strip() for p in (os.environ.get("RAGGY_REFUSAL_GATE") or "").split(","))):
        mode, _, value = part.partition("=")
        try:
            gate[mode.strip()] = float(value)
        except ValueError:
            raise ConfigError(f"RAGGY_REFUSAL_GATE: {mode.strip()} needs a number, got {value!r}") from None
    return gate


def _budgets_env() -> dict:
    """RAGGY_BUDGETS="rerank=300,transform=1500" overrides some stages' p95 budgets (ms)."""
    budgets = dict(DEFAULT_BUDGETS)
    for part in filter(None, (p.strip() for p in (os.environ.get("RAGGY_BUDGETS") or "").split(","))):
        stage, _, ms = part.partition("=")
        stage = stage.strip()
        if stage not in budgets:
            raise ConfigError(f"RAGGY_BUDGETS: unknown stage {stage!r} (stages: {', '.join(budgets)})")
        if not ms.strip().isdigit():
            raise ConfigError(f"RAGGY_BUDGETS: {stage} needs a whole number of milliseconds, got {ms!r}")
        budgets[stage] = int(ms)
    return budgets


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
    pdf: str = "layout"  # layout (headings, tables, page labels) | text (week 3's text layer)
    ocr: bool = True  # OCR scanned PDF pages (needs pdf=layout)
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
    rerank: str = "none"  # none | local (cross-encoder) | llm (the chat model ranks the candidates)
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # local cross-encoder; BAAI/bge-reranker-base is the larger option
    rerank_candidates: int = 20  # how many retrieved chunks the reranker sees
    query_transform: str = "none"  # none | rewrite | multi | hyde (one cached LLM call per question)
    budgets: dict = field(default_factory=lambda: dict(DEFAULT_BUDGETS))  # p95 ms per stage
    pack: bool = False  # merge overlapping passages, order them, fit context_tokens (week 8)
    context_tokens: int = 3000  # budget for the passages in the answer prompt
    citation_check: str = "off"  # off | lexical | llm: verify each [n] against passage n
    citation_threshold: float = 0.5  # lexical: share of a sentence's content words found in its source
    citation_action: str = "report"  # report | strip (remove unsupported markers) | refuse
    refusal_gate: dict = field(default_factory=dict)  # mode -> top-score threshold; empty = off
    images: str = "off"  # off | ocr (text in the image) | vision (a model's description + that text)
    vision_provider: str | None = None  # ollama | openai | anthropic; None: the answer model's provider
    vision_model: str | None = None  # None: the answer model if same provider, else the adapter default
    media: str = "off"  # off | on: transcribe audio and video locally (week 11)
    whisper_model: str = "base"  # tiny | base | small | medium | large-v3
    media_window: int = 60  # seconds per transcript window
    keyframe_seconds: int = 30  # video: one frame per this many seconds (needs RAGGY_IMAGES on)
    image_search: bool = False  # CLIP vectors for every picture: `raggy images "query"` (week 10, optional)

    def chunk_settings(self):
        from raggy.chunking import ChunkSettings

        return ChunkSettings(chunker=self.chunker, tokens=self.chunk_tokens, overlap=self.chunk_overlap_tokens,
                             chars=self.chunk_size, char_overlap=self.chunk_overlap,
                             headers=self.chunk_headers, parent_tokens=self.parent_tokens)

    def load_options(self):
        from raggy.ingest.loaders import LoadOptions

        return LoadOptions(html_extract=self.html_extract, pdf=self.pdf, ocr=self.ocr, images=self.images,
                           keep_images=self.image_search, media=self.media, whisper_model=self.whisper_model,
                           media_window=self.media_window, keyframe_seconds=self.keyframe_seconds)

    def media_key(self) -> dict | None:
        """What shapes a transcript's chunks; None while media is off."""
        if self.media != "on":
            return None
        from raggy.ingest.media import TRANSCRIBE_VERSION

        return {"whisper": self.whisper_model, "window": self.media_window,
                "keyframes": self.keyframe_seconds, "version": TRANSCRIBE_VERSION}

    def vision_key(self) -> dict | None:
        """What decides an image's text, beyond its bytes; None unless a model reads images."""
        if self.images != "vision":
            return None
        from raggy.ingest.images import VISION_VERSION

        return {"provider": self.vision_provider or self.llm_provider, "model": self.vision_model,
                "version": VISION_VERSION}

    def __post_init__(self):
        if self.image_search and self.images == "off":
            raise ConfigError("RAGGY_IMAGE_SEARCH needs pictures to be read: set RAGGY_IMAGES=ocr or vision")
        if self.image_search and self.store != "lance":
            raise ConfigError("RAGGY_IMAGE_SEARCH needs RAGGY_STORE=lance")
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
        llm_provider = os.environ.get("RAGGY_LLM", "ollama")
        llm_model = os.environ.get("RAGGY_LLM_MODEL") or None
        vision_provider = (_choice_env("RAGGY_VISION", "ollama", ("ollama", "openai", "anthropic"))
                           if os.environ.get("RAGGY_VISION") else llm_provider)
        # One model setting is enough when the answer model can see: it reads the images too.
        vision_model = os.environ.get("RAGGY_VISION_MODEL") or (llm_model if vision_provider == llm_provider else None)
        return cls(
            index_dir=Path(os.environ.get("RAGGY_INDEX_DIR") or ".raggy"),  # blank means default, never "."
            embed_model=os.environ.get("RAGGY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            llm_provider=os.environ.get("RAGGY_LLM", "ollama"),
            llm_model=os.environ.get("RAGGY_LLM_MODEL") or None,
            top_k=_positive_int_env("RAGGY_TOP_K", "4"),
            judge_provider=os.environ.get("RAGGY_JUDGE") or _default_judge(),
            judge_model=os.environ.get("RAGGY_JUDGE_MODEL") or None,
            html_extract=_choice_env("RAGGY_HTML_EXTRACT", "main", ("main", "full")),
            pdf=_choice_env("RAGGY_PDF", "layout", ("text", "layout")),
            ocr=_bool_env("RAGGY_OCR", True),
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
            rerank=_choice_env("RAGGY_RERANK", "none", ("none", "local", "llm")),
            rerank_model=os.environ.get("RAGGY_RERANK_MODEL") or "Xenova/ms-marco-MiniLM-L-6-v2",
            rerank_candidates=_positive_int_env("RAGGY_RERANK_CANDIDATES", "20"),
            query_transform=_choice_env("RAGGY_QUERY_TRANSFORM", "none", ("none", "rewrite", "multi", "hyde")),
            budgets=_budgets_env(),
            pack=_bool_env("RAGGY_PACK", False),
            context_tokens=_positive_int_env("RAGGY_CONTEXT_TOKENS", "3000"),
            citation_check=_choice_env("RAGGY_CITATION_CHECK", "off", ("off", "lexical", "llm")),
            citation_threshold=float(os.environ.get("RAGGY_CITATION_THRESHOLD") or 0.5),
            citation_action=_choice_env("RAGGY_CITATION_ACTION", "report", ("report", "strip", "refuse")),
            refusal_gate=_gate_env(),
            images=_choice_env("RAGGY_IMAGES", "off", ("off", "ocr", "vision")),
            vision_provider=vision_provider,
            vision_model=vision_model,
            image_search=_bool_env("RAGGY_IMAGE_SEARCH", False),
            media=_choice_env("RAGGY_MEDIA", "off", ("off", "on")),
            whisper_model=os.environ.get("RAGGY_WHISPER_MODEL") or "base",
            media_window=_positive_int_env("RAGGY_MEDIA_WINDOW", "60"),
            keyframe_seconds=_positive_int_env("RAGGY_KEYFRAME_SECONDS", "30"),
        )
