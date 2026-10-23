"""The whole RAG pipeline in one place.

    ingest:  path ─► load ─► chunk ─► embed ─► store (on disk)
    ask:     question ─► embed ─► search store ─► prompt + LLM ─► cited answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from raggy.chunking.fixed import chunk_fixed
from raggy.config import Config
from raggy.embed.base import Embedder
from raggy.generate.answer import generate_answer
from raggy.generate.llm import LLM, make_llm
from raggy.ingest.loaders import load_path
from raggy.store.numpy_store import IndexNotFoundError, NumpyStore
from raggy.types import Answer, Chunk, Document, Hit


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)


class Raggy:
    def __init__(self, config: Config, embedder: Embedder | None = None, llm: LLM | None = None):
        self.config = config
        self._embedder = embedder
        self._llm = llm

    # Models are built on first use: `raggy sources` shouldn't load an
    # embedding model, and `raggy search` shouldn't need an LLM.
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from raggy.embed.fastembed_embedder import FastEmbedEmbedder

            self._embedder = FastEmbedEmbedder(self.config.embed_model)
        return self._embedder

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = make_llm(self.config.llm_provider, self.config.llm_model)
        return self._llm

    def _open_store(self) -> NumpyStore:
        try:
            return NumpyStore.load(self.config.index_dir)
        except IndexNotFoundError:
            return NumpyStore(self.config.index_dir)

    def ingest(self, paths: list[str]) -> IngestReport:
        docs, skipped = [], []
        for path in paths:
            try:
                docs.extend(load_path(path))
            except FileNotFoundError:
                skipped.append(path)
        report = self.ingest_documents(docs)
        report.skipped = skipped
        return report

    def ingest_documents(self, docs: list[Document], batch_size: int = 256,
                         progress: Callable[[int, int], None] | None = None) -> IngestReport:
        """Chunk, embed and store documents, embedding many chunks per model call.

        One model call per batch is much faster than one per document. A
        document's chunks always stay in one batch, because ``store.add``
        replaces everything from a source it sees again. ``progress`` is
        called after each batch with (chunks embedded, total chunks).
        """
        report = IngestReport()
        store = self._open_store()
        chunked = [c for c in (chunk_fixed(d, self.config.chunk_size, self.config.chunk_overlap)
                               for d in docs) if c]  # chunking is cheap; embedding is the slow part
        total = sum(len(c) for c in chunked)
        batch: list[Chunk] = []

        def flush() -> None:
            if batch:
                vectors = self.embedder.embed_documents([c.text for c in batch])
                store.add(batch, vectors, embedder_name=self.embedder.name)
                report.chunks += len(batch)
                batch.clear()
                if progress:
                    progress(report.chunks, total)

        for chunks in chunked:
            batch.extend(chunks)
            report.documents += 1
            if len(batch) >= batch_size:
                flush()
        flush()
        store.save()
        return report

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        store = NumpyStore.load(self.config.index_dir)  # raises IndexNotFoundError
        query = self.embedder.embed_query(question)
        return store.search(query, k or self.config.top_k, embedder_name=self.embedder.name)

    def ask(self, question: str, k: int | None = None) -> Answer:
        return generate_answer(question, self.search(question, k), self.llm)

    def sources(self) -> dict[str, int]:
        return NumpyStore.load(self.config.index_dir).sources()
