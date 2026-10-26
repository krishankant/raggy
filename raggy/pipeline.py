"""The whole RAG pipeline in one place.

    ingest:  paths/URLs ─► registry: what changed? ─► load ─► dedupe ─► chunk ─► embed ─► store
    ask:     question ─► embed ─► search store ─► prompt + LLM ─► cited answer
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from raggy.chunking import chunk_document
from raggy.config import Config
from raggy.embed.base import Embedder
from raggy.embed.cache import CACHE_NAME as EMBED_CACHE, CachedEmbedder, EmbeddingCache
from raggy.generate.answer import generate_answer
from raggy.generate.prompt import NOT_FOUND
from raggy.generate.llm import LLM, make_llm
from raggy.ingest.dedupe import DuplicateIndex, content_hash, minhash
from raggy.ingest.fetch import fetch_url, is_url
from raggy.ingest.loaders import LoadError, load_file, load_path
from raggy.ingest.normalize import NORMALIZE_VERSION
from raggy.ingest.registry import Record, Registry
from raggy.ingest.walk import iter_files
from raggy.retrieve.fusion import rrf
from raggy.store.base import Filters, IndexNotFoundError, VectorStore, make_store
from raggy.types import Answer, Chunk, Document, Hit, embed_text  # noqa: F401 (embed_text re-exported)


@dataclass
class IngestReport:
    documents: int = 0  # documents embedded this run
    chunks: int = 0  # chunks embedded this run
    skipped: list[str] = field(default_factory=list)  # targets that don't exist
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    duplicates: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (source, reason)
    context_calls: int = 0  # contextual retrieval: LLM calls made this run
    context_cached: int = 0
    context_failed: int = 0
    embedded: int = 0  # texts the model embedded this run
    embed_cached: int = 0  # texts whose vectors came from the embedding cache


def pipeline_key(config: Config, embedder_name: str) -> str:
    """Every setting that changes what a source's chunks look like. A source
    indexed under a different key is re-indexed even if its bytes are the same."""
    return json.dumps({"embedder": embedder_name, "chunking": config.chunk_settings().key(),
                       "chunk_context": config.chunk_context, "html_extract": config.html_extract,
                       "normalize": NORMALIZE_VERSION, "dedupe": config.dedupe, "store": config.store},
                      sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signature(rec: Record) -> np.ndarray | None:
    return None if rec.signature is None else np.frombuffer(rec.signature, dtype=np.uint64)


class Raggy:
    def __init__(self, config: Config, embedder: Embedder | None = None, llm: LLM | None = None):
        self.config = config
        self._embedder = embedder
        self._llm = llm
        self._store: VectorStore | None = None  # opened once for searching; any write closes it
        self._reranker = None  # built on first use, like the embedder
        self._other_rerankers: dict[str, object] = {}
        self._transform_cache = None
        self.transform_stats = {"calls": 0, "cached": 0, "failed": 0}
        self.last_timings: dict[str, float] = {}  # ms per stage of the latest search

    # Models are built on first use: `raggy sources` shouldn't load an
    # embedding model, and `raggy search` shouldn't need an LLM.
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from raggy.embed.fastembed_embedder import FastEmbedEmbedder

            self._embedder = FastEmbedEmbedder(self.config.embed_model)
        if self.config.embed_cache and not isinstance(self._embedder, CachedEmbedder):
            self._embedder = CachedEmbedder(self._embedder,
                                            EmbeddingCache(Path(self.config.index_dir) / EMBED_CACHE))
        return self._embedder

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = make_llm(self.config.llm_provider, self.config.llm_model)
        return self._llm

    def _open_store(self) -> VectorStore:
        self._store = None  # we're about to write: the next search must see the new rows
        return make_store(self.config, create=True)

    def _search_store(self) -> VectorStore:
        """The store, opened once and kept: re-opening per question would re-read the
        numpy store's whole vector file (113 MB at FiQA scale) for every search."""
        if self._store is None:
            self._store = make_store(self.config, create=False)
        return self._store

    def _registry(self) -> Registry:
        return Registry(Path(self.config.index_dir) / "registry.sqlite")

    @property
    def _embedder_name(self) -> str:
        # Don't load a model just to learn its name: an unchanged re-ingest embeds nothing.
        return self._embedder.name if self._embedder is not None else self.config.embed_model

    def ingest(self, paths: list[str], full: bool = False,
               progress: Callable[[int, int], None] | None = None) -> IngestReport:
        """Bring the index up to date with files, folders and URLs.

        New sources are added, changed ones re-indexed, unchanged ones skipped
        (by size and mtime, then by content hash), and files deleted from a
        folder that was walked are removed. ``full=True`` re-reads every file
        instead of trusting size and mtime. A target that doesn't exist is
        reported and removes nothing: an unmounted drive is not a deletion.
        """
        reg = self._registry()
        try:
            return self._sync(reg, paths, full, progress)
        finally:
            reg.close()  # on an exception this rolls back the uncommitted rows

    def _sync(self, reg: Registry, paths: list[str], full: bool,
              progress: Callable[[int, int], None] | None) -> IngestReport:
        report = IngestReport()
        store = self._open_store()
        key, opts = pipeline_key(self.config, self._embedder_name), self.config.load_options()
        status: dict[str, str] = {}  # source -> added | updated | unchanged | duplicate | failed | removed
        pending: dict[str, tuple[Document, str, Record | None, tuple | None]] = {}  # to (re)index
        known = {r.source for r in reg.all()}

        def fail(source: str, reason: str) -> None:
            status[source] = "failed"
            report.failed.append((source, reason))

        def consider(doc: Document, rec: Record | None, stat: tuple | None, force: bool = False) -> None:
            digest = content_hash(doc.text)
            if (not force and rec is not None and rec.content_hash == digest and rec.pipeline_key == key
                    and not self._orphaned(rec, known)):
                if stat is not None:  # same content, new mtime: remember it so the next run skips the read
                    reg.upsert(replace(rec, size=stat[0], mtime_ns=stat[1]))
                status[doc.source] = "unchanged"
                return
            pending[doc.source] = (doc, digest, rec, stat)

        def consider_file(path: Path, force: bool = False) -> None:
            source = str(path.resolve())
            rec = reg.get(source)
            try:
                st = path.stat()
                stat = (st.st_size, st.st_mtime_ns)
                if (not full and not force and rec is not None and rec.pipeline_key == key
                        and (rec.size, rec.mtime_ns) == stat and not self._orphaned(rec, known)):
                    status[source] = "unchanged"
                    return
                doc = load_file(path, opts)
            except (LoadError, OSError) as e:
                return fail(source, str(e))
            consider(doc, rec, stat, force)

        def consider_url(url: str, force: bool = False) -> None:
            try:
                doc = fetch_url(url, opts)
            except LoadError as e:
                return fail(url, str(e))
            consider(doc, reg.get(url), None, force)

        for target in paths:
            if is_url(target):
                consider_url(target)
                continue
            path = Path(target).expanduser()
            if not path.exists():
                report.skipped.append(target)
            elif path.is_file():
                consider_file(path)
            else:
                files = iter_files(path)
                for f in files:
                    consider_file(f)
                walked = {str(f.resolve()) for f in files}
                for rec in reg.under(str(path.resolve())):
                    if rec.source not in walked:
                        status[rec.source] = "removed"

        # A canonical copy that is going away or changing: re-examine its
        # duplicates now, or their content would silently leave the index.
        gone = {s for s, st in status.items() if st == "removed"} | {s for s, p in pending.items() if p[2]}
        for source in sorted(gone):
            for dup in reg.duplicates_of(source):
                if dup.source in status and status[dup.source] != "unchanged" or dup.source in pending:
                    continue
                if dup.kind == "url":
                    consider_url(dup.source, force=True)
                elif Path(dup.source).exists():
                    consider_file(Path(dup.source), force=True)
                else:
                    status[dup.source] = "removed"

        removed = {s for s, st in status.items() if st == "removed"}
        index = DuplicateIndex()
        if self.config.dedupe:
            for rec in reg.all():
                if rec.duplicate_of is None and rec.source not in removed and rec.source not in pending:
                    index.add(rec.source, rec.content_hash, _signature(rec))

        to_embed: list[Document] = []
        rows: dict[str, Record] = {}
        for source in sorted(pending, key=lambda s: (len(s), s)):  # shorter path wins a tie
            doc, digest, rec, stat = pending[source]
            sig = minhash(doc.text)
            original = index.find(digest, sig) if self.config.dedupe else None
            if original is None:
                index.add(source, digest, sig)
                to_embed.append(doc)
                status[source] = "updated" if rec is not None else "added"
            else:
                status[source] = "duplicate"
            m = doc.metadata
            rows[source] = Record(
                source=source, kind=m.get("kind", "file"), content_hash=digest,
                size=stat[0] if stat else None, mtime_ns=stat[1] if stat else None, pipeline_key=key,
                signature=None if sig is None else sig.tobytes(), duplicate_of=original,
                title=m.get("title"), doc_type=m.get("type"), modified=m.get("modified"),
                chunks=0, ingested_at=_now())

        for source, st in status.items():
            if st in ("removed", "duplicate"):
                store.remove_source(source)  # the index only: never the file
        done = self.ingest_documents(to_embed, progress=progress, store=store, save=False)
        report.documents, report.chunks = done.documents, done.chunks
        report.context_calls, report.context_cached = done.context_calls, done.context_cached
        report.context_failed = done.context_failed
        report.embedded, report.embed_cached = done.embedded, done.embed_cached
        counts = store.sources()
        for source, row in rows.items():
            reg.upsert(replace(row, chunks=counts.get(source, 0)))
        for source in removed:
            reg.delete(source)
        store.save()  # first the store ...
        store.finish(build_ann=self.config.ann == "on")
        reg.commit()  # ... then the registry: a crash between the two means redo, never loss

        for st in status.values():
            if st == "duplicate":
                report.duplicates += 1
            elif st != "failed":
                setattr(report, st, getattr(report, st) + 1)
        return report

    @staticmethod
    def _orphaned(rec: Record, known: set[str]) -> bool:
        """A duplicate whose original is no longer registered must be indexed itself."""
        return rec.duplicate_of is not None and rec.duplicate_of not in known

    def remove(self, target: str) -> int:
        """Remove a source, or everything under a folder or URL prefix, from the
        index and the registry. Files on disk are never touched."""
        self._store = None
        store = make_store(self.config, create=False)
        reg = self._registry()
        try:
            return self._remove(store, reg, target)
        finally:
            reg.close()

    @staticmethod
    def _remove(store: VectorStore, reg: Registry, target: str) -> int:
        if is_url(target):
            exact, prefix = target, target.rstrip("/") + "/"
        else:
            exact = str(Path(target).expanduser().resolve())
            prefix = exact.rstrip(os.sep) + os.sep
        sources = {r.source for r in reg.all()} | set(store.sources())
        matches = sorted(s for s in sources if s == exact or s.startswith(prefix))
        for s in matches:
            store.remove_source(s)
            reg.delete(s)
        store.save()
        reg.commit()
        return len(matches)

    def documents(self) -> list[Record]:
        """Registry rows, for `raggy sources`. Empty for an index built before week 3."""
        reg = self._registry()
        try:
            return reg.all()
        finally:
            reg.close()

    def ingest_documents(self, docs: list[Document], batch_size: int = 256,
                         progress: Callable[[int, int], None] | None = None, *,
                         store: VectorStore | None = None, save: bool = True) -> IngestReport:
        """Chunk, embed and store documents, embedding many chunks per model call.

        One model call per batch is much faster than one per document. A
        document's chunks always stay in one batch, because ``store.add``
        replaces everything from a source it sees again. ``progress`` is
        called after each batch with (chunks embedded, total chunks).
        """
        report = IngestReport()
        store = store if store is not None else self._open_store()
        cached = self._embedder if isinstance(self._embedder, CachedEmbedder) else None
        hits0, misses0 = (cached.hits, cached.misses) if cached else (0, 0)
        settings = self.config.chunk_settings()
        count = lambda text: self.embedder.count_tokens(text)  # noqa: E731 (fixed never calls it)
        pairs = [(d, c) for d, c in ((d, chunk_document(d, settings, count)) for d in docs) if c]
        for d, cs in pairs:  # document metadata rides on every chunk, so searches can filter on it
            meta = {k: d.metadata[k] for k in ("type", "title", "modified") if d.metadata.get(k)}
            for c in cs:
                c.meta = dict(meta)
        if self.config.chunk_context == "llm" and pairs:
            from raggy.chunking.contextual import CACHE_NAME, ContextCache, add_llm_context

            cache = ContextCache(Path(self.config.index_dir) / CACHE_NAME)
            for d, cs in pairs:
                st = add_llm_context(d, cs, self.llm, cache, count, self.embedder.max_tokens)
                report.context_calls += st.calls
                report.context_cached += st.cached
                report.context_failed += st.failed
                cache.commit()  # paid-for results are kept even if a later step fails
        chunked = [cs for _, cs in pairs]
        total = sum(len(c) for c in chunked)
        batch: list[Chunk] = []

        def flush() -> None:
            if batch:
                texts = [embed_text(c) for c in batch]
                for c, t in zip(batch, texts):
                    n = self.embedder.count_tokens(t)
                    if n > self.embedder.max_tokens:  # the model would silently drop the rest
                        raise ValueError(f"chunk {c.id} is {n} tokens; {self.embedder.name} reads at most "
                                         f"{self.embedder.max_tokens}. Lower RAGGY_CHUNK_TOKENS.")
                vectors = self.embedder.embed_documents(texts)
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
        cached = self._embedder if isinstance(self._embedder, CachedEmbedder) else cached
        if cached is not None:
            cached.cache.commit()  # vectors are paid for: keep them even if a later step fails
            report.embedded, report.embed_cached = cached.misses - misses0, cached.hits - hits0
        else:
            report.embedded = report.chunks
        if save:
            store.save()
            store.finish(build_ann=self.config.ann == "on")
        return report

    def _transform(self, kind: str, question: str):
        from raggy.retrieve.transforms import CACHE_NAME, TransformCache, transform

        if self._transform_cache is None:
            self._transform_cache = TransformCache(Path(self.config.index_dir) / CACHE_NAME)
        r = transform(kind, question, self.llm, self._transform_cache)
        self._transform_cache.commit()
        self.transform_stats["cached" if r.cached else "calls"] += 1
        self.transform_stats["failed"] += int(r.failed)
        return r

    def context(self, hits: list[Hit]) -> list[Hit]:
        """The passages the answer model reads: the hits, packed if RAGGY_PACK is on."""
        if not self.config.pack:
            return hits
        from raggy.generate.pack import pack

        return pack(hits, lambda text: self.embedder.count_tokens(text), self.config.context_tokens)

    def reranker(self, kind: str | None = None):
        """The configured reranker, or the one named by ``kind`` (a --rerank override)."""
        from raggy.retrieve.rerank import make_reranker

        kind = kind or self.config.rerank
        if kind == self.config.rerank:
            if self._reranker is None:
                self._reranker = make_reranker(self.config, lambda: self.llm, kind)
            return self._reranker
        if kind not in self._other_rerankers:
            self._other_rerankers[kind] = make_reranker(self.config, lambda: self.llm, kind)
        return self._other_rerankers[kind]

    def search(self, question: str, k: int | None = None, filters: Filters | None = None,
               exact: bool | None = None, mode: str | None = None, rerank: str | None = None,
               transform: str | None = None) -> list[Hit]:
        """The k best chunks, or with parent/child on, the k best distinct parents.

        ``mode`` is vector (embeddings), keyword (BM25) or hybrid (both, fused by
        rank with RRF); it defaults to RAGGY_RETRIEVAL. ``filters`` restrict both
        sides to some documents before anything is ranked. Each child found is
        then swapped for its parent, and a parent found twice is kept once."""
        store = self._search_store()  # raises IndexNotFoundError
        k = k or self.config.top_k
        mode = mode or self.config.retrieval
        exact = self.config.ann == "off" if exact is None else exact
        timings: dict[str, float] = {}
        start = time.perf_counter()
        kind = transform or self.config.query_transform
        t = self._transform(kind, question) if kind != "none" else None
        if t is not None:
            timings["transform"] = (time.perf_counter() - start) * 1000
        texts = t.queries if t else [question]  # one search text per variant
        if mode == "keyword":
            vectors = [None] * len(texts)
        elif t is not None and t.hyde_passage:  # HyDE: the vector side searches with the invented answer
            vectors = [self.embedder.embed_documents([t.hyde_passage])[0]]
        else:
            vectors = [self.embedder.embed_query(x) for x in texts]

        def one(text: str, vec, want: int) -> list[Hit]:
            if mode == "vector":
                return store.search(vec, want, embedder_name=self.embedder.name, where=filters, exact=exact)
            if mode == "keyword":
                return store.keyword_search(text, want, where=filters)
            n = max(want, self.config.fusion_candidates)
            v = store.search(vec, n, embedder_name=self.embedder.name, where=filters, exact=exact)
            return rrf([v, store.keyword_search(text, n, where=filters)], k=self.config.rrf_k)[:want]

        def ranked(want: int) -> list[Hit]:
            lists = [one(x, v, want) for x, v in zip(texts, vectors)]
            return lists[0] if len(lists) == 1 else rrf(lists, k=self.config.rrf_k)[:want]

        reranker = self.reranker(rerank)
        if reranker is not None:
            first = ranked

            def ranked(want: int) -> list[Hit]:  # noqa: F811 — retrieve a wider pool, rerank, keep `want`
                n = self.config.rerank_candidates
                pool = first(max(n, want))
                t0 = time.perf_counter()
                # Rerank the top n only; deeper candidates keep their retrieval order behind them.
                out = (reranker.rerank(question, pool[:n]) + pool[n:])[:want]
                timings["rerank"] = timings.get("rerank", 0.0) + (time.perf_counter() - t0) * 1000
                return out

        def done(hits: list[Hit]) -> list[Hit]:
            total = (time.perf_counter() - start) * 1000
            timings["retrieve"] = total - timings.get("transform", 0.0) - timings.get("rerank", 0.0)
            self.last_timings = {k: round(v, 2) for k, v in timings.items()}
            return hits

        if not self.config.parent_tokens:
            return done(ranked(k))
        # Children of one parent often rank together, so k*4 children can hold fewer
        # than k distinct parents. Fetch more until there are k, or nothing more to fetch.
        want = k * 4
        while True:
            hits = ranked(want)
            out, seen = [], set()
            for h in hits:
                c = h.chunk
                if c.parent_text is not None:
                    c = Chunk(id=f"{c.source}#p{c.parent_start}", source=c.source, text=c.parent_text,
                              index=c.index, start=c.parent_start, end=c.parent_end, context=c.context,
                              meta=c.meta)
                if (c.source, c.start) not in seen:
                    seen.add((c.source, c.start))
                    out.append(Hit(chunk=c, score=h.score))
            if len(out) >= k or len(hits) < want or want >= k * 64:
                return done(out[:k])
            want *= 2

    def ask(self, question: str, k: int | None = None, filters: Filters | None = None,
            mode: str | None = None, rerank: str | None = None, transform: str | None = None) -> Answer:
        hits = self.search(question, k, filters, mode=mode, rerank=rerank, transform=transform)
        return self.answer_from(question, self.context(hits))

    def gate_key(self) -> str:
        """Scores mean different things per retrieval mode and reranker: each gets its own threshold."""
        c = self.config
        return c.retrieval + (f"+{c.rerank}" if c.rerank != "none" else "")

    def answer_from(self, question: str, passages: list[Hit]) -> Answer:
        """Generate (and, if configured, verify) an answer from these passages."""
        c = self.config
        threshold = c.refusal_gate.get(self.gate_key())
        if threshold is not None and passages and max(h.score for h in passages) < threshold:
            # Nothing retrieved is close enough to be the answer: say so without paying for a call.
            return Answer(question=question, text=NOT_FOUND, citations=[], hits=passages, verification="gated")
        return generate_answer(question, passages, self.llm, check=c.citation_check,
                               threshold=c.citation_threshold, action=c.citation_action)

    def sources(self) -> dict[str, int]:
        return make_store(self.config, create=False).sources()
