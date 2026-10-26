"""Run a suite against the pipeline and record everything as one JSON-able dict.

For each question: retrieve (timed), score retrieval against the gold spans,
optionally generate an answer (timed), detect refusals, and ask the judge.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from typing import Callable

from raggy.evals.dataset import chunk_gold, resolve_gold
from raggy.evals.judge import JUDGE_VERSION, Judge, Verdict, is_refusal
from raggy.evals.metrics import collapse_to_documents, mean, percentile, retrieval_metrics
from raggy.evals.suites import Suite
from raggy.generate.answer import generate_answer
from raggy.generate.llm import LLMError
from raggy.generate.prompt import source_text
from raggy.ingest.dedupe import drop_duplicates
from raggy.ingest.normalize import NORMALIZE_VERSION
from raggy.store.lance_store import FTS_VERSION
from raggy.pipeline import Raggy
from raggy.types import Document

STAGES = ("transform", "retrieve", "rerank", "generate")  # in the order a question goes through them
SAVED_HITS = 10  # retrieved items kept per question in the run file
# Paid-for work that doesn't depend on the settings that make an index stale:
# a rebuild keeps these files, and a folder holding only them is raggy's own.
CACHE_FILES = {"context-cache.sqlite", "embed-cache.sqlite", "transform-cache.sqlite"}


class UnsafeIndexDirError(RuntimeError):
    """The eval index folder holds files raggy didn't put there, so it won't delete it."""


def _index_exists(config) -> bool:
    from raggy.store.base import IndexNotFoundError, make_store

    try:
        return len(make_store(config, create=False)) > 0
    except IndexNotFoundError:
        return False


def prepare_index(suite: Suite, raggy: Raggy, docs: list[Document] | None = None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Build the suite's index, or reuse it if nothing that shapes it has changed."""
    c = raggy.config
    key = {"fingerprint": suite.fingerprint(), "embedder": raggy.embedder.name,
           "chunk_size": c.chunk_size, "chunk_overlap": c.chunk_overlap,
           "html_extract": c.html_extract, "dedupe": c.dedupe and suite.dedupe, "normalize": NORMALIZE_VERSION,
           "chunking": c.chunk_settings().key(), "chunk_context": c.chunk_context, "store": c.store,
           "ann": c.ann, "fts": FTS_VERSION}
    key_file = c.index_dir / "key.json"
    if key_file.exists() and _index_exists(c):
        if json.loads(key_file.read_text()) == key:
            return {"cached": True}
    if c.index_dir.exists():
        # Stale: built from other data or settings. Only ever delete a folder
        # this function created (it has key.json) or an empty one; anything
        # else may be the user's files, and a wrong path must not destroy them.
        if not key_file.exists() and any(p.name not in CACHE_FILES for p in c.index_dir.iterdir()):
            raise UnsafeIndexDirError(
                f"{c.index_dir} isn't empty and wasn't created by `raggy eval`; "
                "refusing to delete it. Check RAGGY_INDEX_DIR, or remove the folder yourself.")
        # Keep the contextual-retrieval cache: it holds paid-for LLM output that
        # doesn't depend on the settings that made this index stale.
        for child in c.index_dir.iterdir():
            if child.name in CACHE_FILES:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    t0 = time.perf_counter()
    if docs is None:
        docs = suite.documents(c.load_options())
        if c.dedupe and suite.dedupe:
            docs, _ = drop_duplicates(docs)
    report = raggy.ingest_documents(docs, progress=progress)
    seconds = time.perf_counter() - t0
    key_file.write_text(json.dumps(key, indent=2))
    return {"cached": False, "documents": report.documents, "chunks": report.chunks,
            "seconds": round(seconds, 2), "embedded": report.embedded, "embed_cached": report.embed_cached,
            "context_calls": report.context_calls,
            "context_cached": report.context_cached, "context_failed": report.context_failed}


def _relative(sources: list[str], docs: list[Document]) -> list[str]:
    """Show file sources relative to the corpus folder, so run files are portable."""
    try:
        root = os.path.commonpath([d.source for d in docs] + sources)
    except ValueError:
        return sources
    return [os.path.relpath(s, root) if os.path.isabs(s) else s for s in sources]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _with_budgets(m: dict, budgets: dict) -> dict:
    m["over_budget"] = [s for s in STAGES if m.get(f"{s}_ms_p95") is not None and m[f"{s}_ms_p95"] > budgets[s]]
    return m


def _aggregate(records: list[dict]) -> dict:
    answerable = [r for r in records if r["answerable"]]
    order = {"recall": 0, "mrr": 1, "ndcg": 2}  # recall first: did we find it at all?
    keys = sorted({k for r in answerable for k in r.get("retrieval", {})},
                  key=lambda k: (order[k.split("@")[0]], int(k.split("@")[1]) if "@" in k else 0))
    m: dict = {k: mean(r["retrieval"][k] for r in answerable if "retrieval" in r) for k in keys}
    judged = [r["judge"] for r in answerable if r.get("judge")]
    m["correctness"] = mean(j["correctness"] for j in judged)
    m["faithfulness"] = mean(None if j["faithful"] is None else float(j["faithful"]) for j in judged)
    m["citation_precision"] = mean(j["citation_precision"] for j in judged)
    answered = [r for r in records if "refused" in r]
    m["refusal_accuracy"] = mean(float(r["refused"] != r["answerable"]) for r in answered)
    m["unanswerable_refused"] = mean(float(r["refused"]) for r in answered if not r["answerable"])
    m["answerable_refused"] = mean(float(r["refused"]) for r in answered if r["answerable"])
    for k in sorted({k for r in records for k in r.get("ann_overlap", {})}, key=lambda s: int(s[1:])):
        m[f"ann_overlap{k}"] = mean(r["ann_overlap"][k] for r in records if "ann_overlap" in r)
    tokens = [r.get("context_tokens") for r in records]
    m["context_tokens_p50"] = percentile(tokens, 50)  # a bigger context can look like better retrieval
    m["context_tokens_p95"] = percentile(tokens, 95)
    for stage in STAGES:
        times = [r["latency_ms"].get(stage) for r in records]
        if stage in ("retrieve", "generate") or any(t is not None for t in times):
            m[f"{stage}_ms_p50"] = percentile(times, 50)
            m[f"{stage}_ms_p95"] = percentile(times, 95)
    return m


def run_suite(suite: Suite, raggy: Raggy, judge: Judge | None, *, answers: bool = True,
              limit: int | None = None, label: str = "",
              progress: Callable[[int, int], None] | None = None,
              ingest_progress: Callable[[int, int], None] | None = None) -> dict:
    answers = answers and suite.answers
    judge = judge if answers else None
    docs = suite.documents(raggy.config.load_options())
    dups: dict[str, str] = {}
    if raggy.config.dedupe and suite.dedupe:
        docs, dups = drop_duplicates(docs)
    questions = suite.questions()[:limit] if limit else suite.questions()
    gold = resolve_gold(questions, docs)  # fail now, not after 300 model calls
    ingest = prepare_index(suite, raggy, docs, ingest_progress)
    ingest |= {"documents": len(docs), "duplicates": _relative(sorted(dups), docs)}

    n = max(suite.ks)
    fetch = n * 3 if suite.unit == "document" else n  # extra chunks to fill n distinct documents
    records = []
    for i, q in enumerate(questions, start=1):
        rec: dict = {"id": q.id, "question": q.question, "answerable": q.answerable,
                     "tags": q.tags, "latency_ms": {}}
        hits = raggy.search(q.question, k=fetch)
        rec["latency_ms"].update(raggy.last_timings)  # transform / retrieve / rerank, ms
        ranked = (collapse_to_documents(hits) if suite.unit == "document" else hits)[:n]
        covers = [chunk_gold(h.chunk, gold[q.id]) for h in ranked]
        if raggy.config.ann == "on":  # how much of the exact answer did the approximate search find?
            exact = raggy.search(q.question, k=fetch, exact=True)
            exact = (collapse_to_documents(exact) if suite.unit == "document" else exact)[:n]
            key = (lambda h: h.chunk.source) if suite.unit == "document" else (lambda h: h.chunk.id)
            rec["ann_overlap"] = {
                f"@{k}": len({key(h) for h in ranked[:k]} & {key(h) for h in exact[:k]}) / max(1, len(exact[:k]))
                for k in suite.ks}
        rec["retrieved"] = [{"chunk_id": h.chunk.id, "source": h.chunk.source, "score": round(h.score, 4),
                             "gold": sorted(g)} for h, g in zip(ranked[:SAVED_HITS], covers)]
        if q.answerable:
            rec["retrieval"] = retrieval_metrics(covers, len(gold[q.id]), suite.ks)
        context = hits[: raggy.config.top_k]  # what the answer model is (or would be) given
        rec["context_tokens"] = sum(raggy.embedder.count_tokens(source_text(h)) for h in context)
        if answers:
            t0 = time.perf_counter()
            try:
                answer = generate_answer(q.question, context, raggy.llm)
            except LLMError as e:
                rec["error"] = str(e)
            else:
                rec["latency_ms"]["generate"] = round((time.perf_counter() - t0) * 1000, 2)
                rec["answer"] = answer.text
                rec["citations"] = [c.number for c in answer.citations]
                rec["refused"] = is_refusal(answer.text)
                if q.answerable and judge is not None:
                    verdict = (Verdict(correctness=0.0, reason="refused an answerable question")
                               if rec["refused"] else judge.judge(q, answer))
                    rec["judge"] = asdict(verdict)
        records.append(rec)
        if progress:
            progress(i, len(questions))

    tags = sorted({t for q in questions for t in q.tags})
    return {
        "schema": 1,
        "suite": suite.name,
        "label": label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": {
            "embed_model": raggy.embedder.name,
            "chunk_size": raggy.config.chunk_size,
            "chunk_overlap": raggy.config.chunk_overlap,
            "html_extract": raggy.config.html_extract,
            "chunking": raggy.config.chunk_settings().key(),
            "chunk_context": raggy.config.chunk_context,
            "store": raggy.config.store,
            "ann": raggy.config.ann,
            "ann_nprobes": raggy.config.ann_nprobes,
            "ann_refine": raggy.config.ann_refine,
            "retrieval": raggy.config.retrieval,
            "rrf_k": raggy.config.rrf_k,
            "fusion_candidates": raggy.config.fusion_candidates,
            "fts_version": FTS_VERSION,
            "rerank": raggy.config.rerank,
            "rerank_model": raggy.config.rerank_model if raggy.config.rerank == "local" else None,
            "rerank_candidates": raggy.config.rerank_candidates,
            "query_transform": raggy.config.query_transform,
            "budgets": dict(raggy.config.budgets),
            "dedupe": raggy.config.dedupe and suite.dedupe,
            "top_k": raggy.config.top_k,
            "ks": list(suite.ks),
            "unit": suite.unit,
            "llm": raggy.llm.name if answers else None,
            "judge": judge.name if judge else None,
            "judge_version": JUDGE_VERSION if judge else None,
        },
        "ingest": ingest,
        "metrics": _with_budgets(_aggregate(records), raggy.config.budgets),
        "by_tag": {t: _aggregate([r for r in records if t in r["tags"]]) | {"n": sum(t in r["tags"] for r in records)}
                   for t in tags},
        "counts": {
            "questions": len(records),
            "answerable": sum(r["answerable"] for r in records),
            "unanswerable": sum(not r["answerable"] for r in records),
            "errors": sum("error" in r for r in records),
            "judge_errors": sum(bool(r.get("judge", {}).get("error")) for r in records),
        },
        "questions": records,
    }
