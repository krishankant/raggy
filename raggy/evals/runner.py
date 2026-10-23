"""Run a suite against the pipeline and record everything as one JSON-able dict.

For each question: retrieve (timed), score retrieval against the gold spans,
optionally generate an answer (timed), detect refusals, and ask the judge.
"""

from __future__ import annotations

import json
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
from raggy.pipeline import Raggy
from raggy.types import Document

SAVED_HITS = 10  # retrieved items kept per question in the run file


def prepare_index(suite: Suite, raggy: Raggy, docs: list[Document] | None = None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Build the suite's index, or reuse it if nothing that shapes it has changed."""
    c = raggy.config
    key = {"fingerprint": suite.fingerprint(), "embedder": raggy.embedder.name,
           "chunk_size": c.chunk_size, "chunk_overlap": c.chunk_overlap}
    key_file = c.index_dir / "key.json"
    if key_file.exists() and (c.index_dir / "meta.json").exists():
        if json.loads(key_file.read_text()) == key:
            return {"cached": True}
    if c.index_dir.exists():
        shutil.rmtree(c.index_dir)  # stale: built from other data or settings
    t0 = time.perf_counter()
    report = raggy.ingest_documents(docs if docs is not None else suite.documents(), progress=progress)
    seconds = time.perf_counter() - t0
    key_file.write_text(json.dumps(key, indent=2))
    return {"cached": False, "documents": report.documents, "chunks": report.chunks,
            "seconds": round(seconds, 2)}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


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
    for stage in ("retrieve", "generate"):
        times = [r["latency_ms"].get(stage) for r in records]
        m[f"{stage}_ms_p50"] = percentile(times, 50)
        m[f"{stage}_ms_p95"] = percentile(times, 95)
    return m


def run_suite(suite: Suite, raggy: Raggy, judge: Judge | None, *, answers: bool = True,
              limit: int | None = None, label: str = "",
              progress: Callable[[int, int], None] | None = None,
              ingest_progress: Callable[[int, int], None] | None = None) -> dict:
    answers = answers and suite.answers
    judge = judge if answers else None
    docs = suite.documents()
    questions = suite.questions()[:limit] if limit else suite.questions()
    gold = resolve_gold(questions, docs)  # fail now, not after 300 model calls
    ingest = prepare_index(suite, raggy, docs, ingest_progress)

    n = max(suite.ks)
    fetch = n * 3 if suite.unit == "document" else n  # extra chunks to fill n distinct documents
    records = []
    for i, q in enumerate(questions, start=1):
        rec: dict = {"id": q.id, "question": q.question, "answerable": q.answerable,
                     "tags": q.tags, "latency_ms": {}}
        t0 = time.perf_counter()
        hits = raggy.search(q.question, k=fetch)
        rec["latency_ms"]["retrieve"] = round((time.perf_counter() - t0) * 1000, 2)
        ranked = (collapse_to_documents(hits) if suite.unit == "document" else hits)[:n]
        covers = [chunk_gold(h.chunk, gold[q.id]) for h in ranked]
        rec["retrieved"] = [{"chunk_id": h.chunk.id, "source": h.chunk.source, "score": round(h.score, 4),
                             "gold": sorted(g)} for h, g in zip(ranked[:SAVED_HITS], covers)]
        if q.answerable:
            rec["retrieval"] = retrieval_metrics(covers, len(gold[q.id]), suite.ks)
        if answers:
            context = hits[: raggy.config.top_k]
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
            "top_k": raggy.config.top_k,
            "ks": list(suite.ks),
            "unit": suite.unit,
            "llm": raggy.llm.name if answers else None,
            "judge": judge.name if judge else None,
            "judge_version": JUDGE_VERSION if judge else None,
        },
        "ingest": ingest,
        "metrics": _aggregate(records),
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
