"""Command line: raggy ingest | search | ask | sources | eval | compare."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

from raggy.config import Config, ConfigError
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.store.base import EmbedderMismatchError, IndexNotFoundError, KeywordSearchUnavailable


def _build_raggy(config: Config) -> Raggy:  # tests replace this to inject fakes
    return Raggy(config)


def _build_judge(config: Config):  # tests replace this too
    from raggy.evals.judge import Judge
    from raggy.generate.llm import make_llm

    return Judge(make_llm(config.judge_provider, config.judge_model))


def _positive_int(value: str) -> int:
    n = int(value) if value.lstrip("-").isdigit() else 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a whole number of 1 or more, got {value!r}")
    return n


def _iso_date(value: str) -> str:
    from datetime import datetime

    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a date like 2026-01-31, got {value!r}") from None
    return value


def _add_filters(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("filters")
    g.add_argument("--type", dest="doc_type", choices=["markdown", "text", "html", "pdf", "docx", "pptx", "xlsx", "csv", "image", "audio", "video", "subtitles"],
                   help="only documents of this type")
    g.add_argument("--source", help="only documents inside this folder or under this URL")
    g.add_argument("--after", type=_iso_date, help="only documents modified on or after this date")
    p.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default=None,
                   help="retrieval: embeddings, BM25 keywords, or both fused (default: RAGGY_RETRIEVAL)")
    p.add_argument("--transform", choices=["none", "rewrite", "multi", "hyde"], default=None,
                   help="rewrite the question with the chat model before searching (default: RAGGY_QUERY_TRANSFORM)")
    p.add_argument("--rerank", choices=["none", "local", "llm"], default=None,
                   help="rerank the candidates with a cross-encoder or the chat model (default: RAGGY_RERANK)")


def _filters(args):
    from raggy.ingest.fetch import is_url
    from raggy.store.base import Filters

    prefix = args.source
    if prefix and not is_url(prefix):
        path = Path(prefix).expanduser().resolve()
        prefix = str(path) + (os.sep if path.is_dir() else "")  # docs/ must not also match docs2/
    f = Filters(doc_type=args.doc_type, source_prefix=prefix, modified_after=args.after)
    return f or None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="raggy", description="Ask questions about your documents.")
    sub = p.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="index files, folders or URLs (re-runs only redo what changed)")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--full", action="store_true", help="re-read every file instead of trusting size and mtime")
    s = sub.add_parser("search", help="show the chunks retrieval finds, with scores")
    s.add_argument("query")
    s.add_argument("-k", type=_positive_int, default=None)
    _add_filters(s)
    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("question")
    a.add_argument("-k", type=_positive_int, default=None)
    a.add_argument("--show-context", action="store_true", help="also print the retrieved chunks")
    a.add_argument("--timings", action="store_true", help="print how long each stage took")
    a.add_argument("--stream", action="store_true", help="print the answer as it's written")
    a.add_argument("--agent", action="store_true", help="let the model decide what to search next (multi-hop)")
    tr = sub.add_parser("trace", help="show the latest traced answers (RAGGY_TRACE=on)")
    tr.add_argument("-n", type=_positive_int, default=5)
    _add_filters(a)
    sub.add_parser("sources", help="list indexed sources")
    im = sub.add_parser("images", help="find pictures by a description or by another picture (needs RAGGY_IMAGE_SEARCH)")
    im.add_argument("query", nargs="?", help="what the picture shows")
    im.add_argument("--like", type=Path, help="a picture to find similar ones to")
    im.add_argument("-k", type=int, default=5)
    rm = sub.add_parser("remove", help="drop a source, or everything under a folder or URL, from the index")
    rm.add_argument("target")
    from raggy.evals.suites import SUITES

    e = sub.add_parser("eval", help="measure retrieval and answer quality on an eval suite")
    e.add_argument("--suite", choices=sorted(SUITES), default="fictional")
    e.add_argument("--retrieval-only", action="store_true", help="skip answers: no LLM calls, free")
    e.add_argument("--no-judge", action="store_true", help="generate answers but don't grade them")
    e.add_argument("--limit", type=_positive_int, default=None, help="only the first N questions")
    e.add_argument("--label", default="", help="name for this run, e.g. week01")
    e.add_argument("--out", default="evals/runs", help="folder for run JSON files")
    c = sub.add_parser("compare", help="show metric changes between two saved runs")
    c.add_argument("base")
    c.add_argument("new")
    return p


def display_source(source: str) -> str:
    """A URL is shown whole; a file by its name."""
    from raggy.ingest.fetch import is_url

    return source if is_url(source) else Path(source).name


def _print_hits(hits) -> None:
    for n, h in enumerate(hits, start=1):
        text = " ".join(h.chunk.text.split())
        print(f"[{n}] {h.score:.3f}  {display_source(h.chunk.source)} (chunk {h.chunk.index})")
        print(f"    {text[:200]}{'…' if len(text) > 200 else ''}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = Config.from_env()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    raggy = _build_raggy(config)
    try:
        if args.command == "ingest":
            def show(done: int, total: int) -> None:
                print(f"\r  embedded chunks {done}/{total}", end="" if done < total else "\n",
                      file=sys.stderr, flush=True)

            r = raggy.ingest(args.paths, full=args.full, progress=show)
            print(f"Indexed into {raggy.config.index_dir}: {r.added} added, {r.updated} updated, "
                  f"{r.unchanged} unchanged, {r.removed} removed, {r.duplicates} duplicates, "
                  f"{len(r.failed)} failed ({r.chunks} chunks embedded)")
            if r.embedded or r.embed_cached:
                print(f"embeddings: {r.embedded} computed, {r.embed_cached} from cache")
            if raggy.config.chunk_context == "llm":
                print(f"contextual retrieval: {r.context_calls} LLM calls ({r.context_cached} cached, "
                      f"{r.context_failed} failed)")
            if raggy.config.images == "vision" and (r.image_calls or r.image_cached):
                print(f"images: {r.image_calls} described, {r.image_cached} from cache, {r.image_failed} failed"
                      + (" (those use their OCR text only; ingest again to retry)" if r.image_failed else ""))
            for source, reason in r.failed:
                print(f"failed: {reason}", file=sys.stderr)
            for path in r.skipped:
                print(f"not found: {path} (nothing removed; `raggy remove {path}` drops what it held)",
                      file=sys.stderr)
        elif args.command == "search":
            hits = raggy.search(args.query, args.k, _filters(args), mode=args.mode, rerank=args.rerank,
                                transform=args.transform)
            if not hits and _filters(args):
                print("no chunks match these filters", file=sys.stderr)
            _print_hits(hits)
        elif args.command == "ask":
            streamed = []

            def show(piece: str) -> None:
                streamed.append(piece)
                print(piece, end="", flush=True)

            if args.agent:
                raggy.config.agent = True
            answer = raggy.ask(args.question, args.k, _filters(args), mode=args.mode, rerank=args.rerank,
                               transform=args.transform, on_token=show if args.stream else None)
            if args.show_context:
                _print_hits(answer.hits)
                print()
            if args.stream:
                print()
                if answer.text != "".join(streamed).strip():  # the citation check changed it after streaming
                    print(f"\n(after the citation check:)\n{answer.text}")
            else:
                print(answer.text)
            if raggy.last_agent is not None:
                ag = raggy.last_agent
                print(f"\nSearches ({ag.stopped}):")
                for n, step in enumerate(ag.steps, start=1):
                    print(f"  {n}. {step.query}  (+{step.new} new passage{'s' if step.new != 1 else ''})")
                if ag.failures:
                    print(f"⚠ {ag.failures} agent decision(s) failed; it answered from what it had")
            if args.timings:
                print("\nTimings: " + "  ".join(f"{k} {v:.0f} ms" for k, v in raggy.last_timings.items()))
            u = raggy.last_usage
            if answer.cached:
                print("\n(answered from the cache: no model call)")
            elif u.get("calls"):
                spent = f", {raggy.last_cost:.4f}" if raggy.last_cost is not None else ""
                print(f"\nTokens: {u.get('input', 0)} in ({u.get('cache_read', 0)} cached), "
                      f"{u.get('output', 0)} out{spent}")
            if raggy.trace_error:
                print(f"⚠ couldn't write the trace: {raggy.trace_error}", file=sys.stderr)
            for n in answer.unsupported:
                print(f"⚠ citation [{n}] is not supported by its source ({answer.verification} check)")
            if answer.verification == "failed":
                print("⚠ the citation check failed; citations were not verified")
            if answer.citations:
                print("\nSources:")
                for c in answer.citations:
                    where = f", {c.location}" if c.location else ""
                    print(f"[{c.number}] {display_source(c.source)}{where} — \"{c.snippet[:100]}…\"")
        elif args.command == "trace":
            from raggy.trace import Tracer

            rows = Tracer(Path(raggy.config.index_dir) / "traces").latest(args.n)
            if not rows:
                print("no traces yet: set RAGGY_TRACE=on and ask something")
            for r in rows:
                t = r["timings"]
                print(f"{r['time']}  {t.get('total', 0):.0f} ms  {'cached' if r['cached'] else ''}  {r['question']}")
                for h in r["hits"]:
                    where = f" {h['location']}" if h.get("location") else ""
                    print(f"    {h['score']:.3f}  {display_source(h['source'])}{where}")
                print(f"  → {r['answer'][:160]}")
        elif args.command == "sources":
            counts = raggy.sources()  # raises IndexNotFoundError before anything is ingested
            records = raggy.documents()
            if not records:  # an index built before week 3 has no registry
                for source, count in sorted(counts.items()):
                    print(f"{count:5d} chunks  {source}")
            for rec in records:
                note = f"dup of {display_source(rec.duplicate_of)}" if rec.duplicate_of else f"{rec.chunks:5d} chunks"
                print(f"{note:>16}  {rec.doc_type or '?':8}  {(rec.title or '')[:40]:40}  {rec.source}")
        elif args.command == "images":
            if (args.query is None) == (args.like is None):
                print("give a query or --like PICTURE", file=sys.stderr)
                return 2
            hits = raggy.search_images(text=args.query, image=args.like.read_bytes() if args.like else None, k=args.k)
            if not hits:
                print("no pictures indexed yet: run `raggy ingest` with RAGGY_IMAGE_SEARCH=on")
            for h in hits:
                print(f"{h.score:.3f}  {h.label}  ({display_source(h.source)})")
        elif args.command == "remove":
            n = raggy.remove(args.target)
            print(f"Removed {n} source{'s' if n != 1 else ''} from the index (files on disk are untouched)")
        elif args.command == "eval":
            return _eval(args, raggy.config)
        elif args.command == "compare":
            from raggy.evals.report import format_compare, load_run

            print(format_compare(load_run(args.base), load_run(args.new)))
    except IndexNotFoundError:
        print("No index found — run `raggy ingest <path>` first.", file=sys.stderr)
        return 1
    except (LLMError, EmbedderMismatchError, KeywordSearchUnavailable, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _eval(args, config: Config) -> int:
    from raggy.evals.dataset import GoldNotFoundError
    from raggy.evals.report import format_run, save_run
    from raggy.evals.runner import UnsafeIndexDirError, run_suite
    from raggy.evals.suites import SUITES, DatasetError

    # Each suite gets its own index, so an eval never touches your documents' index.
    config = dataclasses.replace(config, index_dir=Path(config.index_dir) / "evals" / args.suite)
    raggy = _build_raggy(config)
    suite = SUITES[args.suite]()
    answers = not args.retrieval_only and suite.answers
    judge = _build_judge(config) if answers and not args.no_judge else None
    if judge is not None:  # say which model grades, before spending minutes on it
        print(f"judge: {judge.name} (set RAGGY_JUDGE / RAGGY_JUDGE_MODEL to change)", file=sys.stderr)

    def counter(what: str):
        def show(i: int, total: int) -> None:
            print(f"\r  {what} {i}/{total}", end="" if i < total else "\n", file=sys.stderr, flush=True)
        return show

    try:
        run = run_suite(suite, raggy, judge, answers=answers, limit=args.limit,
                        label=args.label, progress=counter("question"),
                        ingest_progress=counter("embedded chunks"))
    except (DatasetError, GoldNotFoundError, UnsafeIndexDirError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(format_run(run))
    print(f"\nSaved {save_run(run, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
