"""Command line: raggy ingest | search | ask | sources | eval | compare."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from raggy.config import Config, ConfigError
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.store.numpy_store import EmbedderMismatchError, IndexNotFoundError


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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="raggy", description="Ask questions about your documents.")
    sub = p.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="index files or folders")
    ing.add_argument("paths", nargs="+")
    s = sub.add_parser("search", help="show the chunks retrieval finds, with scores")
    s.add_argument("query")
    s.add_argument("-k", type=_positive_int, default=None)
    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("question")
    a.add_argument("-k", type=_positive_int, default=None)
    a.add_argument("--show-context", action="store_true", help="also print the retrieved chunks")
    sub.add_parser("sources", help="list indexed sources")
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


def _print_hits(hits) -> None:
    for n, h in enumerate(hits, start=1):
        text = " ".join(h.chunk.text.split())
        print(f"[{n}] {h.score:.3f}  {Path(h.chunk.source).name} (chunk {h.chunk.index})")
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
            r = raggy.ingest(args.paths)
            print(f"Indexed {r.documents} documents as {r.chunks} chunks into {raggy.config.index_dir}")
            for path in r.skipped:
                print(f"skipped (not found): {path}", file=sys.stderr)
        elif args.command == "search":
            _print_hits(raggy.search(args.query, args.k))
        elif args.command == "ask":
            answer = raggy.ask(args.question, args.k)
            if args.show_context:
                _print_hits(answer.hits)
                print()
            print(answer.text)
            if answer.citations:
                print("\nSources:")
                for c in answer.citations:
                    print(f"[{c.number}] {Path(c.source).name} — \"{c.snippet[:100]}…\"")
        elif args.command == "sources":
            for source, count in sorted(raggy.sources().items()):
                print(f"{count:5d} chunks  {source}")
        elif args.command == "eval":
            return _eval(args, raggy.config)
        elif args.command == "compare":
            from raggy.evals.report import format_compare, load_run

            print(format_compare(load_run(args.base), load_run(args.new)))
    except IndexNotFoundError:
        print("No index found — run `raggy ingest <path>` first.", file=sys.stderr)
        return 1
    except (LLMError, EmbedderMismatchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _eval(args, config: Config) -> int:
    from raggy.evals.dataset import GoldNotFoundError
    from raggy.evals.report import format_run, save_run
    from raggy.evals.runner import run_suite
    from raggy.evals.suites import SUITES, DatasetError

    # Each suite gets its own index, so an eval never touches your documents' index.
    config = dataclasses.replace(config, index_dir=Path(config.index_dir) / "evals" / args.suite)
    raggy = _build_raggy(config)
    suite = SUITES[args.suite]()
    answers = not args.retrieval_only and suite.answers
    judge = _build_judge(config) if answers and not args.no_judge else None

    def counter(what: str):
        def show(i: int, total: int) -> None:
            print(f"\r  {what} {i}/{total}", end="" if i < total else "\n", file=sys.stderr, flush=True)
        return show

    try:
        run = run_suite(suite, raggy, judge, answers=answers, limit=args.limit,
                        label=args.label, progress=counter("question"),
                        ingest_progress=counter("embedded chunks"))
    except (DatasetError, GoldNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(format_run(run))
    print(f"\nSaved {save_run(run, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
