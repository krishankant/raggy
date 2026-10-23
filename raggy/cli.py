"""Command line: raggy ingest | search | ask | sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raggy.config import Config, ConfigError
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.store.numpy_store import EmbedderMismatchError, IndexNotFoundError


def _build_raggy(config: Config) -> Raggy:  # tests replace this to inject fakes
    return Raggy(config)


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
    except IndexNotFoundError:
        print("No index found — run `raggy ingest <path>` first.", file=sys.stderr)
        return 1
    except (LLMError, EmbedderMismatchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
