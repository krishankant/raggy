"""Structure-aware chunking for Markdown (and the Markdown that week 3's HTML extractor writes).

The document is parsed into blocks (headings, paragraphs, lists, tables,
fenced code) and blocks are packed whole into chunks. So a chunk never ends
mid-sentence, mid-row or mid-line of code unless a single block is bigger
than a whole chunk. Each chunk remembers its section path ("Returns and
refunds › Refunds"), which becomes its contextual header. A heading at the
same or a higher level as the section we're in starts a new chunk, so one
chunk never mixes two sibling sections.

Oversized blocks: a table is split by rows, with the header row carried in
each piece's context; code is split by lines; anything else is split by
sentences, then words, then characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from raggy.chunking.recursive import Unit, _cut, make_units, pack, sentence_spans, split_span
from raggy.chunking.tokens import Counter
from raggy.types import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
SEP = " › "


@dataclass
class Block:
    kind: str  # heading | paragraph | list | table | code
    start: int
    end: int
    level: int = 0
    title: str = ""


def parse_blocks(text: str) -> list[Block]:
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    blocks: list[Block] = []
    i = 0

    def end_of(j: int) -> int:
        return lines[j][0] + len(lines[j][1])

    while i < len(lines):
        start, line = lines[i]
        if not line.strip():
            i += 1
            continue
        if m := _HEADING.match(line):
            blocks.append(Block("heading", start, end_of(i), len(m.group(1)), m.group(2)))
            i += 1
        elif m := _FENCE.match(line):
            j = i + 1
            while j < len(lines) and not lines[j][1].strip().startswith(m.group(1)):
                j += 1
            j = min(j, len(lines) - 1)
            blocks.append(Block("code", start, end_of(j)))
            i = j + 1
        elif line.lstrip().startswith("|"):
            j = i
            while j + 1 < len(lines) and lines[j + 1][1].lstrip().startswith("|"):
                j += 1
            blocks.append(Block("table", start, end_of(j)))
            i = j + 1
        else:
            kind = "list" if _LIST.match(line) else "paragraph"
            j = i
            while j + 1 < len(lines):
                nxt = lines[j + 1][1]
                if (not nxt.strip() or _HEADING.match(nxt) or _FENCE.match(nxt)
                        or nxt.lstrip().startswith("|")):
                    break
                if kind == "paragraph" and _LIST.match(nxt):
                    break
                j += 1
            blocks.append(Block(kind, start, end_of(j)))
            i = j + 1
    return blocks


def _context(title: str, path: list[str]) -> str:
    parts = [title] if title else []
    for p in path:
        if not parts or p != parts[-1]:
            parts.append(p)
    return SEP.join(parts)


def chunk_structure(doc: Document, count: Counter, size: int = 256, overlap: int = 32) -> list[Chunk]:
    text = doc.text
    title = doc.metadata.get("title", "")
    units: list[Unit] = []
    path: list[tuple[int, str]] = []  # (level, heading) of the current section
    has_content = False
    for b in parse_blocks(text):
        if b.kind == "heading":
            # A sibling or higher heading ends the section we're in, so it starts a
            # new chunk. A deeper heading (a subsection) may share its parent's chunk.
            current = path[-1][0] if path else None
            brk = has_content and (current is None or b.level <= current)
            if brk:
                has_content = False
            while path and path[-1][0] >= b.level:
                path.pop()
            path.append((b.level, b.title))
            new = make_units(text, [(b.start, b.end)], count, _context(title, [p[1] for p in path]))
            if new:
                new[0].brk = brk
            units += new
            continue
        ctx = _context(title, [p[1] for p in path])
        if count(text[b.start:b.end]) <= size:
            spans = [(b.start, b.end)]
        elif b.kind == "table":
            rows = _cut(text, b.start, b.end, "\n")
            header = text[rows[0][0]:rows[min(1, len(rows) - 1)][1]].strip()
            ctx = ctx + "\n" + header if header else ctx
            spans = []
            for s, e in rows:
                spans += split_span(text, s, e, count, size, [" ", ""])
        elif b.kind == "code":
            spans = []
            for s, e in _cut(text, b.start, b.end, "\n"):
                spans += split_span(text, s, e, count, size, [" ", ""])
        else:
            spans = sentence_spans(text, b.start, b.end, count, size)
        units += make_units(text, spans, count, ctx)
        has_content = True
    return pack(doc, units, count, size, overlap)
