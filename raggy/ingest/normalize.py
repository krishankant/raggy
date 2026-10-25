"""Make text from any loader look the same before it's hashed, chunked or embedded.

Two copies of a page that differ only in invisible characters or spacing
should hash the same, and a query shouldn't miss "office" because the PDF
spelled it with the "ﬃ" ligature.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZE_VERSION = 1  # bump when the rules change: the registry re-indexes everything

_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))  # soft hyphen, zero-widths, BOM
_INNER_SPACE = re.compile(r"[ \t]+")
_LEADING = re.compile(r"^[ \t]*")
_MANY_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        indent = _LEADING.match(line).group()  # keep indentation: it is meaning in code and lists
        lines.append(indent + _INNER_SPACE.sub(" ", line[len(indent):]).rstrip())
    return _MANY_BLANKS.sub("\n\n", "\n".join(lines)).strip()


_WRAPPED = re.compile(r"([a-z])-\n([a-z])")


def dehyphenate(text: str) -> str:
    """Join words a PDF split across lines ("exam-\\nple"). Lower-case on both
    sides only, so codes such as "BW-\\nCH" and "8-\\nSpeed" are left alone."""
    return _WRAPPED.sub(r"\1\2", text)
