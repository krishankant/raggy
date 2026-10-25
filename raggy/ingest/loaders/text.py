"""Plain text and Markdown: decode as UTF-8, and take Markdown's first heading as the title."""

from __future__ import annotations

import re

from raggy.ingest import loaders

_HEADING = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def load(data: bytes, options) -> tuple[str, str | None]:
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        raise loaders.LoadError("not valid UTF-8 text") from None


def load_markdown(data: bytes, options) -> tuple[str, str | None]:
    text, _ = load(data, options)
    m = _HEADING.search(text)
    return text, m.group(1) if m else None
