"""CSV as a Markdown table; the delimiter (, ; tab |) is sniffed from the first lines."""

from __future__ import annotations

import csv
import io

from raggy.ingest import loaders
from raggy.ingest.loaders.tables import markdown_table


def load(data: bytes, options) -> tuple[str, str | None]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise loaders.LoadError("not valid UTF-8 text") from None
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return markdown_table(list(csv.reader(io.StringIO(text), dialect))), None
