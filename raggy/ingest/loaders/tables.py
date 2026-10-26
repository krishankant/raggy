"""Every table, from any format, becomes a Markdown pipe table.

One shape for all of them means week 4's structure chunker can split any big
table between rows and carry its header row into each piece's context.
"""

from __future__ import annotations


def cell(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return " ".join(str("" if value is None else value).split()).replace("|", "\\|")


def markdown_table(rows) -> str:
    rows = [[cell(c) for c in r] for r in rows if any(c not in (None, "") for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
