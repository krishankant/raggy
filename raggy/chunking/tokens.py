"""Counting tokens, because embedding models have a token limit, not a character limit.

bge-small reads at most 512 tokens (510 plus two special markers) and silently
drops the rest. "800 characters" is anywhere from 150 to 400 tokens depending on
the text, so a character limit is either wasteful or, with codes and numbers,
quietly over the limit. Chunkers take a ``count(text) -> int`` function: the
model's own tokenizer in real runs, ``simple_count`` in tests.
"""

from __future__ import annotations

import re
from typing import Callable

Counter = Callable[[str], int]

_TOKEN = re.compile(r"\w+|[^\w\s]")


def simple_count(text: str) -> int:
    """Words and punctuation marks: close enough to a tokenizer for tests."""
    return len(_TOKEN.findall(text))
