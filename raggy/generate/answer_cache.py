"""Answers to repeated questions, served without a model call (week 12).

The key is everything the model sees and everything that changes what's kept:
the normalised question, each passage's id and text, the model, the system
prompt and the citation-check settings. If the index changes a passage, or
the prompt changes, the key changes: a cached answer is always the same
model's output for byte-identical input.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from raggy.generate.prompt import SYSTEM_PROMPT, source_text
from raggy.types import Hit

ANSWER_CACHE = "answer-cache.sqlite"


def answer_key(question: str, hits: list[Hit], model: str, settings: str) -> str:
    h = hashlib.sha256()
    h.update(" ".join(question.lower().split()).encode())
    for hit in hits:
        h.update(f"\0{hit.chunk.id}\0".encode() + source_text(hit).encode())
    h.update(f"\0{model}\0{settings}\0".encode() + hashlib.sha256(SYSTEM_PROMPT.encode()).digest())
    return h.hexdigest()


class AnswerCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer TEXT NOT NULL)")
        self.conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, answer: dict) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO answers (key, answer) VALUES (?, ?)", (key, json.dumps(answer)))
            self.conn.commit()
