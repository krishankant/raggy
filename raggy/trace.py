"""One JSON line per answer, for debugging and audits (week 12).

With RAGGY_TRACE=on, every `ask` appends a record to
index_dir/traces/YYYY-MM-DD.jsonl: the question, the settings that shaped the
answer, stage timings, what was retrieved (source, location, score), the
answer, its citations and unsupported markers, token usage, cost, and whether
it came from the answer cache. Traces store questions, so tracing is off by
default; the files never leave your machine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from raggy.generate.prompt import location


class Tracer:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def path(self) -> Path:
        return self.folder / (datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl")

    def record(self, *, question, answer, config, timings, usage, cost) -> None:
        row = {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "question": question,
            "settings": {"retrieval": config.retrieval, "rerank": config.rerank, "transform": config.query_transform,
                         "top_k": config.top_k, "pack": config.pack, "citation_check": config.citation_check,
                         "llm": f"{config.llm_provider}:{config.llm_model or ''}"},
            "timings": {k: round(v, 1) for k, v in timings.items()},
            "hits": [{"source": h.chunk.source, "location": location(h.chunk.meta), "score": round(float(h.score), 4),
                      "chunk": h.chunk.id} for h in answer.hits],
            "answer": answer.text,
            "citations": [c.number for c in answer.citations],
            "unsupported": answer.unsupported,
            "cached": answer.cached,
            "usage": usage,
            "cost": cost,
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def latest(self, n: int = 5) -> list[dict]:
        rows = []
        for p in sorted(self.folder.glob("*.jsonl"), reverse=True):
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] + rows
            if len(rows) >= n:
                break
        return rows[-n:]
