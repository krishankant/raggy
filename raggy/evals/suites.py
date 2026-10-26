"""Eval suites: a fixed corpus plus questions with gold evidence.

- fictional: a made-up bike company, so the model can't answer from memory.
  Scores retrieval *and* answers. Small enough to read every question.
- fictional-web: the same company's docs as pages of a web site (menus,
  cookie banner, sidebar, footer) plus two duplicate pages, with the same
  questions. Measures what HTML extraction and dedupe are worth (week 3).
- scifact: a public benchmark (BEIR's copy of SciFact: 5,183 paper abstracts,
  300 test claims with expert relevance labels). Retrieval only, scored per
  document, so the numbers can be checked against published results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from raggy.evals.dataset import EvalQuestion, GoldSpan, load_questions
from raggy.ingest.loaders import LoadOptions, load_path
from raggy.ingest.normalize import normalize_text
from raggy.types import Document

# SciFact is CC BY-NC 2.0: fine for learning, so it's downloaded on demand
# and never committed. The hash pins the exact bytes the baseline used.
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


class DatasetError(Exception):
    """An eval dataset couldn't be fetched or read."""


@dataclass
class Suite:
    name: str
    documents: Callable[..., list[Document]]  # documents(options: LoadOptions | None)
    questions: Callable[[], list[EvalQuestion]]
    fingerprint: Callable[[], str]  # changes whenever the corpus or questions change
    ks: tuple[int, ...]  # cut-offs for recall@k / nDCG@k
    unit: str  # "chunk": score chunks against gold spans; "document": score whole documents
    answers: bool  # does this suite have reference answers to judge?
    dedupe: bool = True  # False for a public benchmark: its corpus is part of its definition


def fictional_suite(root: Path = Path("evals/fictional")) -> Suite:
    root = Path(root)

    def fingerprint() -> str:
        h = hashlib.sha256()
        for f in sorted((root / "docs").rglob("*")) + [root / "questions.jsonl"]:
            if f.is_file():
                h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes())
        return h.hexdigest()

    return Suite(
        name="fictional",
        documents=lambda options=None: load_path(root / "docs", options),
        questions=lambda: load_questions(root / "questions.jsonl"),
        fingerprint=fingerprint,
        ks=(1, 4, 10),
        unit="chunk",
        answers=True,
    )


def load_beir(folder: Path, split: str = "test") -> tuple[list[Document], list[EvalQuestion]]:
    """Read a BEIR-format dataset: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""
    folder = Path(folder)
    docs = []
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = f"{row.get('title', '').strip()}\n\n{row['text'].strip()}".strip()
            docs.append(Document(source=f"scifact:{row['_id']}", text=normalize_text(text)))
    relevant: dict[str, list[str]] = {}
    with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                relevant.setdefault(row["query-id"], []).append(row["corpus-id"])
    questions = []
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in relevant:
                questions.append(EvalQuestion(
                    id=row["_id"], question=row["text"], answerable=True,
                    gold=[GoldSpan(f"scifact:{d}") for d in relevant[row["_id"]]], tags=["scifact"],
                ))
    return docs, questions


def download_scifact(data_dir: Path, url: str = SCIFACT_URL, sha256: str = SCIFACT_SHA256) -> Path:
    """Fetch and unpack SciFact once. Verifies the checksum before trusting a byte."""
    data_dir = Path(data_dir)
    target = data_dir / "scifact"
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    part = data_dir / "scifact.zip.part"
    try:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out)
        except (urllib.error.URLError, OSError) as e:
            raise DatasetError(f"could not download {url}: {e}") from e
        actual = hashlib.sha256(part.read_bytes()).hexdigest()
        if actual != sha256:
            raise DatasetError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
        # Unpack beside the target, then move into place in one step, so an
        # interrupted run never leaves a half-extracted folder that looks done.
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
            with zipfile.ZipFile(part) as z:
                z.extractall(tmp)
            shutil.move(str(Path(tmp) / "scifact"), target)
    finally:
        part.unlink(missing_ok=True)
    return target


def scifact_suite(data_dir: Path = Path("evals/data")) -> Suite:
    @lru_cache(maxsize=1)
    def data():
        return load_beir(download_scifact(data_dir))

    return Suite(
        name="scifact",
        documents=lambda options=None: data()[0],
        questions=lambda: data()[1],
        fingerprint=lambda: SCIFACT_SHA256,
        ks=(1, 10, 100),
        unit="document",
        answers=False,
        dedupe=False,  # dropping abstracts would make our numbers incomparable with published ones
    )


def fictional_web_suite(root: Path = Path("evals/fictional-web"),
                        questions: Path = Path("evals/fictional/questions.jsonl")) -> Suite:
    root, questions = Path(root), Path(questions)
    if not questions.is_absolute() and not questions.exists():
        questions = root.parent / "fictional" / "questions.jsonl"

    def load() -> list[EvalQuestion]:
        qs = load_questions(questions)
        for q in qs:  # same questions, same quotes; the evidence now lives in .html pages
            q.gold = [GoldSpan(g.source.removesuffix(".md") + ".html", g.quote) for g in q.gold]
        return qs

    def fingerprint() -> str:
        h = hashlib.sha256()
        for f in sorted((root / "pages").rglob("*")) + [questions]:
            if f.is_file():
                h.update(f.name.encode() + b"\0" + f.read_bytes())
        return h.hexdigest()

    return Suite(
        name="fictional-web",
        documents=lambda options=None: load_path(root / "pages", options),
        questions=load,
        fingerprint=fingerprint,
        ks=(1, 4, 10),
        unit="chunk",
        answers=True,
    )


SUITES: dict[str, Callable[[], Suite]] = {
    "fictional": fictional_suite,
    "fictional-web": fictional_web_suite,
    "scifact": scifact_suite,
}
