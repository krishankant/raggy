import json
from pathlib import Path

import pytest

from raggy.evals.dataset import (
    EvalQuestion, GoldNotFoundError, GoldSpan, Located, chunk_gold, load_questions, resolve_gold,
)
from raggy.ingest.loaders import load_path
from raggy.types import Chunk, Document

ROOT = Path(__file__).resolve().parents[2]


def q(gold, qid="q1"):
    return EvalQuestion(id=qid, question="?", answerable=True, gold=gold, reference="r")


def test_load_questions_roundtrip(tmp_path):
    rows = [
        {"id": "a", "question": "How long?", "answerable": True, "reference": "14 days",
         "gold": [{"source": "refunds.md", "quote": "14 days"}], "tags": ["lookup"]},
        {"id": "b", "question": "Skateboards?", "answerable": False, "reference": None,
         "gold": [], "tags": ["unanswerable"]},
    ]
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    qs = load_questions(path)
    assert qs[0] == EvalQuestion("a", "How long?", True, [GoldSpan("refunds.md", "14 days")], "14 days", ["lookup"])
    assert qs[1].answerable is False and qs[1].gold == []


def test_resolve_gold_finds_span():
    doc = Document("/x/refunds.md", "Intro. Refunds are paid within 14 days. Bye.")
    loc = resolve_gold([q([GoldSpan("refunds.md", "paid within 14 days")])], [doc])["q1"][0]
    assert doc.text[loc.start:loc.end] == "paid within 14 days" and loc.source == "/x/refunds.md"


def test_resolve_gold_tolerates_whitespace():
    doc = Document("/x/r.md", "Refunds are paid\nwithin   14 days.")
    loc = resolve_gold([q([GoldSpan("r.md", "paid within 14 days")])], [doc])["q1"][0]
    assert doc.text[loc.start:loc.end] == "paid\nwithin   14 days"


def test_resolve_gold_missing_quote_names_question():
    doc = Document("/x/r.md", "Refunds are paid within 14 days.")
    with pytest.raises(GoldNotFoundError, match=r"q7.*r\.md.*within 30 days"):
        resolve_gold([q([GoldSpan("r.md", "within 30 days")], qid="q7")], [doc])


def test_resolve_gold_unknown_source_names_question():
    with pytest.raises(GoldNotFoundError, match="q1.*nope.md"):
        resolve_gold([q([GoldSpan("nope.md", "x")])], [Document("/x/r.md", "x")])


def test_resolve_gold_whole_document():
    doc = Document("scifact:42", "Title\n\nAbstract text.")
    assert resolve_gold([q([GoldSpan("scifact:42")])], [doc])["q1"] == [Located("scifact:42", 0, len(doc.text))]


def test_chunk_gold_overlap_rules():
    gold = [Located("a", 10, 20), Located("b", 0, 5)]
    chunk = lambda s, e, src="a": Chunk("id", src, "t", 0, s, e)
    assert chunk_gold(chunk(0, 11), gold) == {0}
    assert chunk_gold(chunk(0, 10), gold) == set()  # touching is not overlapping
    assert chunk_gold(chunk(20, 30), gold) == set()
    assert chunk_gold(chunk(0, 5, "b"), gold) == {1}


def test_fictional_questions_resolve():
    root = ROOT / "evals" / "fictional"
    qs = load_questions(root / "questions.jsonl")
    located = resolve_gold(qs, load_path(root / "docs"))  # raises on any bad quote
    assert len(qs) >= 36
    assert len({x.id for x in qs}) == len(qs)
    unanswerable = [x for x in qs if not x.answerable]
    assert len(unanswerable) >= 6
    assert all(x.gold == [] and x.reference is None for x in unanswerable)
    assert all(x.gold and x.reference and located[x.id] for x in qs if x.answerable)
    assert all(len(x.tags) >= 1 for x in qs)
