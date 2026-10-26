from pathlib import Path

from raggy.evals.dataset import resolve_gold
from raggy.evals.suites import SUITES, manual_suite

ROOT = Path(__file__).resolve().parents[2]


def test_manual_suite_resolves_and_is_registered():
    s = manual_suite(ROOT / "evals/manual")
    qs = s.questions()
    assert s.name == "manual" and SUITES["manual"]().name == "manual"
    assert len(resolve_gold(qs, s.documents())) == len(qs) == 29
    assert sum(not q.answerable for q in qs) == 5 and len({q.id for q in qs}) == 29
    assert all(not q.gold and q.reference is None for q in qs if not q.answerable)
