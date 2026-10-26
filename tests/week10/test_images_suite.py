from pathlib import Path

from raggy.evals.dataset import resolve_gold
from raggy.evals.suites import SUITES, images_suite
from raggy.ingest.loaders import LoadOptions

ROOT = Path(__file__).resolve().parents[2] / "evals" / "images"


def test_images_gold_resolves_when_images_are_read(monkeypatch):
    monkeypatch.setattr("raggy.ingest.ocr.ocr_image", lambda image: "")  # the headers alone carry the gold
    s = images_suite(ROOT)
    qs = s.questions()
    assert len(qs) == 24 and sum(not q.answerable for q in qs) == 4 and "images" in SUITES
    assert s.strict_gold is False
    resolve_gold(qs, s.documents(LoadOptions(images="ocr")))  # strict here: every header must exist


def test_images_off_loses_every_picture(monkeypatch):
    s = images_suite(ROOT)
    missing: list[str] = []
    resolve_gold(s.questions(), s.documents(LoadOptions()), strict=False, missing=missing)
    assert len(missing) == 20  # with images off, no answer is in any text
