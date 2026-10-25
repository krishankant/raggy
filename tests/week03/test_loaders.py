import os
import time

import pytest

from raggy.evals.dataset import EvalQuestion, GoldSpan, resolve_gold
from raggy.ingest.loaders import LoadError, LoadOptions, load_bytes, load_file
from raggy.types import Document

from .conftest import PAGE, make_pdf, write


def test_html_main_drops_boilerplate():
    doc = load_bytes(PAGE.encode(), "p.html", "html")
    assert "back within three weeks" in doc.text
    assert "| BW-CH-1142 | 8-speed chain |" in doc.text
    for junk in ("cookies", "newsletter", "Returns and refunds", "do not index me"):
        assert junk not in doc.text


def test_html_full_keeps_everything_but_scripts():
    doc = load_bytes(PAGE.encode(), "p.html", "html", LoadOptions(html_extract="full"))
    assert "cookies" in doc.text and "newsletter" in doc.text and "back within three weeks" in doc.text
    assert "do not index me" not in doc.text and "<p>" not in doc.text


def test_html_title_metadata():
    assert load_bytes(PAGE.encode(), "p.html", "html").metadata["title"] == "Spare parts | Brightwater Bikes"
    full = load_bytes(PAGE.encode(), "p.html", "html", LoadOptions(html_extract="full"))
    assert full.metadata["title"] == "Spare parts | Brightwater Bikes"


def test_markdown_title_from_heading(tmp_path):
    doc = load_file(write(tmp_path / "w.md", "Intro line\n\n# Warranty terms\n\nTwo years."))
    assert doc.metadata["title"] == "Warranty terms"
    assert load_file(write(tmp_path / "plain.txt", "no heading")).metadata["title"] == "plain"


def test_pdf_text_and_title(tmp_path):
    f = write(tmp_path / "manual.pdf", make_pdf(["First page about brakes.", "Second page about tyres."],
                                                 title="Owner manual"))
    doc = load_file(f)
    assert "brakes" in doc.text and "tyres" in doc.text
    assert doc.text.index("brakes") < doc.text.index("tyres")
    assert doc.metadata["title"] == "Owner manual" and doc.metadata["type"] == "pdf"


def test_corrupt_pdf_raises_load_error(tmp_path):
    with pytest.raises(LoadError, match="broken.pdf"):
        load_file(write(tmp_path / "broken.pdf", b"%PDF-1.4 this is not really a pdf"))


def test_pdf_without_text_raises_load_error(tmp_path):
    with pytest.raises(LoadError, match="no text"):
        load_file(write(tmp_path / "scan.pdf", make_pdf([""])))


def test_empty_html_raises_load_error():
    with pytest.raises(LoadError, match="no text"):
        load_bytes(b"<html><body><script>x()</script></body></html>", "e.html", "html")


def test_load_file_metadata_fields(tmp_path):
    f = write(tmp_path / "notes.md", "# Notes\n\nhello   world")
    os.utime(f, (time.time(), 1_700_000_000))
    doc = load_file(f)
    assert doc.source == str(f.resolve()) and doc.text == "# Notes\n\nhello world"
    assert doc.metadata == {"name": "notes.md", "title": "Notes", "type": "markdown", "kind": "file",
                            "modified": "2023-11-14T22:13:20Z"}


def test_resolve_gold_ignores_table_pipes():
    q = EvalQuestion("t", "chain code?", True, [GoldSpan("p", "BW-CH-1142 | 8-speed chain | 24 credits")])
    for text in ("| BW-CH-1142 | 8-speed chain | 24 credits |", "BW-CH-1142 8-speed chain\n24 credits"):
        (span,) = resolve_gold([q], [Document("p", text)])["t"]
        assert text[span.start:span.end].startswith("BW-CH-1142") and text[span.start:span.end].endswith("credits")
