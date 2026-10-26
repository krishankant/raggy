import warnings

import pytest

from raggy.ingest.loaders import LoadError, LoadOptions, load_file

from .conftest import FLEET, make_pdf, make_scanned_pdf

LAYOUT = LoadOptions(pdf="layout", ocr=True)


def test_layout_headings_by_font_size(tmp_path):
    doc = load_file(make_pdf(tmp_path / "f.pdf", FLEET), LAYOUT)
    lines = doc.text.splitlines()
    assert "# Fleet price list 2026" in lines and "## Delivery bikes" in lines and "## Service plans" in lines
    assert doc.metadata["title"] == "Fleet price list" and doc.metadata["type"] == "pdf"


def test_layout_ruled_table_as_markdown(tmp_path):
    doc = load_file(make_pdf(tmp_path / "f.pdf", FLEET), LAYOUT)
    assert "| Bikes | Price per bike | Lead time |" in doc.text
    assert "| 5-9 | 4,700 credits | 4 weeks |" in doc.text


def test_table_text_not_duplicated_in_body(tmp_path):
    doc = load_file(make_pdf(tmp_path / "f.pdf", FLEET), LAYOUT)
    assert doc.text.count("4,700 credits") == 1


def test_pages_metadata_offsets(tmp_path):
    doc = load_file(make_pdf(tmp_path / "f.pdf", FLEET), LAYOUT)
    pages = doc.metadata["pages"]
    assert [label for label, _ in pages] == ["p. 1", "p. 2"]
    start2 = pages[1][1]
    assert doc.text[start2:].lstrip().startswith("## Service plans") and pages[0][1] == 0


def test_scanned_page_ocr(tmp_path):
    f = make_scanned_pdf(tmp_path / "scan.pdf", ["WORKSHOP NOTICE", "The Fenwick workshop closes on 14 March."])
    doc = load_file(f, LAYOUT)
    assert "Fenwick workshop closes on 14 March" in doc.text and doc.metadata["pages"] == [("p. 1", 0)]


def test_ocr_off_scanned_pdf_has_no_text(tmp_path):
    f = make_scanned_pdf(tmp_path / "scan.pdf", ["WORKSHOP NOTICE"])
    with pytest.raises(LoadError, match="RAGGY_OCR"):
        load_file(f, LoadOptions(pdf="layout", ocr=False))


def test_corrupt_pdf_layout_raises_load_error(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(LoadError, match="broken.pdf"):
        load_file(f, LAYOUT)


def test_table_cells_escaped(tmp_path):
    f = make_pdf(tmp_path / "t.pdf", [("BodyText", "Codes"), [["Code", "Note"], ["A|B", "line one"]]])
    doc = load_file(f, LAYOUT)
    assert "| A\\|B | line one |" in doc.text


def test_text_mode_is_week3(tmp_path):
    doc = load_file(make_pdf(tmp_path / "f.pdf", FLEET), LoadOptions(pdf="text"))
    assert "Fleet price list 2026" in doc.text and "|" not in doc.text
    assert [label for label, _ in doc.metadata["pages"]] == ["p. 1", "p. 2"]
