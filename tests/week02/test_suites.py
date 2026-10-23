import hashlib
import io
import zipfile

import pytest

from conftest import FIXTURES
from raggy.evals.suites import DatasetError, download_scifact, fictional_suite, load_beir


def test_load_beir_fixture():
    docs, questions = load_beir(FIXTURES / "beir_tiny")
    assert [d.source for d in docs] == ["scifact:d1", "scifact:d2", "scifact:d3"]
    assert docs[0].text.startswith("Coffee and sleep\n\nCaffeine")
    assert [q.id for q in questions] == ["1", "2"]  # query 3 has no labels
    assert [g.source for g in questions[1].gold] == ["scifact:d2", "scifact:d3"]
    assert all(q.answerable and q.gold[0].quote is None for q in questions)


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            z.write(FIXTURES / "beir_tiny" / f, f"scifact/{f}")
    return buf.getvalue()


def test_download_rejects_bad_checksum(tmp_path):
    src = tmp_path / "scifact.zip"
    src.write_bytes(_zip_bytes())
    data = tmp_path / "data"
    with pytest.raises(DatasetError, match="checksum") as e:
        download_scifact(data, url=src.as_uri(), sha256="0" * 64)
    assert src.as_uri() in str(e.value)
    assert not data.exists() or list(data.iterdir()) == []


def test_download_extracts_and_skips_second_time(tmp_path):
    payload = _zip_bytes()
    src = tmp_path / "scifact.zip"
    src.write_bytes(payload)
    data = tmp_path / "data"
    folder = download_scifact(data, url=src.as_uri(), sha256=hashlib.sha256(payload).hexdigest())
    assert (folder / "corpus.jsonl").exists() and folder == data / "scifact"
    src.unlink()  # a second call must not download again
    assert download_scifact(data, url=src.as_uri(), sha256="ignored") == folder


def test_download_unreachable_url_is_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="download"):
        download_scifact(tmp_path / "data", url=(tmp_path / "missing.zip").as_uri(), sha256="0" * 64)


def test_fictional_suite_fingerprint_changes_with_corpus(tmp_path):
    root = tmp_path / "fictional"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.md").write_text("one")
    (root / "questions.jsonl").write_text("")
    before = fictional_suite(root).fingerprint()
    (root / "docs" / "a.md").write_text("two")
    assert fictional_suite(root).fingerprint() != before
    assert fictional_suite(root).unit == "chunk" and fictional_suite(root).ks == (1, 4, 10)
