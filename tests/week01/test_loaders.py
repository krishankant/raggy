import warnings
from pathlib import Path

import pytest

from raggy.ingest.loaders import load_path


def write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_load_single_file(tmp_path):
    f = write(tmp_path / "a.md", "# Hello\nworld")
    docs = load_path(f)
    assert len(docs) == 1
    assert docs[0].source == str(f.resolve())
    assert docs[0].text == "# Hello\nworld"


def test_load_folder_recursive_sorted(tmp_path):
    write(tmp_path / "b.txt", "bee")
    write(tmp_path / "sub" / "a.md", "ay")
    write(tmp_path / "c.markdown", "sea")
    docs = load_path(tmp_path)
    assert {d.text for d in docs} == {"bee", "ay", "sea"}
    assert [d.source for d in docs] == sorted(d.source for d in docs)


def test_load_ignores_unsupported_and_hidden(tmp_path):
    write(tmp_path / "keep.md", "yes")
    write(tmp_path / "image.png", "not text")
    write(tmp_path / ".git" / "notes.md", "hidden dir")
    write(tmp_path / ".hidden.md", "hidden file")
    docs = load_path(tmp_path)
    assert [d.text for d in docs] == ["yes"]


def test_load_empty_folder(tmp_path):
    assert load_path(tmp_path) == []


def test_load_skips_undecodable_file(tmp_path):
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00bad\x81")
    write(tmp_path / "good.md", "fine")
    with pytest.warns(UserWarning, match="bad.md"):
        docs = load_path(tmp_path)
    assert [d.text for d in docs] == ["fine"]


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "nope")
