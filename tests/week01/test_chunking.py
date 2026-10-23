import pytest

from raggy.chunking.fixed import chunk_fixed
from raggy.types import Document


def doc(text: str) -> Document:
    return Document(source="/x/doc.md", text=text)


def test_chunk_offsets_match_text():
    d = doc("abcdefghij" * 30)  # 300 chars
    chunks = chunk_fixed(d, size=100, overlap=20)
    for c in chunks:
        assert d.text[c.start:c.end] == c.text


def test_chunk_covers_text():
    d = doc("".join(chr(97 + i % 26) for i in range(1000)))
    chunks = chunk_fixed(d, size=100, overlap=20)
    rebuilt = chunks[0].text + "".join(c.text[20:] for c in chunks[1:])
    assert rebuilt == d.text


def test_chunk_ids_and_indexes():
    chunks = chunk_fixed(doc("x" * 250), size=100, overlap=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[1].id == "/x/doc.md#1"
    assert all(c.source == "/x/doc.md" for c in chunks)


def test_chunk_short_doc_single_chunk():
    chunks = chunk_fixed(doc("tiny doc"), size=800, overlap=100)
    assert len(chunks) == 1 and chunks[0].text == "tiny doc"


def test_chunk_drops_whitespace_only():
    assert chunk_fixed(doc("   \n\n  "), size=800, overlap=100) == []


def test_chunk_bad_overlap_raises():
    with pytest.raises(ValueError):
        chunk_fixed(doc("abc"), size=100, overlap=100)
