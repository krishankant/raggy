import json
import random
from pathlib import Path

import pytest

from raggy.chunking import ChunkSettings, chunk_document
from raggy.chunking.recursive import chunk_recursive
from raggy.chunking.structure import chunk_structure
from raggy.chunking.tokens import simple_count
from raggy.store.numpy_store import NumpyStore
from raggy.types import Document

ROOT = Path(__file__).resolve().parents[2]
CHUNKERS = {
    "recursive": lambda d, size=64, overlap=8: chunk_recursive(d, simple_count, size, overlap),
    "structure": lambda d, size=64, overlap=8: chunk_structure(d, simple_count, size, overlap),
}


def check(doc, chunks, size):
    last_start = -1
    covered = [False] * len(doc.text)
    for c in chunks:
        assert c.text == doc.text[c.start:c.end], c.id
        assert c.text.strip() and c.text == c.text.strip()
        assert c.start > last_start
        last_start = c.start
        assert simple_count(c.text) <= size, (c.id, simple_count(c.text))
        for i in range(c.start, c.end):
            covered[i] = True
    missing = [i for i, ch in enumerate(doc.text) if not ch.isspace() and not covered[i]]
    assert not missing, doc.text[missing[0]:missing[0] + 40]
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"{doc.source}#{c.index}" for c in chunks)


def random_markdown(seed):
    r = random.Random(seed)
    words = ["brake", "chain", "Heron", "BW-CH-1142", "credits", "the", "warranty", "a", "of", "days",
             "Kestrel", "battery", "x" * r.randint(1, 30)]
    out = []
    for _ in range(r.randint(1, 25)):
        kind = r.choice(["h", "p", "p", "list", "table", "code", "blank", "wall"])
        if kind == "h":
            out.append("#" * r.randint(1, 3) + " " + " ".join(r.choices(words, k=3)))
        elif kind == "p":
            out.append(" ".join(r.choices(words, k=r.randint(1, 120))) + ".")
        elif kind == "list":
            out.append("\n".join(f"- {' '.join(r.choices(words, k=r.randint(1, 15)))}" for _ in range(r.randint(1, 8))))
        elif kind == "table":
            rows = ["| a | b |", "|---|---|"] + [f"| {r.choice(words)} | {' '.join(r.choices(words, k=r.randint(1, 9)))} |"
                                                 for _ in range(r.randint(1, 40))]
            out.append("\n".join(rows))
        elif kind == "code":
            out.append("```\n" + "\n".join("    " + " ".join(r.choices(words, k=r.randint(1, 12))) for _ in range(r.randint(1, 30))) + "\n```")
        elif kind == "wall":
            out.append(" ".join(r.choices(words, k=r.randint(100, 400))))
        else:
            out.append("")
    return Document(f"rand{seed}.md", "\n\n".join(out), {"title": "Random"})


@pytest.mark.parametrize("name", CHUNKERS)
@pytest.mark.parametrize("seed", range(40))
def test_offsets_property_random_markdown(name, seed):
    doc = random_markdown(seed)
    check(doc, CHUNKERS[name](doc), 64)


@pytest.mark.parametrize("name", CHUNKERS)
def test_offsets_exact_on_all_corpora(name):
    from raggy.evals.suites import fictional_suite, fictional_web_suite, load_beir
    docs = fictional_suite(ROOT / "evals/fictional").documents() + fictional_web_suite(ROOT / "evals/fictional-web").documents()
    docs += load_beir(ROOT / "tests/week02/fixtures/beir_tiny")[0]
    for d in docs:
        check(d, CHUNKERS[name](d, 256, 32), 256)


@pytest.mark.parametrize("name", CHUNKERS)
@pytest.mark.parametrize("text", ["", "   \n\n  ", "# Only a heading", "x", "# A\n# B\n## C"])
def test_degenerate_documents(name, text):
    doc = Document("d.md", text, {"title": "D"})
    chunks = CHUNKERS[name](doc)
    check(doc, chunks, 64)
    assert bool(chunks) == bool(text.strip())


def test_oversized_table_splits_by_rows():
    rows = "\n".join(f"| BW-{i:04d} | part number {i} with a longer description | {i} credits |" for i in range(300))
    doc = Document("t.md", "# Parts\n\n| Code | Part | Price |\n|---|---|---|\n" + rows, {"title": "Catalogue"})
    chunks = chunk_structure(doc, simple_count, 128, 16)
    check(doc, chunks, 128)
    assert len(chunks) > 5
    for c in chunks[1:]:
        assert c.text.startswith("| BW-") and c.text.endswith("credits |")  # whole rows only
        assert "| Code | Part | Price |" in c.context  # the header travels in the context


def test_code_block_never_split_mid_line():
    code = "\n".join(f"    value_{i} = compute(value_{i - 1}, factor={i})" for i in range(200))
    doc = Document("c.md", "# Code\n\n```python\n" + code + "\n```\n\nAfter the code.", {"title": "C"})
    for c in chunk_structure(doc, simple_count, 100, 0):
        assert c.start == 0 or doc.text[c.start - 1] in "\n " or doc.text[c.start - 1] == "\n"
        assert doc.text[c.end:c.end + 1] in ("", "\n")


def test_wall_of_text_splits():
    doc = Document("w.txt", " ".join(["word"] * 3000), {"title": "W"})
    for name, f in CHUNKERS.items():
        check(doc, f(doc, 100, 10), 100)


def test_giant_token_hard_split():
    doc = Document("g.txt", "start " + "Z" * 5000 + " end", {"title": "G"})
    for name, f in CHUNKERS.items():
        chunks = f(doc, 50, 5)
        check(doc, chunks, 50)


def test_structure_breaks_at_headings():
    doc = Document("s.md", "# Returns\n\nShort intro.\n\n## Refunds\n\nPaid in 14 days.\n\n# Warranty\n\nTwo years.",
                   {"title": "Help"})
    texts = [c.text for c in chunk_structure(doc, simple_count, 200, 0)]
    assert texts == ["# Returns\n\nShort intro.\n\n## Refunds\n\nPaid in 14 days.", "# Warranty\n\nTwo years."]


def test_structure_context_path():
    doc = Document("s.md", "# Returns and refunds\n\nIntro.\n\n## Refunds\n\n" + "Paid back soon. " * 40
                   + "\n\n## Exchanges\n\nSwap sizes.", {"title": "Returns and refunds"})
    chunks = chunk_structure(doc, simple_count, 60, 0)
    assert chunks[0].context == "Returns and refunds"
    assert any(c.context == "Returns and refunds › Refunds" and c.text.startswith("Paid") for c in chunks)
    assert chunks[-1].context == "Returns and refunds › Exchanges"


def test_recursive_respects_size_and_overlap():
    sentences = [f"Sentence number {i} talks about bikes." for i in range(60)]
    doc = Document("r.txt", " ".join(sentences), {"title": "R"})
    chunks = chunk_recursive(doc, simple_count, 50, 10)
    check(doc, chunks, 50)
    overlapped = sum(1 for a, b in zip(chunks, chunks[1:]) if b.start < a.end)
    assert overlapped >= len(chunks) - 2
    for a, b in zip(chunks, chunks[1:]):
        assert simple_count(doc.text[b.start:min(a.end, b.end)]) <= 10 or b.start >= a.end


def test_chunk_document_dispatch_and_headers():
    doc = Document("s.md", "# A\n\ntext here", {"title": "Doc"})
    with_h = chunk_document(doc, ChunkSettings(chunker="structure", tokens=64, overlap=8), simple_count)
    no_h = chunk_document(doc, ChunkSettings(chunker="structure", tokens=64, overlap=8, headers=False), simple_count)
    assert with_h[0].context == "Doc › A" and no_h[0].context == ""
    fixed = chunk_document(doc, ChunkSettings(chunker="fixed", chars=800, char_overlap=100), simple_count)
    assert fixed[0].text == "# A\n\ntext here" and fixed[0].context == ""


def test_old_chunks_jsonl_loads(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"embedder": "fake", "dim": 2, "version": 1}))
    import numpy as np
    np.save(tmp_path / "vectors.npy", np.ones((1, 2), np.float32))
    (tmp_path / "chunks.jsonl").write_text(json.dumps(
        {"id": "a#0", "source": "a", "text": "hi", "index": 0, "start": 0, "end": 2}) + "\n")
    store = NumpyStore.load(tmp_path)
    assert store.chunks[0].context == "" and store.chunks[0].parent_text is None


def test_parents_span_sibling_sections():
    doc = Document("h.md", "# Handbook\n\n## Hours\n\nOpen late on the first Thursday for the repair evening.\n\n"
                   "## Repair evening\n\nBring a bike and fix it with our tools, free.", {"title": "Handbook"})
    s = ChunkSettings(chunker="structure", tokens=20, overlap=0, parent_tokens=200)
    chunks = chunk_document(doc, s, simple_count)
    child = next(c for c in chunks if "fix it" in c.text)
    assert "first Thursday" in child.parent_text and "fix it" in child.parent_text
    assert child.parent_text == doc.text[child.parent_start:child.parent_end]
