import random

import numpy as np

from raggy.ingest.dedupe import (MIN_WORDS, DuplicateIndex, content_hash, drop_duplicates, minhash,
                                 similarity)
from raggy.types import Document

rng = random.Random(7)
VOCAB = [f"w{i}" for i in range(3000)]


def words(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) for _ in range(n))


BODY = words(200, 1)


def test_exact_duplicate_detected():
    idx = DuplicateIndex()
    idx.add("a.md", content_hash(BODY), minhash(BODY))
    assert idx.find(content_hash(BODY), minhash(BODY)) == "a.md"


def test_near_duplicate_detected():
    edited = BODY.replace(BODY.split()[100], "changed", 1)
    assert content_hash(edited) != content_hash(BODY)
    assert similarity(minhash(BODY), minhash(edited)) > 0.9
    idx = DuplicateIndex()
    idx.add("a.md", content_hash(BODY), minhash(BODY))
    assert idx.find(content_hash(edited), minhash(edited)) == "a.md"


def test_different_docs_not_duplicates():
    footer = words(60, 99)
    a, b = words(120, 2) + " " + footer, words(120, 3) + " " + footer
    idx = DuplicateIndex()
    idx.add("a", content_hash(a), minhash(a))
    assert idx.find(content_hash(b), minhash(b)) is None


def test_short_docs_exact_only():
    short = words(MIN_WORDS - 1, 4)
    assert minhash(short) is None
    idx = DuplicateIndex()
    idx.add("a", content_hash(short), None)
    assert idx.find(content_hash(short), None) == "a"
    assert idx.find(content_hash(short + " extra"), None) is None


def test_drop_duplicates_keeps_shorter_path():
    docs = [Document("archive/returns-2025.md", BODY + " old"), Document("other.md", words(200, 5)),
            Document("returns.md", BODY)]
    kept, dups = drop_duplicates(docs)
    assert [d.source for d in kept] == ["other.md", "returns.md"]
    assert dups == {"archive/returns-2025.md": "returns.md"}


def test_minhash_deterministic():
    assert np.array_equal(minhash(BODY), minhash(BODY))
    assert minhash(BODY).dtype == np.uint64 and minhash(BODY).shape == (128,)


def test_signature_roundtrips_through_bytes():
    sig = minhash(BODY)
    assert np.array_equal(np.frombuffer(sig.tobytes(), dtype=np.uint64), sig)
