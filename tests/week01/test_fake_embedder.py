import numpy as np

from raggy.embed.fake import FakeEmbedder


def test_fake_embedder_normalised_and_deterministic():
    e = FakeEmbedder()
    a = e.embed_documents(["the cat sat", "dogs bark loudly"])
    b = e.embed_documents(["the cat sat", "dogs bark loudly"])
    assert a.shape == (2, e.dim) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)
    assert e.embed_query("the cat sat").shape == (e.dim,)


def test_fake_embedder_overlap_scores_higher():
    e = FakeEmbedder()
    q = e.embed_query("refund policy for orders")
    near, far = e.embed_documents(["our refund policy for damaged orders", "the weather is sunny today"])
    assert q @ near > q @ far
