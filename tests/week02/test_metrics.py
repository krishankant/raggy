import numpy as np
import pytest

from raggy.evals.metrics import collapse_to_documents, mean, percentile, retrieval_metrics
from raggy.types import Chunk, Hit


def test_metrics_hand_computed():
    m = retrieval_metrics([set(), {0}, {0}, {1}], n_gold=2, ks=(1, 3))
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 0.5
    assert m["mrr"] == 0.5
    # DCG@3 = 1/log2(3) (rank 2 finds gold 0; rank 3 repeats it, gain 0)
    # IDCG@3 = 1 + 1/log2(3)  (two gold items, both at the top)
    assert m["ndcg@3"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert m["ndcg@1"] == 0.0


def test_metrics_perfect_ranking():
    m = retrieval_metrics([{0}, {1}], n_gold=2, ks=(1, 2))
    assert m == {"recall@1": 0.5, "ndcg@1": 1.0, "recall@2": 1.0, "ndcg@2": 1.0, "mrr": 1.0}


def test_metrics_no_hits():
    m = retrieval_metrics([], n_gold=1, ks=(4,))
    assert m == {"recall@4": 0.0, "ndcg@4": 0.0, "mrr": 0.0}


def test_percentile_matches_numpy():
    xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
    for p in (50, 95):
        assert percentile(xs, p) == pytest.approx(float(np.percentile(xs, p)))
    assert percentile([], 50) is None


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def _hit(source, i, score):
    return Hit(Chunk(f"{source}#{i}", source, "t", i, 0, 1), score)


def test_collapse_to_documents():
    hits = [_hit("a", 0, 0.9), _hit("a", 1, 0.8), _hit("b", 0, 0.7), _hit("a", 2, 0.6)]
    assert [h.chunk.id for h in collapse_to_documents(hits)] == ["a#0", "b#0"]
