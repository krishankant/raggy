from raggy.retrieve.fusion import rrf
from raggy.types import Chunk, Hit


def h(cid, score=0.0):
    return Hit(Chunk(cid, "s", cid, 0, 0, 1), score)


def test_rrf_sums_and_dedupes():
    out = rrf([[h("a"), h("b"), h("c")], [h("c"), h("a")]], k=60)
    scores = {x.chunk.id: x.score for x in out}
    assert abs(scores["a"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(scores["c"] - (1 / 63 + 1 / 61)) < 1e-12
    assert abs(scores["b"] - 1 / 62) < 1e-12
    assert [x.chunk.id for x in out] == ["a", "c", "b"]


def test_rrf_tie_break_stable():
    # x and y both score 1/61 (each first in one list); x was first in the earlier list... ties go by best rank, then id
    out = rrf([[h("y")], [h("x")]], k=60)
    assert [o.chunk.id for o in out] == ["x", "y"]


def test_rrf_single_ranking_preserves_order():
    assert [o.chunk.id for o in rrf([[h("c"), h("a"), h("b")]])] == ["c", "a", "b"]


def test_rrf_empty():
    assert rrf([[], []]) == []
