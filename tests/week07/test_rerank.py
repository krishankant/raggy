import json

import pytest

from raggy import cli
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.generate.prompt import source_text
from raggy.pipeline import Raggy
from raggy.retrieve.rerank import CrossEncoderReranker, LLMReranker, make_reranker
from raggy.types import Chunk, Document, Hit


def hits(*texts):
    return [Hit(Chunk(f"s#{i}", "s", t, i, 0, len(t), context=f"Doc › S{i}"), 1.0 - i / 10) for i, t in enumerate(texts)]


class FakeCrossEncoder:
    """Scores a passage by how many times it says 'gold'; records what it was given."""

    def __init__(self):
        self.seen = []

    def rerank(self, query, documents, batch_size=16):
        docs = list(documents)
        self.seen.append((query, docs))
        return [float(d.count("gold")) for d in docs]


def cross(model=None):
    r = CrossEncoderReranker("fake-model")
    r._model = model or FakeCrossEncoder()
    return r


def test_rerank_reorders_by_score():
    out = cross().rerank("q", hits("nothing", "gold gold", "gold"))
    assert [h.chunk.text for h in out] == ["gold gold", "gold", "nothing"]
    assert [h.score for h in out] == [2.0, 1.0, 0.0]


def test_cross_encoder_input_order():
    m = FakeCrossEncoder()
    hs = hits("first text", "second text")
    cross(m).rerank("the question", hs)
    assert m.seen == [("the question", [source_text(h) for h in hs])]
    assert m.seen[0][1][0] == "Doc › S0\nfirst text"  # section path, then the chunk


def test_cross_encoder_failure_keeps_order():
    class Broken:
        def rerank(self, *a, **k):
            raise RuntimeError("model download failed")
    r = cross(Broken())
    hs = hits("a", "b")
    assert r.rerank("q", hs) == hs and r.failures == 1


def llm_reranker(reply):
    return LLMReranker(FakeLLM(reply))


def test_llm_reranker_orders_by_json():
    r = llm_reranker(json.dumps({"ranking": [3, 1, 2]}))
    assert [h.chunk.text for h in r.rerank("q", hits("a", "b", "c"))] == ["c", "a", "b"]


def test_llm_reranker_prompt_lists_numbered_passages():
    llm = FakeLLM(json.dumps({"ranking": [1]}))
    LLMReranker(llm).rerank("why?", hits("alpha"))
    system, user = llm.calls[0]
    assert "why?" in user and "[1] Doc › S0\nalpha" in user and "ranking" in user


def test_llm_reranker_partial_permutation():
    r = llm_reranker(json.dumps({"ranking": [2, 9, 2, "x"]}))
    assert [h.chunk.text for h in r.rerank("q", hits("a", "b", "c"))] == ["b", "a", "c"]


@pytest.mark.parametrize("reply", ["not json at all", json.dumps({"order": [1]}), json.dumps({"ranking": "1,2"})])
def test_llm_reranker_garbage_keeps_order(reply):
    r = llm_reranker(reply)
    hs = hits("a", "b")
    assert r.rerank("q", hs) == hs and r.failures == 1


def test_llm_reranker_llm_error_keeps_order():
    r = LLMReranker(FakeLLM(error=LLMError("rate limited")))
    hs = hits("a", "b")
    assert r.rerank("q", hs) == hs and r.failures == 1


def test_make_reranker():
    cfg = lambda **k: Config(index_dir="x", embed_model="fake", llm_provider="fake", llm_model=None, **k)
    assert make_reranker(cfg(), lambda: None) is None
    assert make_reranker(cfg(rerank="local"), lambda: None).name == "local:Xenova/ms-marco-MiniLM-L-6-v2"
    assert make_reranker(cfg(rerank="llm"), lambda: FakeLLM()).name == "llm:fake:scripted"


DOCS = [Document(f"/d/{i}.md", f"Plain page {i} about bikes and bikes.", {"type": "markdown"}) for i in range(10)] + [
    Document("/d/gold.md", "gold gold gold: the page that answers bikes.", {"type": "markdown"})]


def make(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0} | cfg))
    r = Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("ok [1]."))
    return r


def test_rerank_in_search(tmp_path):
    r = make(tmp_path, rerank="local", rerank_candidates=11)
    r._reranker = cross()
    r.ingest_documents(DOCS)
    assert r.search("bikes", k=1)[0].chunk.source == "/d/gold.md"
    model = r._reranker._model
    assert len(model.seen[0][1]) == 11  # it saw rerank_candidates passages, not just k


def test_rerank_before_parent_expansion(tmp_path):
    long = Document("/d/m.md", "# M\n\n" + "\n\n".join(f"## S{i}\n\nplain section {i} text" for i in range(20))
                    + "\n\n## Gold\n\ngold gold gold", {"type": "markdown"})
    # The reranker only reorders what retrieval returns, so give it every child here:
    # the test is about *when* reranking happens (before parents), not about recall.
    r = make(tmp_path, chunker="structure", chunk_tokens=20, chunk_overlap_tokens=0, parent_tokens=60, rerank="local",
             rerank_candidates=200)
    r._reranker = cross()
    r.ingest_documents([long])
    top = r.search("section", k=1)[0]
    assert "#p" in top.chunk.id and "gold gold gold" in top.chunk.text
    assert all("#p" not in p.split("\n")[0] for _, ps in r._reranker._model.seen for p in ps)  # children were reranked


def test_rerank_flag_cli(monkeypatch, tmp_path, capsys):
    r = make(tmp_path)
    r.ingest_documents(DOCS)
    monkeypatch.setattr(cli, "_build_raggy", lambda config: r)
    monkeypatch.setattr("raggy.retrieve.rerank.CrossEncoderReranker._load", lambda self: FakeCrossEncoder())
    assert cli.main(["search", "bikes", "-k", "1", "--rerank", "local"]) == 0
    assert "gold.md" in capsys.readouterr().out


def test_rerank_only_top_candidates_rest_keep_order(tmp_path):
    r = make(tmp_path, rerank="local", rerank_candidates=3)
    r._reranker = cross()
    r.ingest_documents(DOCS)
    hits = r.search("bikes", k=8)
    assert len(hits) == 8
    assert all(len(ps) == 3 for _, ps in r._reranker._model.seen)  # only the top 3 were scored
