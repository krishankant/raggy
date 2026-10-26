import numpy as np
import pytest

from raggy import cli
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.runner import prepare_index
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.store.base import Filters
from raggy.types import Document

from week04.test_pipeline_chunking import tiny_suite


class Decoy(FakeEmbedder):
    """Meaningless vectors: everything mentioning 'decoy' is closest to every query."""

    def _vec(self, text):
        v = np.zeros(self.dim, np.float32)
        v[0 if "decoy" in text else 1] = 1.0
        v[2] = 0.1
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec("decoy")


DOCS = [Document(f"/d/decoy{i}.md", f"decoy page {i} about nothing in particular", {"type": "markdown"})
        for i in range(8)] + [
    Document("/d/parts.md", "The BW-CH-1142 chain fits the Heron Commuter.", {"type": "markdown"}),
    Document("/d/parts.pdf", "Spare BW-CH-1142 chains are 24 credits.", {"type": "pdf"})]


def make(tmp_path, emb=None, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"chunker": "fixed", "parent_tokens": 0} | cfg))
    r = Raggy(c, embedder=emb or Decoy(), llm=FakeLLM("ok [1]."))
    return r


def test_hybrid_finds_code_vector_misses(tmp_path):
    r = make(tmp_path)
    r.ingest_documents(DOCS)
    top = lambda mode: [h.chunk.source for h in r.search("BW-CH-1142", k=2, mode=mode)]
    assert not any("parts" in s for s in top("vector"))
    assert all("parts" in s for s in top("keyword"))
    assert any("parts" in s for s in top("hybrid"))


def test_hybrid_respects_filters(tmp_path):
    r = make(tmp_path)
    r.ingest_documents(DOCS)
    hits = r.search("BW-CH-1142", k=4, filters=Filters(doc_type="pdf"), mode="hybrid")
    assert [h.chunk.source for h in hits] == ["/d/parts.pdf"]


def test_hybrid_with_parents(tmp_path):
    long = Document("/d/manual.md", "# Manual\n\n" + "\n\n".join(
        f"## Section {i}\n\nPart BW-CH-{1100 + i} is described here in detail." for i in range(30)), {"type": "markdown"})
    r = make(tmp_path, emb=FakeEmbedder(), chunker="structure", chunk_tokens=20, chunk_overlap_tokens=0,
             parent_tokens=200)
    r.ingest_documents([long])
    hits = r.search("BW-CH-1105 described", k=3, mode="hybrid")
    keys = [(h.chunk.source, h.chunk.start) for h in hits]
    assert len(hits) == 3 and len(set(keys)) == 3 and all("#p" in h.chunk.id for h in hits)


def test_default_mode_from_config(tmp_path):
    r = make(tmp_path, retrieval="keyword")
    r.ingest_documents(DOCS)
    assert all("parts" in h.chunk.source for h in r.search("BW-CH-1142", k=2))


def test_mode_flag_cli(monkeypatch, tmp_path, capsys):
    r = make(tmp_path)
    r.ingest_documents(DOCS)
    monkeypatch.setattr(cli, "_build_raggy", lambda config: r)
    assert cli.main(["search", "BW-CH-1142", "-k", "2", "--mode", "keyword"]) == 0
    assert "parts" in capsys.readouterr().out
    assert cli.main(["ask", "BW-CH-1142?", "--mode", "hybrid"]) == 0


def test_numpy_store_hybrid_errors_cleanly(monkeypatch, tmp_path, capsys):
    r = make(tmp_path, store="numpy")
    r.ingest_documents(DOCS)
    monkeypatch.setattr(cli, "_build_raggy", lambda config: r)
    assert cli.main(["search", "x", "--mode", "hybrid"]) == 1
    assert "RAGGY_STORE=lance" in capsys.readouterr().err


def test_eval_key_has_retrieval_settings(tmp_path):
    def mk(**kw):
        c = Config(index_dir=tmp_path / "ev", embed_model="fake", llm_provider="fake", llm_model=None, **kw)
        return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM())
    assert prepare_index(tiny_suite(), mk())["cached"] is False
    assert prepare_index(tiny_suite(), mk())["cached"] is True
    # The index is the same for every mode (both sides are always built), so changing mode reuses it...
    assert prepare_index(tiny_suite(), mk(retrieval="hybrid"))["cached"] is True
    from raggy.evals.runner import run_suite
    run = run_suite(tiny_suite(), mk(retrieval="hybrid", rrf_k=30), None, answers=False)
    assert (run["config"]["retrieval"], run["config"]["rrf_k"], run["config"]["fusion_candidates"]) == ("hybrid", 30, 50)
    assert run["config"]["fts_version"] == 1
