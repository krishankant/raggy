import numpy as np

from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.dataset import EvalQuestion, GoldSpan
from raggy.evals.runner import prepare_index, run_suite
from raggy.evals.suites import Suite
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy
from raggy.store.lance_store import ANN_MIN_ROWS, LanceStore
from raggy.types import Chunk, Document

emb = FakeEmbedder()
WORDS = [f"w{i}" for i in range(400)]


def many_docs(n):
    rng = np.random.default_rng(0)
    return [Document(f"/c/doc{i}.md", " ".join(rng.choice(WORDS, 12)), {"type": "markdown"}) for i in range(n)]


def cfg(tmp_path, **kw):
    return Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
                  chunker="fixed", parent_tokens=0, **kw)


def test_ann_index_built_on_ingest_when_on(tmp_path):
    r = Raggy(cfg(tmp_path, ann="on"), embedder=emb, llm=FakeLLM())
    r.ingest_documents(many_docs(ANN_MIN_ROWS + 200))
    assert LanceStore(tmp_path / "index" / "lance").has_index()
    off = Raggy(cfg(tmp_path / "off"), embedder=emb, llm=FakeLLM())
    off.ingest_documents(many_docs(ANN_MIN_ROWS + 200))
    assert not LanceStore(tmp_path / "off" / "index" / "lance").has_index()


def test_small_corpus_skips_index(tmp_path):
    Raggy(cfg(tmp_path, ann="on"), embedder=emb, llm=FakeLLM()).ingest_documents(many_docs(50))
    assert not LanceStore(tmp_path / "index" / "lance").has_index()


def test_ann_finds_rows_added_after_index(tmp_path):
    r = Raggy(cfg(tmp_path, ann="on"), embedder=emb, llm=FakeLLM())
    r.ingest_documents(many_docs(ANN_MIN_ROWS + 200))
    r.ingest_documents([Document("/c/new.md", "zebra quokka narwhal unique words", {"type": "markdown"})])
    hit = r.search("zebra quokka narwhal", k=1)[0]
    assert hit.chunk.source == "/c/new.md"


def test_exact_bypasses_index(tmp_path):
    r = Raggy(cfg(tmp_path, ann="on"), embedder=emb, llm=FakeLLM())
    docs = many_docs(ANN_MIN_ROWS + 200)
    r.ingest_documents(docs)
    s = LanceStore(tmp_path / "index" / "lance")
    q = emb.embed_query(docs[7].text)
    exact = s.search(q, 10, emb.name, exact=True)
    vectors = emb.embed_documents([d.text for d in docs])
    best = np.argsort(-(vectors @ q))[:10]
    assert [h.chunk.source for h in exact][:1] == [docs[best[0]].source]
    assert np.isclose(exact[0].score, float(vectors[best[0]] @ q), atol=1e-5)


def suite(docs):
    qs = [EvalQuestion(f"q{i}", docs[i].text, True, [GoldSpan(docs[i].source)]) for i in range(0, 60, 6)]
    return Suite(name="ann", documents=lambda options=None: list(docs), questions=lambda: qs,
                 fingerprint=lambda: "v1", ks=(1, 10), unit="document", answers=False, dedupe=False)


def test_run_records_ann_overlap(tmp_path):
    docs = many_docs(ANN_MIN_ROWS + 200)
    run = run_suite(suite(docs), Raggy(cfg(tmp_path, ann="on"), embedder=emb, llm=FakeLLM()), None, answers=False)
    m = run["metrics"]
    assert 0.0 <= m["ann_overlap@10"] <= 1.0 and 0.0 <= m["ann_overlap@1"] <= 1.0
    assert all("ann_overlap" in q for q in run["questions"])
    assert run["config"]["ann"] == "on" and run["config"]["ann_nprobes"] == 20
    exact = run_suite(suite(docs), Raggy(cfg(tmp_path / "x"), embedder=emb, llm=FakeLLM()), None, answers=False)
    assert "ann_overlap@10" not in exact["metrics"]


def test_ann_settings_in_keys(tmp_path):
    docs = many_docs(30)
    r = Raggy(cfg(tmp_path), embedder=emb, llm=FakeLLM())
    assert prepare_index(suite(docs + many_docs(60)[30:]), r)["cached"] is False
    assert prepare_index(suite(docs + many_docs(60)[30:]), Raggy(cfg(tmp_path), embedder=emb, llm=FakeLLM()))["cached"] is True
    assert prepare_index(suite(docs + many_docs(60)[30:]), Raggy(cfg(tmp_path, ann="on"), embedder=emb, llm=FakeLLM()))["cached"] is False
    from raggy.pipeline import pipeline_key  # ANN changes no chunk: it isn't a reason to re-index files
    assert pipeline_key(cfg(tmp_path), "m") == pipeline_key(cfg(tmp_path, ann="on"), "m")
