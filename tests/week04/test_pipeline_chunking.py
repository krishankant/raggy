import pytest

from raggy.chunking import ChunkSettings, chunk_document
from raggy.chunking.fixed import chunk_fixed
from raggy.chunking.tokens import simple_count
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.judge import Judge
from raggy.evals.runner import prepare_index, run_suite
from raggy.generate.fake import FakeLLM
from raggy.generate.prompt import build_user_prompt
from raggy.pipeline import Raggy, embed_text
from raggy.store.numpy_store import NumpyStore
from raggy.types import Chunk, Document, Hit

from raggy.evals.dataset import EvalQuestion, GoldSpan
from raggy.evals.suites import Suite


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def tiny_suite():
    docs = [Document("refunds.md", "# Refunds\n\nRefunds are paid back within 14 days of the return arriving."),
            Document("cats.md", "The office cat is called Biscuit and sleeps on the printer.")]
    qs = [EvalQuestion("a1", "when are refunds paid back", True, [GoldSpan("refunds.md", "within 14 days")], "14 days")]
    return Suite(name="tiny", documents=lambda options=None: list(docs), questions=lambda: list(qs),
                 fingerprint=lambda: "v1", ks=(1, 2), unit="chunk", answers=True)

SECTIONS = "# Help\n\n## Refunds\n\nRefunds are paid within 14 days of the return arriving.\n\n" \
           "## Bikes\n\n" + " ".join(f"The Heron model {i} is a city bike." for i in range(40))


def make(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None, **cfg)
    return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("Paid within 14 days [1]."))


class Recording(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.seen = []

    def embed_documents(self, texts):
        self.seen += texts
        return super().embed_documents(texts)


def test_headers_are_embedded_not_in_span(tmp_path):
    write(tmp_path / "d" / "help.md", SECTIONS)
    emb = Recording()
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               chunker="structure", chunk_tokens=64, chunk_overlap_tokens=8)
    Raggy(c, embedder=emb, llm=FakeLLM()).ingest([str(tmp_path / "d")])
    assert any(t.startswith("Help › Refunds\n\n## Refunds") or t.startswith("Help\n\n# Help") for t in emb.seen)
    store = NumpyStore.load(tmp_path / "index")
    doc_text = (tmp_path / "d" / "help.md").read_text()
    for ch in store.chunks:
        assert ch.text == doc_text[ch.start:ch.end] and "›" not in ch.text


def test_embed_text():
    c = Chunk("a#0", "a", "body", 0, 0, 4, context="Doc › Sec")
    assert embed_text(c) == "Doc › Sec\n\nbody"
    assert embed_text(Chunk("a#0", "a", "body", 0, 0, 4)) == "body"


def test_embed_text_over_limit_raises(tmp_path):
    class Tiny(FakeEmbedder):
        max_tokens = 5
    r = make(tmp_path)
    r._embedder = Tiny()
    r._embedder.max_tokens = 5
    with pytest.raises(ValueError, match="a#0.*tokens"):
        r.ingest_documents([Document("a", "one two three four five six seven")])


def test_parent_expansion_dedupes(tmp_path):
    write(tmp_path / "d" / "help.md", SECTIONS)
    r = make(tmp_path, chunker="structure", chunk_tokens=20, chunk_overlap_tokens=0, parent_tokens=400)
    r.ingest([str(tmp_path / "d")])
    hits = r.search("Heron model city bike", k=3)
    keys = [(h.chunk.source, h.chunk.start) for h in hits]
    assert len(keys) == len(set(keys))
    store = NumpyStore.load(tmp_path / "index")
    assert all(any(h.chunk.text == c.parent_text for c in store.chunks) for h in hits)
    assert all(h.chunk.id.split("#")[1].startswith("p") for h in hits)


def test_top_k_counts_parents(tmp_path):
    for i in range(5):
        write(tmp_path / "d" / f"doc{i}.md", f"# Doc {i}\n\n" + " ".join(f"Bike fact {j} for doc {i}." for j in range(30)))
    r = make(tmp_path, chunker="structure", chunk_tokens=20, chunk_overlap_tokens=0, parent_tokens=200)
    r.ingest([str(tmp_path / "d")])
    assert len(r.search("bike fact", k=4)) == 4


def test_prompt_shows_section_path():
    h = Hit(Chunk("/x/help.md#0", "/x/help.md", "Paid in 14 days.", 0, 0, 16, context="Help › Refunds"), 0.9)
    p = build_user_prompt("when?", [h])
    assert "[1] (source: help.md)\nHelp › Refunds\nPaid in 14 days." in p


def test_judge_sees_same_text_as_model():
    from raggy.evals.judge import _grounding_prompt
    from raggy.types import Answer
    from raggy.evals.dataset import EvalQuestion
    h = Hit(Chunk("/x/help.md#0", "/x/help.md", "Paid in 14 days.", 0, 0, 16, context="Help › Refunds"), 0.9)
    q = EvalQuestion("q", "when?", True, [], "14 days")
    out = _grounding_prompt(q, Answer("when?", "14 days [1]", [], [h]))
    assert "Help › Refunds\nPaid in 14 days." in out


def test_chunker_change_reindexes(tmp_path):
    write(tmp_path / "d" / "help.md", SECTIONS)
    make(tmp_path).ingest([str(tmp_path / "d")])
    assert make(tmp_path).ingest([str(tmp_path / "d")]).unchanged == 1
    for cfg in ({"chunker": "structure"}, {"chunker": "structure", "chunk_headers": False},
                {"chunker": "structure", "chunk_headers": False, "parent_tokens": 512},
                {"chunker": "structure", "chunk_headers": False, "parent_tokens": 512, "chunk_tokens": 128}):
        assert make(tmp_path, **cfg).ingest([str(tmp_path / "d")]).updated == 1, cfg


def test_eval_key_has_chunk_settings(tmp_path):
    suite = tiny_suite()
    assert prepare_index(suite, make(tmp_path))["cached"] is False
    assert prepare_index(suite, make(tmp_path))["cached"] is True
    assert prepare_index(suite, make(tmp_path, chunker="recursive"))["cached"] is False
    assert prepare_index(suite, make(tmp_path, chunker="recursive", chunk_context="llm"))["cached"] is False


def test_run_records_context_tokens_and_settings(tmp_path):
    run = run_suite(tiny_suite(), make(tmp_path, chunker="structure"), None, answers=False)
    assert run["metrics"]["context_tokens_p50"] > 0
    assert all(r["context_tokens"] > 0 for r in run["questions"])
    assert run["config"]["chunking"]["chunker"] == "structure" and run["config"]["chunk_context"] == "none"


def test_fixed_chunker_reproduces_week3():
    d = Document("/a.md", SECTIONS, {"title": "Help"})
    got = chunk_document(d, ChunkSettings(chunker="fixed", chars=300, char_overlap=50), simple_count)
    assert got == chunk_fixed(d, 300, 50)
