from raggy.chunking.tokens import simple_count
from raggy.generate.pack import pack
from raggy.generate.prompt import build_user_prompt
from raggy.types import Chunk, Hit


def h(source, start, text, score, context=""):
    return Hit(Chunk(f"{source}#{start}", source, text, 0, start, start + len(text), context=context), score)


DOC = "Alpha one two. Beta three four. Gamma five six. Delta seven eight."


def test_pack_merges_overlapping_same_source():
    a = h("/d/a.md", 0, DOC[0:31], 0.9, "A")   # "Alpha ... Beta three four."
    b = h("/d/a.md", 15, DOC[15:47], 0.7)     # "Beta ... Gamma five six." (overlaps a)
    c = h("/d/b.md", 0, "Other doc.", 0.8)
    out = pack([a, c, b], simple_count, 1000)
    assert [x.chunk.source for x in out] == ["/d/a.md", "/d/b.md"]
    merged = out[0].chunk
    assert (merged.start, merged.end, merged.text) == (0, 47, DOC[0:47])
    assert out[0].score == 0.9 and merged.context == "A"


def test_pack_merges_touching_spans():
    a, b = h("/d/a.md", 0, DOC[0:15], 0.5), h("/d/a.md", 15, DOC[15:31], 0.6)
    (m,) = pack([b, a], simple_count, 1000)
    assert m.chunk.text == DOC[0:31] and m.score == 0.6


def test_pack_keeps_document_order_within_source():
    late, early = h("/d/a.md", 48, DOC[48:], 0.9), h("/d/a.md", 0, DOC[0:14], 0.4)
    out = pack([late, early], simple_count, 1000)
    assert [x.chunk.start for x in out] == [0, 48]


def test_pack_orders_sources_by_best_score():
    out = pack([h("/d/x.md", 0, "low", 0.2), h("/d/y.md", 0, "high", 0.9), h("/d/x.md", 10, "best x", 0.95)],
               simple_count, 1000)
    assert [x.chunk.source for x in out] == ["/d/x.md", "/d/x.md", "/d/y.md"]


def test_pack_fits_budget_dropping_lowest():
    hs = [h(f"/d/{i}.md", 0, " ".join(["word"] * 40), 1.0 - i / 10) for i in range(5)]
    out = pack(hs, simple_count, 130)
    assert [x.chunk.source for x in out] == ["/d/0.md", "/d/1.md", "/d/2.md"]
    assert simple_count(build_user_prompt("q", out)) <= 130 + 40  # budget counts passages; prompt adds a little


def test_pack_oversized_single_passage():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    (only,) = pack([h("/d/a.md", 0, text, 0.9)], simple_count, 60)
    assert simple_count(only.chunk.text) <= 60 and only.chunk.text.endswith(".")
    assert text.startswith(only.chunk.text) and only.chunk.end == only.chunk.start + len(only.chunk.text)


def test_pack_empty():
    assert pack([], simple_count, 100) == []


def test_context_off_is_unchanged_and_on_packs(tmp_path):
    from raggy.config import Config
    from raggy.embed.fake import FakeEmbedder
    from raggy.generate.fake import FakeLLM
    from raggy.pipeline import Raggy
    hs = [h("/d/a.md", 0, DOC[0:31], 0.9), h("/d/a.md", 15, DOC[15:47], 0.7)]
    cfg = lambda **k: Config(index_dir=tmp_path, embed_model="fake", llm_provider="fake", llm_model=None, **k)
    off = Raggy(cfg(pack=False), embedder=FakeEmbedder(), llm=FakeLLM())
    assert off.context(hs) == hs
    on = Raggy(cfg(pack=True, context_tokens=1000), embedder=FakeEmbedder(), llm=FakeLLM())
    assert [p.chunk.text for p in on.context(hs)] == [DOC[0:47]]


def test_runner_uses_packed_context(tmp_path, monkeypatch):
    from raggy.config import Config
    from raggy.embed.fake import FakeEmbedder
    from raggy.evals.runner import run_suite
    from raggy.generate.fake import FakeLLM
    from raggy.pipeline import Raggy
    from week04.test_pipeline_chunking import tiny_suite
    seen = []
    r = Raggy(Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None, pack=True),
              embedder=FakeEmbedder(), llm=FakeLLM("ok [1]."))
    real = r.context
    monkeypatch.setattr(r, "context", lambda hits: seen.append(len(hits)) or real(hits))
    run = run_suite(tiny_suite(), r, None, answers=True)
    assert seen and run["config"]["pack"] is True and run["config"]["context_tokens"] == 3000
