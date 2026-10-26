import threading

from raggy.chunking.contextual import CONTEXT_VERSION, ContextCache, add_llm_context, cache_key
from raggy.chunking.tokens import simple_count
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.evals.runner import prepare_index
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.pipeline import Raggy
from raggy.types import Chunk, Document

from .test_pipeline_chunking import tiny_suite, write

DOC = Document("/d/returns.md", "# Returns\n\nYou can return bikes.\n\n## Refunds\n\nPaid within 14 days.", {"title": "Returns"})


def chunks():
    return [Chunk("/d/returns.md#0", "/d/returns.md", "# Returns\n\nYou can return bikes.", 0, 0, 31, context="Returns"),
            Chunk("/d/returns.md#1", "/d/returns.md", "## Refunds\n\nPaid within 14 days.", 1, 33, 65,
                  context="Returns › Refunds")]


class Counting(FakeLLM):
    def __init__(self, reply="From the Brightwater returns policy, about refund timing.", fail_on=None):
        super().__init__(reply)
        self.n, self.prompts, self.fail_on = 0, [], fail_on
        self._lock = threading.Lock()

    def complete(self, system, user, cache_prefix=None):
        user = (cache_prefix or "") + user
        with self._lock:
            self.n += 1
            self.prompts.append(user)
        if self.fail_on and self.fail_on in user.split("<chunk>")[1]:
            raise LLMError("rate limited")
        return super().complete(system, user)


def test_context_prepended_and_cached(tmp_path):
    llm, cache = Counting(), ContextCache(tmp_path / "c.sqlite")
    cs = chunks()
    stats = add_llm_context(DOC, cs, llm, cache)
    assert (stats.calls, stats.cached, stats.failed) == (2, 0, 0)
    assert cs[1].context == "From the Brightwater returns policy, about refund timing.\nReturns › Refunds"
    cache.commit()
    again = chunks()
    stats = add_llm_context(DOC, again, Counting(), ContextCache(tmp_path / "c.sqlite"))
    assert (stats.calls, stats.cached) == (0, 2) and again[1].context == cs[1].context


def test_llm_context_failure_falls_back_and_is_reported(tmp_path):
    cache = ContextCache(tmp_path / "c.sqlite")
    cs = chunks()
    stats = add_llm_context(DOC, cs, Counting(fail_on="Paid within"), cache)
    assert (stats.calls, stats.failed) == (2, 1)
    assert cs[1].context == "Returns › Refunds"  # header only
    cache.commit()
    retry = add_llm_context(DOC, chunks(), Counting(), ContextCache(tmp_path / "c.sqlite"))
    assert (retry.calls, retry.cached) == (1, 1)  # the failure was not cached


def test_cache_key_includes_model_and_version():
    c = chunks()[0]
    k = cache_key(DOC, c, "openai:gpt-5.5")
    assert k != cache_key(DOC, c, "openai:gpt-5.4")
    assert k != cache_key(Document(DOC.source, DOC.text + " changed", {}), c, "openai:gpt-5.5")
    assert CONTEXT_VERSION == "ctx-v1"


def test_document_first_in_prompt(tmp_path):
    llm = Counting()
    add_llm_context(DOC, chunks(), llm, ContextCache(tmp_path / "c.sqlite"))
    p = llm.prompts[0]
    assert p.startswith("<document>\n# Returns") and p.index("</document>") < p.index("<chunk>")


def test_context_trimmed_to_fit_token_limit(tmp_path):
    cs = chunks()
    add_llm_context(DOC, cs, Counting(reply="word " * 600), ContextCache(tmp_path / "c.sqlite"),
                    count=simple_count, max_tokens=100)
    for c in cs:
        assert simple_count(f"{c.context}\n\n{c.text}") <= 100


def test_ingest_reports_context_calls(tmp_path):
    write(tmp_path / "d" / "returns.md", DOC.text)
    llm = Counting()

    def make():
        c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
                   chunker="structure", chunk_context="llm")
        return Raggy(c, embedder=FakeEmbedder(), llm=llm)
    r = make().ingest([str(tmp_path / "d")])
    assert r.context_calls == llm.n > 0 and r.context_failed == 0
    write(tmp_path / "d" / "returns.md", DOC.text + "\n\nNew line.")
    before = llm.n
    r = make().ingest([str(tmp_path / "d")])
    assert r.context_calls == llm.n - before  # the doc changed, so its hash and keys changed


def test_eval_rebuild_keeps_context_cache(tmp_path):
    def make(**cfg):
        c = Config(index_dir=tmp_path / "ev", embed_model="fake", llm_provider="fake", llm_model=None,
                   chunker="structure", chunk_context="llm", **cfg)
        return Raggy(c, embedder=FakeEmbedder(), llm=Counting())
    prepare_index(tiny_suite(), make())
    assert (tmp_path / "ev" / "context-cache.sqlite").exists()
    r = make(chunk_headers=False)
    prepare_index(tiny_suite(), r)  # settings changed: rebuilt
    assert (tmp_path / "ev" / "context-cache.sqlite").exists() and r.llm.n == 0


def test_eval_folder_with_only_the_cache_is_rebuilt(tmp_path):
    from raggy.chunking.contextual import CACHE_NAME, ContextCache
    ContextCache(tmp_path / "ev" / CACHE_NAME).commit()  # a first build that died after paying for contexts
    c = Config(index_dir=tmp_path / "ev", embed_model="fake", llm_provider="fake", llm_model=None)
    assert prepare_index(tiny_suite(), Raggy(c, embedder=FakeEmbedder(), llm=Counting()))["cached"] is False
