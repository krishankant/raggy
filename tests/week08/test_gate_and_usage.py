from types import SimpleNamespace as NS

from raggy.chunking.contextual import ContextCache, add_llm_context
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.anthropic_llm import AnthropicLLM
from raggy.generate.fake import FakeLLM
from raggy.generate.openai_llm import OpenAILLM
from raggy.generate.prompt import NOT_FOUND
from raggy.pipeline import Raggy
from raggy.types import Chunk, Document, Hit


class FakeAnthropic:
    def __init__(self):
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text="ok")],
                  usage=NS(input_tokens=40, output_tokens=5, cache_read_input_tokens=600,
                           cache_creation_input_tokens=0))


def test_anthropic_cache_prefix_block_shape():
    llm = AnthropicLLM()
    llm._client = FakeAnthropic()
    llm.complete("sys", "the chunk", cache_prefix="<document>long text</document>")
    content = llm._client.requests[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "<document>long text</document>", "cache_control": {"type": "ephemeral"}}
    assert content[1] == {"type": "text", "text": "the chunk"}
    llm.complete("sys", "plain")
    assert llm._client.requests[1]["messages"][0]["content"] == "plain"  # no prefix: unchanged request
    assert llm.usage == {"calls": 2, "input": 80, "output": 10, "cache_read": 1200, "cache_write": 0}


class FakeOpenAI:
    def __init__(self):
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        return NS(choices=[NS(message=NS(content="ok"))],
                  usage=NS(prompt_tokens=1500, completion_tokens=7, prompt_tokens_details=NS(cached_tokens=1024)))


def test_openai_cache_prefix_concatenates():
    llm = OpenAILLM("gpt-x")
    llm._client = FakeOpenAI()
    llm.complete("sys", "the chunk", cache_prefix="<document>long</document>\n\n")
    assert llm._client.requests[0]["messages"][1]["content"] == "<document>long</document>\n\nthe chunk"
    assert llm.usage == {"calls": 1, "input": 1500, "output": 7, "cache_read": 1024, "cache_write": 0}


def test_fake_llm_accepts_cache_prefix():
    llm = FakeLLM("x")
    llm.complete("s", "u", cache_prefix="p:")
    assert llm.calls == [("s", "p:u")]


def test_contextual_passes_document_as_prefix(tmp_path):
    doc = Document("/d/a.md", "# A\n\nAlpha text.", {"title": "A"})
    chunk = Chunk("/d/a.md#0", "/d/a.md", "Alpha text.", 0, 5, 16, context="A")
    seen = []

    class Spy(FakeLLM):
        def complete(self, system, user, cache_prefix=None):
            seen.append((cache_prefix, user))
            return "ctx"
    add_llm_context(doc, [chunk], Spy(), ContextCache(tmp_path / "c.sqlite"))
    prefix, user = seen[0]
    assert prefix.startswith("<document>\n# A") and prefix.rstrip().endswith("</document>")
    assert user.startswith("<chunk>") and "Alpha text." in user


def make(tmp_path, **cfg):
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               chunker="fixed", parent_tokens=0, **cfg)
    llm = FakeLLM("Answer [1].")
    return Raggy(c, embedder=FakeEmbedder(), llm=llm), llm


def hit(score):
    return Hit(Chunk("a#0", "a", "text", 0, 0, 4), score)


def test_gate_refuses_below_threshold_without_llm_call(tmp_path):
    r, llm = make(tmp_path, refusal_gate={"vector": 0.6})
    a = r.answer_from("q", [hit(0.4), hit(0.3)])
    assert a.text == NOT_FOUND and llm.calls == [] and a.verification == "gated"
    b = r.answer_from("q", [hit(0.7)])
    assert b.text == "Answer [1]." and len(llm.calls) == 1


def test_gate_off_without_calibration_for_mode(tmp_path):
    r, llm = make(tmp_path, refusal_gate={"vector": 0.6}, retrieval="hybrid")
    assert r.answer_from("q", [hit(0.01)]).text == "Answer [1]."  # hybrid isn't calibrated: no gate
    r2, _ = make(tmp_path / "2", refusal_gate={"vector+local": 0.6}, rerank="local")
    assert r2.gate_key() == "vector+local"


def test_gate_env_parsing(monkeypatch):
    monkeypatch.setattr("dotenv.find_dotenv", lambda **kw: "")
    monkeypatch.setenv("RAGGY_REFUSAL_GATE", "vector=0.62, hybrid=0.031")
    assert Config.from_env().refusal_gate == {"vector": 0.62, "hybrid": 0.031}


def test_runner_records_llm_usage(tmp_path):
    from raggy.evals.runner import run_suite
    from week04.test_pipeline_chunking import tiny_suite
    r, llm = make(tmp_path)
    llm.usage = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    real = llm.complete

    def counting(system, user, cache_prefix=None):
        llm.usage["calls"] += 1
        llm.usage["input"] += 10
        return real(system, user, cache_prefix)
    llm.complete = counting
    run = run_suite(tiny_suite(), r, None, answers=True)
    assert run["llm_usage"]["calls"] == 1 and run["llm_usage"]["input"] == 10  # tiny_suite has one question


def test_calibrate_gate_on_half_measures_on_other():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cal", Path(__file__).resolve().parents[2] / "evals" / "calibrate_gate.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    q = lambda i, ans, top: {"id": f"q{i}", "answerable": ans, "retrieved": [{"score": top}]}
    run = {"questions": [q(0, True, 0.80), q(1, True, 0.70), q(2, False, 0.40), q(3, False, 0.50),
                         q(4, True, 0.65), q(5, True, 0.90), q(6, False, 0.72), q(7, False, 0.30)]}
    r = cal.calibrate(run)
    assert r["threshold"] == 0.70  # lowest answerable top score in the odd half (q1 0.70, q5 0.90)
    assert r["calibrated_on"] == 4 and r["held_out"] == 4
    # held-out (even): q0 ans 0.80 ok, q2 unans 0.40 gated, q4 ans 0.65 wrongly gated, q6 unans 0.72 not gated
    assert (r["unanswerable_gated"], r["unanswerable"], r["answerable_gated"], r["answerable"]) == (1, 2, 1, 2)
