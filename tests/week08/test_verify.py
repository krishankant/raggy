import json

from raggy.generate.fake import FakeLLM
from raggy.generate.prompt import NOT_FOUND
from raggy.generate.verify import apply_action, lexical_support, verify
from raggy.types import Chunk, Hit


def p(text, n=0):
    return Hit(Chunk(f"s#{n}", f"/d/{n}.md", text, 0, 0, len(text)), 1.0)


P = [p("Refunds are paid back within 14 days of the return arriving.", 0),
     p("The office cat is called Biscuit and sleeps on the printer.", 1)]


def test_lexical_supported():
    assert lexical_support("Refunds are paid within 14 days.", P[0].chunk.text) >= 0.8


def test_lexical_unsupported():
    assert lexical_support("Refunds take 30 days by bank transfer.", P[1].chunk.text) < 0.3


def test_verify_marks_each_citation():
    v = verify("Refunds arrive within 14 days [1]. The cat is called Biscuit [1].", P, method="lexical")
    assert [(c.n, c.supported) for c in v.checks] == [(1, True), (1, False)]
    assert v.unsupported == [] and v.method == "lexical"  # [1] is supported somewhere, so not listed


def test_verify_handles_runs_and_out_of_range():
    v = verify("Refunds take 14 days [1][2]. Biscuit sleeps on the printer [2, 7].", P, method="lexical")
    assert [(c.sentence, c.n, c.supported) for c in v.checks] == [(0, 1, True), (0, 2, False), (1, 2, True)]
    # [7] isn't a passage: never checked, never invented


def test_action_strip_removes_marker_only():
    text = "Refunds take 14 days [1][2]. The cat is Biscuit [2]."
    v = verify(text, P, method="lexical")
    out = apply_action(text, v, "strip")
    assert out == "Refunds take 14 days [1]. The cat is Biscuit [2]."


def test_action_refuse_when_none_supported():
    text = "Refunds go by bank transfer in 30 working days [2]."
    v = verify(text, P, method="lexical")
    assert apply_action(text, v, "refuse") == NOT_FOUND
    assert apply_action(text, v, "report") == text


def test_refusal_is_not_verified():
    v = verify(NOT_FOUND, P, method="lexical")
    assert v.checks == [] and apply_action(NOT_FOUND, v, "refuse") == NOT_FOUND


def test_llm_verifier_parses_verdicts():
    llm = FakeLLM(json.dumps({"verdicts": [{"i": 0, "supported": True}, {"i": 1, "supported": False}]}))
    v = verify("Refunds take two weeks [1]. The cat is Biscuit [1].", P, method="llm", llm=llm)
    assert [(c.n, c.supported) for c in v.checks] == [(1, True), (1, False)]
    system, user = llm.calls[0]
    assert "two weeks" in user and "Refunds are paid back" in user


def test_llm_verifier_garbage_keeps_answer():
    text = "Refunds take two weeks [1]."
    v = verify(text, P, method="llm", llm=FakeLLM("no idea"))
    assert v.failed and v.checks == [] and apply_action(text, v, "strip") == text


def test_ask_applies_configured_check(tmp_path):
    from raggy.config import Config
    from raggy.embed.fake import FakeEmbedder
    from raggy.pipeline import Raggy
    from raggy.types import Document
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               chunker="fixed", parent_tokens=0, citation_check="lexical", citation_action="strip")
    r = Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("Refunds arrive within 14 days [1]. Cats like boxes [1]."))
    r.ingest_documents([Document("/d/r.md", "Refunds are paid back within 14 days of the return arriving.")])
    a = r.ask("refunds?", k=1)
    assert a.text == "Refunds arrive within 14 days [1]. Cats like boxes." and a.verification == "lexical"
    assert a.unsupported == []  # [1] stays supported by the first sentence


def test_cli_prints_unsupported_warning(tmp_path, monkeypatch, capsys):
    from raggy import cli
    from raggy.config import Config
    from raggy.embed.fake import FakeEmbedder
    from raggy.pipeline import Raggy
    from raggy.types import Document
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               chunker="fixed", parent_tokens=0, citation_check="lexical")
    r = Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM("Cats like boxes [1]."))
    r.ingest_documents([Document("/d/r.md", "Refunds are paid back within 14 days.")])
    monkeypatch.setattr(cli, "_build_raggy", lambda config: r)
    assert cli.main(["ask", "refunds?"]) == 0
    assert "citation [1] is not supported by its source" in capsys.readouterr().out
