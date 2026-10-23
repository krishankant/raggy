from raggy.generate.prompt import NOT_FOUND, SYSTEM_PROMPT, build_user_prompt, parse_citations
from raggy.types import Chunk, Hit


def hit(n: int, text: str, source: str = "/docs/a.md") -> Hit:
    return Hit(Chunk(id=f"{source}#{n}", source=source, text=text, index=n, start=0, end=len(text)), score=1.0 - n / 10)


def test_system_prompt_demands_grounding_and_not_found_sentence():
    assert NOT_FOUND in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT


def test_prompt_numbers_sources_in_rank_order():
    p = build_user_prompt("Who?", [hit(0, "first text", "/d/one.md"), hit(1, "second text", "/d/two.md")])
    assert p.index("[1] (source: one.md)") < p.index("first text") < p.index("[2] (source: two.md)") < p.index("second text")
    assert p.rstrip().endswith("Question: Who?")


def test_parse_citations_dedup_order_and_range():
    hits = [hit(0, "zero"), hit(1, "one " * 100)]
    cites = parse_citations("A [2]. B [1][2]. C [9] and [0].", hits)
    assert [c.number for c in cites] == [2, 1]
    assert cites[0].chunk_id == "/docs/a.md#1"
    assert len(cites[0].snippet) <= 160


def test_parse_citations_handles_comma_lists():
    hits = [hit(0, "zero"), hit(1, "one")]
    assert [c.number for c in parse_citations("X [1, 2].", hits)] == [1, 2]
