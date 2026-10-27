"""Agentic retrieval (bonus week): the model decides whether it has enough, and what to search next.

One search pass finds passages that look like the question. A multi-hop
question ("which centre repairs the motor of the bike that seats two
children?") needs a fact the question doesn't name: first that the bike is
the Ebb, then who repairs the Ebb's motor. The second search can only be
written after reading the first result.

So retrieval becomes a loop, the same loop as harnessy's week 2 agent loop,
with one tool:

    search the question  →  show the model what was found
        → it replies {"action": "search", "query": "..."}    → search, repeat
        → it replies {"action": "answer"}                   → answer from everything found
        → it replies {"action": "not_found"}                → refuse, no answer call

The decisions are JSON in plain text, not a provider's native tool calls,
so the loop works the same on Ollama, OpenAI and Claude (and with the fake
model in tests). The answer itself goes through the normal cited-answer
path, so citations, checks and the answer cache all still apply.

Every way the loop can end is recorded (`stopped`): answer, not_found,
limit (RAGGY_AGENT_STEPS searches), repeat (a search it already made),
unreadable (a reply that isn't a decision) or error (the decision call
failed). The last two are counted as failures; neither loses the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raggy.evals.judge import parse_json_object
from raggy.generate.llm import LLMError
from raggy.generate.prompt import NOT_FOUND
from raggy.generate.prompt import _source_label as source_label
from raggy.types import Answer, Hit

AGENT_VERSION = "agent-v1"
PASSAGE_CHARS = 700  # per passage in the decision prompt: enough to judge, cheap to repeat each step
MAX_PASSAGES = 12

AGENT_SYSTEM = """You help answer a question from a company's documents by deciding what to search for.
You see the question, the searches made so far, and the passages they found.

Decide one thing:
- If the passages contain everything needed to answer, reply {"action": "answer"}.
- If something is missing, reply {"action": "search", "query": "<a short search for the missing fact>"}.
  Use what the passages taught you: if they name the product, place or person the question
  describes, search for that name. Never repeat a search already made.
- If the documents clearly don't cover the question, reply {"action": "not_found"}.

Reply with JSON only."""


@dataclass
class Step:
    query: str
    hits: list[Hit]  # what this search returned (before de-duplication)
    new: int  # passages it added that earlier searches hadn't found


@dataclass
class AgentResult:
    answer: Answer
    passages: list[Hit]  # every distinct passage found, in the order found
    steps: list[Step] = field(default_factory=list)
    stopped: str = "answer"  # answer | not_found | limit | repeat | unreadable | error
    failures: int = 0  # unreadable or failed decisions


def parse_action(text: str) -> tuple[str, str | None] | None:
    """("search", query), ("answer", None) or ("not_found", None); None if the reply isn't one."""
    try:
        obj = parse_json_object(text)
    except ValueError:
        return None
    action = str(obj.get("action", "")).strip().lower()
    if action == "search":
        query = " ".join(str(obj.get("query") or "").split())
        return ("search", query) if query else None
    if action in ("answer", "not_found"):
        return action, None
    return None


def _prompt(question: str, steps: list[Step], passages: list[Hit]) -> str:
    searches = "\n".join(f"- {s.query}" for s in steps)
    shown = "\n\n".join(f"[{n}] {source_label(h)}\n{h.chunk.text[:PASSAGE_CHARS]}"
                        for n, h in enumerate(passages, start=1))
    return f"Question: {question}\n\nSearches so far:\n{searches}\n\nPassages found:\n\n{shown or '(none)'}\n"


def run_agent(raggy, question: str, k: int | None = None, use_cache: bool = True, on_token=None) -> AgentResult:
    c = raggy.config
    k = k or c.top_k
    passages: list[Hit] = []
    seen: set[str] = set()
    steps: list[Step] = []
    failures, stopped = 0, "limit"

    def search(query: str) -> None:
        hits = raggy.search(query, k=k)
        fresh = [h for h in hits if h.chunk.id not in seen]
        for h in fresh:
            seen.add(h.chunk.id)
        passages.extend(fresh)
        steps.append(Step(query, hits, len(fresh)))

    search(question)  # the first search is the question itself: no decision needed to make it
    while len(steps) < c.agent_steps and len(passages) < MAX_PASSAGES:
        try:
            reply = raggy.llm.complete(AGENT_SYSTEM, _prompt(question, steps, passages))
        except LLMError:
            failures, stopped = failures + 1, "error"
            break
        action = parse_action(reply)
        if action is None:
            failures, stopped = failures + 1, "unreadable"
            break
        kind, query = action
        if kind != "search":
            stopped = kind
            break
        if " ".join(query.lower().split()) in {" ".join(s.query.lower().split()) for s in steps}:
            stopped = "repeat"
            break
        search(query)
    if stopped == "not_found":
        answer = Answer(question=question, text=NOT_FOUND, citations=[], hits=passages)
    else:
        answer = raggy.answer_from(question, passages[:MAX_PASSAGES], on_token=on_token, use_cache=use_cache)
    return AgentResult(answer, passages[:MAX_PASSAGES], steps, stopped, failures)
