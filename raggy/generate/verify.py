"""Check each citation against the passage it points at.

A model can cite [2] for a sentence that passage 2 doesn't support: the fact
came from passage 1, or from memory. Week 2's judge measures this after the
fact; this module checks it on every answer, before the user sees it.

- lexical (free, instant): the share of the sentence's content words (numbers
  included) that appear in the cited passage. Paraphrase lowers it; a wrong
  number or a different topic lowers it a lot.
- llm (one call per answer): a model reads each claim beside its cited
  passage and says supported or not.

Then an action: report (keep the text, list the unsupported markers), strip
(delete just the unsupported markers) or refuse (no supported citation left
means the answer isn't grounded: use the fixed refusal sentence instead).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from raggy.evals.judge import parse_json_object
from raggy.generate.llm import LLM, LLMError
from raggy.generate.prompt import NOT_FOUND
from raggy.types import Hit

_CITE = re.compile(r"\s?\[(\d+(?:\s*,\s*\d+)*)\]")
_BOUNDARY = re.compile(r"(?<=[.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])*)\s+(?=\S)")
_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been being it its
this that these those there their they them you your we our i me my he she his her not no do does did can could
will would should may might must have has had within into about than then so such what which who whom when where
why how all any each also just only more most very""".split())


@dataclass
class Check:
    sentence: int  # index of the sentence in the answer
    n: int  # the cited passage number
    score: float | None  # lexical support; None for the llm method
    supported: bool


@dataclass
class Verification:
    method: str
    checks: list[Check] = field(default_factory=list)
    failed: bool = False  # the llm verifier couldn't produce verdicts

    @property
    def unsupported(self) -> list[int]:
        """Passage numbers that weren't supported in any sentence citing them."""
        bad = {c.n for c in self.checks if not c.supported}
        good = {c.n for c in self.checks if c.supported}
        return sorted(bad - good)


def sentences(text: str) -> list[tuple[int, int]]:
    """Spans of sentences; markers after the full stop stay with their sentence."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        spans.append((start, m.end(1)))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(_CITE.sub(" ", text).lower()):
        if w in STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def lexical_support(sentence: str, passage: str) -> float:
    claim = _words(sentence)
    return 1.0 if not claim else len(claim & _words(passage)) / len(claim)


def _cited(text: str, passages: list[Hit]) -> list[tuple[int, int, str]]:
    """(sentence index, passage number, sentence text) for every in-range marker."""
    out = []
    for i, (s, e) in enumerate(sentences(text)):
        sent = text[s:e]
        for group in _CITE.findall(sent):
            for part in group.split(","):
                n = int(part)
                if 1 <= n <= len(passages) and (i, n, sent) not in out:
                    out.append((i, n, sent))
    return out


LLM_SYSTEM = ("You check citations. For each claim, decide whether the cited source states what the claim says "
              "(paraphrase is fine; extra facts not in the source are not). Reply with JSON only.")


def verify(text: str, passages: list[Hit], method: str = "lexical", threshold: float = 0.5,
           llm: LLM | None = None) -> Verification:
    v = Verification(method=method)
    if NOT_FOUND.lower() in text.lower():
        return v  # a refusal makes no claims
    cited = _cited(text, passages)
    if method == "lexical":
        for i, n, sent in cited:
            score = lexical_support(sent, passages[n - 1].chunk.text)
            v.checks.append(Check(i, n, round(score, 3), score >= threshold))
        return v
    if not cited:
        return v
    claims = "\n\n".join(f"Claim {k}: {_CITE.sub('', sent).strip()}\nSource [{n}]: {passages[n - 1].chunk.text}"
                         for k, (i, n, sent) in enumerate(cited))
    user = claims + '\n\nReturn {"verdicts": [{"i": claim number, "supported": true or false}, ...]}'
    try:
        verdicts = parse_json_object(llm.complete(LLM_SYSTEM, user))["verdicts"]
        by_i = {int(x["i"]): bool(x["supported"]) for x in verdicts}
    except (LLMError, ValueError, KeyError, TypeError):
        v.failed = True
        return v
    for k, (i, n, sent) in enumerate(cited):
        v.checks.append(Check(i, n, None, by_i.get(k, True)))  # no verdict for a claim: leave it alone
    return v


def apply_action(text: str, v: Verification, action: str) -> str:
    if action == "report" or v.failed or not v.checks:
        return text
    if action == "refuse":
        return text if any(c.supported for c in v.checks) else NOT_FOUND
    bad = {(c.sentence, c.n) for c in v.checks if not c.supported}
    out = []
    for i, (s, e) in enumerate(sentences(text)):
        def keep(m: re.Match, i=i) -> str:
            nums = [p.strip() for p in m.group(1).split(",") if (i, int(p)) not in bad]
            return f"{m.group(0)[0] if m.group(0)[0] == ' ' else ''}[{', '.join(nums)}]" if nums else ""
        out.append((s, e, _CITE.sub(keep, text[s:e])))
    rebuilt, pos = [], 0
    for s, e, new in out:
        rebuilt.append(text[pos:s] + new)
        pos = e
    rebuilt.append(text[pos:])
    return "".join(rebuilt)
