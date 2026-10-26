"""Save, load, print and compare eval runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def save_run(run: dict, out_dir: str | Path) -> Path:
    folder = Path(out_dir) / run["suite"]
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(run["created"]).strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-{run['label']}" if run["label"] else stamp
    path, n = folder / f"{stem}.json", 2
    while path.exists():
        path, n = folder / f"{stem}-{n}.json", n + 1
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x) -> str:
    if x is None:
        return "—"
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.3f}"


_GROUPS = [
    ("Retrieval (answerable questions)", lambda k: "@" in k or k == "mrr"),
    ("Answers", lambda k: k in {"correctness", "faithfulness", "citation_precision", "refusal_accuracy",
                                "unanswerable_refused", "answerable_refused"}),
    ("Latency (ms)", lambda k: k.endswith(("_p50", "_p95"))),
]


def format_run(run: dict) -> str:
    c, counts, m = run["config"], run["counts"], run["metrics"]
    lines = [
        f"Suite {run['suite']}: {counts['questions']} questions "
        f"({counts['answerable']} answerable, {counts['unanswerable']} unanswerable)"
        + (f"  label: {run['label']}" if run["label"] else ""),
        f"Embedder {c['embed_model']} · chunks {c['chunk_size']}/{c['chunk_overlap']} · top_k {c['top_k']}"
        f" · scored per {c['unit']}",
        f"LLM {c['llm'] or '—'} · judge {c['judge'] or '—'}"
        + (f" ({c['judge_version']})" if c["judge_version"] else ""),
    ]
    ing = run["ingest"]
    lines.append("Index: cached" if ing.get("cached") else
                 f"Index: {ing['documents']} documents → {ing['chunks']} chunks in {ing['seconds']} s")
    for title, keep in _GROUPS:
        rows = [(k, v) for k, v in m.items() if keep(k)]
        if rows and any(v is not None for _, v in rows):
            lines.append(f"\n{title}")
            lines += [f"  {k:<22}{_fmt(v):>8}" for k, v in rows]
    labels = {"transform": "query transform", "rerank": "rerank"}
    for stage, n in (run.get("fallbacks") or {}).items():
        if n:
            total = (run.get("fallback_of") or {}).get(stage, n)
            lines.append(f"\n⚠ {n} of {total} {labels.get(stage, stage)} calls failed and fell back: "
                         "these numbers are not that technique's (check the model / API key)")
    budgets = c.get("budgets") or {}
    stages = [s for s in ("transform", "retrieve", "rerank", "generate") if m.get(f"{s}_ms_p95") is not None]
    if budgets and stages:
        lines.append(f"\n{'Stage':<12}{'p50 ms':>9}{'p95 ms':>9}{'budget':>9}")
        for s in stages:
            ok = "✓" if m[f"{s}_ms_p95"] <= budgets.get(s, float("inf")) else "✗"
            lines.append(f"  {s:<10}{_fmt(m[f'{s}_ms_p50']):>9}{_fmt(m[f'{s}_ms_p95']):>9}{budgets.get(s, '—'):>8} {ok}")
    rk = next((k for k in m if k.startswith("recall@") and k != "recall@1"), "recall@1")
    lines.append(f"\nBy tag            n  {rk:>9}      mrr  correct")
    for tag, t in run["by_tag"].items():
        lines.append(f"  {tag:<14}{t['n']:>3}  {_fmt(t.get(rk)):>9} {_fmt(t.get('mrr')):>8} {_fmt(t.get('correctness')):>8}")
    if counts["errors"] or counts["judge_errors"]:
        lines.append(f"\n{counts['errors']} questions errored, {counts['judge_errors']} judge errors:")
        for q in run["questions"]:
            err = q.get("error") or (q.get("judge") or {}).get("error")
            if err:
                lines.append(f"  {q['id']}: {err[:140]}")
    return "\n".join(lines)


def format_compare(base: dict, new: dict) -> str:
    names = (base["label"] or "base", new["label"] or "new")
    w = max(10, *(len(n) + 2 for n in names))  # long labels must not run into each other
    lines = [f"{'metric':<22}{names[0]:>{w}}{names[1]:>{w}}{'delta':>10}"]
    for k in base["metrics"]:
        if k not in new["metrics"]:
            continue
        a, b = base["metrics"][k], new["metrics"][k]
        if isinstance(a, list) or isinstance(b, list):  # e.g. over_budget: shown below, not subtracted
            continue
        delta = f"{b - a:+.3f}" if a is not None and b is not None else "—"
        lines.append(f"{k:<22}{_fmt(a):>{w}}{_fmt(b):>{w}}{delta:>10}")
    for side, run in (("base", base), ("new", new)):
        if run["metrics"].get("over_budget"):
            lines.append(f"{side}: over budget at p95: {', '.join(run['metrics']['over_budget'])}")
    for side, run in (("base", base), ("new", new)):
        c = run.get("config", {})
        if c:
            lines.append(f"{side}: {c.get('embed_model')} · chunks {c.get('chunk_size')}/{c.get('chunk_overlap')}"
                         f" · llm {c.get('llm')} · judge {c.get('judge_version')}"
                         + (f" · html {c['html_extract']} · dedupe {'on' if c.get('dedupe') else 'off'}"
                            if "html_extract" in c else ""))
    if base.get("config", {}).get("judge_version") != new.get("config", {}).get("judge_version"):
        lines.append("note: judge versions differ, so answer scores aren't comparable")
    return "\n".join(lines)
