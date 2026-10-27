"""The course's final benchmark (week 12): every suite, under each profile, in one table.

    uv run python evals/benchmark.py                       # default and accurate, retrieval only
    uv run python evals/benchmark.py --profiles fast accurate --answers

Each (suite, profile) pair is one `raggy eval` run, saved under evals/runs/
as usual with the label bench-<profile>. The table is written to
evals/baselines/week12/benchmark.md. Suites that need models you don't have
(media needs RAGGY_MEDIA, which the `accurate` profile turns on) are run
anyway; a run that fails is reported in the table, not hidden.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SUITES = ["fictional", "fictional-web", "manual", "rich", "images", "media", "scifact", "fiqa"]
DOC_SCORED = {"scifact", "fiqa"}
OUT = Path("evals/baselines/week12/benchmark.md")


def run(suite: str, profile: str, answers: bool) -> dict | None:
    env = dict(os.environ)
    env.pop("RAGGY_PROFILE", None)
    if profile != "default":
        env["RAGGY_PROFILE"] = profile
    label = f"bench-{profile}"
    cmd = ["uv", "run", "raggy", "eval", "--suite", suite, "--label", label]
    if not answers:
        cmd.append("--retrieval-only")
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        print(f"  {suite} / {profile}: FAILED\n{done.stderr[-400:]}", file=sys.stderr)
        return None
    runs = sorted(Path("evals/runs", suite).glob(f"*-{label}.json"))
    return json.loads(runs[-1].read_text()) if runs else None


def cell(run: dict | None, suite: str, answers: bool) -> str:
    if run is None:
        return "failed"
    m = run["metrics"]
    keys = ["ndcg@10", "recall@100"] if suite in DOC_SCORED else ["recall@1", "recall@4", "mrr"]
    if answers and m.get("correctness") is not None:
        keys.append("correctness")
    vals = " / ".join(f"{m[k]:.3f}" for k in keys if m.get(k) is not None)
    p95 = m.get("retrieve_ms_p95")
    warn = " ⚠" if run.get("fallbacks") and any(run["fallbacks"].values()) else ""
    return f"{vals} ({p95:.0f} ms){warn}" if p95 is not None else vals + warn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["default", "accurate"])
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--answers", action="store_true", help="also generate and judge answers (paid calls)")
    args = ap.parse_args()
    rows = []
    for suite in args.suites:
        print(f"{suite} …", file=sys.stderr)
        rows.append((suite, [cell(run(suite, p, args.answers), suite, args.answers) for p in args.profiles]))
    head = "| suite | " + " | ".join(args.profiles) + " |"
    lines = [head, "|" + " --- |" * (len(args.profiles) + 1)]
    lines += [f"| {s} | " + " | ".join(cells) + " |" for s, cells in rows]
    note = ("Cells: recall@1 / recall@4 / MRR (nDCG@10 / recall@100 for SciFact and FiQA)"
            + (" / correctness" if args.answers else "") + ", then retrieval p95. ⚠ = a stage fell back.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# Final benchmark\n\n" + note + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
