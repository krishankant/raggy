"""Pick a refusal-gate threshold from a retrieval run, without grading it on the same questions (week 8).

The gate answers "not in the documents" without calling the model when the
best retrieved score is below a threshold. The threshold is the lowest top score
of any *answerable* question in one half (odd positions), so that half loses
none; then it's measured on the other half (even positions), which it never saw.

    uv run python evals/calibrate_gate.py evals/runs/fictional/<run>.json [more runs...]
"""

import json
import sys


def calibrate(run: dict) -> dict:
    qs = run["questions"]
    calib, held = qs[1::2], qs[0::2]
    top = lambda q: q["retrieved"][0]["score"] if q.get("retrieved") else 0.0
    answerable_tops = [top(q) for q in calib if q["answerable"]]
    t = min(answerable_tops) if answerable_tops else 0.0
    ans = [q for q in held if q["answerable"]]
    unans = [q for q in held if not q["answerable"]]
    return {"threshold": round(t, 4), "calibrated_on": len(calib), "held_out": len(held),
            "unanswerable_gated": sum(top(q) < t for q in unans), "unanswerable": len(unans),
            "answerable_gated": sum(top(q) < t for q in ans), "answerable": len(ans)}


def main() -> None:
    merged = {"questions": []}
    for path in sys.argv[1:]:
        merged["questions"] += json.load(open(path))["questions"]
    r = calibrate(merged)
    print(f"threshold {r['threshold']} (from {r['calibrated_on']} questions)")
    print(f"held out ({r['held_out']}): unanswerable gated {r['unanswerable_gated']}/{r['unanswerable']}, "
          f"answerable wrongly gated {r['answerable_gated']}/{r['answerable']}")


if __name__ == "__main__":
    main()
