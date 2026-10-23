from conftest import make_raggy
from raggy.evals.report import format_compare, format_run, load_run, save_run
from raggy.evals.runner import run_suite


def test_save_and_load_run(tmp_path, suite):
    run = run_suite(suite, make_raggy(tmp_path / "idx"), None, label="week01")
    path = save_run(run, tmp_path / "runs")
    assert path.parent == tmp_path / "runs" / "tiny" and path.name.endswith("-week01.json")
    assert load_run(path) == run


def test_format_run_mentions_key_numbers(tmp_path, suite):
    text = format_run(run_suite(suite, make_raggy(tmp_path / "idx"), None))
    assert "recall@1" in text and "refusal_accuracy" in text and "lookup" in text


def test_format_compare_shows_delta():
    base = {"suite": "s", "label": "a", "metrics": {"recall@4": 0.5, "mrr": 0.4, "correctness": None}}
    new = {"suite": "s", "label": "b", "metrics": {"recall@4": 0.75, "mrr": 0.3, "correctness": 0.9}}
    text = format_compare(base, new)
    assert "+0.250" in text and "-0.100" in text
    assert "correctness" in text  # shown even when one side is missing
