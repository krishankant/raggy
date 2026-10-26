from raggy.evals.report import format_compare


def run(label, recall, **cfg):
    return {"label": label, "metrics": {"recall@4": recall},
            "config": {"embed_model": "m", "chunk_size": 800, "chunk_overlap": 100, "llm": None,
                       "judge_version": None, **cfg}}


def test_compare_long_labels_stay_in_columns():
    out = format_compare(run("week03-full-dedupe0", 0.758), run("week03-main-dedupe1", 0.939))
    header, row = out.splitlines()[:2]
    assert header.split() == ["metric", "week03-full-dedupe0", "week03-main-dedupe1", "delta"]
    assert row.split() == ["recall@4", "0.758", "0.939", "+0.181"]


def test_compare_shows_ingest_settings():
    out = format_compare(run("a", 0.7, html_extract="full", dedupe=False), run("b", 0.9, html_extract="main", dedupe=True))
    assert "html full · dedupe off" in out and "html main · dedupe on" in out
