from conftest import judge_llm, make_raggy, tiny_suite
from raggy.evals.judge import Judge
from raggy.evals.runner import prepare_index, run_suite
from raggy.generate.llm import LLMError
from raggy.generate.prompt import NOT_FOUND


def test_prepare_index_caches_then_rebuilds_when_settings_change(tmp_path, suite):
    idx = tmp_path / "idx"
    assert prepare_index(suite, make_raggy(idx))["cached"] is False
    assert prepare_index(suite, make_raggy(idx))["cached"] is True
    assert prepare_index(suite, make_raggy(idx, chunk_size=20))["cached"] is False
    assert prepare_index(tiny_suite(fingerprint="v2"), make_raggy(idx, chunk_size=20))["cached"] is False


def test_run_suite_scores_retrieval_answers_and_refusals(tmp_path, suite):
    def reply(system, user):
        return NOT_FOUND if "skateboards" in user else "It's in the source [1]."
    r = make_raggy(tmp_path / "idx", reply=reply)
    run = run_suite(suite, r, Judge(judge_llm()), label="t")
    m = run["metrics"]
    assert run["schema"] == 1 and run["suite"] == "tiny" and run["label"] == "t"
    assert run["counts"] == {"questions": 3, "answerable": 2, "unanswerable": 1, "errors": 0, "judge_errors": 0}
    assert m["recall@1"] == 1.0 and m["mrr"] == 1.0
    assert m["correctness"] == 1.0 and m["faithfulness"] == 1.0 and m["citation_precision"] == 1.0
    assert m["refusal_accuracy"] == 1.0 and m["unanswerable_refused"] == 1.0 and m["answerable_refused"] == 0.0
    assert m["retrieve_ms_p50"] is not None and m["generate_ms_p95"] is not None
    assert run["by_tag"]["lookup"]["recall@1"] == 1.0
    assert run["config"]["judge_version"] == "judge-v1" and run["config"]["llm"] == "fake:scripted"
    q = run["questions"][0]
    assert q["retrieved"][0]["gold"] == [0] and "retrieve" in q["latency_ms"]


def test_refused_answerable_scores_zero_without_judge_calls(tmp_path, suite):
    judge = judge_llm()
    run = run_suite(suite, make_raggy(tmp_path / "idx", reply=NOT_FOUND), Judge(judge))
    assert run["metrics"]["correctness"] == 0.0 and run["metrics"]["answerable_refused"] == 1.0
    assert judge.calls == []


def test_run_records_llm_error_and_continues(tmp_path, suite):
    run = run_suite(suite, make_raggy(tmp_path / "idx", error=LLMError("ollama is not running")), None)
    assert run["counts"]["errors"] == 3
    assert run["questions"][0]["error"] == "ollama is not running"
    assert run["metrics"]["recall@1"] == 1.0  # retrieval still scored


def test_retrieval_only_makes_no_llm_calls(tmp_path, suite):
    r = make_raggy(tmp_path / "idx")
    run = run_suite(suite, r, None, answers=False)
    assert r.llm.calls == [] and run["config"]["llm"] is None
    assert run["metrics"]["correctness"] is None and run["metrics"]["recall@2"] == 1.0


def test_run_document_unit_collapses(tmp_path):
    r = make_raggy(tmp_path / "idx", chunk_size=20)  # several chunks per document
    run = run_suite(tiny_suite(unit="document", answers=False), r, None)
    sources = [x["source"] for x in run["questions"][0]["retrieved"]]
    assert len(sources) == len(set(sources)) == 2


def test_limit_takes_first_questions(tmp_path, suite):
    run = run_suite(suite, make_raggy(tmp_path / "idx"), None, answers=False, limit=1)
    assert [q["id"] for q in run["questions"]] == ["a1"]


def test_metric_order_is_recall_mrr_ndcg(tmp_path, suite):
    keys = list(run_suite(suite, make_raggy(tmp_path / "idx"), None, answers=False)["metrics"])
    assert keys[:5] == ["recall@1", "recall@2", "mrr", "ndcg@1", "ndcg@2"]


def test_prepare_index_refuses_to_delete_a_folder_it_did_not_create(tmp_path, suite):
    import pytest
    from raggy.evals.runner import UnsafeIndexDirError
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "questions.jsonl").write_text("precious")
    with pytest.raises(UnsafeIndexDirError, match="idx"):
        prepare_index(suite, make_raggy(idx))
    assert (idx / "questions.jsonl").read_text() == "precious"


def test_prepare_index_still_rebuilds_its_own_folder(tmp_path, suite):
    idx = tmp_path / "idx"
    prepare_index(suite, make_raggy(idx))
    assert prepare_index(suite, make_raggy(idx, chunk_size=20))["cached"] is False
