import dotenv
import pytest

from conftest import judge_llm, make_raggy, tiny_suite
from raggy import cli
from raggy.evals import suites
from raggy.evals.judge import Judge
from raggy.evals.report import save_run
from raggy.evals.runner import run_suite


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("RAGGY_INDEX_DIR", str(tmp_path / ".raggy"))
    monkeypatch.setitem(suites.SUITES, "tiny", tiny_suite)
    built = {}

    def build(cfg):
        built["config"] = cfg
        return make_raggy(cfg.index_dir)
    monkeypatch.setattr(cli, "_build_raggy", build)
    monkeypatch.setattr(cli, "_build_judge", lambda cfg: Judge(judge_llm()))
    return built


def test_cli_eval_runs_and_saves(env, tmp_path, capsys):
    assert cli.main(["eval", "--suite", "tiny", "--label", "x", "--out", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "recall@1" in out and "saved" in out.lower()
    assert list((tmp_path / "runs" / "tiny").glob("*-x.json"))
    assert env["config"].index_dir == tmp_path / ".raggy" / "evals" / "tiny"  # never the user's index


def test_cli_eval_rejects_unknown_suite(env, capsys):
    with pytest.raises(SystemExit):
        cli.main(["eval", "--suite", "nope"])


def test_cli_eval_dataset_error_is_one_line(env, monkeypatch, capsys):
    def broken(options=None):
        raise suites.DatasetError("could not download https://x: offline")
    monkeypatch.setitem(suites.SUITES, "tiny", lambda: tiny_suite().__class__(
        name="tiny", documents=broken, questions=broken, fingerprint=lambda: "z",
        ks=(1,), unit="chunk", answers=False))
    assert cli.main(["eval", "--suite", "tiny", "--out", "runs"]) == 1
    assert "offline" in capsys.readouterr().err


def test_cli_compare(env, tmp_path, capsys):
    a = save_run(run_suite(tiny_suite(), make_raggy(tmp_path / "i1"), None, label="a"), tmp_path / "r")
    b = save_run(run_suite(tiny_suite(), make_raggy(tmp_path / "i2"), None, label="b"), tmp_path / "r")
    assert cli.main(["compare", str(a), str(b)]) == 0
    assert "recall@1" in capsys.readouterr().out


def test_cli_eval_announces_judge_before_running(env, tmp_path, capsys):
    cli.main(["eval", "--suite", "tiny", "--out", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert "judge: fake:scripted" in err and "RAGGY_JUDGE" in err
