import dotenv
import pytest

from raggy import cli
from raggy.config import Config, ConfigError


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)  # never read the real .env
    for var in ("RAGGY_JUDGE", "RAGGY_JUDGE_MODEL", "ANTHROPIC_API_KEY", "RAGGY_TOP_K"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_judge_defaults_to_anthropic_with_key(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Config.from_env().judge_provider == "anthropic"


def test_judge_falls_back_to_ollama_without_key(clean_env):
    assert Config.from_env().judge_provider == "ollama"


def test_judge_env_override(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
    clean_env.setenv("RAGGY_JUDGE", "openai")
    clean_env.setenv("RAGGY_JUDGE_MODEL", "gpt-x")
    cfg = Config.from_env()
    assert (cfg.judge_provider, cfg.judge_model) == ("openai", "gpt-x")


@pytest.mark.parametrize("bad", ["abc", "0", "-3", ""])
def test_bad_top_k_raises_config_error(clean_env, bad):
    clean_env.setenv("RAGGY_TOP_K", bad)
    with pytest.raises(ConfigError, match="RAGGY_TOP_K"):
        Config.from_env()


def test_cli_bad_top_k_one_line_error(clean_env, capsys):
    clean_env.setenv("RAGGY_TOP_K", "abc")
    assert cli.main(["sources"]) == 1
    err = capsys.readouterr().err
    assert "RAGGY_TOP_K" in err and "Traceback" not in err
