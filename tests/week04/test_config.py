import pytest

from raggy.chunking import ChunkSettings
from raggy.config import Config, ConfigError


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.find_dotenv", lambda **kw: "")
    for k in ("RAGGY_CHUNKER", "RAGGY_CHUNK_TOKENS", "RAGGY_CHUNK_OVERLAP_TOKENS", "RAGGY_CHUNK_HEADERS",
              "RAGGY_PARENT_TOKENS", "RAGGY_CHUNK_CONTEXT"):
        monkeypatch.delenv(k, raising=False)


def test_chunk_settings_from_env(monkeypatch):
    monkeypatch.setenv("RAGGY_CHUNKER", "recursive")
    monkeypatch.setenv("RAGGY_CHUNK_TOKENS", "128")
    monkeypatch.setenv("RAGGY_CHUNK_OVERLAP_TOKENS", "16")
    monkeypatch.setenv("RAGGY_CHUNK_HEADERS", "0")
    monkeypatch.setenv("RAGGY_PARENT_TOKENS", "512")
    s = Config.from_env().chunk_settings()
    assert s == ChunkSettings(chunker="recursive", tokens=128, overlap=16, chars=800, char_overlap=100,
                              headers=False, parent_tokens=512)


@pytest.mark.parametrize("env,match", [
    ({"RAGGY_CHUNKER": "semantic"}, "RAGGY_CHUNKER"),
    ({"RAGGY_CHUNK_TOKENS": "64", "RAGGY_CHUNK_OVERLAP_TOKENS": "64"}, "smaller than"),
    ({"RAGGY_CHUNK_TOKENS": "500"}, "512"),
    ({"RAGGY_PARENT_TOKENS": "100"}, "RAGGY_PARENT_TOKENS"),
    ({"RAGGY_CHUNK_CONTEXT": "gpt"}, "RAGGY_CHUNK_CONTEXT"),
])
def test_bad_chunk_settings(monkeypatch, env, match):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ConfigError, match=match):
        Config.from_env()


def test_settings_key_is_stable_dict():
    k = ChunkSettings().key()
    assert set(k) == {"chunker", "tokens", "overlap", "chars", "char_overlap", "headers", "parent_tokens"}
