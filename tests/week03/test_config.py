import pytest

import raggy.config as config_mod
from raggy.config import Config, ConfigError
from raggy.ingest.loaders import LoadOptions


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.find_dotenv", lambda **kw: "")
    for k in ("RAGGY_HTML_EXTRACT", "RAGGY_DEDUPE"):
        monkeypatch.delenv(k, raising=False)


def test_ingest_defaults():
    c = Config.from_env()
    assert (c.html_extract, c.dedupe) == ("main", True)
    assert c.load_options() == LoadOptions(html_extract="main")


def test_bad_html_extract_raises(monkeypatch):
    monkeypatch.setenv("RAGGY_HTML_EXTRACT", "everything")
    with pytest.raises(ConfigError, match="RAGGY_HTML_EXTRACT"):
        Config.from_env()


@pytest.mark.parametrize("raw,expected", [("0", False), ("false", False), ("No", False),
                                          ("1", True), ("yes", True), ("", True)])
def test_dedupe_flag_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("RAGGY_DEDUPE", raw)
    assert Config.from_env().dedupe is expected


def test_bad_dedupe_raises(monkeypatch):
    monkeypatch.setenv("RAGGY_DEDUPE", "maybe")
    with pytest.raises(ConfigError, match="RAGGY_DEDUPE"):
        Config.from_env()
