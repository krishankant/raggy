from raggy.ingest.normalize import dehyphenate, normalize_text


def test_normalize_collapses_inner_space_keeps_indent():
    text = "a   b\t\tc   \n    def f():\n        return  1"
    assert normalize_text(text) == "a b c\n    def f():\n        return 1"


def test_normalize_nfkc_and_invisible_chars():
    assert normalize_text("﻿ofﬁce co­op​era‍tive") == "office cooperative"


def test_normalize_newlines():
    assert normalize_text("a\r\nb\rc\n\n\n\n\nd\n") == "a\nb\nc\n\nd"


def test_normalize_idempotent():
    messy = "  Title \r\n\r\n\r\n  body  text​  \n"
    once = normalize_text(messy)
    assert normalize_text(once) == once


def test_dehyphenate_only_lowercase_wraps():
    assert dehyphenate("exam-\nple and 8-\nSpeed and BW-\nCH") == "example and 8-\nSpeed and BW-\nCH"
