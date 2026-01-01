from __future__ import annotations

import pytest

pytest.importorskip("sqlmodel")

from services.pipeline import sanitize_header
from services.ai import is_spam_or_unsafe


def test_sanitize_header_removes_crlf():
    assert sanitize_header("Hello\r\nWorld") == "Hello World"
    assert sanitize_header("Test\n\n") == "Test"


def test_spam_or_unsafe_short_or_keywords():
    assert is_spam_or_unsafe(" ") is True
    assert is_spam_or_unsafe("buy viagra now") is True
    assert is_spam_or_unsafe("legit message with enough length") is False
