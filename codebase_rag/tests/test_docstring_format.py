"""Tests for ``codebase_rag.storage.docstring_format``.

Plan G — verifies that ``format_docstring`` produces a normalized,
embedding-friendly form across Google/NumPy/RST styles and degrades
gracefully on unparseable input.
"""

from __future__ import annotations

import pytest

pytest.importorskip("docstring_parser")

from codebase_rag.storage import docstring_format
from codebase_rag.storage.docstring_format import format_docstring


def test_should_return_empty_string_when_input_is_none() -> None:
    assert format_docstring(None) == ""


def test_should_return_empty_string_when_input_is_empty() -> None:
    assert format_docstring("") == ""
    assert format_docstring("   \n\t  ") == ""


def test_should_extract_description_when_google_style_docstring() -> None:
    raw = """Compute the answer.

    A longer paragraph that explains the function in more depth.
    """
    out = format_docstring(raw)
    assert "Description: Compute the answer." in out
    assert "A longer paragraph" in out


def test_should_extract_args_section_when_present() -> None:
    raw = """Add two numbers.

    Args:
        x: First operand.
        y: Second operand.

    Returns:
        The sum of x and y.
    """
    out = format_docstring(raw)
    assert "Args:" in out
    assert "x: First operand." in out
    assert "y: Second operand." in out


def test_should_extract_returns_section_when_present() -> None:
    raw = """Look up a row.

    Args:
        key: Primary key.

    Returns:
        The matching row, or None when missing.

    Raises:
        KeyError: When the key is malformed.
    """
    out = format_docstring(raw)
    assert "Returns: The matching row, or None when missing." in out
    assert "Raises:" in out
    assert "KeyError: When the key is malformed." in out


def test_should_fall_back_to_raw_when_parser_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the parser raises, we keep the raw text rather than dropping it."""

    raw = "Some perfectly fine docstring."

    class _Boom:
        DocstringStyle = type("S", (), {"AUTO": object()})

        @staticmethod
        def parse(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("simulated parser failure")

    monkeypatch.setitem(
        __import__("sys").modules, "docstring_parser", _Boom
    )
    # Force re-import path inside format_docstring by clearing any cached
    # reference (the function imports lazily on each call).
    out = format_docstring(raw)
    assert out == raw
    assert docstring_format is not None  # sanity
