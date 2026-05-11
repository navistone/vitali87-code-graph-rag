"""BUC-1603: CALLS edges carry call-site provenance (file + line + column).

These tests assert that every CALLS edge emitted by the call processor
includes ``file_path``, ``line_start``, and ``col_start`` properties whose
values match the source location of the tree-sitter call node.

We exercise three caller shapes — module-level call, function-to-function,
and method-to-method — to cover the three internal dispatchers in
``CallProcessor._ingest_function_calls`` (module / function / class paths).
A fourth test exercises the schema migration's idempotent ALTER path so an
already-indexed DB does not error on re-migration.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from codebase_rag import constants as cs
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers

if TYPE_CHECKING:
    from tree_sitter import Parser

    from codebase_rag.types_defs import LanguageQueries


@pytest.fixture
def parsers_and_queries() -> tuple[
    dict[cs.SupportedLanguage, Parser],
    dict[cs.SupportedLanguage, LanguageQueries],
]:
    parsers, queries = load_parsers()
    return parsers, queries


def _get_calls(mock_ingestor: MagicMock) -> list:
    """Filter the mock's recorded relationships down to CALLS edges only."""
    return [
        c
        for c in mock_ingestor.ensure_relationship_batch.call_args_list
        if c.args[1] == cs.RelationshipType.CALLS
    ]


def _props(call) -> dict:
    """Pull the ``properties`` kwarg from a recorded ensure_relationship_batch call.

    The call_processor always passes provenance via kwargs (``properties=...``)
    so this helper short-circuits the positional/keyword inspection.
    """
    return call.kwargs.get("properties") or {}


class TestCallsProvenance:
    """Every CALLS edge must carry file_path / line_start / col_start."""

    def test_module_level_call_carries_provenance(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """Module-level ``foo()`` at a known line number should ingest with
        ``line_start`` matching the source line (1-indexed).
        """
        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "module_call.py"
        # Write source where ``foo()`` is on a deterministic line.  Line 1 is
        # blank, line 2 is ``def foo(): pass``, line 3 is blank, line 4 is
        # the bare ``foo()`` call we expect to see in CALLS.
        test_file.write_text(
            encoding="utf-8",
            data=(
                "\n"
                "def foo():\n"
                "    pass\n"
                "foo()\n"
            ),
        )

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        calls = _get_calls(mock_ingestor)
        foo_calls = [c for c in calls if "foo" in c.args[2][2]]
        assert foo_calls, "expected at least one CALLS edge for foo()"

        props = _props(foo_calls[0])
        assert props.get("file_path") == "module_call.py", (
            f"file_path mismatch: {props!r}"
        )
        # The ``foo()`` call is on the 4th line (1-indexed).
        assert props.get("line_start") == 4, f"line_start mismatch: {props!r}"
        # First column on the line.
        assert props.get("col_start") == 0, f"col_start mismatch: {props!r}"

    def test_function_to_function_call_provenance(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """``main`` calls ``helper`` from inside the function body — the
        CALLS edge should record the line of the call expression, not the
        line of the function definition.
        """
        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "func_to_func.py"
        # Line 1: blank
        # Line 2: ``def helper(): pass``
        # Line 3: blank
        # Line 4: ``def main():``
        # Line 5: ``    helper()``  <-- the call we care about
        test_file.write_text(
            encoding="utf-8",
            data=(
                "\n"
                "def helper():\n"
                "    pass\n"
                "def main():\n"
                "    helper()\n"
            ),
        )

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        calls = _get_calls(mock_ingestor)
        helper_calls = [
            c
            for c in calls
            if "helper" in c.args[2][2] and "main" in c.args[0][2]
        ]
        assert helper_calls, (
            "expected at least one CALLS edge from main -> helper"
        )

        props = _props(helper_calls[0])
        assert props.get("file_path") == "func_to_func.py"
        # Indented call on line 5, column 4.
        assert props.get("line_start") == 5, f"line_start mismatch: {props!r}"
        assert props.get("col_start") == 4, f"col_start mismatch: {props!r}"

    def test_method_to_method_call_provenance(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """``self.helper()`` inside a class method should land on the line
        of the call expression and carry the indented column offset.
        """
        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "method_call.py"
        # Line 1: blank
        # Line 2: ``class Widget:``
        # Line 3: ``    def helper(self): pass``
        # Line 4: ``    def main(self):``
        # Line 5: ``        self.helper()``  <-- this is the call
        test_file.write_text(
            encoding="utf-8",
            data=(
                "\n"
                "class Widget:\n"
                "    def helper(self):\n"
                "        pass\n"
                "    def main(self):\n"
                "        self.helper()\n"
            ),
        )

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        calls = _get_calls(mock_ingestor)
        self_helper_calls = [
            c
            for c in calls
            if "helper" in c.args[2][2] and "main" in c.args[0][2]
        ]
        assert self_helper_calls, (
            "expected at least one CALLS edge for self.helper()"
        )

        props = _props(self_helper_calls[0])
        assert props.get("file_path") == "method_call.py"
        # The body indented twice (class + method body) = 8 spaces.
        assert props.get("line_start") == 6, f"line_start mismatch: {props!r}"
        assert props.get("col_start") == 8, f"col_start mismatch: {props!r}"

    def test_every_calls_edge_has_provenance_keys(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """Regression — every recorded CALLS edge must carry the three
        provenance keys.  A missing key here means a code path inside
        call_processor forgot to thread ``call_site_file_path`` through.
        """
        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "mixed.py"
        test_file.write_text(
            encoding="utf-8",
            data=(
                "def f(): pass\n"
                "def g():\n"
                "    f()\n"
                "class C:\n"
                "    def m(self):\n"
                "        f()\n"
                "        self.m()\n"
                "g()\n"
            ),
        )

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        calls = _get_calls(mock_ingestor)
        assert calls, "fixture should produce at least one CALLS edge"

        for c in calls:
            props = _props(c)
            assert "file_path" in props, f"missing file_path on: {c!r}"
            assert "line_start" in props, f"missing line_start on: {c!r}"
            assert "col_start" in props, f"missing col_start on: {c!r}"
            # 1-indexed line numbers — must be >= 1 for any real call.
            assert props["line_start"] >= 1, (
                f"line_start should be 1-indexed, got {props['line_start']!r}"
            )
            # Column is 0-indexed and can sit at 0.
            assert props["col_start"] >= 0


class TestSchemaMigrationIdempotency:
    """The schema ALTER path must be idempotent — running migrate() twice
    on the same DB should not raise, even on the BUC-1603 columns."""

    def test_migrate_is_idempotent(self, tmp_path: Path) -> None:
        from codebase_rag.services.ladybug_schema import migrate

        db_path = str(tmp_path / "idempotent.ladybug.db")
        migrate(db_path)
        # Second run must not raise; the columns are already present
        # inline in the CALLS DDL, so the ALTERs hit the
        # idempotent "already exists" branch.
        migrate(db_path)
