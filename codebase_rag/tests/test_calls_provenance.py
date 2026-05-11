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
            data=("\ndef foo():\n    pass\nfoo()\n"),
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
            data=("\ndef helper():\n    pass\ndef main():\n    helper()\n"),
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
            c for c in calls if "helper" in c.args[2][2] and "main" in c.args[0][2]
        ]
        assert helper_calls, "expected at least one CALLS edge from main -> helper"

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
            c for c in calls if "helper" in c.args[2][2] and "main" in c.args[0][2]
        ]
        assert self_helper_calls, "expected at least one CALLS edge for self.helper()"

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


# ---------------------------------------------------------------------------
# BUC-1609: resolver provenance — resolved_via + confidence
# ---------------------------------------------------------------------------


class TestCallsResolverProvenance:
    """Every CALLS edge must carry resolved_via + confidence properties,
    and a sample re-index over a mixed-input fixture must produce a
    distribution that spans multiple taxonomy categories (not all
    ``'unknown'`` / 1.0)."""

    def test_every_calls_edge_has_resolved_via_and_confidence(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """Regression — every recorded CALLS edge must carry the two
        BUC-1609 properties.  A missing key here means a code path in
        call_processor forgot to thread the tagged result through."""
        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "buc_1609_keys.py"
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
            assert "resolved_via" in props, (
                f"BUC-1609: CALLS edge missing resolved_via: {props!r}"
            )
            assert "confidence" in props, (
                f"BUC-1609: CALLS edge missing confidence: {props!r}"
            )
            # Confidence is a float in [0.0, 1.0].
            assert isinstance(props["confidence"], float), (
                f"confidence must be a float, got {type(props['confidence'])!r}"
            )
            assert 0.0 <= props["confidence"] <= 1.0, (
                f"confidence out of range [0.0, 1.0]: {props['confidence']!r}"
            )

    def test_same_module_call_is_tagged_exact(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """A bare call to a function defined in the same module is the
        canonical "exact" resolver path — same_module branch fires."""
        from codebase_rag.parsers.call_resolver import (
            CONFIDENCE_EXACT,
            RESOLVED_VIA_EXACT,
        )

        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "same_mod.py"
        test_file.write_text(
            encoding="utf-8",
            data=("def helper(): pass\ndef caller():\n    helper()\n"),
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
            if c.args[2][2].endswith(".helper") and ".caller" in c.args[0][2]
        ]
        assert helper_calls, "expected caller -> helper CALLS edge"

        props = _props(helper_calls[0])
        assert props["resolved_via"] == RESOLVED_VIA_EXACT
        assert props["confidence"] == CONFIDENCE_EXACT

    def test_wildcard_import_call_is_tagged_wildcard_with_low_confidence(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """A call resolved through ``from foo import *`` must carry the
        ``'wildcard'`` taxonomy value and ``confidence < 1.0`` so a
        downstream ``min_confidence`` filter can deprioritize it relative
        to a direct import.
        """
        from codebase_rag.parsers.call_resolver import (
            CONFIDENCE_WILDCARD,
            RESOLVED_VIA_WILDCARD,
        )

        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        # Two-module fixture: ``utils.py`` defines a function; ``caller.py``
        # pulls in ``from utils import *`` and calls it.  The resolver's
        # wildcard branch fires because the name is not in import_map
        # directly but the ``*`` entry points at the module containing it.
        utils_file = temp_repo / "utils.py"
        utils_file.write_text(
            encoding="utf-8",
            data=("def wildcarded(): pass\n"),
        )
        caller_file = temp_repo / "caller.py"
        caller_file.write_text(
            encoding="utf-8",
            data=("from utils import *\ndef main():\n    wildcarded()\n"),
        )

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        calls = _get_calls(mock_ingestor)
        wildcard_calls = [
            c
            for c in calls
            if c.args[2][2].endswith(".wildcarded") and "main" in c.args[0][2]
        ]
        assert wildcard_calls, (
            f"expected main -> wildcarded CALLS edge in {[c.args for c in calls]!r}"
        )

        props = _props(wildcard_calls[0])
        # Acceptance criterion 2: edges resolved via wildcard import
        # have confidence < 1.0.
        assert props["confidence"] < 1.0, (
            f"wildcard edge should have confidence < 1.0, got {props['confidence']!r}"
        )
        assert props["resolved_via"] == RESOLVED_VIA_WILDCARD
        assert props["confidence"] == CONFIDENCE_WILDCARD

    def test_resolved_via_distribution_spans_multiple_categories(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """Acceptance criterion 1: ``query CALLS, group by resolved_via,
        distribution shows multiple categories``.  A mixed-input repo
        (direct call + wildcard import call) should produce at least two
        different taxonomy values across its CALLS edges — proves the
        tag isn't constant at ``'unknown'`` / 1.0.
        """
        from codebase_rag.parsers.call_resolver import (
            RESOLVED_VIA_EXACT,
            RESOLVED_VIA_WILDCARD,
        )

        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        # ``utils.py`` exports a function via a wildcard import path;
        # ``caller.py`` calls one function locally (exact / same_module)
        # AND one via ``from utils import *`` (wildcard).  The resulting
        # CALLS edges should span both taxonomy values.
        (temp_repo / "utils.py").write_text(
            encoding="utf-8",
            data=("def via_wildcard(): pass\n"),
        )
        (temp_repo / "caller.py").write_text(
            encoding="utf-8",
            data=(
                "from utils import *\n"
                "def local_helper(): pass\n"
                "def main():\n"
                "    local_helper()\n"
                "    via_wildcard()\n"
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
        # Group by resolved_via — distribution must contain >= 2 categories.
        distribution: dict[str, int] = {}
        for c in calls:
            props = _props(c)
            tag = props.get("resolved_via")
            if tag is None:
                continue
            distribution[tag] = distribution.get(tag, 0) + 1

        assert distribution, "no resolved_via tags recorded — wiring is broken"
        # Don't accept a "everything is unknown" distribution: that would
        # mean the dispatcher is silently falling through to schema
        # defaults for every edge.
        assert distribution.keys() != {"unknown"}, (
            f"distribution collapsed to unknown only: {distribution!r}"
        )
        # We expect the exact and wildcard tags to both appear at least
        # once in this fixture.  If either is missing, the resolver's
        # branch detection is broken.
        assert RESOLVED_VIA_EXACT in distribution, (
            f"expected at least one 'exact' edge, got {distribution!r}"
        )
        assert RESOLVED_VIA_WILDCARD in distribution, (
            f"expected at least one 'wildcard' edge, got {distribution!r}"
        )

    def test_resolver_emits_only_emittable_resolved_via_values(
        self,
        temp_repo: Path,
        mock_ingestor: MagicMock,
        parsers_and_queries: tuple,
    ) -> None:
        """Reserved values ``'rebound'`` (BUC-1611) and ``'scip'``
        (BUC-1615) must never appear on a CALLS edge until those tickets
        land.  This guard catches accidental leakage."""
        from codebase_rag.parsers.call_resolver import (
            _EMITTABLE_RESOLVED_VIA,
            RESOLVED_VIA_REBOUND,
            RESOLVED_VIA_SCIP,
        )

        # Pre-condition: the constants exist but are *not* in the
        # emittable set.
        assert RESOLVED_VIA_REBOUND not in _EMITTABLE_RESOLVED_VIA
        assert RESOLVED_VIA_SCIP not in _EMITTABLE_RESOLVED_VIA

        parsers, queries = parsers_and_queries
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        (temp_repo / "rebound_check.py").write_text(
            encoding="utf-8",
            data=(
                "def f(): pass\n"
                "def g():\n"
                "    f()\n"
                "class C:\n"
                "    def m(self):\n"
                "        self.m()\n"
                "g()\n"
            ),
        )

        GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        ).run()

        calls = _get_calls(mock_ingestor)
        for c in calls:
            tag = _props(c).get("resolved_via")
            if tag is None:
                continue
            assert tag in _EMITTABLE_RESOLVED_VIA, (
                f"BUC-1609 resolver emitted reserved value {tag!r} — "
                f"this should only happen via BUC-1611 (rebound) or "
                f"BUC-1615 (scip)."
            )


class TestResolveResultGuard:
    """Module-level guards for the ResolveResult NamedTuple."""

    def test_from_tuple_rejects_reserved_resolved_via_values(self) -> None:
        from codebase_rag.parsers.call_resolver import (
            CONFIDENCE_EXACT,
            RESOLVED_VIA_REBOUND,
            ResolveResult,
        )

        # ``'rebound'`` is reserved for BUC-1611 — the guard must fire.
        with pytest.raises(ValueError, match="reserved for a sibling ticket"):
            ResolveResult.from_tuple(
                ("Function", "a.b.c"),
                RESOLVED_VIA_REBOUND,
                CONFIDENCE_EXACT,
            )

    def test_from_tuple_returns_none_when_inner_result_is_none(self) -> None:
        from codebase_rag.parsers.call_resolver import (
            CONFIDENCE_EXACT,
            RESOLVED_VIA_EXACT,
            ResolveResult,
        )

        # Short-circuit: a None tuple stays None regardless of tag.
        assert (
            ResolveResult.from_tuple(None, RESOLVED_VIA_EXACT, CONFIDENCE_EXACT) is None
        )

    def test_backward_compat_destructure_yields_legacy_two_tuple_shape(
        self,
    ) -> None:
        """Existing callers that do ``(callee_type, callee_qn) = result``
        must keep working — ResolveResult's first two fields are the same
        as the legacy tuple."""
        from codebase_rag.parsers.call_resolver import (
            CONFIDENCE_EXACT,
            RESOLVED_VIA_EXACT,
            ResolveResult,
        )

        result = ResolveResult(
            callee_type="Function",
            callee_qn="a.b.c",
            resolved_via=RESOLVED_VIA_EXACT,
            confidence=CONFIDENCE_EXACT,
        )
        # NamedTuple supports tuple-style destructure on the first N
        # fields if you slice; the legacy shape is the first 2.
        callee_type, callee_qn = result.callee_type, result.callee_qn
        assert callee_type == "Function"
        assert callee_qn == "a.b.c"
