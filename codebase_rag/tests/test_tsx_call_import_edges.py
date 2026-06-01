"""Regression test: .tsx files must produce call and import edges.

Before fix: .tsx files were parsed with the `typescript` grammar (which does not
handle JSX syntax), yielding zero call/import edges even though the `contains`
(structural) edges were created correctly.

After fix: .tsx files are dispatched to SupportedLanguage.TSX whose LanguageSpec
uses the `language_tsx` grammar from tree_sitter_typescript.  That grammar handles
JSX, so the standard call_expression / import_statement queries fire and edges are
extracted — matching the behaviour .ts files always had.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag.constants import RelationshipType, SupportedLanguage
from codebase_rag.tests.conftest import create_and_run_updater, get_relationships


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TSX_FIXTURE = """\
import { useState } from "react";
import { Foo } from "./foo";

function baz(): string {
    return "hello";
}

function Bar(): JSX.Element {
    const [count, setCount] = useState(0);
    const result = baz();
    return <div className={String(count)}>{result}</div>;
}
"""


def _make_tsx_repo(tmp_path: Path) -> Path:
    """Create a minimal repo that contains a single .tsx file."""
    repo = tmp_path / "tsx_repo"
    repo.mkdir()
    (repo / "comp.tsx").write_text(TSX_FIXTURE, encoding="utf-8")
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTsxCallImportEdges:
    """Ensure .tsx files produce call and import edges (regression for the
    tsx-grammar routing bug where the typescript grammar was used instead of tsx)."""

    def test_tsx_grammar_loads(self) -> None:
        """SupportedLanguage.TSX parser is available after load_parsers()."""
        from codebase_rag.parser_loader import load_parsers

        parsers, queries = load_parsers()
        assert SupportedLanguage.TSX in parsers, (
            "TSX parser not loaded — tree-sitter-typescript must expose language_tsx"
        )
        assert SupportedLanguage.TSX in queries, "TSX queries not built"

    def test_tsx_extension_maps_to_tsx_language(self) -> None:
        """The .tsx extension must resolve to SupportedLanguage.TSX, not TS."""
        from codebase_rag.language_spec import get_language_for_extension

        lang = get_language_for_extension(".tsx")
        assert lang is SupportedLanguage.TSX, (
            f".tsx extension maps to {lang!r}, expected SupportedLanguage.TSX"
        )

    def test_ts_extension_still_maps_to_ts(self) -> None:
        """The .ts extension must still resolve to SupportedLanguage.TS (no regression)."""
        from codebase_rag.language_spec import get_language_for_extension

        lang = get_language_for_extension(".ts")
        assert lang is SupportedLanguage.TS, (
            f".ts extension maps to {lang!r}, expected SupportedLanguage.TS"
        )

    def test_tsx_produces_import_edges(
        self, tmp_path: Path, mock_ingestor: MagicMock
    ) -> None:
        """A .tsx file with import statements must produce IMPORTS relationships."""
        repo = _make_tsx_repo(tmp_path)
        create_and_run_updater(
            repo, mock_ingestor, skip_if_missing=str(SupportedLanguage.TSX)
        )

        imports = get_relationships(mock_ingestor, RelationshipType.IMPORTS)
        assert len(imports) > 0, (
            "No IMPORTS edges extracted from .tsx file. "
            "The tsx grammar routing is likely still broken."
        )

    def test_tsx_produces_call_edges(
        self, tmp_path: Path, mock_ingestor: MagicMock
    ) -> None:
        """A .tsx file where function Bar() calls baz() must produce a CALLS edge."""
        repo = _make_tsx_repo(tmp_path)
        create_and_run_updater(
            repo, mock_ingestor, skip_if_missing=str(SupportedLanguage.TSX)
        )

        calls = get_relationships(mock_ingestor, RelationshipType.CALLS)
        assert len(calls) > 0, (
            "No CALLS edges extracted from .tsx file. "
            "The tsx grammar routing is likely still broken."
        )

        # Specifically: Bar calls baz — verify that edge exists
        call_pairs = {
            (c.args[0], c.args[2]) for c in calls
        }
        bar_to_baz = any(
            "Bar" in str(src) and "baz" in str(dst)
            for src, dst in call_pairs
        )
        assert bar_to_baz, (
            f"Expected a CALLS edge from Bar to baz, got: {call_pairs}"
        )

    def test_tsx_import_from_react_present(
        self, tmp_path: Path, mock_ingestor: MagicMock
    ) -> None:
        """The 'import from react' statement must yield an IMPORTS edge to 'react'."""
        repo = _make_tsx_repo(tmp_path)
        create_and_run_updater(
            repo, mock_ingestor, skip_if_missing=str(SupportedLanguage.TSX)
        )

        imports = get_relationships(mock_ingestor, RelationshipType.IMPORTS)
        import_targets = {str(c.args[2]) for c in imports}
        assert any("react" in t.lower() for t in import_targets), (
            f"No import edge pointing at 'react' found. Import targets: {import_targets}"
        )

    def test_ts_call_import_edges_unchanged(
        self, tmp_path: Path, mock_ingestor: MagicMock
    ) -> None:
        """Regression: .ts files must still produce call/import edges (no regression)."""
        repo = tmp_path / "ts_repo"
        repo.mkdir()
        (repo / "utils.ts").write_text(
            'import { readFileSync } from "fs";\n'
            'function helper(x: string): string { return x.trim(); }\n'
            'export function run() { return helper("hello"); }\n',
            encoding="utf-8",
        )
        create_and_run_updater(
            repo, mock_ingestor, skip_if_missing=str(SupportedLanguage.TS)
        )

        calls = get_relationships(mock_ingestor, RelationshipType.CALLS)
        imports = get_relationships(mock_ingestor, RelationshipType.IMPORTS)
        assert len(calls) > 0, "No CALLS edges from .ts file — TS regression"
        assert len(imports) > 0, "No IMPORTS edges from .ts file — TS regression"
