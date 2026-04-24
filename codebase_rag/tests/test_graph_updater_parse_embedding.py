from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag import constants as cs
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.types_defs import EmbeddingQueryResult, ResultRow


@pytest.fixture
def graph_updater(temp_repo: Path, mock_ingestor: MagicMock) -> GraphUpdater:
    parsers, queries = load_parsers()
    return GraphUpdater(
        ingestor=mock_ingestor,
        repo_path=temp_repo,
        parsers=parsers,
        queries=queries,
    )


class TestParseEmbeddingResult:
    def test_valid_input_all_fields(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 42,
            cs.KEY_QUALIFIED_NAME: "myproject.module.func",
            cs.KEY_START_LINE: 10,
            cs.KEY_END_LINE: 20,
            cs.KEY_PATH: "src/module.py",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result[cs.KEY_NODE_ID] == 42
        assert result[cs.KEY_QUALIFIED_NAME] == "myproject.module.func"
        assert result[cs.KEY_START_LINE] == 10
        assert result[cs.KEY_END_LINE] == 20
        assert result[cs.KEY_PATH] == "src/module.py"

    def test_valid_input_required_fields_only(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 1,
            cs.KEY_QUALIFIED_NAME: "pkg.func",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result[cs.KEY_NODE_ID] == 1
        assert result[cs.KEY_QUALIFIED_NAME] == "pkg.func"
        assert result[cs.KEY_START_LINE] is None
        assert result[cs.KEY_END_LINE] is None
        assert result[cs.KEY_PATH] is None

    def test_missing_node_id_returns_none(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {
            cs.KEY_QUALIFIED_NAME: "pkg.func",
            cs.KEY_START_LINE: 5,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is None

    def test_missing_qualified_name_returns_none(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 42,
            cs.KEY_START_LINE: 5,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is None

    def test_node_id_not_int_returns_none(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: "not_an_int",
            cs.KEY_QUALIFIED_NAME: "pkg.func",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is None

    def test_qualified_name_not_str_returns_none(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 42,
            cs.KEY_QUALIFIED_NAME: 12345,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is None

    def test_start_line_not_int_becomes_none(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 42,
            cs.KEY_QUALIFIED_NAME: "pkg.func",
            cs.KEY_START_LINE: "ten",
            cs.KEY_END_LINE: 20,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result[cs.KEY_START_LINE] is None
        assert result[cs.KEY_END_LINE] == 20

    def test_end_line_not_int_becomes_none(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 42,
            cs.KEY_QUALIFIED_NAME: "pkg.func",
            cs.KEY_START_LINE: 10,
            cs.KEY_END_LINE: "twenty",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result[cs.KEY_START_LINE] == 10
        assert result[cs.KEY_END_LINE] is None

    def test_path_not_str_becomes_none(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 42,
            cs.KEY_QUALIFIED_NAME: "pkg.func",
            cs.KEY_PATH: 12345,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result[cs.KEY_PATH] is None

    def test_empty_dict_returns_none(self, graph_updater: GraphUpdater) -> None:
        row: ResultRow = {}

        result = graph_updater._parse_embedding_result(row)

        assert result is None

    def test_none_values_for_required_fields_returns_none(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: None,
            cs.KEY_QUALIFIED_NAME: None,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is None

    def test_result_is_embedding_query_result_type(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 1,
            cs.KEY_QUALIFIED_NAME: "test.func",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        expected: EmbeddingQueryResult = {
            "node_id": 1,
            "qualified_name": "test.func",
            "start_line": None,
            "end_line": None,
            "path": None,
            "docstring": None,
        }
        assert result == expected

    def test_result_includes_docstring_when_present(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 1,
            cs.KEY_QUALIFIED_NAME: "test.func",
            cs.KEY_DOCSTRING: "Return x plus one.",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result["docstring"] == "Return x plus one."

    def test_result_docstring_none_when_missing(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 1,
            cs.KEY_QUALIFIED_NAME: "test.func",
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result["docstring"] is None

    def test_result_docstring_none_when_not_a_string(
        self, graph_updater: GraphUpdater
    ) -> None:
        row: ResultRow = {
            cs.KEY_NODE_ID: 1,
            cs.KEY_QUALIFIED_NAME: "test.func",
            cs.KEY_DOCSTRING: 42,
        }

        result = graph_updater._parse_embedding_result(row)

        assert result is not None
        assert result["docstring"] is None


class TestBuildEmbedText:
    def test_returns_source_unchanged_when_no_docstring(self) -> None:
        source = "def add(a, b):\n    return a + b"

        result = GraphUpdater._build_embed_text(source, None)

        assert result == source

    def test_returns_source_unchanged_when_empty_docstring(self) -> None:
        source = "def add(a, b):\n    return a + b"

        result = GraphUpdater._build_embed_text(source, "")

        assert result == source

    def test_prepends_docstring_as_comment(self) -> None:
        source = "def add(a, b):\n    return a + b"
        docstring = "Return the sum of a and b."

        result = GraphUpdater._build_embed_text(source, docstring)

        assert result == f"# {docstring}\n{source}"
        assert result.startswith("# Return the sum of a and b.\n")
        assert result.endswith(source)

    def test_preserves_multiline_docstring(self) -> None:
        source = "def foo(): pass"
        docstring = "Line one.\nLine two."

        result = GraphUpdater._build_embed_text(source, docstring)

        assert docstring in result
        assert result.endswith(source)
