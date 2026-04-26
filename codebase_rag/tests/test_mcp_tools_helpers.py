"""Tests for MCPToolsRegistry helper methods.

Updated for LadybugDB (CI-5): node_id is now a string (qualified_name)
instead of an integer id(n).
"""
from unittest.mock import MagicMock, patch

from codebase_rag import constants as cs

_PATCH_DELETE = "codebase_rag.mcp.tools.delete_project_embeddings"


def _make_registry(mock_ingestor: MagicMock) -> MagicMock:
    from codebase_rag.mcp.tools import MCPToolsRegistry

    registry = MagicMock(spec=MCPToolsRegistry)
    registry.ingestor = mock_ingestor
    registry._get_project_node_ids = MCPToolsRegistry._get_project_node_ids.__get__(
        registry
    )
    registry._cleanup_project_embeddings = (
        MCPToolsRegistry._cleanup_project_embeddings.__get__(registry)
    )
    return registry


class TestGetProjectNodeIds:
    def test_returns_string_ids(self) -> None:
        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {cs.KEY_NODE_ID: "proj.func_a"},
            {cs.KEY_NODE_ID: "proj.func_b"},
            {cs.KEY_NODE_ID: "proj.MyClass.method"},
        ]
        registry = _make_registry(mock_ingestor)

        result = registry._get_project_node_ids("myproject")

        assert result == ["proj.func_a", "proj.func_b", "proj.MyClass.method"]
        mock_ingestor.fetch_all.assert_called_once_with(
            cs.CYPHER_QUERY_PROJECT_NODE_IDS,
            {cs.KEY_PROJECT_NAME: "myproject"},
        )

    def test_filters_non_string_ids(self) -> None:
        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {cs.KEY_NODE_ID: "proj.func_a"},
            {cs.KEY_NODE_ID: 42},           # integer → filtered out
            {cs.KEY_NODE_ID: None},          # None → filtered out
            {cs.KEY_NODE_ID: "proj.func_b"},
        ]
        registry = _make_registry(mock_ingestor)

        result = registry._get_project_node_ids("proj")

        assert result == ["proj.func_a", "proj.func_b"]

    def test_returns_empty_when_no_rows(self) -> None:
        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = []
        registry = _make_registry(mock_ingestor)

        result = registry._get_project_node_ids("empty")

        assert result == []

    def test_skips_rows_missing_key(self) -> None:
        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {"other_key": "something"},
            {cs.KEY_NODE_ID: "proj.func"},
        ]
        registry = _make_registry(mock_ingestor)

        result = registry._get_project_node_ids("proj")

        assert result == ["proj.func"]


class TestCleanupProjectEmbeddings:
    def test_calls_delete_with_node_ids(self) -> None:
        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {cs.KEY_NODE_ID: "proj.func_a"},
            {cs.KEY_NODE_ID: "proj.func_b"},
        ]
        registry = _make_registry(mock_ingestor)

        with patch(_PATCH_DELETE) as mock_delete:
            registry._cleanup_project_embeddings("myproject")

        mock_delete.assert_called_once_with("myproject", ["proj.func_a", "proj.func_b"])

    def test_calls_delete_with_empty_list_when_no_nodes(self) -> None:
        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = []
        registry = _make_registry(mock_ingestor)

        with patch(_PATCH_DELETE) as mock_delete:
            registry._cleanup_project_embeddings("empty_proj")

        mock_delete.assert_called_once_with("empty_proj", [])
