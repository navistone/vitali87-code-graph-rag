"""Batching behavior tests for LadybugIngestor.

Updated for CI-4: replaced mgclient/cursor-based assertions with
LadybugIngestor-specific behavior (file DB path, per-node queries).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from codebase_rag.services.ladybug_ingestor import LadybugIngestor


def _empty_result() -> MagicMock:
    """Return a mock QueryResult that yields no rows."""
    r = MagicMock()
    r.get_column_names.return_value = []
    r.has_next.return_value = False
    return r


def _create_ingestor_with_mocked_connection(
    batch_size: int = 2,
) -> tuple[LadybugIngestor, MagicMock]:
    """Create a LadybugIngestor with conn.execute mocked."""
    ingestor = LadybugIngestor(db_path=":memory:", batch_size=batch_size)
    conn_mock = MagicMock()
    conn_mock.execute.return_value = _empty_result()
    ingestor.conn = conn_mock
    return ingestor, conn_mock


def test_node_batch_flushes_when_threshold_reached() -> None:
    ingestor, conn_mock = _create_ingestor_with_mocked_connection()

    ingestor.ensure_node_batch("File", {"path": "a", "name": "a.txt"})
    assert len(ingestor.node_buffer) == 1
    conn_mock.execute.assert_not_called()

    ingestor.ensure_node_batch("File", {"path": "b", "name": "b.txt"})

    # Buffer should be empty after flush
    assert len(ingestor.node_buffer) == 0
    conn_mock.execute.assert_called()


def test_node_batch_preserves_per_row_properties() -> None:
    ingestor, conn_mock = _create_ingestor_with_mocked_connection()

    ingestor.ensure_node_batch(
        "Function",
        {"qualified_name": "demo.fn1", "name": "fn1", "decorators": ["@a"]},
    )
    ingestor.ensure_node_batch(
        "Function",
        {"qualified_name": "demo.fn2", "name": "fn2", "decorators": []},
    )

    # Buffer should be flushed (batch_size=2)
    assert len(ingestor.node_buffer) == 0
    # One MERGE query per node
    assert conn_mock.execute.call_count == 2

    # Both calls should be MERGE queries targeting the Function label
    for call in conn_mock.execute.call_args_list:
        query = call[0][0]
        assert "Function" in query


def test_relationship_batch_flushes_after_threshold_and_respects_node_flush() -> None:
    ingestor, conn_mock = _create_ingestor_with_mocked_connection()

    with patch.object(
        LadybugIngestor, "flush_nodes", wraps=ingestor.flush_nodes
    ) as flush_nodes_spy:
        ingestor.ensure_relationship_batch(
            ("Module", "qualified_name", "proj.module1"),
            "CONTAINS_FILE",
            ("File", "path", "file1"),
        )
        assert ingestor._rel_count == 1

        ingestor.ensure_relationship_batch(
            ("Module", "qualified_name", "proj.module2"),
            "CONTAINS_FILE",
            ("File", "path", "file2"),
        )

        # flush_nodes is called before flush_relationships when threshold reached
        assert flush_nodes_spy.call_count == 1

    assert ingestor._rel_count == 0
    # Relationship queries should have been executed
    assert conn_mock.execute.call_count >= 2
