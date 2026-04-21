from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from loguru import logger

from codebase_rag.services.graph_service import MemgraphIngestor


def _empty_result() -> MagicMock:
    return MagicMock(
        get_column_names=MagicMock(return_value=[]),
        has_next=MagicMock(return_value=False),
    )


@pytest.fixture
def graph_service() -> MemgraphIngestor:
    """Create a LadybugIngestor (MemgraphIngestor alias) with mocked connection."""
    ingestor = MemgraphIngestor(db_path=":memory:", batch_size=100)
    conn_mock = MagicMock()
    conn_mock.execute.return_value = _empty_result()
    ingestor.conn = conn_mock
    return ingestor


@pytest.fixture
def log_messages() -> Generator[list[str], None, None]:
    """Capture log messages using a custom sink."""
    messages: list[str] = []

    def sink(message: Any) -> None:
        messages.append(str(message))

    handler_id = logger.add(sink, format="{message}")
    yield messages
    logger.remove(handler_id)


def _make_conn_execute_side_effect(succeed_count: int) -> Any:
    """Return a side_effect for conn.execute that succeeds N times then raises."""
    call_state: dict[str, int] = {"count": 0}

    def side_effect(query: str, params: dict[str, Any] | None = None) -> MagicMock:
        call_state["count"] += 1
        if call_state["count"] <= succeed_count:
            return _empty_result()
        raise RuntimeError("Cannot MATCH target node — not found")

    return side_effect


def test_calls_failure_logging_single_batch(
    graph_service: MemgraphIngestor, log_messages: list[str]
) -> None:
    """Test that CALLS failures are logged correctly for a single batch.

    In LadybugDB, failures are detected via exceptions from conn.execute.
    The first relationship succeeds; the remaining two fail (target nodes missing).
    """
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassA.methodA()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.ClassB.methodB()"),
    )
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassA.methodA()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.NonExistent.missing()"),
    )
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassC.methodC()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.AlsoMissing.gone()"),
    )

    graph_service.conn.execute.side_effect = _make_conn_execute_side_effect(
        succeed_count=1
    )
    graph_service.flush_relationships()

    log_text = "\n".join(log_messages)
    assert "Failed to create 2 CALLS relationships" in log_text
    assert "nodes may not exist" in log_text

    assert "Sample 1:" in log_text or "Sample 2:" in log_text


def test_calls_failure_logging_multiple_batches(
    graph_service: MemgraphIngestor, log_messages: list[str]
) -> None:
    """Test that each flush call independently logs CALLS failures.

    Two separate flush() calls each with 1 failure should each log
    "Failed to create 1 CALLS relationships".
    """
    # First batch: 1 success, 1 failure
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassA.methodA()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.ClassB.methodB()"),
    )
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassA.methodA()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.Missing1.missing1()"),
    )

    graph_service.conn.execute.side_effect = _make_conn_execute_side_effect(
        succeed_count=1
    )
    graph_service.flush_relationships()

    # Second batch: 1 success, 1 failure
    graph_service.ensure_relationship_batch(
        ("Function", "qualified_name", "project.module.funcA"),
        "CALLS",
        ("Function", "qualified_name", "project.module.funcB"),
    )
    graph_service.ensure_relationship_batch(
        ("Function", "qualified_name", "project.module.funcA"),
        "CALLS",
        ("Function", "qualified_name", "project.module.missing2"),
    )

    graph_service.conn.execute.side_effect = _make_conn_execute_side_effect(
        succeed_count=1
    )
    graph_service.flush_relationships()

    log_text = "\n".join(log_messages)
    failure_count = log_text.count("Failed to create 1 CALLS relationships")

    assert failure_count == 2, (
        f"Expected 2 batches to each report 1 failure, but found {failure_count} occurrences in logs:\n{log_text}"
    )


def test_calls_success_no_failure_logging(
    graph_service: MemgraphIngestor, log_messages: list[str]
) -> None:
    """Test that no failure logs are emitted when all CALLS succeed."""
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassA.methodA()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.ClassB.methodB()"),
    )
    graph_service.ensure_relationship_batch(
        ("Method", "qualified_name", "project.module.ClassC.methodC()"),
        "CALLS",
        ("Method", "qualified_name", "project.module.ClassD.methodD()"),
    )

    # conn.execute already returns _empty_result() by default from the fixture
    graph_service.flush_relationships()

    log_text = "\n".join(log_messages)
    assert "Failed to create" not in log_text
    assert "nodes may not exist" not in log_text


def test_non_calls_relationships_no_failure_logging(
    graph_service: MemgraphIngestor, log_messages: list[str]
) -> None:
    """Test that non-CALLS relationship failures do not emit CALLS-specific logs."""
    graph_service.ensure_relationship_batch(
        ("Module", "qualified_name", "project.moduleA"),
        "IMPORTS",
        ("Module", "qualified_name", "project.moduleB"),
    )
    graph_service.ensure_relationship_batch(
        ("Module", "qualified_name", "project.moduleA"),
        "IMPORTS",
        ("Module", "qualified_name", "project.missing"),
    )

    graph_service.conn.execute.side_effect = _make_conn_execute_side_effect(
        succeed_count=1
    )
    graph_service.flush_relationships()

    log_text = "\n".join(log_messages)
    # CALLS-specific failure log must not be present for non-CALLS failures
    assert "Failed to create" not in log_text or "CALLS" not in log_text
