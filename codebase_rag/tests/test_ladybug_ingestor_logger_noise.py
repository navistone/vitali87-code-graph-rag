"""BUC-1620 — regression test: ladybug_ingestor must not emit ERROR-level
log records on a normal indexing pass.

Before the fix, ``_execute_query`` and ``_execute_batch`` logged the offending
Cypher query / params / error at ``ERROR`` level whenever the per-row fallback
in ``flush_relationships`` (or the idempotent CREATE in ``flush_nodes``)
caught a benign exception.  The caller silently treated those as soft-success,
so from the operator's perspective the index ran cleanly — yet logs were
flooded with misleading ``ERROR | ...`` lines, masking the real failures
during triage.

The fix downgrades both internal helpers to ``logger.debug``.  Callers that
treat the raised exception as a real failure still log a ``warning`` /
``error`` at their level (e.g. ``MG_CALLS_FAILED``, ``MG_LABEL_FLUSH_ERROR``,
``MG_REL_FLUSH_ERROR``) — that behaviour is unchanged and is covered by
``test_graph_service_calls_failure_logging.py``.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from loguru import logger

from codebase_rag.services.ladybug_ingestor import LadybugIngestor


def _empty_result() -> MagicMock:
    return MagicMock(
        get_column_names=MagicMock(return_value=[]),
        has_next=MagicMock(return_value=False),
    )


@pytest.fixture
def ingestor() -> LadybugIngestor:
    """LadybugIngestor with a mocked connection."""
    inst = LadybugIngestor(db_path=":memory:", batch_size=100)
    conn_mock = MagicMock()
    conn_mock.execute.return_value = _empty_result()
    inst.conn = conn_mock
    return inst


@pytest.fixture
def error_records() -> Generator[list[dict[str, Any]], None, None]:
    """Capture every loguru record at ERROR level or higher."""
    records: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        rec = message.record
        records.append({"level": rec["level"].name, "message": rec["message"]})

    handler_id = logger.add(sink, level="ERROR")
    yield records
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Regression: clean index pass — zero ERROR records
# ---------------------------------------------------------------------------


def test_clean_index_pass_emits_zero_error_records(
    ingestor: LadybugIngestor, error_records: list[dict[str, Any]]
) -> None:
    """Happy path: a clean index pass (all queries succeed) must emit zero
    ERROR-level records.

    This guards against an accidental promotion of ``_execute_query`` /
    ``_execute_batch`` diagnostic logs back to ERROR level.
    """
    ingestor.ensure_node_batch("Module", {"qualified_name": "project.modA"})
    ingestor.ensure_node_batch("Module", {"qualified_name": "project.modB"})
    ingestor.flush_nodes()

    ingestor.ensure_relationship_batch(
        ("Module", "qualified_name", "project.modA"),
        "IMPORTS",
        ("Module", "qualified_name", "project.modB"),
    )
    ingestor.flush_relationships()

    error_levels = [r["level"] for r in error_records]
    assert error_levels == [], (
        f"Expected zero ERROR records on a clean index pass; got {error_records!r}"
    )


def test_execute_query_failure_does_not_emit_error_log_directly(
    ingestor: LadybugIngestor, error_records: list[dict[str, Any]]
) -> None:
    """Even when ``_execute_query`` raises a non-benign exception, it must
    not emit an ERROR-level record itself.  Escalation is the caller's job.

    Rationale: ``_execute_query`` cannot know whether its caller (per-row
    fallback in ``flush_relationships``, idempotent CREATE in ``flush_nodes``)
    treats the raised exception as a soft-success.  Logging at ERROR here
    produces the operational noise BUC-1620 is fixing.
    """
    ingestor.conn.execute.side_effect = RuntimeError(
        "Cannot MATCH target node — not found"
    )

    with pytest.raises(RuntimeError):
        ingestor._execute_query("MATCH (n) RETURN n", {"x": 1})

    assert error_records == [], (
        f"_execute_query must not emit ERROR records on exception; "
        f"got {error_records!r}"
    )


def test_execute_batch_failure_does_not_emit_error_log_directly(
    ingestor: LadybugIngestor, error_records: list[dict[str, Any]]
) -> None:
    """Same contract for ``_execute_batch``: it must not emit ERROR-level
    records itself.  ``flush_relationships`` already escalates with a
    ``logger.warning`` (MG_REL_FLUSH_ERROR) when a batch fails non-benignly.
    """
    ingestor.conn.execute.side_effect = RuntimeError("hash join key not found")

    with pytest.raises(RuntimeError):
        ingestor._execute_batch(
            "MATCH (a) MATCH (b) MERGE (a)-[:R]->(b)",
            [{"from_val": "x", "to_val": "y"}],
        )

    assert error_records == [], (
        f"_execute_batch must not emit ERROR records on exception; "
        f"got {error_records!r}"
    )


def test_real_failures_still_escalate_at_caller_level(
    ingestor: LadybugIngestor, error_records: list[dict[str, Any]]
) -> None:
    """Sanity check: real failures must still surface to the operator.

    ``flush_nodes`` calls ``_execute_query`` for a non-MERGE CREATE path,
    catches the exception, and (when the error is NOT the idempotent
    "already exists" / "constraint" pattern) logs at ERROR via
    ``MG_LABEL_FLUSH_ERROR``.  That escalation path is unaffected by this
    fix and must remain ERROR-level.
    """
    # Force a non-benign failure on every execute() call.
    ingestor.conn.execute.side_effect = RuntimeError("disk I/O failure")
    ingestor._use_merge = False  # take the CREATE-only path

    ingestor.ensure_node_batch(
        "Module", {"qualified_name": "project.modA", "value": "ok"}
    )
    ingestor.flush_nodes()

    error_messages = [r["message"] for r in error_records]
    assert any("disk I/O failure" in m for m in error_messages), (
        f"Expected caller-level ERROR for a non-benign flush failure; "
        f"got {error_records!r}"
    )
