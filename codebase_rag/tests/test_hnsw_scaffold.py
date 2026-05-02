"""Tests for Phase 8 HNSW scaffold (codebase_rag.storage.vector_store).

Covers:
- _hnsw_active returns False on a fresh DB
- create_hnsw_index is idempotent (run twice, no error)
- search_similar returns identical top-1 with HNSW on vs off on a 10-row corpus

Default activation gates:
  HNSW_ENABLED env = false  (default)
  repo_metadata "hnsw_active" key absent / "false"

These tests NEVER flip HNSW_ENABLED=true globally; each HNSW-on test sets the
env var narrowly via monkeypatch so the global default is always false.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("duckdb", reason="duckdb not installed")

from codebase_rag.storage.vector_store import (
    EmbeddingRow,
    _hnsw_active,
    bulk_insert,
    create_hnsw_index,
    open_or_create,
    search_similar,
    write_metadata,
)

_DIM = 768


def _unit_vec(index: int, dim: int = _DIM) -> list[float]:
    """Return a unit vector with 1.0 at ``index`` and 0.0 elsewhere."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_rows(n: int) -> list[EmbeddingRow]:
    """Return ``n`` embedding rows with orthogonal unit vectors."""
    return [
        EmbeddingRow(
            qualified_name=f"pkg.mod.fn_{i}",
            embedding=_unit_vec(i % _DIM),
            file_path=f"/repo/mod_{i}.py",
            start_line=i * 10 + 1,
            end_line=i * 10 + 9,
            symbol_type="Function",
        )
        for i in range(n)
    ]


@pytest.fixture()
def fresh_conn(tmp_path: Path):
    """Open a fresh DuckDB connection to a temp .duck file."""
    conn = open_or_create(str(tmp_path / "hnsw_test.duck"))
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# _hnsw_active — gate behaviour
# ---------------------------------------------------------------------------


def test_should_return_false_when_hnsw_enabled_env_is_not_set(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return False when HNSW_ENABLED env var is absent."""
    monkeypatch.delenv("HNSW_ENABLED", raising=False)
    assert _hnsw_active(fresh_conn) is False


def test_should_return_false_when_hnsw_enabled_is_false(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return False when HNSW_ENABLED=false (explicit default)."""
    monkeypatch.setenv("HNSW_ENABLED", "false")
    assert _hnsw_active(fresh_conn) is False


def test_should_return_false_on_fresh_db_even_when_global_flag_is_true(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return False when global flag true but repo_metadata key is absent."""
    monkeypatch.setenv("HNSW_ENABLED", "true")
    # No hnsw_active key written yet
    assert _hnsw_active(fresh_conn) is False


def test_should_return_false_when_repo_flag_is_false_string(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return False when hnsw_active metadata is 'false'."""
    monkeypatch.setenv("HNSW_ENABLED", "true")
    write_metadata(fresh_conn, hnsw_active="false")
    assert _hnsw_active(fresh_conn) is False


def test_should_return_true_when_both_gates_are_open(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return True only when HNSW_ENABLED=true AND hnsw_active='true'."""
    monkeypatch.setenv("HNSW_ENABLED", "true")
    write_metadata(fresh_conn, hnsw_active="true")
    assert _hnsw_active(fresh_conn) is True


# ---------------------------------------------------------------------------
# create_hnsw_index — idempotency
# ---------------------------------------------------------------------------


def test_should_create_index_without_error_on_empty_table(
    fresh_conn,
) -> None:
    """should create HNSW index successfully when embeddings table is empty."""
    # Use the live "embeddings" table name
    create_hnsw_index(fresh_conn, table="embeddings", col="embedding")


def test_should_be_idempotent_when_called_twice(
    fresh_conn,
) -> None:
    """should not raise when create_hnsw_index is called twice on same table."""
    create_hnsw_index(fresh_conn, table="embeddings", col="embedding")
    # Second call must be a no-op (CREATE INDEX IF NOT EXISTS)
    create_hnsw_index(fresh_conn, table="embeddings", col="embedding")


def test_should_be_idempotent_after_rows_inserted(
    fresh_conn,
) -> None:
    """should create index idempotently when table has data."""
    bulk_insert(fresh_conn, _make_rows(5))
    create_hnsw_index(fresh_conn, table="embeddings", col="embedding")
    create_hnsw_index(fresh_conn, table="embeddings", col="embedding")


# ---------------------------------------------------------------------------
# search_similar — smoke equivalence: HNSW on vs off, 10-row corpus
# ---------------------------------------------------------------------------


def _populate_10_rows(conn) -> None:
    """Insert 10 rows with orthogonal unit vectors into ``conn``."""
    bulk_insert(conn, _make_rows(10))


def test_should_return_same_top1_with_hnsw_off(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return correct top-1 result when HNSW is disabled (default)."""
    monkeypatch.setenv("HNSW_ENABLED", "false")
    _populate_10_rows(fresh_conn)

    query = _unit_vec(0)
    results = search_similar(fresh_conn, query, k=1)
    assert len(results) == 1
    assert results[0].qualified_name == "pkg.mod.fn_0"
    assert results[0].score > 0.99


def test_should_return_same_top1_with_hnsw_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should return identical top-1 with HNSW on as with HNSW off on 10-row corpus."""
    # Fresh DB so we can create the HNSW index before inserting rows
    conn = open_or_create(str(tmp_path / "hnsw_on.duck"))
    try:
        # Enable both gates
        monkeypatch.setenv("HNSW_ENABLED", "true")
        write_metadata(conn, hnsw_active="true")

        # Build the HNSW index before populating (valid for DuckDB VSS)
        create_hnsw_index(conn, table="embeddings", col="embedding")

        _populate_10_rows(conn)

        query = _unit_vec(3)
        results = search_similar(conn, query, k=1)
        assert len(results) == 1
        assert results[0].qualified_name == "pkg.mod.fn_3"
        assert results[0].score > 0.99
    finally:
        conn.close()


def test_should_agree_on_top1_between_hnsw_on_and_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should produce identical top-1 qualified_name with HNSW on vs off."""
    rows = _make_rows(10)
    query = _unit_vec(5)

    # --- HNSW OFF ---
    conn_off = open_or_create(str(tmp_path / "off.duck"))
    monkeypatch.setenv("HNSW_ENABLED", "false")
    bulk_insert(conn_off, rows)
    off_results = search_similar(conn_off, query, k=1)
    conn_off.close()

    # --- HNSW ON ---
    conn_on = open_or_create(str(tmp_path / "on.duck"))
    monkeypatch.setenv("HNSW_ENABLED", "true")
    write_metadata(conn_on, hnsw_active="true")
    create_hnsw_index(conn_on, table="embeddings", col="embedding")
    bulk_insert(conn_on, rows)
    on_results = search_similar(conn_on, query, k=1)
    conn_on.close()

    assert off_results[0].qualified_name == on_results[0].qualified_name
    assert off_results[0].qualified_name == "pkg.mod.fn_5"


# ---------------------------------------------------------------------------
# Default flag invariant — confirm default_behavior_unchanged
# ---------------------------------------------------------------------------


def test_should_not_use_hnsw_path_by_default(
    fresh_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should use brute-force path when HNSW_ENABLED is not set (regression guard)."""
    monkeypatch.delenv("HNSW_ENABLED", raising=False)
    bulk_insert(fresh_conn, _make_rows(5))
    # No VSS extension loaded, no index — must succeed via brute-force
    results = search_similar(fresh_conn, _unit_vec(2), k=1)
    assert len(results) == 1
    assert results[0].qualified_name == "pkg.mod.fn_2"
