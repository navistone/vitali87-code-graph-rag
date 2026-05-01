"""Unit tests for codebase_rag.storage.vector_store (DuckDB backend).

Tests follow the Testing Rules:
- Names: "should [result] when [condition]"
- Pure interface tests against real DuckDB (no mock drift)
- Error conditions and edge cases covered alongside happy paths
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

# Skip the entire module when duckdb is not installed — the semantic
# optional-dependency group is not required for the core parser/graph tests.
pytest.importorskip("duckdb", reason="duckdb not installed")

from codebase_rag.storage.vector_store import (
    EmbeddingRow,
    SearchResult,
    bulk_insert,
    insert_embedding,
    open_or_create,
    read_all_metadata,
    read_metadata,
    row_count,
    search_similar,
    write_metadata,
)

_DIM = 768


def _unit_vec(index: int, dim: int = _DIM) -> list[float]:
    """Return a unit vector with 1.0 at ``index`` and 0.0 elsewhere."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _normalise(v: list[float]) -> list[float]:
    """Return an L2-normalised copy of ``v``; zero vectors returned unchanged."""
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0.0:
        return v
    return [x / mag for x in v]


@pytest.fixture()
def tmp_vec_db(tmp_path: Path):
    """Yield a fresh open_or_create connection to a temp .duck file."""
    db_file = tmp_path / "test.duck"
    conn = open_or_create(str(db_file))
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# open_or_create
# ---------------------------------------------------------------------------


def test_should_create_schema_when_file_is_new(tmp_path: Path) -> None:
    db_file = tmp_path / "fresh.duck"
    conn = open_or_create(str(db_file))
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    conn.close()
    assert "repo_metadata" in tables
    assert "embeddings" in tables
    assert "centrality" in tables
    assert db_file.exists()


def test_should_open_existing_file_without_recreating_schema(tmp_path: Path) -> None:
    db_file = tmp_path / "existing.duck"
    conn1 = open_or_create(str(db_file))
    row = EmbeddingRow(
        qualified_name="pkg.mod.fn",
        embedding=_unit_vec(0),
        file_path="/repo/mod.py",
        start_line=1,
        end_line=10,
        symbol_type="Function",
    )
    bulk_insert(conn1, [row])
    conn1.close()

    conn2 = open_or_create(str(db_file))
    assert row_count(conn2) == 1
    conn2.close()


def test_should_raise_when_duckdb_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    saved = sys.modules.pop("duckdb", None)
    try:
        monkeypatch.setitem(sys.modules, "duckdb", None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="duckdb"):
            open_or_create(str(tmp_path / "x.duck"))
    finally:
        if saved is not None:
            sys.modules["duckdb"] = saved
        else:
            sys.modules.pop("duckdb", None)


# ---------------------------------------------------------------------------
# insert_embedding / bulk_insert
# ---------------------------------------------------------------------------


def test_should_insert_single_row_when_called_with_valid_row(
    tmp_vec_db,
) -> None:
    row = EmbeddingRow(
        qualified_name="myapp.utils.retry",
        embedding=_unit_vec(1),
        file_path="/repo/utils.py",
        start_line=20,
        end_line=35,
        symbol_type="Function",
    )
    insert_embedding(tmp_vec_db, row)
    assert row_count(tmp_vec_db) == 1


def test_should_return_zero_when_bulk_insert_receives_empty_list(
    tmp_vec_db,
) -> None:
    result = bulk_insert(tmp_vec_db, [])
    assert result == 0
    assert row_count(tmp_vec_db) == 0


def test_should_insert_all_rows_when_bulk_insert_receives_10_rows(
    tmp_vec_db,
) -> None:
    rows = [
        EmbeddingRow(
            qualified_name=f"pkg.mod.fn_{i}",
            embedding=_unit_vec(i % _DIM),
            file_path=f"/repo/mod_{i}.py",
            start_line=i * 10 + 1,
            end_line=i * 10 + 9,
            symbol_type="Function",
        )
        for i in range(10)
    ]
    n = bulk_insert(tmp_vec_db, rows)
    assert n == 10
    assert row_count(tmp_vec_db) == 10


def test_should_replace_existing_row_when_qualified_name_collides(
    tmp_vec_db,
) -> None:
    qname = "pkg.mod.fn"
    row_v1 = EmbeddingRow(
        qualified_name=qname,
        embedding=_unit_vec(0),
        file_path="/repo/mod.py",
        start_line=1,
        end_line=5,
        symbol_type="Function",
    )
    row_v2 = EmbeddingRow(
        qualified_name=qname,
        embedding=_unit_vec(1),
        file_path="/repo/mod_new.py",
        start_line=10,
        end_line=20,
        symbol_type="Method",
    )
    bulk_insert(tmp_vec_db, [row_v1])
    bulk_insert(tmp_vec_db, [row_v2])
    assert row_count(tmp_vec_db) == 1


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------


def test_should_return_empty_list_when_store_is_empty(tmp_vec_db) -> None:
    results = search_similar(tmp_vec_db, _unit_vec(0), k=5)
    assert results == []


def test_should_return_k_results_when_store_has_more_than_k_rows(
    tmp_vec_db,
) -> None:
    rows = [
        EmbeddingRow(
            qualified_name=f"fn_{i}",
            embedding=_unit_vec(i),
            file_path="/repo/f.py",
            start_line=i,
            end_line=i + 1,
            symbol_type="Function",
        )
        for i in range(20)
    ]
    bulk_insert(tmp_vec_db, rows)
    results = search_similar(tmp_vec_db, _unit_vec(0), k=5)
    assert len(results) == 5


def test_should_rank_identical_vector_first_when_it_is_in_store(
    tmp_vec_db,
) -> None:
    target_vec = _unit_vec(0)
    other_vec = _unit_vec(1)

    bulk_insert(
        tmp_vec_db,
        [
            EmbeddingRow("target", target_vec, "/a.py", 1, 10, "Function"),
            EmbeddingRow("other", other_vec, "/b.py", 1, 10, "Function"),
        ],
    )
    results = search_similar(tmp_vec_db, target_vec, k=2)
    assert len(results) == 2
    assert results[0].qualified_name == "target"
    assert results[0].score > results[1].score


def test_should_return_SearchResult_with_correct_fields(tmp_vec_db) -> None:
    row = EmbeddingRow(
        qualified_name="pkg.utils.helper",
        embedding=_unit_vec(5),
        file_path="/repo/utils.py",
        start_line=42,
        end_line=60,
        symbol_type="Method",
    )
    bulk_insert(tmp_vec_db, [row])
    results = search_similar(tmp_vec_db, _unit_vec(5), k=1)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert r.qualified_name == "pkg.utils.helper"
    assert r.file_path == "/repo/utils.py"
    assert r.start_line == 42
    assert r.end_line == 60
    assert r.score > 0.99  # near-identical unit vectors → cosine ≈ 1


def test_should_return_score_in_valid_range_for_orthogonal_vectors(
    tmp_vec_db,
) -> None:
    bulk_insert(
        tmp_vec_db,
        [EmbeddingRow("fn_a", _unit_vec(0), "/a.py", 1, 5, "Function")],
    )
    # Query with a vector orthogonal to the stored one
    results = search_similar(tmp_vec_db, _unit_vec(1), k=1)
    assert len(results) == 1
    # cos(90°) = 0 → distance = 1 → score = 0
    assert -1.0 <= results[0].score <= 1.0


# ---------------------------------------------------------------------------
# write_metadata / read_metadata / read_all_metadata
# ---------------------------------------------------------------------------


def test_should_store_and_retrieve_metadata_key(tmp_vec_db) -> None:
    write_metadata(tmp_vec_db, last_indexed_at="1700000000")
    assert read_metadata(tmp_vec_db, "last_indexed_at") == "1700000000"


def test_should_return_default_when_key_missing(tmp_vec_db) -> None:
    assert read_metadata(tmp_vec_db, "nonexistent") is None
    assert read_metadata(tmp_vec_db, "nonexistent", default="fallback") == "fallback"


def test_should_overwrite_existing_key_when_write_metadata_called_twice(
    tmp_vec_db,
) -> None:
    write_metadata(tmp_vec_db, root_path="/old")
    write_metadata(tmp_vec_db, root_path="/new")
    assert read_metadata(tmp_vec_db, "root_path") == "/new"


def test_should_return_all_keys_when_read_all_metadata_called(
    tmp_vec_db,
) -> None:
    write_metadata(
        tmp_vec_db,
        last_indexed_at="1700000000",
        root_path="/repo",
        node_count="100",
        rel_count="200",
        last_job_id="abc-123",
        schema_version="1.5",
    )
    meta = read_all_metadata(tmp_vec_db)
    assert meta["last_indexed_at"] == "1700000000"
    assert meta["root_path"] == "/repo"
    assert meta["node_count"] == "100"
    assert meta["rel_count"] == "200"
    assert meta["last_job_id"] == "abc-123"
    assert meta["schema_version"] == "1.5"


def test_should_return_empty_dict_when_no_metadata_written(tmp_vec_db) -> None:
    assert read_all_metadata(tmp_vec_db) == {}


# ---------------------------------------------------------------------------
# row_count
# ---------------------------------------------------------------------------


def test_should_return_zero_when_table_is_empty(tmp_vec_db) -> None:
    assert row_count(tmp_vec_db) == 0


def test_should_return_correct_count_after_inserts(tmp_vec_db) -> None:
    rows = [
        EmbeddingRow(f"fn_{i}", _unit_vec(i % _DIM), "/f.py", i, i + 1, "Function")
        for i in range(7)
    ]
    bulk_insert(tmp_vec_db, rows)
    assert row_count(tmp_vec_db) == 7
