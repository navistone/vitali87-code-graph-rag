"""Unit tests for centrality storage in codebase_rag.storage.vector_store.

Plan J adds a ``centrality`` KV-style table that stores PageRank scores
keyed by qualified_name so that semantic search can fuse cosine similarity
with graph centrality.

Tests follow the Testing Rules:
- Names: "should [result] when [condition]"
- Real DuckDB store; no mocks
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb", reason="duckdb not installed")

from codebase_rag.storage.vector_store import (
    clear_centrality,
    open_or_create,
    read_centrality,
    write_centrality,
)


@pytest.fixture()
def tmp_vec_db(tmp_path: Path):
    """Yield a fresh open_or_create connection to a temp .duck file."""
    db_file = tmp_path / "test.duck"
    conn = open_or_create(str(db_file))
    yield conn
    conn.close()


def test_should_create_centrality_table_when_db_is_new(tmp_vec_db) -> None:
    tables = {
        r[0]
        for r in tmp_vec_db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    assert "centrality" in tables


def test_should_write_and_read_pagerank_scores(tmp_vec_db) -> None:
    scores = {
        "pkg.mod.fn_a": 1.0,
        "pkg.mod.fn_b": 0.5,
        "pkg.mod.fn_c": 0.25,
    }
    n = write_centrality(tmp_vec_db, scores)
    assert n == 3
    out = read_centrality(
        tmp_vec_db, ["pkg.mod.fn_a", "pkg.mod.fn_b", "pkg.mod.fn_c"]
    )
    assert out == pytest.approx({
        "pkg.mod.fn_a": 1.0,
        "pkg.mod.fn_b": 0.5,
        "pkg.mod.fn_c": 0.25,
    })


def test_should_overwrite_existing_score_when_qualified_name_collides(
    tmp_vec_db,
) -> None:
    write_centrality(tmp_vec_db, {"pkg.mod.fn": 0.10})
    write_centrality(tmp_vec_db, {"pkg.mod.fn": 0.90})
    out = read_centrality(tmp_vec_db, ["pkg.mod.fn"])
    assert out == pytest.approx({"pkg.mod.fn": 0.90})


def test_should_return_empty_dict_when_qualified_names_not_found(
    tmp_vec_db,
) -> None:
    write_centrality(tmp_vec_db, {"pkg.mod.fn_a": 0.7})
    out = read_centrality(tmp_vec_db, ["does.not.exist", "also.missing"])
    assert out == {}


def test_should_clear_all_scores_when_clear_centrality_called(tmp_vec_db) -> None:
    write_centrality(
        tmp_vec_db,
        {"pkg.mod.fn_a": 1.0, "pkg.mod.fn_b": 0.5},
    )
    clear_centrality(tmp_vec_db)
    out = read_centrality(tmp_vec_db, ["pkg.mod.fn_a", "pkg.mod.fn_b"])
    assert out == {}


def test_should_return_empty_dict_when_qualified_names_list_is_empty(
    tmp_vec_db,
) -> None:
    write_centrality(tmp_vec_db, {"pkg.mod.fn_a": 0.7})
    assert read_centrality(tmp_vec_db, []) == {}


def test_should_return_zero_when_write_centrality_receives_empty_dict(
    tmp_vec_db,
) -> None:
    assert write_centrality(tmp_vec_db, {}) == 0
