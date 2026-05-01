"""Unit tests for codebase_rag.storage.vector_store_arrow (opt-in Arrow path).

Tests follow the Testing Rules:
- Names: "should [result] when [condition]"
- Real DuckDB + real pyarrow (no mock drift)
- Equivalence with the executemany fallback is the explicit contract:
  same input must produce identical rows on disk and identical search ranking.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("duckdb", reason="duckdb not installed")
pytest.importorskip("pyarrow", reason="pyarrow not installed (optional [arrow] extra)")

from codebase_rag.storage.vector_store import (
    EmbeddingRow,
    open_or_create,
    row_count,
    search_similar,
)
from codebase_rag.storage.vector_store_arrow import bulk_insert_arrow

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
    db_file = tmp_path / "arrow.duck"
    conn = open_or_create(str(db_file))
    yield conn
    conn.close()


def _row(qn: str, idx: int) -> EmbeddingRow:
    """Build a synthetic EmbeddingRow with a unit vector at dimension ``idx``."""
    return EmbeddingRow(
        qualified_name=qn,
        embedding=_unit_vec(idx),
        file_path=f"/repo/{qn.split('.', maxsplit=1)[0]}.py",
        start_line=idx,
        end_line=idx + 5,
        symbol_type="Function",
    )


def test_should_insert_rows_and_count_when_called_with_valid_data(tmp_vec_db) -> None:
    rows = [_row(f"m.fn_{i}", i) for i in range(5)]
    inserted = bulk_insert_arrow(tmp_vec_db, rows)
    assert inserted == 5
    assert row_count(tmp_vec_db) == 5


def test_should_be_no_op_when_rows_is_empty(tmp_vec_db) -> None:
    assert bulk_insert_arrow(tmp_vec_db, []) == 0
    assert row_count(tmp_vec_db) == 0


def test_should_upsert_when_qualified_name_already_exists(tmp_vec_db) -> None:
    # Initial insert.
    bulk_insert_arrow(tmp_vec_db, [_row("m.fn_a", 1)])
    # Re-insert with a different vector.
    new_row = EmbeddingRow(
        qualified_name="m.fn_a",
        embedding=_unit_vec(2),
        file_path="/repo/m.py",
        start_line=99,
        end_line=109,
        symbol_type="Function",
    )
    bulk_insert_arrow(tmp_vec_db, [new_row])
    assert row_count(tmp_vec_db) == 1
    # The post-upsert row should be the most-similar match for unit_vec(2).
    hits = search_similar(tmp_vec_db, _unit_vec(2), k=1)
    assert hits[0].qualified_name == "m.fn_a"
    assert hits[0].start_line == 99


def test_should_l2_normalise_embeddings_so_cosine_matches_inner_product(tmp_vec_db) -> None:
    # Insert a vector with magnitude 5 (un-normalised) and a unit vector.
    big = [5.0] + [0.0] * (_DIM - 1)
    rows = [
        EmbeddingRow(
            qualified_name="m.big",
            embedding=big,
            file_path="/repo/m.py",
            start_line=1,
            end_line=2,
            symbol_type="Function",
        ),
        _row("m.unit_other", 1),  # orthogonal unit vector
    ]
    bulk_insert_arrow(tmp_vec_db, rows)

    # Querying with a unit vector aligned to the "big" direction must yield
    # cosine similarity 1.0 (not 5.0) — proving normalisation happened.
    hits = search_similar(tmp_vec_db, _unit_vec(0), k=2)
    top = next(h for h in hits if h.qualified_name == "m.big")
    assert math.isclose(top.score, 1.0, abs_tol=1e-5)


def test_should_produce_same_ranking_as_executemany_path(tmp_path: Path) -> None:
    """Equivalence test: arrow path must rank identically to executemany."""
    # Build two DBs from identical input and ensure search results match.
    rows = [_row(f"m.fn_{i}", i) for i in range(20)]

    db_arrow = open_or_create(str(tmp_path / "arrow.duck"))
    db_exec = open_or_create(str(tmp_path / "exec.duck"))
    try:
        bulk_insert_arrow(db_arrow, rows)

        # Force the executemany fallback by simulating a missing pyarrow.
        # The cleanest way: call the same SQL the fallback emits, but
        # we can simply re-use the public bulk_insert with an env trick.
        # Instead: rely on the documented contract and compare orderings
        # via search results.
        # Temporarily monkeypatch import to force fallback path.
        import builtins

        from codebase_rag.storage import vector_store as _vs

        real_import = builtins.__import__

        def fake_import(name: str, *a, **kw):
            if name == "pyarrow":
                raise ImportError("simulated absence")
            return real_import(name, *a, **kw)

        builtins.__import__ = fake_import
        try:
            _vs.bulk_insert(db_exec, rows)
        finally:
            builtins.__import__ = real_import

        query = _unit_vec(7)
        a_hits = [h.qualified_name for h in search_similar(db_arrow, query, k=5)]
        e_hits = [h.qualified_name for h in search_similar(db_exec, query, k=5)]
        assert a_hits == e_hits
    finally:
        db_arrow.close()
        db_exec.close()
