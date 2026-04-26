"""DuckDB-backed per-repo vector store (v5.3 §6.5 + §8.4).

Replaces the previous SQLite + sqlite-vec implementation with a single
``.duck`` file per repo holding ``FLOAT[768]`` embedding vectors plus
symbol metadata, repo metadata, and PageRank centrality scores.

v5.3 §17.2 explicitly rejects sqlite-vec; §8.4 mandates DuckDB
``FLOAT[768]`` columns and the built-in ``array_cosine_distance``
function with L2-normalised embeddings written at insert time so that
cosine similarity is equivalent to inner product at query time.

Schema
------
    embeddings(qualified_name TEXT PK, embedding FLOAT[768],
               symbol_type TEXT, file_path TEXT,
               start_line INTEGER, end_line INTEGER,
               indexed_at BIGINT)
    repo_metadata(key TEXT PK, value TEXT NOT NULL, updated_at BIGINT)
    centrality(qualified_name TEXT PK, pagerank REAL NOT NULL,
               updated_at BIGINT)

Public API (preserved from the previous backend so callers don't change):
    open_or_create(path) -> duckdb.DuckDBPyConnection
    insert_embedding(conn, row)
    bulk_insert(conn, rows) -> int
    search_similar(conn, query_vec, k) -> list[SearchResult]
    write_metadata(conn, **fields)
    read_metadata(conn, key, default=None)
    read_all_metadata(conn) -> dict[str, str]
    write_centrality(conn, scores) -> int
    read_centrality(conn, qualified_names) -> dict[str, float]
    clear_centrality(conn)
    row_count(conn) -> int

DuckDB connections are not safe to share across threads; callers should
open a per-request connection (or a per-thread cursor) and close it
when done.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb  # noqa: F401  (type-only; actual import is lazy)

_EMBEDDING_DIM = 768


@dataclass
class EmbeddingRow:
    """One row in the ``embeddings`` table.

    The embedding vector is L2-normalised at write time (see ``bulk_insert``)
    so that cosine similarity equals the inner product at query time.
    """

    qualified_name: str
    embedding: list[float]
    file_path: str
    start_line: int
    end_line: int
    symbol_type: str
    indexed_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class SearchResult:
    """One result from ``search_similar``."""

    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    score: float  # cosine similarity in [-1, 1]; higher = more similar


def _l2_normalise(vec: list[float]) -> list[float]:
    """Return a unit-norm copy of ``vec``.

    Zero vectors are returned unchanged so callers don't have to special-case
    them; sqrt of a tiny float still produces a defined result.
    """
    mag = math.sqrt(sum(x * x for x in vec))
    if mag == 0.0:
        return list(vec)
    return [x / mag for x in vec]


def open_or_create(path: str | Path) -> Any:
    """Open (or create) a ``.duck`` file and ensure the schema exists.

    Args:
        path: Filesystem path to the DuckDB file.  Parent directory is
            created automatically if missing.

    Returns:
        duckdb.DuckDBPyConnection: An open connection with the schema applied.
        Callers are responsible for closing it.

    Raises:
        RuntimeError: When the ``duckdb`` module is not installed.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover — surfaced clearly to callers
        raise RuntimeError(
            "duckdb is not installed. Add `duckdb>=1.1.0` to your project."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(path))
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            qualified_name TEXT PRIMARY KEY,
            embedding      FLOAT[{_EMBEDDING_DIM}],
            symbol_type    TEXT,
            file_path      TEXT,
            start_line     INTEGER,
            end_line       INTEGER,
            indexed_at     BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_metadata (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at BIGINT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS centrality (
            qualified_name TEXT PRIMARY KEY,
            pagerank       REAL NOT NULL,
            updated_at     BIGINT NOT NULL
        )
        """
    )
    return conn


def insert_embedding(conn: Any, row: EmbeddingRow) -> None:
    """Insert or replace a single embedding row.

    The embedding vector is L2-normalised before write so that
    ``1 - array_cosine_distance(...)`` at query time is the inner product
    against unit vectors (matches v5.3 §8.4 exactly).

    Args:
        conn: Open connection from ``open_or_create``.
        row: Embedding data including the 768-dim vector and metadata.
    """
    bulk_insert(conn, [row])


def bulk_insert(conn: Any, rows: list[EmbeddingRow]) -> int:
    """Insert (upsert) many rows inside a single transaction.

    Args:
        conn: Open connection from ``open_or_create``.
        rows: Embedding rows to insert.  Empty list is a no-op.

    Returns:
        int: Number of rows inserted.
    """
    if not rows:
        return 0

    now = int(time.time())
    conn.execute("BEGIN")
    try:
        for row in rows:
            normalised = _l2_normalise(row.embedding)
            conn.execute(
                "DELETE FROM embeddings WHERE qualified_name = ?",
                (row.qualified_name,),
            )
            conn.execute(
                """
                INSERT INTO embeddings
                    (qualified_name, embedding, symbol_type, file_path,
                     start_line, end_line, indexed_at)
                VALUES (?, ?::FLOAT[768], ?, ?, ?, ?, ?)
                """,
                (
                    row.qualified_name,
                    normalised,
                    row.symbol_type,
                    row.file_path,
                    row.start_line,
                    row.end_line,
                    row.indexed_at or now,
                ),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(rows)


def search_similar(
    conn: Any,
    query_vec: list[float],
    k: int = 10,
) -> list[SearchResult]:
    """Return the top-k most cosine-similar symbols.

    Stored embeddings are L2-normalised at write time, so for a normalised
    query vector ``1 - array_cosine_distance(stored, query)`` yields the
    cosine similarity in ``[-1, 1]`` (higher = more similar).  We normalise
    the query here defensively in case callers pass an un-normalised vector.

    Args:
        conn: Open connection from ``open_or_create``.
        query_vec: 768-dim query embedding.
        k: Max number of nearest neighbours to return.

    Returns:
        list[SearchResult]: Ranked results, highest similarity first.
    """
    normalised = _l2_normalise(query_vec)
    rows = conn.execute(
        """
        SELECT qualified_name, file_path, start_line, end_line,
               1.0 - array_cosine_distance(embedding, ?::FLOAT[768]) AS score
        FROM embeddings
        ORDER BY score DESC
        LIMIT ?
        """,
        (normalised, int(k)),
    ).fetchall()
    return [
        SearchResult(
            qualified_name=r[0],
            file_path=r[1] or "",
            start_line=int(r[2]) if r[2] is not None else 0,
            end_line=int(r[3]) if r[3] is not None else 0,
            score=float(r[4]),
        )
        for r in rows
    ]


def write_metadata(conn: Any, **fields: Any) -> None:
    """Upsert key-value pairs into ``repo_metadata``.

    Args:
        conn: Open connection from ``open_or_create``.
        **fields: Arbitrary key=value pairs.  All values are coerced to str.
    """
    now = int(time.time())
    conn.execute("BEGIN")
    try:
        for key, value in fields.items():
            conn.execute(
                "DELETE FROM repo_metadata WHERE key = ?",
                (key,),
            )
            conn.execute(
                """
                INSERT INTO repo_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, str(value), now),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def read_metadata(
    conn: Any, key: str, default: str | None = None
) -> str | None:
    """Read a single value from ``repo_metadata``.

    Args:
        conn: Open connection from ``open_or_create``.
        key: Metadata key (e.g. ``"last_indexed_at"``).
        default: Value to return when the key is absent.

    Returns:
        str | None: The stored value, or ``default`` when not found.
    """
    row = conn.execute(
        "SELECT value FROM repo_metadata WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else default


def read_all_metadata(conn: Any) -> dict[str, str]:
    """Return all rows from ``repo_metadata`` as a plain dict.

    Args:
        conn: Open connection from ``open_or_create``.

    Returns:
        dict[str, str]: All key/value pairs, empty dict on miss.
    """
    rows = conn.execute("SELECT key, value FROM repo_metadata").fetchall()
    return {r[0]: r[1] for r in rows}


def write_centrality(conn: Any, scores: dict[str, float]) -> int:
    """Bulk upsert PageRank centrality scores.

    Args:
        conn: Open connection from ``open_or_create``.
        scores: Mapping of ``qualified_name`` → normalised PageRank score
            in ``[0.0, 1.0]``.  Empty dict is a no-op.

    Returns:
        int: Number of rows written.
    """
    if not scores:
        return 0
    now = int(time.time())
    conn.execute("BEGIN")
    try:
        for qname, score in scores.items():
            conn.execute(
                "DELETE FROM centrality WHERE qualified_name = ?",
                (qname,),
            )
            conn.execute(
                """
                INSERT INTO centrality (qualified_name, pagerank, updated_at)
                VALUES (?, ?, ?)
                """,
                (qname, float(score), now),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(scores)


def read_centrality(
    conn: Any, qualified_names: list[str]
) -> dict[str, float]:
    """Return PageRank scores for the requested qualified names.

    Args:
        conn: Open connection from ``open_or_create``.
        qualified_names: Symbol names to look up.  Missing keys are absent
            from the returned dict — callers should default to 0.0.

    Returns:
        dict[str, float]: Subset of ``qualified_names`` that have a stored
        score, mapped to the score.
    """
    if not qualified_names:
        return {}
    placeholders = ",".join("?" for _ in qualified_names)
    rows = conn.execute(
        f"SELECT qualified_name, pagerank FROM centrality "
        f"WHERE qualified_name IN ({placeholders})",
        tuple(qualified_names),
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def clear_centrality(conn: Any) -> None:
    """Delete every row from the ``centrality`` table.

    Used before recomputing scores so stale qualified names from a previous
    indexing run don't linger after files are deleted upstream.

    Args:
        conn: Open connection from ``open_or_create``.
    """
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM centrality")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def row_count(conn: Any) -> int:
    """Return the number of embeddings stored.

    Args:
        conn: Open connection from ``open_or_create``.

    Returns:
        int: Row count (0 when table is empty or missing).
    """
    try:
        result = conn.execute("SELECT count(*) FROM embeddings").fetchone()
        return int(result[0]) if result else 0
    except Exception:
        return 0
