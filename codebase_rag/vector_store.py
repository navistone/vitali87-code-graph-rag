"""CI-5: LadybugDB-native vector store — replaces Qdrant.

Embeddings are stored in a dedicated ``Embedding`` node table (separate from
the structural ``Function``/``Method`` tables).  Using a separate table allows
the pass-4 *generate-embeddings* loop to use ``DETACH DELETE`` + ``CREATE``
without touching structural nodes or their relationships — the only pattern
that avoids the LadybugDB ``"cannot SET an indexed vector column"`` runtime
error.

Public API is intentionally identical to the old Qdrant-backed module so that
callers (``graph_updater.py``, ``semantic_search.py``) need only minimal
changes.

Key API change: ``search_embeddings`` now returns ``list[tuple[str, float]]``
(``qualified_name``, score) instead of ``list[tuple[int, float]]``
(``node_id``, score).  Integer node IDs were a Memgraph internal detail that
has no equivalent in LadybugDB.
"""
from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager

from loguru import logger

from . import logs as ls
from .config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextmanager
def _open_conn() -> Generator[object, None, None]:
    """Open a LadybugDB connection and ensure it is closed on exit.

    Yields the ``lb.Connection`` object.  Both the connection *and* the
    underlying ``lb.Database`` are deleted in the ``finally`` block so that
    kuzu releases its file-lock on the embedded database file promptly.

    Usage::

        with _open_conn() as conn:
            conn.execute(...)
    """
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(settings.LADYBUG_DB_PATH)
    conn = lb.Connection(db)
    # Load the VECTOR extension so QUERY_VECTOR_INDEX is available.
    try:
        conn.execute("INSTALL VECTOR")
    except Exception:
        pass
    try:
        conn.execute("LOAD EXTENSION VECTOR")
    except Exception as e:
        logger.warning(f"vector_store: could not load VECTOR extension: {e}")
    try:
        yield conn
    finally:
        del conn
        del db


def _result_to_rows(result: object) -> list[dict]:  # type: ignore[override]
    rows = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def close_qdrant_client() -> None:
    """No-op: LadybugDB uses ephemeral file connections — nothing to close."""


def store_embedding(
    node_id: str | int, embedding: list[float], qualified_name: str
) -> None:
    store_embedding_batch([(node_id, embedding, qualified_name)])


def store_embedding_batch(
    points: Sequence[tuple[str | int, list[float], str]],
) -> int:
    """Write embeddings to the LadybugDB ``Embedding`` node table.

    Uses ``DETACH DELETE`` + ``CREATE`` (the delete-then-insert pattern that
    LadybugDB requires for vector-indexed columns).  The ``node_id`` field
    from the original Qdrant API is accepted but ignored — ``qualified_name``
    is the primary key.
    """
    if not points:
        return 0
    stored = 0
    with _open_conn() as conn:
        for _, embedding, qualified_name in points:
            try:
                # Determine whether this is a Function or Method (used for join
                # back to the structural graph during semantic search).
                node_type = "Function"  # default; refined below
                try:
                    r = _result_to_rows(
                        conn.execute(  # type: ignore[attr-defined]
                            "MATCH (n:Method {qualified_name: $qn}) RETURN n.qualified_name LIMIT 1",
                            {"qn": qualified_name},
                        )
                    )
                    if r:
                        node_type = "Method"
                except Exception:
                    pass

                # Delete any existing embedding entry (idempotent).
                conn.execute(  # type: ignore[attr-defined]
                    "MATCH (e:Embedding {qualified_name: $qn}) DETACH DELETE e",
                    {"qn": qualified_name},
                )
                # Insert fresh embedding.
                conn.execute(  # type: ignore[attr-defined]
                    "CREATE (e:Embedding {qualified_name: $qn, node_type: $nt, embedding: $emb})",
                    {"qn": qualified_name, "nt": node_type, "emb": embedding},
                )
                stored += 1
            except Exception as e:
                logger.warning(f"vector_store: failed to store embedding for {qualified_name}: {e}")

    logger.debug(ls.EMBEDDING_BATCH_STORED.format(count=stored))
    return stored


def delete_project_embeddings(
    project_name: str, node_ids: Sequence[str | int]
) -> None:
    """Delete all Embedding nodes whose ``qualified_name`` belongs to the project."""
    with _open_conn() as conn:
        try:
            conn.execute(  # type: ignore[attr-defined]
                "MATCH (e:Embedding) WHERE e.qualified_name STARTS WITH ($project_name + '.') DETACH DELETE e",
                {"project_name": project_name},
            )
            logger.info(ls.QDRANT_DELETE_PROJECT_DONE.format(project=project_name))
        except Exception as e:
            logger.warning(
                ls.QDRANT_DELETE_PROJECT_FAILED.format(project=project_name, error=e)
            )


def verify_stored_ids(expected_ids: set[str | int]) -> set[str | int]:
    """Return the subset of ``expected_ids`` (qualified_names) that are stored.

    The original Qdrant API accepted ``set[int]``; we accept ``set[str]`` or
    ``set[int]`` but always interpret values as ``qualified_name`` strings.
    All integers are returned as-is (treated as fully verified) so that
    ``graph_updater._reconcile_embeddings`` is a no-op for integer IDs.
    """
    if not expected_ids:
        return set()

    str_ids = {str(i) for i in expected_ids if isinstance(i, str)}
    int_ids = {i for i in expected_ids if isinstance(i, int)}

    if not str_ids:
        # Legacy integer IDs — no meaningful way to verify; pretend all stored.
        return expected_ids

    found: set[str | int] = set()
    with _open_conn() as conn:
        try:
            results = _result_to_rows(
                conn.execute(  # type: ignore[attr-defined]
                    "MATCH (e:Embedding) WHERE e.qualified_name IN $qns RETURN e.qualified_name AS qn",
                    {"qns": list(str_ids)},
                )
            )
            found.update(r["qn"] for r in results)
        except Exception as e:
            logger.warning(f"vector_store: verify_stored_ids failed: {e}")

    found.update(int_ids)  # integers pass through as verified
    return found


def search_embeddings(
    query_embedding: list[float], top_k: int | None = None
) -> list[tuple[str, float]]:
    """Return the top-k most similar embeddings.

    Returns ``list[tuple[qualified_name, score]]`` where score ∈ [0, 1].
    (LadybugDB returns a ``distance`` value; we convert to similarity.)
    """
    effective_top_k = top_k if top_k is not None else settings.VECTOR_TOP_K
    with _open_conn() as conn:
        try:
            results = _result_to_rows(
                conn.execute(  # type: ignore[attr-defined]
                    "CALL QUERY_VECTOR_INDEX('Embedding', 'embed_idx', $vec, $k) "
                    "RETURN node.qualified_name AS qualified_name, distance",
                    {"vec": query_embedding, "k": effective_top_k},
                )
            )
            # LadybugDB returns cosine distance (0 = identical, 2 = opposite).
            # Convert to similarity score ∈ [0, 1].
            return [
                (str(r["qualified_name"]), max(0.0, 1.0 - float(r["distance"]) / 2.0))
                for r in results
                if r.get("qualified_name") is not None
            ]
        except Exception as e:
            logger.warning(ls.EMBEDDING_SEARCH_FAILED.format(error=e))
            return []
