"""Numpy-based embedding store — replaces the LadybugDB vector index.

Embeddings are stored in per-repo numpy files alongside the DB file:
  {db_dir}/{slug}.embeddings.npy        — float32 matrix (N × 768)
  {db_dir}/{slug}.embeddings_idx.json   — list of qualified_names (index → qn)

This avoids any LadybugDB vector extension dependency so the DB can be
opened in any fresh process without pre-loading the VECTOR extension.
Semantic search uses numpy cosine similarity — plenty fast for codebases
up to ~50k functions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emb_paths(db_path: str) -> tuple[Path, Path]:
    """Return (npy_path, idx_path) for the given DB file."""
    base = Path(db_path).with_suffix("")
    return base.with_suffix(".embeddings.npy"), base.with_suffix(".embeddings_idx.json")


# In-memory accumulator used during the embedding pass.
# Maps qualified_name -> np.ndarray(768,) float32.
_pending: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def store_embedding(
    node_id: object,
    vector: list[float] | None,
    qualified_name: str,
) -> None:
    """Store a single embedding in the in-memory accumulator.

    Convenience wrapper around ``store_embedding_batch`` for single-item
    calls.  ``flush_embeddings()`` must be called once all embeddings are
    accumulated to persist them to disk.

    Args:
        node_id: Ignored — kept for API compatibility with callers that used
            the old Qdrant-backed store which required an integer point ID.
        vector: 768-dim float list.  Ignored if ``None``.
        qualified_name: Fully-qualified symbol name (the lookup key).
    """
    if vector is not None:
        _pending[qualified_name] = np.asarray(vector, dtype=np.float32)


def store_embedding_batch(
    embeddings: list[tuple[str, list[float], str]],
    *args: object,
    **kwargs: object,
) -> int:
    """Accumulate embeddings in memory during the embedding pass.

    Accepts the legacy 3-tuple signature ``(node_id, vector, qualified_name)``
    used by ``graph_updater._generate_semantic_embeddings()``.  ``node_id`` is
    ignored — ``qualified_name`` is the key.

    ``flush_embeddings()`` must be called once after all batches to persist to
    disk, **or** pass ``db_path`` to auto-flush immediately.

    Returns:
        Number of embeddings accepted (non-None vectors).
    """
    count = 0
    for item in embeddings:
        if len(item) == 3:
            # Legacy signature: (node_id, vector, qualified_name)
            _node_id, vec, qn = item  # type: ignore[misc]
        else:
            # New signature: (qualified_name, vector)
            qn, vec = item[0], item[1]  # type: ignore[misc]

        if vec is not None:
            _pending[qn] = np.asarray(vec, dtype=np.float32)
            count += 1

    logger.debug("Accumulated %d embeddings in memory (total pending: %d)", count, len(_pending))
    return count


def flush_embeddings(db_path: str | None = None) -> int:
    """Write accumulated embeddings to disk and clear the in-memory store.

    Args:
        db_path: Path to the repo's .db file. Defaults to settings.LADYBUG_DB_PATH.

    Returns:
        Number of embeddings persisted.
    """
    path = db_path or settings.LADYBUG_DB_PATH
    npy_path, idx_path = _emb_paths(path)

    if not _pending:
        logger.debug("flush_embeddings: nothing to write")
        return 0

    qns = list(_pending.keys())
    matrix = np.stack([_pending[q] for q in qns]).astype(np.float32)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), matrix)
    idx_path.write_text(json.dumps(qns))
    count = len(qns)
    _pending.clear()
    logger.info("Flushed %d embeddings to %s", count, npy_path)
    return count


def search_embeddings(
    query_vector: np.ndarray | list[float],
    k: int = 10,
    db_path: str | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Find the top-k most similar embeddings using cosine similarity.

    If there are pending (unflushed) embeddings in memory, they are flushed
    to disk at ``db_path`` before the search so callers that omit an explicit
    ``flush_embeddings()`` call still get correct results.

    Args:
        query_vector: 768-dim float vector.
        k: Number of results to return.
        db_path: Path to the repo's .db file. Defaults to settings.LADYBUG_DB_PATH.
        top_k: Alias for ``k`` (legacy kwarg).

    Returns:
        List of (qualified_name, score) tuples, descending by score.
    """
    # Accept legacy ``top_k`` kwarg for backward compatibility with callers
    # that used the old LadybugDB-backed API.
    effective_k = top_k if top_k is not None else k

    path = db_path or settings.LADYBUG_DB_PATH

    # Auto-flush any pending embeddings so searches work immediately after storing.
    if _pending:
        flush_embeddings(path)

    npy_path, idx_path = _emb_paths(path)

    if not npy_path.exists() or not idx_path.exists():
        logger.warning("search_embeddings: no embedding file at %s", npy_path)
        return []

    matrix = np.load(str(npy_path))  # (N, 768) float32
    qns: list[str] = json.loads(idx_path.read_text())

    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    mat_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    scores = mat_norms @ q_norm  # (N,)

    n_results = int(min(effective_k, len(scores)))
    top_idx = np.argpartition(scores, -n_results)[-n_results:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return [(qns[i], float(scores[i])) for i in top_idx]


def verify_stored_ids(
    expected_ids: set[object],
    db_path: str | None = None,
) -> set[object]:
    """Return the subset of ``expected_ids`` that have stored embeddings.

    Integer IDs (legacy Qdrant point IDs) are returned as-is — they cannot be
    checked against the numpy index and are assumed present for backward
    compatibility with callers that stored embeddings before the Qdrant→numpy
    migration.

    String IDs are checked against both the in-memory ``_pending`` accumulator
    and the on-disk index so calls made before ``flush_embeddings()`` still
    return correct results.

    Args:
        expected_ids: Set of qualified names (str) or legacy integer IDs.
        db_path: Path to the repo's .db file. Defaults to settings.LADYBUG_DB_PATH.

    Returns:
        Subset of expected_ids that are present.
    """
    if not expected_ids:
        return set()

    # Legacy integer IDs pass through unconditionally.
    int_ids = {i for i in expected_ids if isinstance(i, int)}
    str_ids = {i for i in expected_ids if isinstance(i, str)}

    if not str_ids:
        return int_ids

    path = db_path or settings.LADYBUG_DB_PATH
    _npy_path, idx_path = _emb_paths(path)

    # Check in-memory pending first (covers calls before flush).
    found_in_pending = {qn for qn in str_ids if qn in _pending}

    # Then check on-disk index.
    found_on_disk: set[str] = set()
    if idx_path.exists():
        try:
            stored = set(json.loads(idx_path.read_text()))
            found_on_disk = str_ids & stored
        except Exception as exc:
            logger.warning("verify_stored_ids: could not read index: %s", exc)

    return int_ids | found_in_pending | found_on_disk


def delete_project_embeddings(
    project_name: str,
    node_ids: list[object],  # noqa: ARG001  (legacy param, unused in numpy impl)
    db_path: str | None = None,
) -> None:
    """Delete all embeddings whose qualified_name starts with ``project_name``.

    Also clears any pending (unflushed) embeddings for the same project.
    The ``node_ids`` parameter is accepted for API compatibility with the old
    Qdrant-backed store but is not used — deletion is name-prefix based.

    Args:
        project_name: Repo slug / project prefix (e.g. ``"myproject"``).
        node_ids: Ignored.  Kept for call-site compatibility.
        db_path: Path to the repo's .db file. Defaults to settings.LADYBUG_DB_PATH.
    """
    prefix = project_name + "."

    # Remove from in-memory accumulator.
    for qn in list(_pending.keys()):
        if qn.startswith(prefix) or qn == project_name:
            del _pending[qn]

    # Remove from on-disk index + matrix.
    path = db_path or settings.LADYBUG_DB_PATH
    npy_path, idx_path = _emb_paths(path)

    if not npy_path.exists() or not idx_path.exists():
        return

    try:
        qns: list[str] = json.loads(idx_path.read_text())
    except Exception as exc:
        logger.warning("delete_project_embeddings: could not read index: %s", exc)
        return

    keep_mask = [
        not (q.startswith(prefix) or q == project_name) for q in qns
    ]
    kept_qns = [q for q, keep in zip(qns, keep_mask) if keep]

    if len(kept_qns) == len(qns):
        return  # nothing to delete

    if not kept_qns:
        npy_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)
        logger.info("Deleted all embeddings for project '%s'", project_name)
        return

    matrix = np.load(str(npy_path))
    kept_matrix = matrix[keep_mask]
    np.save(str(npy_path), kept_matrix)
    idx_path.write_text(json.dumps(kept_qns))
    removed = len(qns) - len(kept_qns)
    logger.info("Deleted %d embeddings for project '%s'", removed, project_name)
