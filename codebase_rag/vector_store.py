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


def close_qdrant_client() -> None:
    """No-op — kept for API compatibility."""
    pass


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
    disk.

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

    Args:
        query_vector: 768-dim float vector.
        k: Number of results to return.
        db_path: Path to the repo's .db file. Defaults to settings.LADYBUG_DB_PATH.

    Returns:
        List of (qualified_name, score) tuples, descending by score.
    """
    # Accept legacy ``top_k`` kwarg for backward compatibility with callers
    # that used the old LadybugDB-backed API.
    effective_k = top_k if top_k is not None else k

    path = db_path or settings.LADYBUG_DB_PATH
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
    expected_ids: set[str],
    db_path: str | None = None,
) -> set[str]:
    """Return the subset of ``expected_ids`` that have stored embeddings.

    Args:
        expected_ids: Set of qualified names to check.
        db_path: Path to the repo's .db file. Defaults to settings.LADYBUG_DB_PATH.

    Returns:
        Subset of expected_ids that are present in the embedding index.
    """
    if not expected_ids:
        return set()

    path = db_path or settings.LADYBUG_DB_PATH
    _npy_path, idx_path = _emb_paths(path)

    if not idx_path.exists():
        return set()

    try:
        stored = set(json.loads(idx_path.read_text()))
    except Exception as exc:
        logger.warning("verify_stored_ids: could not read index: %s", exc)
        return set()

    return expected_ids & stored
