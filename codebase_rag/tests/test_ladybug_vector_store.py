"""test_ladybug_vector_store.py — LadybugDB-backed vector store unit + integration tests.

Replaces the Qdrant-based ``test_vector_store.py`` tests (all skipped after CI-5).

These tests exercise:
- ``store_embedding`` / ``store_embedding_batch`` — write embeddings to LadybugDB
- ``search_embeddings`` — cosine similarity via QUERY_VECTOR_INDEX
- ``verify_stored_ids`` — check which qualified_names are stored
- ``delete_project_embeddings`` — remove embeddings by project prefix

Each test spins up a fresh LadybugDB in a temp directory and patches
``settings.LADYBUG_DB_PATH`` so the global config points to the test DB.
No Docker, no Qdrant, no network required.
"""
from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if real_ladybug is not installed
# (developer machines that haven't run `uv sync` yet).
# ---------------------------------------------------------------------------
try:
    import real_ladybug as lb  # type: ignore[import-untyped]  # noqa: F401
    _HAS_LADYBUG = True
except ImportError:
    _HAS_LADYBUG = False

pytestmark = pytest.mark.skipif(
    not _HAS_LADYBUG,
    reason="real_ladybug not installed — run `uv sync` first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ladybug_db(tmp_path: Path) -> Generator[str, None, None]:
    """Create a fresh LadybugDB, run the schema migration, yield the db path."""
    db_path = str(tmp_path / "test_graph.db")

    # Run the schema migration so Embedding table and vector index exist.
    from codebase_rag.services.ladybug_schema import migrate
    migrate(db_path)

    yield db_path

    # Cleanup (tmp_path is auto-cleaned by pytest, but be explicit)
    shutil.rmtree(db_path, ignore_errors=True)


@pytest.fixture()
def vs(ladybug_db: str):
    """Patch settings.LADYBUG_DB_PATH and return the vector_store module."""
    with patch("codebase_rag.vector_store.settings") as mock_settings:
        mock_settings.LADYBUG_DB_PATH = ladybug_db
        mock_settings.VECTOR_TOP_K = 5
        import codebase_rag.vector_store as _vs
        yield _vs


# ---------------------------------------------------------------------------
# store_embedding / store_embedding_batch
# ---------------------------------------------------------------------------

class TestStoreEmbedding:
    def test_store_single_embedding_returns_without_error(self, vs, ladybug_db: str) -> None:
        """store_embedding must not raise on a valid input."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import store_embedding
            # Should not raise
            store_embedding(1, [0.1] * 768, "project.module.func")

    def test_store_batch_returns_count(self, ladybug_db: str) -> None:
        """store_embedding_batch must return the number of embeddings stored."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import store_embedding_batch
            points = [
                (1, [0.1] * 768, "proj.mod.fn1"),
                (2, [0.2] * 768, "proj.mod.fn2"),
            ]
            count = store_embedding_batch(points)
        assert count == 2

    def test_store_batch_empty_returns_zero(self, ladybug_db: str) -> None:
        """store_embedding_batch with an empty list must return 0."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import store_embedding_batch
            count = store_embedding_batch([])
        assert count == 0

    def test_store_idempotent_upsert(self, ladybug_db: str) -> None:
        """Storing the same qualified_name twice must not raise; latest wins."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                store_embedding_batch,
                verify_stored_ids,
            )
            emb_v1 = [1.0] + [0.0] * 767
            emb_v2 = [0.0, 1.0] + [0.0] * 766
            store_embedding_batch([(1, emb_v1, "proj.fn")])
            store_embedding_batch([(1, emb_v2, "proj.fn")])  # should not raise
            # Should still exist (upsert)
            found = verify_stored_ids({"proj.fn"})
        assert "proj.fn" in found


# ---------------------------------------------------------------------------
# verify_stored_ids
# ---------------------------------------------------------------------------

class TestVerifyStoredIds:
    def test_empty_input_returns_empty_set(self, ladybug_db: str) -> None:
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import verify_stored_ids
            result = verify_stored_ids(set())
        assert result == set()

    def test_stored_id_is_found(self, ladybug_db: str) -> None:
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                store_embedding_batch,
                verify_stored_ids,
            )
            store_embedding_batch([(1, [0.5] * 768, "myproj.mod.alpha")])
            found = verify_stored_ids({"myproj.mod.alpha"})
        assert "myproj.mod.alpha" in found

    def test_absent_id_not_returned(self, ladybug_db: str) -> None:
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import verify_stored_ids
            found = verify_stored_ids({"ghost.module.fn"})
        assert "ghost.module.fn" not in found

    def test_integer_ids_pass_through(self, ladybug_db: str) -> None:
        """Legacy integer IDs must be returned as-is (verified = True)."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import verify_stored_ids
            result = verify_stored_ids({42, 99})
        assert 42 in result
        assert 99 in result


# ---------------------------------------------------------------------------
# search_embeddings
# ---------------------------------------------------------------------------

class TestSearchEmbeddings:
    def test_search_empty_db_returns_empty_list(self, ladybug_db: str) -> None:
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import search_embeddings
            results = search_embeddings([0.5] * 768)
        assert results == []

    def test_search_returns_closest_embedding(self, ladybug_db: str) -> None:
        """Vector closest to the query must score highest."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                search_embeddings,
                store_embedding_batch,
            )

            emb_a = [1.0] + [0.0] * 767          # unit vector in dim 0
            emb_b = [0.0, 1.0] + [0.0] * 766     # unit vector in dim 1
            store_embedding_batch([
                (1, emb_a, "proj.fn_a"),
                (2, emb_b, "proj.fn_b"),
            ])

            # Query closest to emb_a
            query = [0.99, 0.01] + [0.0] * 766
            results = search_embeddings(query, top_k=2)

        assert len(results) == 2
        # First result (highest score) should be fn_a
        top_qn, top_score = results[0]
        assert top_qn == "proj.fn_a", f"Expected fn_a first, got {top_qn}"
        assert 0.0 <= top_score <= 1.0

    def test_search_score_between_zero_and_one(self, ladybug_db: str) -> None:
        """All returned scores must be in [0, 1]."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                search_embeddings,
                store_embedding_batch,
            )

            emb = [0.5] * 768
            store_embedding_batch([(1, emb, "proj.fn")])
            results = search_embeddings([0.5] * 768, top_k=1)

        for _qn, score in results:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_search_respects_top_k(self, ladybug_db: str) -> None:
        """search_embeddings must return at most top_k results."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                search_embeddings,
                store_embedding_batch,
            )

            points = [(i, [float(i % 10)] * 768, f"proj.fn_{i}") for i in range(10)]
            store_embedding_batch(points)
            results = search_embeddings([1.0] * 768, top_k=3)

        assert len(results) <= 3

    def test_search_result_is_list_of_tuples(self, ladybug_db: str) -> None:
        """Return type must be list[tuple[str, float]]."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                search_embeddings,
                store_embedding_batch,
            )

            store_embedding_batch([(1, [0.1] * 768, "proj.fn")])
            results = search_embeddings([0.1] * 768, top_k=1)

        assert isinstance(results, list)
        if results:
            qn, score = results[0]
            assert isinstance(qn, str)
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# delete_project_embeddings
# ---------------------------------------------------------------------------

class TestDeleteProjectEmbeddings:
    def test_delete_removes_matching_embeddings(self, ladybug_db: str) -> None:
        """Embeddings whose qualified_name starts with project_name are removed."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import (
                delete_project_embeddings,
                store_embedding_batch,
                verify_stored_ids,
            )
            store_embedding_batch([
                (1, [0.1] * 768, "myproject.mod.fn1"),
                (2, [0.2] * 768, "myproject.mod.fn2"),
                (3, [0.3] * 768, "other.mod.fn3"),
            ])
            delete_project_embeddings("myproject", [])
            remaining = verify_stored_ids({"myproject.mod.fn1", "myproject.mod.fn2", "other.mod.fn3"})

        assert "myproject.mod.fn1" not in remaining
        assert "myproject.mod.fn2" not in remaining
        assert "other.mod.fn3" in remaining

    def test_delete_nonexistent_project_is_noop(self, ladybug_db: str) -> None:
        """Deleting embeddings for an unknown project must not raise."""
        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = ladybug_db
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import delete_project_embeddings
            # Should not raise
            delete_project_embeddings("ghost_project", [])
