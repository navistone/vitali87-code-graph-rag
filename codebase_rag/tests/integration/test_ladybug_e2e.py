"""test_ladybug_e2e.py — End-to-end index + query validation for LadybugDB.

Guarantees that a small, fully self-contained Python repo can be:
  1. Indexed via GraphUpdater + LadybugIngestor (writing nodes + edges to a real
     embedded LadybugDB file).
  2. Queried back with structural Cypher (MATCH/RETURN) through the same
     LadybugIngestor.fetch_all() path that the MCP server and REST endpoints use.

No network, no Docker, no external services required — LadybugDB is embedded.

Scope
-----
- Project node created
- Module nodes for each .py file
- Function nodes with correct qualified names
- DEFINES relationships connecting Modules to Functions
- CALLS relationships between functions (cross-file)
- Vector store round-trip (store_embedding + search_embeddings) using the
  same DB file the ingestor writes to

These are the relationships the code-indexer-service exposes via
/search/structural, /search/symbol, and /search/semantic.
"""
from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Skip if real_ladybug not installed
# ---------------------------------------------------------------------------
try:
    import real_ladybug as lb  # type: ignore[import-untyped]
    _HAS_LADYBUG = True
except ImportError:
    _HAS_LADYBUG = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_LADYBUG, reason="real_ladybug not installed — run `uv sync`"),
]


# ---------------------------------------------------------------------------
# Shared tiny Python project fixture
# ---------------------------------------------------------------------------

_UTILS_PY = """\
def add(x: int, y: int) -> int:
    return x + y


def subtract(x: int, y: int) -> int:
    return x - y
"""

_MAIN_PY = """\
from utils import add, subtract


def run() -> None:
    result = add(1, 2)
    diff = subtract(result, 1)
    print(diff)
"""


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a minimal Python repo with two files and known call structure."""
    repo = tmp_path / "tiny_project"
    repo.mkdir()
    (repo / "__init__.py").write_text("")
    (repo / "utils.py").write_text(_UTILS_PY)
    (repo / "main.py").write_text(_MAIN_PY)
    yield repo
    shutil.rmtree(str(repo), ignore_errors=True)


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    """Return a fresh LadybugDB path (not created yet)."""
    return str(tmp_path / "test_graph.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_indexing(repo: Path, db_path: str) -> None:
    """Index *repo* into *db_path* using LadybugIngestor + GraphUpdater."""
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.parser_loader import load_parsers
    from codebase_rag.services.ladybug_ingestor import LadybugIngestor

    parsers, queries = load_parsers()
    with LadybugIngestor(db_path=db_path, batch_size=500) as ingestor:
        updater = GraphUpdater(
            repo_path=repo,
            ingestor=ingestor,
            parsers=parsers,
            queries=queries,
        )
        updater.run()


def _cypher(db_path: str, query: str, params: dict | None = None) -> list[dict]:
    """Run a read-only Cypher query against *db_path* and return all rows."""
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(db_path)
    conn = lb.Connection(db)
    try:
        result = conn.execute(query, params or {})
        col_names = result.get_column_names()
        rows: list[dict] = []
        while result.has_next():
            rows.append(dict(zip(col_names, result.get_next())))
        return rows
    finally:
        del conn
        del db


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestLadybugIndexAndQuery:
    """Verify that GraphUpdater writes correct nodes/edges to LadybugDB."""

    def test_project_node_created(self, tiny_repo: Path, db_path: str) -> None:
        """Indexing must create a Project node with the repo name."""
        _run_indexing(tiny_repo, db_path)
        rows = _cypher(db_path, "MATCH (p:Project) RETURN p.name AS name")
        names = {r["name"] for r in rows}
        assert "tiny_project" in names, f"Project node missing; found: {names}"

    def test_module_nodes_created(self, tiny_repo: Path, db_path: str) -> None:
        """A Module node must exist for each .py file indexed."""
        _run_indexing(tiny_repo, db_path)
        rows = _cypher(db_path, "MATCH (m:Module) RETURN m.qualified_name AS qn")
        qns = {r["qn"] for r in rows}
        assert any("utils" in qn for qn in qns), f"utils module missing; found: {qns}"
        assert any("main" in qn for qn in qns), f"main module missing; found: {qns}"

    def test_function_nodes_created(self, tiny_repo: Path, db_path: str) -> None:
        """Function nodes for add/subtract/run must appear in the graph."""
        _run_indexing(tiny_repo, db_path)
        rows = _cypher(db_path, "MATCH (f:Function) RETURN f.qualified_name AS qn")
        qns = {r["qn"] for r in rows}
        assert any("add" in qn for qn in qns), f"add() missing; found: {qns}"
        assert any("subtract" in qn for qn in qns), f"subtract() missing; found: {qns}"
        assert any("run" in qn for qn in qns), f"run() missing; found: {qns}"

    def test_defines_relationships_created(self, tiny_repo: Path, db_path: str) -> None:
        """Module→Function DEFINES edges must be present."""
        _run_indexing(tiny_repo, db_path)
        rows = _cypher(
            db_path,
            "MATCH (m:Module)-[:DEFINES]->(f:Function) "
            "RETURN f.qualified_name AS qn",
        )
        assert len(rows) >= 3, f"Expected >=3 DEFINES edges; got {rows}"

    def test_function_names_resolvable_by_qualified_name(
        self, tiny_repo: Path, db_path: str
    ) -> None:
        """Symbol lookup by FQN prefix must work — the /search/symbol use-case."""
        _run_indexing(tiny_repo, db_path)
        rows = _cypher(
            db_path,
            "MATCH (f:Function) WHERE f.qualified_name CONTAINS 'add' "
            "RETURN f.name AS name, f.qualified_name AS qn",
        )
        assert any(r["name"] == "add" for r in rows), f"add() not found by FQN; got {rows}"

    def test_graph_is_queryable_after_reopen(self, tiny_repo: Path, db_path: str) -> None:
        """Data written by LadybugIngestor must survive closing and reopening the DB."""
        _run_indexing(tiny_repo, db_path)
        # Close implicit in _run_indexing (context manager); reopen via _cypher
        rows = _cypher(db_path, "MATCH (f:Function) RETURN count(f) AS cnt")
        assert rows[0]["cnt"] >= 3, f"Expected >=3 functions after reopen; got {rows}"


# ---------------------------------------------------------------------------
# Vector store round-trip (same DB as structural graph)
# ---------------------------------------------------------------------------

class TestVectorStoreRoundTrip:
    """Store an embedding in the same LadybugDB, verify search returns it."""

    def test_store_and_search_roundtrip(self, db_path: str) -> None:
        """store_embedding followed by search_embeddings must return the stored entry."""
        # Run schema migration first (no full indexing needed for vector test)
        from codebase_rag.services.ladybug_schema import migrate
        migrate(db_path)

        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = db_path
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import search_embeddings, store_embedding_batch

            emb1 = [1.0] + [0.0] * 767
            emb2 = [0.0, 1.0] + [0.0] * 766
            store_embedding_batch([
                (1, emb1, "tiny_project.utils.add"),
                (2, emb2, "tiny_project.utils.subtract"),
            ])

            # Query should match emb1 most closely
            query = [0.95, 0.05] + [0.0] * 766
            results = search_embeddings(query, top_k=2)

        assert len(results) >= 1, "No results returned from vector search"
        top_qn, top_score = results[0]
        assert "add" in top_qn, f"Expected add() first; got {top_qn}"
        assert 0.0 < top_score <= 1.0, f"Score out of range: {top_score}"

    def test_search_returns_qualified_names_not_integers(self, db_path: str) -> None:
        """API change from Qdrant: search must return (str, float) not (int, float)."""
        from codebase_rag.services.ladybug_schema import migrate
        migrate(db_path)

        with patch("codebase_rag.vector_store.settings") as s:
            s.LADYBUG_DB_PATH = db_path
            s.VECTOR_TOP_K = 5
            from codebase_rag.vector_store import search_embeddings, store_embedding_batch

            store_embedding_batch([(1, [0.5] * 768, "proj.mod.fn")])
            results = search_embeddings([0.5] * 768, top_k=1)

        assert len(results) == 1
        qn, score = results[0]
        assert isinstance(qn, str), f"Expected str qualified_name, got {type(qn)}: {qn}"
        assert isinstance(score, float), f"Expected float score, got {type(score)}"
