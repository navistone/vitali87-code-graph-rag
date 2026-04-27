"""Integration test fixtures for LadybugDB-backed ingestors.

Ported from the legacy Memgraph/mgclient conftest (v5.3 LadybugDB swap).
The previous implementation spun up a Memgraph Docker container via
testcontainers and connected with mgclient over Bolt/7687.

LadybugDB is an embedded graph store — no Docker, no network. The
``memgraph_ingestor`` fixture now creates a fresh LadybugDB file under
``tmp_path`` (function-scoped) so each test gets a clean, isolated database.
The ``memgraph_container`` and ``memgraph_connection`` fixtures are removed;
integration tests that used them have been updated to use
``memgraph_ingestor`` directly.
"""
from __future__ import annotations

import pytest

from codebase_rag.services.ladybug_ingestor import LadybugIngestor

if False:  # TYPE_CHECKING equivalent for type stubs only
    pass


@pytest.fixture(scope="function")
def memgraph_ingestor(tmp_path) -> LadybugIngestor:  # type: ignore[return]
    """Yield a fresh LadybugIngestor connected to a temporary DB file.

    The DB is created under pytest's ``tmp_path`` so it is automatically
    cleaned up after each test.  The ingestor is entered (``__enter__``
    runs the schema migration) and exited (``__exit__`` flushes + closes)
    around the test body via a generator fixture.

    Any nodes written during the test are isolated to this DB file and are
    discarded when the fixture tears down.
    """
    db_path = str(tmp_path / "test_integration.ladybug.db")
    ingestor = LadybugIngestor(db_path=db_path, batch_size=100, use_merge=True)
    ingestor.__enter__()
    try:
        yield ingestor
    finally:
        ingestor.__exit__(None, None, None)
