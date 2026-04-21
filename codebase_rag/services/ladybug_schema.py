"""CI-3: LadybugDB schema migration (DEV-1171).

Declares all node tables, relationship tables, and vector indexes that
match the Code-Graph-RAG schema. Must be run once before any ingestion.
Safe to call on an existing DB — every DDL uses ``IF NOT EXISTS`` guards
and vector index creation is wrapped in try/except to handle the
already-exists case gracefully.

Schema layout:
    Node tables
        Project, Package, Folder, File, Module, Class, Function, Method,
        Interface, Enum, ExternalPackage, Embedding.

        The ``Embedding`` table is deliberately separated from
        ``Function``/``Method`` so pass-4 (``generate-embeddings``) can
        DELETE+CREATE rows without tripping LadybugDB's "cannot SET an
        indexed vector column" constraint during MERGE.

    Relationship tables
        CONTAINS_FILE, CONTAINS_FOLDER, CONTAINS_PACKAGE, CONTAINS_MODULE,
        DEFINES, DEFINES_METHOD, CALLS, IMPORTS, INHERITS, IMPLEMENTS,
        OVERRIDES, BELONGS_TO.

    Vector indexes
        Only one — on ``Embedding.embedding`` — used by semantic search.
"""
from __future__ import annotations

import real_ladybug as lb
from loguru import logger

# ---------------------------------------------------------------------------
# Node table definitions
# ---------------------------------------------------------------------------
# Order matters only indirectly: rel tables below reference these node
# tables, so node DDL is always executed first in migrate().
_NODE_TABLES: list[str] = [
    """CREATE NODE TABLE IF NOT EXISTS Project(
        name STRING,
        root_path STRING,
        PRIMARY KEY (name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Package(
        qualified_name STRING,
        name STRING,
        path STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Folder(
        path STRING,
        name STRING,
        PRIMARY KEY (path)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS File(
        path STRING,
        name STRING,
        extension STRING,
        PRIMARY KEY (path)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Module(
        qualified_name STRING,
        name STRING,
        path STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Class(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Function(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        source_code STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Method(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        source_code STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Interface(
        qualified_name STRING,
        name STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Enum(
        qualified_name STRING,
        name STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS ExternalPackage(
        qualified_name STRING,
        name STRING,
        PRIMARY KEY (qualified_name)
    )""",
    # Separate embedding table — vector-indexed column lives here so that
    # pass-4 (generate-embeddings) can DELETE + CREATE without touching the
    # structural node or its relationships.
    """CREATE NODE TABLE IF NOT EXISTS Embedding(
        qualified_name STRING,
        node_type STRING,
        embedding FLOAT[768],
        PRIMARY KEY (qualified_name)
    )""",
]

# ---------------------------------------------------------------------------
# Relationship table definitions
# ---------------------------------------------------------------------------
# Each REL TABLE may have multiple FROM/TO pairs because LadybugDB requires
# declaring every valid endpoint combination up front (unlike some graph
# DBs that allow ad-hoc typing at edge-insert time).
_REL_TABLES: list[str] = [
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_FILE(
        FROM Project TO File,
        FROM Package TO File,
        FROM Folder TO File
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_FOLDER(
        FROM Project TO Folder,
        FROM Folder TO Folder
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_PACKAGE(
        FROM Project TO Package,
        FROM Package TO Package
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_MODULE(
        FROM Project TO Module,
        FROM Package TO Module
    )""",
    """CREATE REL TABLE IF NOT EXISTS DEFINES(
        FROM Module TO Class,
        FROM Module TO Function,
        FROM Module TO Interface,
        FROM Module TO Enum
    )""",
    """CREATE REL TABLE IF NOT EXISTS DEFINES_METHOD(
        FROM Class TO Method
    )""",
    """CREATE REL TABLE IF NOT EXISTS CALLS(
        FROM Function TO Function,
        FROM Function TO Method,
        FROM Method TO Function,
        FROM Method TO Method,
        FROM Module TO Function,
        FROM Module TO Method
    )""",
    """CREATE REL TABLE IF NOT EXISTS IMPORTS(
        FROM Module TO Module,
        FROM Module TO ExternalPackage
    )""",
    """CREATE REL TABLE IF NOT EXISTS INHERITS(
        FROM Class TO Class
    )""",
    """CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(
        FROM Class TO Interface
    )""",
    """CREATE REL TABLE IF NOT EXISTS OVERRIDES(
        FROM Method TO Method
    )""",
    """CREATE REL TABLE IF NOT EXISTS BELONGS_TO(
        FROM File TO Module,
        FROM File TO Package
    )""",
]

# ---------------------------------------------------------------------------
# Vector indexes — run after node tables
# ---------------------------------------------------------------------------
_VECTOR_INDEXES: list[tuple[str, str, str]] = [
    # (table_name, index_name, vector_column)
    # CALL CREATE_VECTOR_INDEX(table, index_name, vector_col)
    # Single index on the dedicated Embedding table; Function and Method
    # no longer carry the embedding column directly, which avoids the
    # "cannot SET an indexed vector column" constraint during MERGE.
    ("Embedding", "embed_idx", "embedding"),
]


def migrate(db_path: str) -> None:
    """Run schema migration on the given LadybugDB database path.

    Idempotent — safe to call on an existing database. Node/rel tables use
    ``IF NOT EXISTS``; vector indexes are created in a try/except to handle
    the already-exists case gracefully (LadybugDB does not support
    ``CREATE ... IF NOT EXISTS`` for vector indexes).

    Args:
        db_path: Filesystem path to the LadybugDB database file. Created if
            it does not exist.
    """
    logger.info(f"Running LadybugDB schema migration on: {db_path}")
    db = lb.Database(db_path)
    conn = lb.Connection(db)

    # Node DDL first — rel tables below reference these types.
    for ddl in _NODE_TABLES:
        # Extract the table name from the DDL for logging only — LadybugDB
        # does not echo the created object name back to the caller.
        table_name = ddl.split("TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        conn.execute(ddl)
        logger.debug(f"  Node table: {table_name}")

    for ddl in _REL_TABLES:
        table_name = ddl.split("TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        conn.execute(ddl)
        logger.debug(f"  Rel table: {table_name}")

    # Load the VECTOR extension (required for CREATE_VECTOR_INDEX /
    # QUERY_VECTOR_INDEX). INSTALL fetches the binary if not present; LOAD
    # registers it with this connection. Both are guarded: INSTALL is
    # idempotent and LOAD is non-fatal so the core schema still builds even
    # on systems where the extension is unavailable — semantic search just
    # won't work until the extension is present.
    try:
        conn.execute("INSTALL VECTOR")
        logger.debug("  VECTOR extension installed")
    except Exception:
        pass  # already installed
    try:
        conn.execute("LOAD EXTENSION VECTOR")
        logger.debug("  VECTOR extension loaded")
    except Exception as e:
        logger.warning(f"  Could not load VECTOR extension: {e} — semantic search will be unavailable")

    for node_table, idx_name, prop in _VECTOR_INDEXES:
        try:
            conn.execute(f"CALL CREATE_VECTOR_INDEX('{node_table}', '{idx_name}', '{prop}')")
            logger.debug(f"  Vector index: {idx_name} on {node_table}.{prop}")
        except Exception as e:
            # There's no IF NOT EXISTS for vector indexes, so detect the
            # already-exists case by error substring and treat it as success.
            if "already exists" in str(e).lower():
                logger.debug(f"  Vector index {idx_name} already exists — skipping")
            else:
                logger.warning(f"  Vector index {idx_name}: {e}")

    logger.info("LadybugDB schema migration complete ✓")
