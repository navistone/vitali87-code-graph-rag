"""CI-3: LadybugDB schema migration (DEV-1171).

Declares all node tables, relationship tables, and vector indexes
that match the Code-Graph-RAG schema. Must be run once before any
ingestion. Safe to call on an existing DB (IF NOT EXISTS guards).
"""
from __future__ import annotations

import real_ladybug as lb
from loguru import logger

# ---------------------------------------------------------------------------
# Node table definitions
# ---------------------------------------------------------------------------
_NODE_TABLES: list[str] = [
    """CREATE NODE TABLE IF NOT EXISTS Project(
        name STRING,
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
        FROM Method TO Method
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

    Idempotent — safe to call on an existing database. Node/rel tables
    use IF NOT EXISTS; vector indexes are created in a try/except to
    handle the already-exists case gracefully.
    """
    logger.info(f"Running LadybugDB schema migration on: {db_path}")
    db = lb.Database(db_path)
    conn = lb.Connection(db)

    for ddl in _NODE_TABLES:
        table_name = ddl.split("TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        conn.execute(ddl)
        logger.debug(f"  Node table: {table_name}")

    for ddl in _REL_TABLES:
        table_name = ddl.split("TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        conn.execute(ddl)
        logger.debug(f"  Rel table: {table_name}")

    # Load the VECTOR extension (required for CREATE_VECTOR_INDEX / QUERY_VECTOR_INDEX)
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
            if "already exists" in str(e).lower():
                logger.debug(f"  Vector index {idx_name} already exists — skipping")
            else:
                logger.warning(f"  Vector index {idx_name}: {e}")

    logger.info("LadybugDB schema migration complete ✓")
