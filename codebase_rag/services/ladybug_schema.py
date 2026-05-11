"""CI-3: LadybugDB schema migration (DEV-1171).

Declares all node tables and relationship tables that match the
Code-Graph-RAG schema. Must be run once before any ingestion.
Safe to call on an existing DB — every DDL uses ``IF NOT EXISTS`` guards.

Embeddings are stored in per-repo numpy files alongside the DB file
(see ``vector_store.py``), not in LadybugDB. This avoids the chicken-and-egg
problem where opening a DB with a persisted vector index requires the VECTOR
extension pre-loaded, but the extension can only be loaded after the DB is
opened.

Schema layout:
    Node tables
        Project, Package, Folder, File, Module, Class, Function, Method,
        Interface, Enum, ExternalPackage.

    Relationship tables
        CONTAINS_FILE, CONTAINS_FOLDER, CONTAINS_PACKAGE, CONTAINS_MODULE,
        DEFINES, DEFINES_METHOD, CALLS, IMPORTS, INHERITS, IMPLEMENTS,
        OVERRIDES, BELONGS_TO.
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
        is_exported BOOL,
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
    # Interface / Enum mirror Class's shape — C# emits the full property
    # set (decorators, start_line, docstring, is_exported) on these node
    # types.  Without the columns declared, every flush silently drops the
    # node with a "Binder exception: Cannot find property decorators"
    # error and stalls ingestion with log spam.
    """CREATE NODE TABLE IF NOT EXISTS Interface(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Enum(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    # ExternalPackage uses ``name`` as its natural identifier (e.g.
    # "NSwag.AspNetCore") — code-graph-rag's C# parser emits ``name`` as the
    # only key.  Matching PK to the parser's contract here avoids a "Create
    # node expects primary key qualified_name" error on every external-dep
    # reference, which would poison every IMPORTS-edge batch.
    """CREATE NODE TABLE IF NOT EXISTS ExternalPackage(
        name STRING,
        qualified_name STRING,
        PRIMARY KEY (name)
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
    # BUC-1603: CALLS edges carry resolver provenance so downstream consumers
    # (blast-radius, mergeAndRank) can deprioritize low-confidence bindings
    # like the trie fallback. ``resolved_via`` is the resolver-path tag
    # (e.g. "direct_import", "same_module", "trie_fallback") and
    # ``confidence`` is one of {"high", "medium", "low", "unknown"}.
    # Pre-existing rows (rows written before this migration ran) carry the
    # DEFAULT 'unknown' on both columns — see ``_REL_ALTERS`` below.
    """CREATE REL TABLE IF NOT EXISTS CALLS(
        FROM Function TO Function,
        FROM Function TO Method,
        FROM Method TO Function,
        FROM Method TO Method,
        FROM Module TO Function,
        FROM Module TO Method,
        resolved_via STRING DEFAULT 'unknown',
        confidence STRING DEFAULT 'unknown'
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
# Backfill ALTERs for existing databases
# ---------------------------------------------------------------------------
# LadybugDB (Kuzu fork) supports `ALTER REL TABLE <name> ADD <col> <type>`
# but, unlike `CREATE`, it has no `IF NOT EXISTS` guard.  We run these in a
# try/except block and treat "already exists" / "duplicate column" errors as
# success.  Each ALTER below is the migration counterpart for a column that
# was added in-place after the initial CALLS DDL shipped — for fresh DBs the
# columns are already present (declared inline in ``_REL_TABLES``) so the
# ALTER is a no-op; for already-indexed DBs (BUC-1603 deployed onto an
# existing graph) the ALTER backfills the missing columns with DEFAULT
# 'unknown', preserving every CALLS row.
_REL_ALTERS: list[str] = [
    "ALTER TABLE CALLS ADD resolved_via STRING DEFAULT 'unknown'",
    "ALTER TABLE CALLS ADD confidence STRING DEFAULT 'unknown'",
]

# Substrings that indicate "the column you tried to add already exists" —
# LadybugDB's error messages vary across versions, so we match on common
# fragments rather than an exact string.
_ALTER_IDEMPOTENT_SUBSTRINGS: tuple[str, ...] = (
    "already exists",
    "duplicate",
    "already has property",
)


def migrate(db_path: str) -> None:
    """Run schema migration on the given LadybugDB database path.

    Idempotent — safe to call on an existing database. Node/rel tables use
    ``IF NOT EXISTS`` guards.  No VECTOR extension is loaded here; embeddings
    are stored in per-repo numpy files (see ``vector_store.py``).

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

    # BUC-1603: backfill resolved_via + confidence on CALLS for DBs that
    # were created before this migration ran. Each ALTER is independent —
    # a failure on one column should not prevent the next from being tried.
    for alter_ddl in _REL_ALTERS:
        try:
            conn.execute(alter_ddl)
            logger.debug(f"  Applied ALTER: {alter_ddl}")
        except Exception as e:
            err_str = str(e).lower()
            if any(s in err_str for s in _ALTER_IDEMPOTENT_SUBSTRINGS):
                logger.debug(f"  ALTER skipped (column already present): {alter_ddl}")
            else:
                # Hard fail: we can't silently swallow a schema migration
                # error that isn't the idempotent already-exists case.
                logger.error(f"  ALTER failed: {alter_ddl}: {e}")
                raise

    logger.info("LadybugDB schema migration complete ✓")
