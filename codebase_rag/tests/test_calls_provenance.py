"""BUC-1603: CALLS edge resolver-provenance tags.

These tests cover three layers:

1. **CallResolver tagging** — every resolver path returns a ``ResolveResult``
   with the canonical ``resolved_via`` + ``confidence`` tag from the audit
   mapping.  Each of the 6 mapped paths gets a dedicated test; one
   additional test asserts the backward-compat default ("unknown" /
   "unknown") for pre-BUC-1603 rows surfaces through the schema migration.

2. **CallProcessor wiring** — the integration path used by GraphUpdater
   threads the tag through to ``ensure_relationship_batch`` as
   ``properties={"resolved_via": ..., "confidence": ...}``.

3. **Schema migration** — running ``migrate`` against a fresh DB and then
   against the same DB a second time is a no-op for the ALTER step
   (idempotent), and the CALLS table accepts rows with the new properties.

The resolver-unit tests use the same ``MockFunctionRegistry`` pattern as
``test_call_resolver.py``; the schema migration tests skip when the
``real_ladybug`` extension is not installed locally (developer setup).
"""
from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Generator, ItemsView, KeysView
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag import constants as cs
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.parsers.call_resolver import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_UNKNOWN,
    RESOLVED_VIA_DIRECT_IMPORT,
    RESOLVED_VIA_INHERITED,
    RESOLVED_VIA_SAME_MODULE,
    RESOLVED_VIA_SUPER,
    RESOLVED_VIA_TRIE_FALLBACK,
    RESOLVED_VIA_TYPE_INFERRED,
    RESOLVED_VIA_UNKNOWN,
    CallResolver,
    ResolveResult,
)
from codebase_rag.parsers.import_processor import ImportProcessor
from codebase_rag.parsers.type_inference import TypeInferenceEngine
from codebase_rag.types_defs import NodeType, QualifiedName


# ---------------------------------------------------------------------------
# Shared fixtures (mirror test_call_resolver.py's MockFunctionRegistry so
# we don't pull in the full GraphUpdater for unit tests).
# ---------------------------------------------------------------------------


class MockFunctionRegistry:
    def __init__(self) -> None:
        self._data: dict[QualifiedName, NodeType] = {}
        self._suffix_index: dict[str, list[QualifiedName]] = defaultdict(list)

    def __contains__(self, qn: QualifiedName) -> bool:
        return qn in self._data

    def __getitem__(self, qn: QualifiedName) -> NodeType:
        return self._data[qn]

    def __setitem__(self, qn: QualifiedName, func_type: NodeType) -> None:
        self._data[qn] = func_type
        parts = qn.split(cs.SEPARATOR_DOT)
        for i in range(len(parts)):
            suffix = cs.SEPARATOR_DOT.join(parts[i:])
            if qn not in self._suffix_index[suffix]:
                self._suffix_index[suffix].append(qn)

    def get(
        self, qn: QualifiedName, default: NodeType | None = None
    ) -> NodeType | None:
        return self._data.get(qn, default)

    def keys(self) -> KeysView[QualifiedName]:
        return self._data.keys()

    def items(self) -> ItemsView[QualifiedName, NodeType]:
        return self._data.items()

    def find_with_prefix(self, prefix: str) -> list[tuple[QualifiedName, NodeType]]:
        return [(k, v) for k, v in self._data.items() if k.startswith(prefix)]

    def find_ending_with(self, suffix: str) -> list[QualifiedName]:
        return self._suffix_index.get(suffix, [])


@pytest.fixture
def mock_function_registry() -> MockFunctionRegistry:
    return MockFunctionRegistry()


@pytest.fixture
def mock_import_processor(temp_repo: Path) -> ImportProcessor:
    return ImportProcessor(repo_path=temp_repo, project_name="test_project")


@pytest.fixture
def mock_ast_cache() -> MagicMock:
    cache = MagicMock()
    cache.__contains__ = MagicMock(return_value=False)
    cache.__getitem__ = MagicMock(return_value=(None, None))
    cache.__setitem__ = MagicMock()
    return cache


@pytest.fixture
def mock_type_inference(
    mock_import_processor: ImportProcessor,
    mock_function_registry: MockFunctionRegistry,
    mock_ast_cache: MagicMock,
    temp_repo: Path,
) -> TypeInferenceEngine:
    return TypeInferenceEngine(
        import_processor=mock_import_processor,
        function_registry=mock_function_registry,
        repo_path=temp_repo,
        project_name="test_project",
        ast_cache=mock_ast_cache,
        queries={},
        module_qn_to_file_path={},
        class_inheritance={},
        simple_name_lookup=defaultdict(set),
    )


@pytest.fixture
def call_resolver(
    mock_function_registry: MockFunctionRegistry,
    mock_import_processor: ImportProcessor,
    mock_type_inference: TypeInferenceEngine,
) -> CallResolver:
    return CallResolver(
        function_registry=mock_function_registry,
        import_processor=mock_import_processor,
        type_inference=mock_type_inference,
        class_inheritance={},
    )


# ---------------------------------------------------------------------------
# 1. Resolver-level tagging (audit map: 6 paths)
# ---------------------------------------------------------------------------


class TestResolverProvenanceTags:
    """One test per resolver path in the BUC-1603 tag map."""

    def test_should_tag_direct_import_high_when_call_resolves_through_import_map(
        self, call_resolver: CallResolver
    ) -> None:
        # Direct-import path: the call name is in import_map, the imported
        # qn is in the registry, so _try_resolve_direct_import fires first.
        call_resolver.function_registry["external.module.imported_func"] = (
            NodeType.FUNCTION
        )
        call_resolver.import_processor.import_mapping["test_project.caller"] = {
            "imported_func": "external.module.imported_func"
        }

        result = call_resolver.resolve_function_call_with_provenance(
            "imported_func", "test_project.caller"
        )

        assert result is not None
        assert result.callee_qn == "external.module.imported_func"
        assert result.resolved_via == RESOLVED_VIA_DIRECT_IMPORT
        assert result.confidence == CONFIDENCE_HIGH

    def test_should_tag_same_module_high_when_call_lives_alongside_caller(
        self, call_resolver: CallResolver
    ) -> None:
        # Same-module path: no import binding, no method chain — the call
        # name resolves under module_qn directly.
        call_resolver.function_registry["test_project.mod.local_helper"] = (
            NodeType.FUNCTION
        )

        result = call_resolver.resolve_function_call_with_provenance(
            "local_helper", "test_project.mod"
        )

        assert result is not None
        assert result.callee_qn == "test_project.mod.local_helper"
        assert result.resolved_via == RESOLVED_VIA_SAME_MODULE
        assert result.confidence == CONFIDENCE_HIGH

    def test_should_tag_type_inferred_medium_when_local_var_type_drives_binding(
        self, call_resolver: CallResolver
    ) -> None:
        # Type-inferred path: an object's local var type names a known
        # class, the method exists directly on that class.  The qualified
        # call goes through _try_resolve_via_local_type → _try_method_on_class.
        call_resolver.function_registry["test_project.mod.MyClass"] = NodeType.CLASS
        call_resolver.function_registry["test_project.mod.MyClass.do_thing"] = (
            NodeType.METHOD
        )
        # _try_resolve_via_imports requires module_qn in import_mapping —
        # an empty map still triggers the qualified-call path.
        call_resolver.import_processor.import_mapping["test_project.mod"] = {}
        local_var_types = {"obj": "test_project.mod.MyClass"}

        result = call_resolver.resolve_function_call_with_provenance(
            "obj.do_thing", "test_project.mod", local_var_types=local_var_types
        )

        assert result is not None
        assert result.callee_qn == "test_project.mod.MyClass.do_thing"
        assert result.resolved_via == RESOLVED_VIA_TYPE_INFERRED
        assert result.confidence == CONFIDENCE_MEDIUM

    def test_should_tag_inherited_high_when_method_resolves_through_parent_class(
        self, call_resolver: CallResolver
    ) -> None:
        # Inherited path: object's class doesn't have the method directly,
        # but a parent class does.  _resolve_inherited_method walks the
        # class_inheritance map.  Tag must be "inherited" not "type_inferred".
        call_resolver.function_registry["test_project.mod.Parent"] = NodeType.CLASS
        call_resolver.function_registry["test_project.mod.Child"] = NodeType.CLASS
        call_resolver.function_registry["test_project.mod.Parent.shared_method"] = (
            NodeType.METHOD
        )
        # NOTE: Child does NOT define shared_method — inheritance must walk
        # to Parent for the binding to succeed.
        call_resolver.class_inheritance["test_project.mod.Child"] = [
            "test_project.mod.Parent"
        ]
        call_resolver.import_processor.import_mapping["test_project.mod"] = {}
        local_var_types = {"obj": "test_project.mod.Child"}

        result = call_resolver.resolve_function_call_with_provenance(
            "obj.shared_method", "test_project.mod", local_var_types=local_var_types
        )

        assert result is not None
        assert result.callee_qn == "test_project.mod.Parent.shared_method"
        assert result.resolved_via == RESOLVED_VIA_INHERITED
        assert result.confidence == CONFIDENCE_HIGH

    def test_should_tag_super_high_when_call_uses_super_keyword(
        self, call_resolver: CallResolver
    ) -> None:
        # super() path: dispatcher recognizes the super-keyword prefix and
        # routes through _resolve_super_call (which itself uses
        # _resolve_inherited_method internally, but the audit table says
        # the user-facing tag is "super").
        call_resolver.function_registry["test_project.mod.Parent"] = NodeType.CLASS
        call_resolver.function_registry["test_project.mod.Parent.do_thing"] = (
            NodeType.METHOD
        )
        call_resolver.class_inheritance["test_project.mod.Child"] = [
            "test_project.mod.Parent"
        ]

        result = call_resolver.resolve_function_call_with_provenance(
            "super.do_thing",
            "test_project.mod",
            class_context="test_project.mod.Child",
        )

        assert result is not None
        assert result.callee_qn == "test_project.mod.Parent.do_thing"
        assert result.resolved_via == RESOLVED_VIA_SUPER
        assert result.confidence == CONFIDENCE_HIGH

    def test_should_tag_trie_fallback_low_when_no_strict_resolver_path_succeeds(
        self, call_resolver: CallResolver
    ) -> None:
        # Trie fallback: nothing matches direct/same-module — the resolver
        # falls back to the trie index, which picks the closest qn by
        # import distance.  Tag must reflect the fuzzy nature with low
        # confidence so downstream consumers can deprioritize.
        call_resolver.function_registry["unrelated.pkg.lonely_func"] = (
            NodeType.FUNCTION
        )

        result = call_resolver.resolve_function_call_with_provenance(
            "lonely_func", "test_project.caller"
        )

        assert result is not None
        assert result.callee_qn == "unrelated.pkg.lonely_func"
        assert result.resolved_via == RESOLVED_VIA_TRIE_FALLBACK
        assert result.confidence == CONFIDENCE_LOW


# ---------------------------------------------------------------------------
# 2. Backward compatibility — default tags on pre-migration rows
# ---------------------------------------------------------------------------


class TestBackwardCompatDefaults:
    """Rows ingested before BUC-1603 carry the schema-level DEFAULT
    'unknown' on both columns.  Downstream consumers should treat unknown
    as 'don't filter on this row — re-resolve if confidence matters.'"""

    def test_should_expose_unknown_unknown_as_named_constants_for_default_value(
        self,
    ) -> None:
        # Sanity check: the migration uses string literals 'unknown' /
        # 'unknown' in the DDL, and the module-level constants must agree
        # so callers can compare against them without duplicating strings.
        assert RESOLVED_VIA_UNKNOWN == "unknown"
        assert CONFIDENCE_UNKNOWN == "unknown"

    def test_should_construct_resolve_result_for_a_legacy_unknown_row(self) -> None:
        # A "row read back from the DB" scenario: a row created before
        # BUC-1603 surfaces with both columns at 'unknown'.  Verify the
        # ResolveResult NamedTuple wraps it cleanly so consumer code can
        # compare against the named constants.
        legacy_row = ResolveResult(
            callee_type="Function",
            callee_qn="some.pre.existing.qn",
            resolved_via=RESOLVED_VIA_UNKNOWN,
            confidence=CONFIDENCE_UNKNOWN,
        )
        assert legacy_row.resolved_via == "unknown"
        assert legacy_row.confidence == "unknown"


# ---------------------------------------------------------------------------
# 3. CallProcessor wiring — properties reach ensure_relationship_batch
# ---------------------------------------------------------------------------


class TestCallProcessorPropagation:
    """End-to-end at the parser level: a small Python repo flows through
    GraphUpdater + CallProcessor and the resulting CALLS calls to the
    ingestor mock carry ``properties={'resolved_via': ..., 'confidence': ...}``.
    """

    def test_should_pass_resolved_via_and_confidence_to_ensure_relationship_batch(
        self, temp_repo: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "test_module.py"
        test_file.write_text(
            encoding="utf-8",
            data=(
                "def helper():\n"
                "    pass\n"
                "\n"
                "def main():\n"
                "    helper()\n"
            ),
        )

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        calls = [
            c
            for c in mock_ingestor.ensure_relationship_batch.call_args_list
            if c.args[1] == cs.RelationshipType.CALLS
        ]
        assert calls, "Expected at least one CALLS edge from helper() in main()"

        # Every CALLS edge must carry resolved_via + confidence in
        # properties=...; the absence of either is a regression.
        for call_args in calls:
            props = call_args.kwargs.get("properties")
            assert props is not None, (
                f"CALLS edge missing properties kwarg: {call_args}"
            )
            assert "resolved_via" in props, (
                f"CALLS edge missing resolved_via: {props}"
            )
            assert "confidence" in props, f"CALLS edge missing confidence: {props}"
            # main → helper is a same-module call; tag must be same_module/high.
            target_qn = call_args.args[2][2]
            if target_qn.endswith(".helper"):
                assert props["resolved_via"] == RESOLVED_VIA_SAME_MODULE
                assert props["confidence"] == CONFIDENCE_HIGH

    def test_should_produce_stable_tags_when_indexing_the_same_repo_twice(
        self, temp_repo: Path
    ) -> None:
        """Re-indexing must produce identical tags — the resolver is
        deterministic so the same repo input → same provenance output.
        This is the contract downstream consumers rely on when caching
        confidence buckets across reindex cycles."""
        # Use the same _MockIngestor shape as conftest so the call lists
        # are populated the same way GraphUpdater expects.  conftest is
        # loaded into the test package via pytest's plugin discovery
        # rather than a normal Python module, so import from the package
        # path directly.
        from codebase_rag.tests.conftest import (  # type: ignore[attr-defined]
            _MockIngestor,
        )

        parsers, queries = load_parsers()
        if cs.SupportedLanguage.PYTHON not in parsers:
            pytest.skip("Python parser not available")

        test_file = temp_repo / "test_module.py"
        test_file.write_text(
            encoding="utf-8",
            data=(
                "def helper():\n"
                "    pass\n"
                "\n"
                "def main():\n"
                "    helper()\n"
            ),
        )

        def _collect_tags(ingestor: _MockIngestor) -> list[tuple[str, str, str]]:
            tags: list[tuple[str, str, str]] = []
            for c in ingestor.ensure_relationship_batch.call_args_list:
                if c.args[1] != cs.RelationshipType.CALLS:
                    continue
                props = c.kwargs.get("properties")
                if props is None:
                    continue
                tags.append(
                    (c.args[2][2], props["resolved_via"], props["confidence"])
                )
            return sorted(tags)

        # Build two fully independent mock ingestors and run separate
        # GraphUpdater instances against each.
        ingestor_a = _MockIngestor()
        GraphUpdater(
            ingestor=ingestor_a,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        ).run()
        first_pass = _collect_tags(ingestor_a)

        ingestor_b = _MockIngestor()
        # force=True bypasses the on-disk hash cache so the second pass
        # actually re-parses (otherwise GraphUpdater short-circuits on
        # unchanged files and emits zero CALLS edges).
        GraphUpdater(
            ingestor=ingestor_b,
            repo_path=temp_repo,
            parsers=parsers,
            queries=queries,
        ).run(force=True)
        second_pass = _collect_tags(ingestor_b)

        # Don't compare empty lists — the test would silently pass if
        # nothing was indexed.
        assert first_pass, "Expected at least one CALLS edge"
        assert first_pass == second_pass, (
            "Re-indexing produced different provenance tags — resolver is non-deterministic"
        )


# ---------------------------------------------------------------------------
# 4. Schema migration — applies-to-existing-DB scenario
# ---------------------------------------------------------------------------

try:
    import real_ladybug as lb  # type: ignore[import-untyped]  # noqa: F401

    _HAS_LADYBUG = True
except ImportError:
    _HAS_LADYBUG = False


@pytest.fixture()
def fresh_ladybug_db(tmp_path: Path) -> Generator[str, None, None]:
    """Create a fresh LadybugDB, run the schema migration once, yield path."""
    db_path = str(tmp_path / "test_calls_provenance.db")
    from codebase_rag.services.ladybug_schema import migrate

    migrate(db_path)
    yield db_path
    shutil.rmtree(db_path, ignore_errors=True)


@pytest.mark.skipif(
    not _HAS_LADYBUG, reason="real_ladybug not installed — run `uv sync` first"
)
class TestSchemaMigration:
    """The migration must:

    1. Create the CALLS rel table with resolved_via + confidence columns
       (declared inline in _REL_TABLES).
    2. Be safely re-runnable on the same DB — the ALTER fallback path
       must treat 'column already exists' as a no-op.
    3. Add resolved_via + confidence to a DB whose CALLS table predates
       BUC-1603 — this is the production deployment scenario.
    4. Preserve existing CALLS rows when those columns are added.
    """

    def test_should_create_calls_table_with_new_columns_on_fresh_db(
        self, fresh_ladybug_db: str
    ) -> None:
        db = lb.Database(fresh_ladybug_db)
        conn = lb.Connection(db)
        # Query the CALLS rel table for its property schema.  Kuzu / Ladybug
        # surfaces table metadata via the system catalog: we just attempt
        # a write that references both columns and rely on the engine
        # accepting it as proof the columns exist.
        # Need source + target nodes first.
        conn.execute(
            "CREATE (:Function {qualified_name: 'a.fn', name: 'fn', "
            "decorators: [], start_line: 1, end_line: 2, docstring: '', "
            "source_code: '', is_exported: false})"
        )
        conn.execute(
            "CREATE (:Function {qualified_name: 'b.fn', name: 'fn', "
            "decorators: [], start_line: 1, end_line: 2, docstring: '', "
            "source_code: '', is_exported: false})"
        )
        conn.execute(
            "MATCH (a:Function {qualified_name: 'a.fn'}), "
            "(b:Function {qualified_name: 'b.fn'}) "
            "CREATE (a)-[:CALLS {resolved_via: 'same_module', "
            "confidence: 'high'}]->(b)"
        )
        # Read it back and verify the columns were persisted.
        result = conn.execute(
            "MATCH (a:Function)-[r:CALLS]->(b:Function) "
            "RETURN r.resolved_via AS rv, r.confidence AS conf"
        )
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        assert rows, "Expected one CALLS row"
        # Each row is a list of column values in the same order as RETURN.
        assert rows[0][0] == "same_module"
        assert rows[0][1] == "high"

    def test_should_be_idempotent_when_migration_runs_twice(
        self, fresh_ladybug_db: str
    ) -> None:
        # First migration ran in the fixture; running it again should not
        # raise — the ALTER step matches on 'already exists' substrings and
        # skips silently.
        from codebase_rag.services.ladybug_schema import migrate

        migrate(fresh_ladybug_db)  # Must not raise.

    def test_should_preserve_existing_rows_when_alter_runs_on_populated_db(
        self, tmp_path: Path
    ) -> None:
        """The production scenario: deploying BUC-1603 to a DB that
        already has CALLS rows from a previous indexing run.  After the
        migration applies, every pre-existing row must still be there
        (and show 'unknown' for the new columns via DEFAULT)."""
        from codebase_rag.services.ladybug_schema import (
            _NODE_TABLES,
            _REL_ALTERS,
            migrate,
        )

        db_path = str(tmp_path / "pre_buc_1603.db")

        # Phase 1: simulate a DB created BEFORE BUC-1603 — same node DDL,
        # but CALLS without the new columns.
        db = lb.Database(db_path)
        conn = lb.Connection(db)
        for node_ddl in _NODE_TABLES:
            conn.execute(node_ddl)
        conn.execute(
            "CREATE REL TABLE IF NOT EXISTS CALLS("
            "FROM Function TO Function, "
            "FROM Function TO Method, "
            "FROM Method TO Function, "
            "FROM Method TO Method, "
            "FROM Module TO Function, "
            "FROM Module TO Method)"
        )
        conn.execute(
            "CREATE (:Function {qualified_name: 'legacy.caller', name: 'caller', "
            "decorators: [], start_line: 1, end_line: 2, docstring: '', "
            "source_code: '', is_exported: false})"
        )
        conn.execute(
            "CREATE (:Function {qualified_name: 'legacy.callee', name: 'callee', "
            "decorators: [], start_line: 1, end_line: 2, docstring: '', "
            "source_code: '', is_exported: false})"
        )
        conn.execute(
            "MATCH (a:Function {qualified_name: 'legacy.caller'}), "
            "(b:Function {qualified_name: 'legacy.callee'}) "
            "CREATE (a)-[:CALLS]->(b)"
        )
        # Close handles so the next migration can open the DB freshly.
        if hasattr(conn, "close"):
            conn.close()
        if hasattr(db, "close"):
            db.close()

        # Phase 2: apply BUC-1603 migration via the ALTER path.
        # We call migrate() which runs the full DDL (idempotent) and the
        # ALTERs.  The legacy CALLS row must survive.
        migrate(db_path)

        # Phase 3: verify the legacy row is still there and the new
        # columns surface with DEFAULT 'unknown'.
        db2 = lb.Database(db_path)
        conn2 = lb.Connection(db2)
        result = conn2.execute(
            "MATCH (a:Function {qualified_name: 'legacy.caller'})"
            "-[r:CALLS]->(b:Function {qualified_name: 'legacy.callee'}) "
            "RETURN r.resolved_via AS rv, r.confidence AS conf"
        )
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        # We don't want to lose the row.
        assert rows, (
            "Legacy CALLS row was lost during BUC-1603 migration — data loss!"
        )
        # Per the DEFAULT clause in _REL_ALTERS, both columns should be
        # 'unknown' for rows written before the migration.
        assert rows[0][0] == "unknown"
        assert rows[0][1] == "unknown"
        # Ensure the ALTERs were defined in the first place — without
        # them, the test above would still pass when CALLS was already
        # created with the columns.  Belt-and-braces guard.
        assert any("resolved_via" in a for a in _REL_ALTERS)
        assert any("confidence" in a for a in _REL_ALTERS)
