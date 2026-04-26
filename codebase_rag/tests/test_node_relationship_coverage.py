"""Test that every NodeLabel and RelationshipType is reachable through the
LadybugIngestor flush path.

Ported from the legacy MemgraphIngestor coverage test (M11 docs sync). The
v5.3 fork uses LadybugDB as the embedded graph store, so the API differs:

    * Constructor: ``LadybugIngestor(db_path=..., batch_size=..., use_merge=...)``
    * Per-node writes go through ``self._execute_query(query, params)`` rather
      than a ``cursor.execute(query)`` round-trip on a pymgclient cursor.
    * Relationships are flushed in grouped UNWIND batches via
      ``self._execute_batch``.
    * Constraints are declared in the schema DDL via ``PRIMARY KEY``, so
      ``ensure_constraints()`` is a no-op runtime hook (preserved for
      interface parity with the old MemgraphIngestor consumers).

The constants under test (``NodeLabel``, ``RelationshipType``,
``_NODE_LABEL_UNIQUE_KEYS``, ``NODE_UNIQUE_CONSTRAINTS``, ``UniqueKeyType``,
``NodeType``) are storage-engine independent — they describe the graph
*shape* and are imported from ``codebase_rag.constants``.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from codebase_rag.constants import (
    _NODE_LABEL_UNIQUE_KEYS,
    KEY_NAME,
    KEY_PATH,
    KEY_QUALIFIED_NAME,
    NODE_UNIQUE_CONSTRAINTS,
    NodeLabel,
    RelationshipType,
    UniqueKeyType,
)
from codebase_rag.services.ladybug_ingestor import LadybugIngestor
from codebase_rag.types_defs import NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ingestor(tmp_path) -> LadybugIngestor:
    """Build an ingestor without running the real LadybugDB migration.

    The flush-coverage tests below patch ``_execute_query`` so the real
    connection is never touched. Setting ``conn`` to a sentinel object is
    enough to satisfy the connection-guard inside the public flush methods.
    """
    ingestor = LadybugIngestor(
        db_path=str(tmp_path / "coverage.ladybug.db"),
        batch_size=10,
        use_merge=True,
    )
    # ``_execute_query`` raises when ``self.conn`` is None; patch path bypasses
    # the call but the guard still runs in some helpers, so we hand it a
    # sentinel rather than a real connection.
    ingestor.conn = object()  # type: ignore[assignment]
    return ingestor


# ---------------------------------------------------------------------------
# Constants integrity (storage-engine independent)
# ---------------------------------------------------------------------------


class TestNodeLabelCoverage:
    def test_all_node_labels_have_unique_key_mapping(self) -> None:
        missing = set(NodeLabel) - set(_NODE_LABEL_UNIQUE_KEYS.keys())

        assert not missing, (
            f"NodeLabel(s) missing from _NODE_LABEL_UNIQUE_KEYS: {missing}. "
            "Every NodeLabel MUST have a unique key defined."
        )

    def test_all_node_labels_in_constraints(self) -> None:
        missing = {label.value for label in NodeLabel} - set(
            NODE_UNIQUE_CONSTRAINTS.keys()
        )

        assert not missing, (
            f"NodeLabel value(s) missing from NODE_UNIQUE_CONSTRAINTS: {missing}. "
            "This would cause nodes to be silently dropped during flush."
        )

    def test_all_node_types_in_constraints(self) -> None:
        missing = {node_type.value for node_type in NodeType} - set(
            NODE_UNIQUE_CONSTRAINTS.keys()
        )

        assert not missing, (
            f"NodeType value(s) missing from NODE_UNIQUE_CONSTRAINTS: {missing}. "
            "This would cause nodes to be silently dropped during flush."
        )

    def test_node_unique_constraints_derived_from_single_source(self) -> None:
        expected = {
            label.value: key.value for label, key in _NODE_LABEL_UNIQUE_KEYS.items()
        }

        assert NODE_UNIQUE_CONSTRAINTS == expected, (
            "NODE_UNIQUE_CONSTRAINTS must be derived from _NODE_LABEL_UNIQUE_KEYS. "
            "Do not maintain NODE_UNIQUE_CONSTRAINTS manually."
        )

    def test_unique_key_types_are_valid(self) -> None:
        valid_keys = set(UniqueKeyType)

        for label, key in _NODE_LABEL_UNIQUE_KEYS.items():
            assert key in valid_keys, (
                f"Invalid unique key type {key} for {label}. "
                f"Must be one of {valid_keys}."
            )


class TestNodeLabelConstraintConsistency:
    @pytest.mark.parametrize("label", list(NodeLabel))
    def test_each_node_label_has_constraint(self, label: NodeLabel) -> None:
        assert label.value in NODE_UNIQUE_CONSTRAINTS, (
            f"NodeLabel.{label.name} ({label.value}) missing from NODE_UNIQUE_CONSTRAINTS. "
            "This would cause nodes of this type to be silently dropped."
        )

    @pytest.mark.parametrize("node_type", list(NodeType))
    def test_each_node_type_has_constraint(self, node_type: NodeType) -> None:
        assert node_type.value in NODE_UNIQUE_CONSTRAINTS, (
            f"NodeType.{node_type.name} ({node_type.value}) missing from NODE_UNIQUE_CONSTRAINTS. "
            "This would cause nodes of this type to be silently dropped."
        )


# ---------------------------------------------------------------------------
# Flush coverage — every NodeLabel reachable through the public path
# ---------------------------------------------------------------------------


class TestFlushNodesForAllNodeLabels:
    @pytest.mark.parametrize("label", list(NodeLabel))
    def test_each_node_label_can_be_flushed(self, label: NodeLabel, tmp_path) -> None:
        ingestor = _make_ingestor(tmp_path)
        unique_key = NODE_UNIQUE_CONSTRAINTS[label.value]
        node_props = {unique_key: f"test_{label.value}_id", KEY_NAME: "test"}

        with patch.object(LadybugIngestor, "_execute_query") as mock_exec:
            ingestor.node_buffer.append((label.value, node_props))
            ingestor.flush_nodes()

        # One node → exactly one Cypher write through the LadybugDB executor.
        mock_exec.assert_called_once()
        assert ingestor.node_buffer == []

    @pytest.mark.parametrize("node_type", list(NodeType))
    def test_each_node_type_can_be_flushed(self, node_type: NodeType, tmp_path) -> None:
        ingestor = _make_ingestor(tmp_path)
        unique_key = NODE_UNIQUE_CONSTRAINTS[node_type.value]
        node_props = {unique_key: f"test_{node_type.value}_id", KEY_NAME: "test"}

        with patch.object(LadybugIngestor, "_execute_query") as mock_exec:
            ingestor.node_buffer.append((node_type.value, node_props))
            ingestor.flush_nodes()

        mock_exec.assert_called_once()
        assert ingestor.node_buffer == []


class TestFlushRelationshipsForAllTypes:
    @pytest.mark.parametrize("rel_type", list(RelationshipType))
    def test_each_relationship_type_can_be_flushed(
        self, rel_type: RelationshipType, tmp_path
    ) -> None:
        ingestor = _make_ingestor(tmp_path)

        # LadybugIngestor groups relationships by (pattern, prop-shape) and
        # flushes them as a single UNWIND batch through ``_execute_batch``.
        # We patch that boundary to assert exactly one batched write fires.
        with patch.object(LadybugIngestor, "_execute_batch") as mock_batch:
            ingestor.ensure_relationship_batch(
                (NodeLabel.MODULE.value, KEY_QUALIFIED_NAME, "module.test"),
                rel_type.value,
                (NodeLabel.FUNCTION.value, KEY_QUALIFIED_NAME, "module.test.func"),
            )
            ingestor.flush_relationships()

        mock_batch.assert_called_once()
        assert ingestor._rel_count == 0


# ---------------------------------------------------------------------------
# Property-name correctness
# ---------------------------------------------------------------------------


class TestUniqueKeyPropertyNames:
    def test_name_unique_key_uses_correct_property(self) -> None:
        for label in NodeLabel:
            key = _NODE_LABEL_UNIQUE_KEYS[label]
            if key == UniqueKeyType.NAME:
                assert NODE_UNIQUE_CONSTRAINTS[label.value] == KEY_NAME

    def test_path_unique_key_uses_correct_property(self) -> None:
        for label in NodeLabel:
            key = _NODE_LABEL_UNIQUE_KEYS[label]
            if key == UniqueKeyType.PATH:
                assert NODE_UNIQUE_CONSTRAINTS[label.value] == KEY_PATH

    def test_qualified_name_unique_key_uses_correct_property(self) -> None:
        for label in NodeLabel:
            key = _NODE_LABEL_UNIQUE_KEYS[label]
            if key == UniqueKeyType.QUALIFIED_NAME:
                assert NODE_UNIQUE_CONSTRAINTS[label.value] == KEY_QUALIFIED_NAME


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestNodeLabelEnumCompleteness:
    def test_node_label_count_matches_constraints_count(self) -> None:
        assert len(NodeLabel) == len(NODE_UNIQUE_CONSTRAINTS), (
            f"NodeLabel has {len(NodeLabel)} values but "
            f"NODE_UNIQUE_CONSTRAINTS has {len(NODE_UNIQUE_CONSTRAINTS)} entries. "
            "These must match."
        )

    def test_node_type_is_subset_of_node_label(self) -> None:
        node_type_values = {t.value for t in NodeType}
        node_label_values = {label.value for label in NodeLabel}

        extra_in_node_type = node_type_values - node_label_values

        assert not extra_in_node_type, (
            f"NodeType has values not in NodeLabel: {extra_in_node_type}. "
            "NodeType must be a subset of NodeLabel."
        )


class TestRelationshipTypeCompleteness:
    def test_relationship_types_are_uppercase(self) -> None:
        for rel_type in RelationshipType:
            assert rel_type.value == rel_type.value.upper(), (
                f"RelationshipType.{rel_type.name} has value '{rel_type.value}' "
                "which is not uppercase. Relationship types must be uppercase."
            )

    def test_relationship_type_values_match_names(self) -> None:
        for rel_type in RelationshipType:
            assert rel_type.name == rel_type.value, (
                f"RelationshipType.{rel_type.name} has mismatched value '{rel_type.value}'. "
                "Name and value should match for relationship types."
            )


# ---------------------------------------------------------------------------
# Defensive flush: missing unique key must skip rather than crash
# ---------------------------------------------------------------------------


class TestNodeBufferFlushWithMissingKey:
    @pytest.mark.parametrize("label", list(NodeLabel))
    def test_node_without_unique_key_is_skipped_not_crashed(
        self, label: NodeLabel, tmp_path
    ) -> None:
        ingestor = _make_ingestor(tmp_path)
        node_props = {KEY_NAME: "test_without_unique_key"}

        with patch.object(LadybugIngestor, "_execute_query") as mock_exec:
            ingestor.node_buffer.append((label.value, node_props))
            ingestor.flush_nodes()

        # Buffer must be drained even when every row in it was skipped.
        assert ingestor.node_buffer == []
        # If KEY_NAME happens to be the unique key for this label, the row
        # is valid; otherwise it gets logged + skipped without an execute.
        unique_key = NODE_UNIQUE_CONSTRAINTS[label.value]
        if unique_key != KEY_NAME:
            mock_exec.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_constraints — LadybugDB declares constraints in the schema DDL,
# so the runtime hook is intentionally a no-op (preserved for interface
# parity with consumers that still call it).
# ---------------------------------------------------------------------------


class TestEnsureConstraintsIsNoOp:
    def test_ensure_constraints_does_not_execute_runtime_queries(self, tmp_path) -> None:
        ingestor = _make_ingestor(tmp_path)

        with patch.object(LadybugIngestor, "_execute_query") as mock_exec:
            ingestor.ensure_constraints()

        mock_exec.assert_not_called()


# ---------------------------------------------------------------------------
# Import-time validation of the constants module
# ---------------------------------------------------------------------------


class TestImportTimeValidation:
    def test_import_time_validation_catches_missing_keys(self) -> None:
        code = """
from enum import StrEnum

class UniqueKeyType(StrEnum):
    NAME = "name"
    QUALIFIED_NAME = "qualified_name"

class NodeLabel(StrEnum):
    PROJECT = "Project"
    NEW_MISSING_LABEL = "NewMissingLabel"

_NODE_LABEL_UNIQUE_KEYS = {
    NodeLabel.PROJECT: UniqueKeyType.NAME,
}

_missing_keys = set(NodeLabel) - set(_NODE_LABEL_UNIQUE_KEYS.keys())
if _missing_keys:
    raise RuntimeError(
        f"NodeLabel(s) missing from _NODE_LABEL_UNIQUE_KEYS: {_missing_keys}"
    )
"""
        with pytest.raises(RuntimeError, match="missing from _NODE_LABEL_UNIQUE_KEYS"):
            exec(code)
