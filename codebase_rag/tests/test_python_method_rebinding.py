"""BUC-1611: module-level method rebinding (Python monkey-patching).

These tests exercise the full pipeline end-to-end through GraphUpdater:
  * A REBINDS edge is emitted with the correct (Module, Method, new_target)
    triple when a module's top-level scope does ``Klass.method = other``.
  * The CALLS edge from a caller of ``Klass().method()`` is rerouted to
    the replacement function and tagged ``resolved_via='rebound'``.
  * When the same attribute is rebound twice, the *latest* (last source
    line / last-registered) wins.
  * Rebinding to a function that isn't in the graph (external library,
    lambda, literal) falls back gracefully: no REBINDS edge, no CALLS
    rerouting, no crash.
  * Instance-level attribute assignment (``widget.foo = bar``) is NOT
    a rebinding — verified by a negative-control test that ensures the
    class-level resolution is untouched.

Test harness uses the same ``temp_repo`` + ``mock_ingestor`` fixtures
as the rest of the Python-pipeline tests so behaviour is verified at
the same boundary every consumer sees (the recorded mock call list).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag import constants as cs
from codebase_rag.tests.conftest import get_relationships, run_updater


def _rebinds(mock_ingestor: MagicMock) -> list:
    return get_relationships(mock_ingestor, cs.RelationshipType.REBINDS)


def _calls(mock_ingestor: MagicMock) -> list:
    return get_relationships(mock_ingestor, cs.RelationshipType.CALLS)


def _props(call) -> dict:
    return call.kwargs.get("properties") or {}


# ---------------------------------------------------------------------------
# Helpers — build a tiny multi-file Python project rooted at ``base``.
# ---------------------------------------------------------------------------


def _write(base: Path, rel: str, src: str) -> None:
    """Write a Python source file under ``base`` and make any parent dirs."""
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src, encoding="utf-8")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestSingleRebinding:
    """``Widget.render = custom_render`` swaps the callee everywhere."""

    @pytest.fixture
    def proj(self, temp_repo: Path) -> Path:
        # Project layout:
        #   pkg/__init__.py
        #   pkg/original.py    — defines class Widget with .render
        #   pkg/elsewhere.py   — defines plain function custom_render
        #   pkg/patcher.py     — does the rebinding at module scope
        #   pkg/consumer.py    — calls Widget().render(); should be rerouted
        root = temp_repo / "proj"
        root.mkdir()
        _write(root, "__init__.py", "")
        _write(
            root,
            "original.py",
            (
                "class Widget:\n"
                "    def render(self):\n"
                "        return 'default'\n"
            ),
        )
        _write(
            root,
            "elsewhere.py",
            (
                "def custom_render(self):\n"
                "    return 'custom'\n"
            ),
        )
        _write(
            root,
            "patcher.py",
            (
                "from proj.original import Widget\n"
                "from proj.elsewhere import custom_render\n"
                "\n"
                "Widget.render = custom_render\n"
            ),
        )
        _write(
            root,
            "consumer.py",
            (
                "from proj.original import Widget\n"
                "\n"
                "def use_widget():\n"
                "    w = Widget()\n"
                "    return w.render()\n"
            ),
        )
        return temp_repo

    def test_should_emit_rebinds_edge_when_module_assigns_class_attribute(
        self, proj: Path, mock_ingestor: MagicMock
    ) -> None:
        run_updater(proj, mock_ingestor)

        rebinds = _rebinds(mock_ingestor)
        assert len(rebinds) == 1, (
            f"Expected exactly 1 REBINDS edge for Widget.render = custom_render, "
            f"got {len(rebinds)}: {[r.args for r in rebinds]}"
        )

        edge = rebinds[0]
        from_tuple, _rel, to_tuple = edge.args[0], edge.args[1], edge.args[2]
        # From: the rebinding module (patcher.py).
        assert from_tuple[0] == cs.NodeLabel.MODULE
        assert from_tuple[2].endswith("patcher")
        # To: the *original* Widget.render method qname.
        assert to_tuple[0] == cs.NodeLabel.METHOD
        assert to_tuple[2].endswith("original.Widget.render")
        # Property carries the replacement target.
        props = _props(edge)
        assert props["new_target"].endswith("elsewhere.custom_render")
        assert props["file_path"].endswith("patcher.py")
        assert props["line_start"] > 0

    def test_should_reroute_caller_calls_edge_when_widget_render_is_rebound(
        self, proj: Path, mock_ingestor: MagicMock
    ) -> None:
        run_updater(proj, mock_ingestor)

        # Find the CALLS edge from consumer.use_widget — there should be
        # at least one whose callee qname is now the replacement target
        # (NOT the original Widget.render).
        widget_calls = [
            c
            for c in _calls(mock_ingestor)
            if c.args[0][2].endswith("consumer.use_widget")
        ]
        # use_widget contains: Widget() and w.render() — at least one of
        # the latter should be tagged as rebound.  We don't care about
        # the Widget() instantiation call here.
        render_calls = [c for c in widget_calls if "render" in c.args[2][2]]
        assert render_calls, (
            f"Expected at least one CALLS edge from use_widget to a "
            f"'render' callee, got {[c.args[2] for c in widget_calls]}"
        )

        # The renamed callee should point at custom_render, not the
        # original Widget.render.
        rerouted = [
            c
            for c in render_calls
            if c.args[2][2].endswith("custom_render")
        ]
        assert rerouted, (
            f"Expected the render call to be rerouted to custom_render; "
            f"got callees: {[c.args[2][2] for c in render_calls]}"
        )
        assert _props(rerouted[0]).get("resolved_via") == "rebound", (
            f"CALLS edge for a rebound resolution must carry "
            f"resolved_via='rebound'; got props={_props(rerouted[0])}"
        )


class TestMultipleRebindings:
    """When the same attribute is rebound twice, latest wins."""

    @pytest.fixture
    def proj(self, temp_repo: Path) -> Path:
        root = temp_repo / "proj"
        root.mkdir()
        _write(root, "__init__.py", "")
        _write(
            root,
            "original.py",
            (
                "class Widget:\n"
                "    def render(self):\n"
                "        return 'default'\n"
            ),
        )
        _write(
            root,
            "elsewhere.py",
            (
                "def first(self):\n"
                "    return 'first'\n"
                "\n"
                "def second(self):\n"
                "    return 'second'\n"
            ),
        )
        # Both rebindings live in the same module, so "latest wins" is
        # the LAST assignment in source order — ``second`` must win.
        _write(
            root,
            "patcher.py",
            (
                "from proj.original import Widget\n"
                "from proj.elsewhere import first, second\n"
                "\n"
                "Widget.render = first\n"
                "Widget.render = second\n"
            ),
        )
        _write(
            root,
            "consumer.py",
            (
                "from proj.original import Widget\n"
                "\n"
                "def use_widget():\n"
                "    w = Widget()\n"
                "    return w.render()\n"
            ),
        )
        return temp_repo

    def test_should_pick_latest_rebinding_when_attribute_is_reassigned(
        self, proj: Path, mock_ingestor: MagicMock
    ) -> None:
        run_updater(proj, mock_ingestor)

        # Both rebindings are recorded as REBINDS edges (audit trail),
        # but the CALLS rerouting picks the latest.
        rebinds = _rebinds(mock_ingestor)
        targets = sorted(_props(r).get("new_target") for r in rebinds)
        assert any(t and t.endswith("first") for t in targets)
        assert any(t and t.endswith("second") for t in targets)

        # Caller's CALLS edge for the render() call site is rerouted —
        # so we look for an edge from use_widget tagged with
        # ``resolved_via='rebound'`` and verify its callee is ``second``
        # (the LATER assignment in source order — line 5 vs line 4).
        rerouted = [
            c
            for c in _calls(mock_ingestor)
            if c.args[0][2].endswith("consumer.use_widget")
            and _props(c).get("resolved_via") == "rebound"
        ]
        assert rerouted, (
            "Expected at least one CALLS edge from use_widget tagged "
            "resolved_via='rebound' after Widget.render was monkey-patched twice"
        )
        # And every rerouted callee must be ``second`` (not ``first``) —
        # latest wins.
        for c in rerouted:
            assert c.args[2][2].endswith("elsewhere.second"), (
                f"Latest rebinding (``second``) should win; got "
                f"callee {c.args[2][2]}"
            )


class TestExternalRhsFallsBackGracefully:
    """Rebinding to a function not in the graph must not crash or fabricate edges."""

    @pytest.fixture
    def proj(self, temp_repo: Path) -> Path:
        root = temp_repo / "proj"
        root.mkdir()
        _write(root, "__init__.py", "")
        _write(
            root,
            "original.py",
            (
                "class Widget:\n"
                "    def render(self):\n"
                "        return 'default'\n"
            ),
        )
        # RHS is a lambda — not a function in the registry.
        _write(
            root,
            "patcher.py",
            (
                "from proj.original import Widget\n"
                "\n"
                "Widget.render = lambda self: 'lambda'\n"
            ),
        )
        _write(
            root,
            "consumer.py",
            (
                "from proj.original import Widget\n"
                "\n"
                "def use_widget():\n"
                "    w = Widget()\n"
                "    return w.render()\n"
            ),
        )
        return temp_repo

    def test_should_drop_rebind_when_rhs_is_not_in_graph(
        self, proj: Path, mock_ingestor: MagicMock
    ) -> None:
        # Indexing must complete without raising.
        run_updater(proj, mock_ingestor)

        # No REBINDS edge is emitted for a lambda RHS (we can't point
        # CALLS at something the graph doesn't contain).
        assert _rebinds(mock_ingestor) == [], (
            f"Lambda RHS should NOT produce a REBINDS edge; got "
            f"{[r.args for r in _rebinds(mock_ingestor)]}"
        )

        # The consumer's render() call still resolves the *original*
        # way — i.e. it's NOT tagged as rebound.
        render_calls = [
            c
            for c in _calls(mock_ingestor)
            if c.args[0][2].endswith("consumer.use_widget")
            and "render" in c.args[2][2]
        ]
        for c in render_calls:
            assert _props(c).get("resolved_via") != "rebound", (
                f"With no in-graph RHS, the CALLS edge must NOT be "
                f"tagged as rebound; got props={_props(c)}"
            )


class TestNoFalsePositiveOnInstanceAttributeAssignment:
    """``widget.render = ...`` (lowercase instance) is NOT a class rebind."""

    @pytest.fixture
    def proj(self, temp_repo: Path) -> Path:
        root = temp_repo / "proj"
        root.mkdir()
        _write(root, "__init__.py", "")
        _write(
            root,
            "original.py",
            (
                "class Widget:\n"
                "    def render(self):\n"
                "        return 'default'\n"
            ),
        )
        _write(
            root,
            "elsewhere.py",
            (
                "def custom_render(self):\n"
                "    return 'custom'\n"
            ),
        )
        # ``widget`` is an INSTANCE — lowercase identifier that doesn't
        # resolve to a Class qname in the registry.  The processor must
        # reject this as a class rebinding.
        _write(
            root,
            "patcher.py",
            (
                "from proj.original import Widget\n"
                "from proj.elsewhere import custom_render\n"
                "\n"
                "widget = Widget()\n"
                "widget.render = custom_render\n"
            ),
        )
        _write(
            root,
            "consumer.py",
            (
                "from proj.original import Widget\n"
                "\n"
                "def use_widget():\n"
                "    w = Widget()\n"
                "    return w.render()\n"
            ),
        )
        return temp_repo

    def test_should_not_emit_rebinds_for_instance_attribute_assignment(
        self, proj: Path, mock_ingestor: MagicMock
    ) -> None:
        run_updater(proj, mock_ingestor)

        # Critical: zero REBINDS edges from the patcher module, because
        # ``widget.render = ...`` is an instance attribute, not a class
        # rebinding.
        rebinds = _rebinds(mock_ingestor)
        assert rebinds == [], (
            f"Instance attribute assignment must NOT be recorded as a "
            f"class rebinding; got {[r.args for r in rebinds]}"
        )

        # And the consumer's render() call must point at the ORIGINAL
        # Widget.render (no rebound tag).
        render_calls = [
            c
            for c in _calls(mock_ingestor)
            if c.args[0][2].endswith("consumer.use_widget")
            and "render" in c.args[2][2]
        ]
        for c in render_calls:
            assert _props(c).get("resolved_via") != "rebound", (
                f"Instance-level assignment must not affect class-level "
                f"call resolution; got props={_props(c)}"
            )


class TestRebindRegistryUnit:
    """Direct unit tests on the in-memory ``RebindRegistry``."""

    def test_should_return_latest_when_multiple_rebinds_registered(self) -> None:
        from codebase_rag.parsers.rebind_processor import Rebind, RebindRegistry
        from codebase_rag.types_defs import NodeType

        reg = RebindRegistry()
        first = Rebind(
            class_qn="proj.original.Widget",
            attribute="render",
            new_target_qn="proj.elsewhere.first",
            new_target_type=NodeType.FUNCTION,
            module_qn="proj.patcher",
            file_path="proj/patcher.py",
            line_start=4,
        )
        second = Rebind(
            class_qn="proj.original.Widget",
            attribute="render",
            new_target_qn="proj.elsewhere.second",
            new_target_type=NodeType.FUNCTION,
            module_qn="proj.patcher",
            file_path="proj/patcher.py",
            line_start=5,
        )
        reg.add(first)
        reg.add(second)

        latest = reg.latest_for("proj.original.Widget.render")
        assert latest is not None
        assert latest.new_target_qn == "proj.elsewhere.second"

    def test_should_return_none_when_no_rebind_registered(self) -> None:
        from codebase_rag.parsers.rebind_processor import RebindRegistry

        reg = RebindRegistry()
        assert reg.latest_for("proj.original.Widget.render") is None

    def test_should_be_idempotent_on_identical_registration(self) -> None:
        from codebase_rag.parsers.rebind_processor import Rebind, RebindRegistry
        from codebase_rag.types_defs import NodeType

        reg = RebindRegistry()
        rebind = Rebind(
            class_qn="proj.original.Widget",
            attribute="render",
            new_target_qn="proj.elsewhere.custom_render",
            new_target_type=NodeType.FUNCTION,
            module_qn="proj.patcher",
            file_path="proj/patcher.py",
            line_start=4,
        )
        reg.add(rebind)
        reg.add(rebind)  # exact duplicate — should coalesce
        assert len(reg) == 1
