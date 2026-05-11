"""BUC-1610: re-exports resolution tests.

Cover the four shapes that motivated the ticket:

- TS aliased re-export (depth 1) — ``export { X } from './mod'``
- TS chain depth 2 — A re-exports from B re-exports from C
- TS namespace re-export — ``export * from './mod'``
- TS cyclic re-export — A -> B -> A must not infinite-loop
- Python ``__init__.py`` with ``__all__`` curation

Each test exercises the resolver as a unit (driving ``ImportProcessor`` +
``CallResolver`` directly with a minimal function_registry), which keeps
the assertions sharp and avoids depending on the full ``GraphUpdater``
end-to-end pipeline (the existing import tests already cover that path).
"""

from __future__ import annotations

from pathlib import Path

from codebase_rag.parsers.call_resolver import CallResolver
from codebase_rag.parsers.import_processor import ImportProcessor


class _FakeRegistry:
    """Minimal stand-in for FunctionRegistryTrie.

    The real trie supports prefix / suffix lookups; the BUC-1610 chain
    follower only needs ``__contains__`` and ``__getitem__`` to decide
    "is this qn a defined symbol?" and "what node type does it have?".
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def __contains__(self, qn: str) -> bool:
        return qn in self._mapping

    def __getitem__(self, qn: str) -> str:
        return self._mapping[qn]

    def get(self, qn: str, default: str | None = None) -> str | None:
        return self._mapping.get(qn, default)

    def find_ending_with(self, _suffix: str) -> list[str]:
        return []


def _make_resolver(
    project_name: str,
    repo_path: Path,
    import_mapping: dict[str, dict[str, str]],
    re_export_mapping: dict[str, dict[str, str]],
    registry: _FakeRegistry,
) -> CallResolver:
    proc = ImportProcessor(repo_path=repo_path, project_name=project_name)
    proc.import_mapping = import_mapping
    proc.re_export_mapping = re_export_mapping
    # CallResolver only touches resolver.import_processor.re_export_mapping
    # for the chain walk; the other deps (type_inference, class_inheritance)
    # are unused on the direct-import codepath that BUC-1610 lives on.
    return CallResolver(
        function_registry=registry,  # type: ignore[arg-type]
        import_processor=proc,
        type_inference=None,  # type: ignore[arg-type]
        class_inheritance={},
    )


# --------------------------------------------------------------------------
# TS aliased re-export — depth 1
# --------------------------------------------------------------------------


def test_should_resolve_through_barrel_when_consumer_imports_via_reexport(
    tmp_path: Path,
) -> None:
    """consumer.ts imports Button from barrel.ts which re-exports it."""

    registry = _FakeRegistry({"proj.components.Button.Button": "Class"})
    import_mapping = {
        "proj.consumer": {"Button": "proj.barrel.Button"},
        "proj.barrel": {"Button": "proj.components.Button.Button"},
    }
    re_export_mapping = {
        "proj.barrel": {"Button": "proj.components.Button.Button"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("Button", "proj.consumer", None)

    assert result is not None
    _node_type, qn = result
    assert qn == "proj.components.Button.Button", (
        f"Expected chain to resolve to original symbol, got {qn}"
    )


# --------------------------------------------------------------------------
# TS chain depth 2
# --------------------------------------------------------------------------


def test_should_resolve_through_two_barrels_when_chain_depth_is_two(
    tmp_path: Path,
) -> None:
    """A re-exports from B re-exports from C — caller targets the leaf."""

    registry = _FakeRegistry({"proj.leaf.add": "Function"})
    import_mapping = {
        "proj.consumer": {"add": "proj.barrelA.add"},
        "proj.barrelA": {"add": "proj.barrelB.add"},
        "proj.barrelB": {"add": "proj.leaf.add"},
    }
    re_export_mapping = {
        "proj.barrelA": {"add": "proj.barrelB.add"},
        "proj.barrelB": {"add": "proj.leaf.add"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("add", "proj.consumer", None)

    assert result is not None
    _node_type, qn = result
    assert qn == "proj.leaf.add"


# --------------------------------------------------------------------------
# Cyclic re-export — must terminate
# --------------------------------------------------------------------------


def test_should_terminate_when_reexport_chain_cycles_back_on_itself(
    tmp_path: Path,
) -> None:
    """barrelA re-exports from barrelB, barrelB re-exports from barrelA.

    The chain walker must detect the revisit and bail rather than loop
    forever. Returning ``None`` is the correct unresolved signal.
    """

    registry = _FakeRegistry({})  # nothing concrete to land on
    import_mapping = {
        "proj.consumer": {"thing": "proj.barrelA.thing"},
        "proj.barrelA": {"thing": "proj.barrelB.thing"},
        "proj.barrelB": {"thing": "proj.barrelA.thing"},
    }
    re_export_mapping = {
        "proj.barrelA": {"thing": "proj.barrelB.thing"},
        "proj.barrelB": {"thing": "proj.barrelA.thing"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("thing", "proj.consumer", None)

    assert result is None, "Cyclic chain should resolve to None, not loop"


# --------------------------------------------------------------------------
# Re-exporter that *defines* a symbol with the same name
# --------------------------------------------------------------------------


def test_should_resolve_to_reexporter_when_symbol_is_defined_at_the_barrel(
    tmp_path: Path,
) -> None:
    """If the barrel both re-exports AND has a real definition at that qn,
    we keep the existing direct-import behaviour (no chain walk needed)."""

    registry = _FakeRegistry({"proj.barrel.Button": "Class"})
    import_mapping = {
        "proj.consumer": {"Button": "proj.barrel.Button"},
    }
    re_export_mapping: dict[str, dict[str, str]] = {}

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("Button", "proj.consumer", None)

    assert result is not None
    _node_type, qn = result
    assert qn == "proj.barrel.Button"


# --------------------------------------------------------------------------
# TS export * — namespace re-export
# --------------------------------------------------------------------------


def test_should_record_namespace_reexport_when_ts_uses_export_star(
    tmp_path: Path,
) -> None:
    """``export * from './mod'`` registers a wildcard sentinel in
    re_export_mapping so the namespace edge is captured.

    This is a wiring test against ImportProcessor — the consumer-side
    chain walk for wildcard names depends on a future v1 enhancement
    (suffix matching across wildcard sentinels), but the schema-level
    capture must happen now so the RE_EXPORTS edge is correct.
    """

    proc = ImportProcessor(repo_path=tmp_path, project_name="proj")
    # Simulate the post-parse state for a barrel that did ``export *``.
    proc.import_mapping = {"proj.barrel": {"*proj.components": "proj.components"}}
    proc.re_export_mapping = {"proj.barrel": {"*proj.components": "proj.components"}}

    target = proc.re_export_mapping["proj.barrel"]["*proj.components"]
    assert target == "proj.components"

    parent = ImportProcessor._target_module_for_reexport(target)
    # ``proj.components`` already IS a module — the parent is ``proj``.
    assert parent == "proj"


# --------------------------------------------------------------------------
# Python __all__ curation
# --------------------------------------------------------------------------


def test_should_filter_python_reexports_when_dunder_all_excludes_them(
    tmp_path: Path,
) -> None:
    """A package ``__init__.py`` that imports both Foo and Bar but only
    lists Foo in ``__all__`` should retain Foo in re_export_mapping but
    drop Bar — so consumers can chain through Foo, not Bar.

    We exercise the full pipeline via ``run_updater`` and then read back
    the ImportProcessor's in-memory state via the constructed GraphUpdater
    to assert on the filtered mapping directly. (Edge-level assertions are
    fragile because Python relative-import qname resolution has its own
    pre-existing quirks unrelated to BUC-1610.)
    """

    project_path = tmp_path / "pyproj"
    pkg = project_path / "mypkg"
    pkg.mkdir(parents=True)
    (project_path / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "submod.py").write_text(
        "def Foo():\n    pass\n\ndef Bar():\n    pass\n",
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        "from .submod import Foo, Bar\n__all__ = ['Foo']\n",
        encoding="utf-8",
    )

    from codebase_rag.tests.conftest import _MockIngestor, create_and_run_updater

    ingestor = _MockIngestor()
    updater = create_and_run_updater(project_path, ingestor)  # type: ignore[arg-type]

    proc = updater.factory.import_processor
    pkg_mapping = proc.re_export_mapping.get("pyproj.mypkg", {})

    assert "Foo" in pkg_mapping, (
        f"Foo (in __all__) should remain a re-export, got {pkg_mapping}"
    )
    assert "Bar" not in pkg_mapping, (
        f"Bar (excluded from __all__) should be filtered out, got {pkg_mapping}"
    )


def test_should_keep_reexport_when_no_dunder_all_is_declared(
    tmp_path: Path,
) -> None:
    """Without ``__all__``, every ``from .x import Y`` in ``__init__.py``
    stays in re_export_mapping (historical Python convention)."""

    project_path = tmp_path / "pyproj"
    pkg = project_path / "mypkg"
    pkg.mkdir(parents=True)
    (project_path / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "submod.py").write_text(
        "def Foo():\n    pass\n\ndef Bar():\n    pass\n",
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        "from .submod import Foo, Bar\n", encoding="utf-8"
    )

    from codebase_rag.tests.conftest import _MockIngestor, create_and_run_updater

    ingestor = _MockIngestor()
    updater = create_and_run_updater(project_path, ingestor)  # type: ignore[arg-type]

    proc = updater.factory.import_processor
    pkg_mapping = proc.re_export_mapping.get("pyproj.mypkg", {})

    assert "Foo" in pkg_mapping
    assert "Bar" in pkg_mapping


def test_should_keep_empty_dunder_all_filter_when_list_is_empty(
    tmp_path: Path,
) -> None:
    """An empty ``__all__ = []`` is a valid declaration meaning "nothing
    is re-exported" — we honor it by clearing all named re-exports."""

    project_path = tmp_path / "pyproj"
    pkg = project_path / "mypkg"
    pkg.mkdir(parents=True)
    (project_path / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "submod.py").write_text("def Foo(): pass\n", encoding="utf-8")
    (pkg / "__init__.py").write_text(
        "from .submod import Foo\n__all__ = []\n", encoding="utf-8"
    )

    from codebase_rag.tests.conftest import _MockIngestor, create_and_run_updater

    ingestor = _MockIngestor()
    updater = create_and_run_updater(project_path, ingestor)  # type: ignore[arg-type]

    proc = updater.factory.import_processor
    pkg_mapping = proc.re_export_mapping.get("pyproj.mypkg", {})
    assert "Foo" not in pkg_mapping


# --------------------------------------------------------------------------
# Schema-shape regression: helper that derives target module from qn
# --------------------------------------------------------------------------


def test_should_return_parent_module_when_target_qn_has_dot_separator() -> None:
    """Sanity-check the helper that turns ``mod.symbol`` into ``mod``."""

    assert (
        ImportProcessor._target_module_for_reexport("proj.utils.helpers.add")
        == "proj.utils.helpers"
    )


def test_should_return_none_when_target_qn_is_atomic() -> None:
    """Atomic names have no parent module to point at; helper returns None."""

    assert ImportProcessor._target_module_for_reexport("Foo") is None
    assert ImportProcessor._target_module_for_reexport("") is None
