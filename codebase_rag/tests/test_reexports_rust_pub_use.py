"""BUC-1618: Rust ``pub use`` re-export resolution.

Mirror of BUC-1610 (TS barrels + Python ``__all__``) and BUC-1617
(wildcard consumer-side walk) for Rust.  Pins three shapes:

* Named ``pub use crate::sub::Foo`` reaches the leaf definition through
  the re-exporter.
* Glob ``pub use crate::sub::*`` re-exports every public name; consumers
  resolve through via the BUC-1617 wildcard-chain walker.
* Cycle (A ``pub use B::*``, B ``pub use A::*``) terminates cleanly,
  bounded by the shared 16-hop ceiling and visited set.

We exercise the Rust parser end-to-end via ``run_updater`` so we cover
the new visibility detection + path-resolution logic.  The chain-walk
assertions are kept sharp by inspecting the ``re_export_mapping`` the
parser produces directly, plus a focused unit test against
``ImportProcessor._resolve_rust_full_path_qn``.

A non-``pub use`` baseline guards against regressing private ``use``
declarations into re-exports.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag.parsers.call_resolver import CallResolver
from codebase_rag.parsers.import_processor import ImportProcessor
from codebase_rag.tests.conftest import run_updater


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _FakeRegistry:
    """Minimal stand-in for FunctionRegistryTrie (BUC-1610 pattern)."""

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


def _run_and_extract_processor(
    project_path: Path,
) -> ImportProcessor:
    """Run the updater over ``project_path`` and return the processor.

    We grab the ``ImportProcessor`` instance off the ``GraphUpdater`` so
    the test can inspect ``re_export_mapping`` directly — the chain-walk
    behaviour is already covered by ``test_reexports_wildcard_consumer``,
    so for BUC-1618 the contract under test is "did the parser register
    the right re-export sentinels?".
    """
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.parser_loader import load_parsers

    mock_ingestor = MagicMock()
    parsers, queries = load_parsers()
    updater = GraphUpdater(
        ingestor=mock_ingestor,
        repo_path=project_path,
        parsers=parsers,
        queries=queries,
    )
    updater.run()
    return updater.factory.import_processor


def _make_rust_fixture(
    base: Path,
    name: str,
    files: dict[str, str],
) -> Path:
    """Materialize a Cargo project with the given src/ layout."""
    project = base / name
    project.mkdir()
    (project / "Cargo.toml").write_text(
        encoding="utf-8",
        data=(
            "[package]\n"
            f'name = "{name}"\n'
            'version = "0.1.0"\n'
            'edition = "2021"\n'
        ),
    )
    src = project / "src"
    src.mkdir()
    for rel_path, body in files.items():
        target = src / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return project


# --------------------------------------------------------------------------
# Named `pub use` re-exports
# --------------------------------------------------------------------------


def test_should_register_named_reexport_when_pub_use_targets_leaf_symbol(
    temp_repo: Path,
) -> None:
    """``pub use crate::sub::Foo`` in lib.rs surfaces a re-export sentinel."""

    project = _make_rust_fixture(
        temp_repo,
        "rust_named_reexport",
        {
            "sub.rs": "pub struct Foo;\n",
            "lib.rs": "pub mod sub;\npub use crate::sub::Foo;\n",
        },
    )

    processor = _run_and_extract_processor(project)

    lib_qn = f"{project.name}.src.lib"
    site = processor.re_export_mapping.get(lib_qn, {})

    assert "Foo" in site, (
        f"Expected 'Foo' in lib's re-export map, got keys={list(site.keys())}"
    )
    assert site["Foo"] == f"{project.name}.src.sub.Foo", (
        f"Expected dotted project-qualified target, got {site['Foo']}"
    )


def test_should_not_register_reexport_when_use_is_private(
    temp_repo: Path,
) -> None:
    """Bare ``use crate::sub::Foo`` (no ``pub``) is a private import."""

    project = _make_rust_fixture(
        temp_repo,
        "rust_private_use",
        {
            "sub.rs": "pub struct Foo;\n",
            "lib.rs": "pub mod sub;\nuse crate::sub::Foo;\n",
        },
    )

    processor = _run_and_extract_processor(project)

    lib_qn = f"{project.name}.src.lib"
    site = processor.re_export_mapping.get(lib_qn, {})

    assert "Foo" not in site, (
        f"Private `use` must not produce a re-export sentinel; "
        f"got site={site}"
    )


def test_should_register_reexport_for_pub_crate_use(
    temp_repo: Path,
) -> None:
    """``pub(crate) use`` still routes consumers through the re-exporter."""

    project = _make_rust_fixture(
        temp_repo,
        "rust_pub_crate_use",
        {
            "sub.rs": "pub fn greet() {}\n",
            "lib.rs": "pub mod sub;\npub(crate) use crate::sub::greet;\n",
        },
    )

    processor = _run_and_extract_processor(project)

    lib_qn = f"{project.name}.src.lib"
    site = processor.re_export_mapping.get(lib_qn, {})

    assert "greet" in site, (
        "pub(crate) use must be treated as a re-export for graph topology; "
        f"got site keys={list(site.keys())}"
    )


def test_should_register_named_reexports_from_use_list(
    temp_repo: Path,
) -> None:
    """``pub use crate::sub::{A, B}`` registers both names."""

    project = _make_rust_fixture(
        temp_repo,
        "rust_use_list_reexport",
        {
            "sub.rs": "pub struct A;\npub struct B;\n",
            "lib.rs": "pub mod sub;\npub use crate::sub::{A, B};\n",
        },
    )

    processor = _run_and_extract_processor(project)

    lib_qn = f"{project.name}.src.lib"
    site = processor.re_export_mapping.get(lib_qn, {})

    assert site.get("A") == f"{project.name}.src.sub.A"
    assert site.get("B") == f"{project.name}.src.sub.B"


# --------------------------------------------------------------------------
# Glob `pub use ::*` — wildcard sentinel
# --------------------------------------------------------------------------


def test_should_register_wildcard_sentinel_when_pub_use_glob(
    temp_repo: Path,
) -> None:
    """``pub use crate::sub::*`` registers ``*<target_module>`` sentinel."""

    project = _make_rust_fixture(
        temp_repo,
        "rust_glob_reexport",
        {
            "sub.rs": "pub struct Foo;\npub fn helper() {}\n",
            "lib.rs": "pub mod sub;\npub use crate::sub::*;\n",
        },
    )

    processor = _run_and_extract_processor(project)

    lib_qn = f"{project.name}.src.lib"
    site = processor.re_export_mapping.get(lib_qn, {})
    sub_module_qn = f"{project.name}.src.sub"
    wildcard_key = f"*{sub_module_qn}"

    assert wildcard_key in site, (
        "Glob pub use must register a *<dotted_target_module> sentinel "
        f"matching the form BUC-1617 walks; got keys={list(site.keys())}"
    )
    assert site[wildcard_key] == sub_module_qn


def test_should_resolve_consumer_through_wildcard_pub_use_chain(
    temp_repo: Path,
) -> None:
    """End-to-end: a named call resolves through a glob pub use chain.

    This exercises the BUC-1617 wildcard walker against a *Rust*-shaped
    re-export site (the new code in this PR registers it the same way
    Python/TS do, so the consumer-side walker engages unchanged).
    """

    project = "rust_glob_consumer"
    registry = _FakeRegistry({f"{project}.src.utils.helper": "Function"})
    import_mapping: dict[str, dict[str, str]] = {
        f"{project}.src.consumer": {"helper": f"{project}.src.barrel.helper"},
        f"{project}.src.barrel": {},
        f"{project}.src.utils": {},
    }
    re_export_mapping: dict[str, dict[str, str]] = {
        f"{project}.src.consumer": {},
        f"{project}.src.barrel": {
            f"*{project}.src.utils": f"{project}.src.utils",
        },
        f"{project}.src.utils": {},
    }

    proc = ImportProcessor(repo_path=Path("/tmp"), project_name=project)
    proc.import_mapping = import_mapping
    proc.re_export_mapping = re_export_mapping
    resolver = CallResolver(
        function_registry=registry,  # type: ignore[arg-type]
        import_processor=proc,
        type_inference=None,  # type: ignore[arg-type]
        class_inheritance={},
    )

    result = resolver._try_resolve_via_imports(
        "helper", f"{project}.src.consumer", None
    )

    assert result is not None, (
        "Wildcard pub use chain must resolve: consumer imports `helper` "
        "from barrel which `pub use utils::*`s through to utils.helper"
    )
    _node, qn = result
    assert qn == f"{project}.src.utils.helper"


# --------------------------------------------------------------------------
# Cycle safety
# --------------------------------------------------------------------------


def test_should_not_infinite_loop_when_pub_use_glob_cycle(
    temp_repo: Path,
) -> None:
    """A <-> B cyclic glob ``pub use`` must terminate the chain walker.

    Constructs a Rust fixture with two modules each globbing the other,
    then drives a consumer-side lookup that probes a symbol neither side
    actually defines — the walker must hit the visited-set guard and
    return None rather than spinning the 16-hop budget into a loop.
    """

    project = _make_rust_fixture(
        temp_repo,
        "rust_cyclic_glob",
        {
            "a.rs": "pub use crate::b::*;\n",
            "b.rs": "pub use crate::a::*;\n",
            "lib.rs": "pub mod a;\npub mod b;\n",
        },
    )

    processor = _run_and_extract_processor(project)

    a_qn = f"{project.name}.src.a"
    b_qn = f"{project.name}.src.b"
    a_site = processor.re_export_mapping.get(a_qn, {})
    b_site = processor.re_export_mapping.get(b_qn, {})

    assert f"*{b_qn}" in a_site, (
        f"Expected a's wildcard sentinel pointing at b, got {a_site}"
    )
    assert f"*{a_qn}" in b_site, (
        f"Expected b's wildcard sentinel pointing at a, got {b_site}"
    )

    # Drive the walker — it must terminate (None) rather than loop.
    registry = _FakeRegistry({})
    resolver = CallResolver(
        function_registry=registry,  # type: ignore[arg-type]
        import_processor=processor,
        type_inference=None,  # type: ignore[arg-type]
        class_inheritance={},
    )

    # Probe a symbol that doesn't exist anywhere in the cycle.  Without
    # the BUC-1610 visited-set guard this would walk a -> b -> a forever
    # (or until the 16-hop ceiling — either way, the contract is "return
    # None deterministically").
    result = resolver._follow_reexport_chain(f"{a_qn}.nope", "nope")
    assert result is None


# --------------------------------------------------------------------------
# Path resolution unit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rust_path,module_qn,expected",
    [
        # Simple crate-rooted path
        (
            "crate::sub::Foo",
            "proj.src.lib",
            "proj.src.sub.Foo",
        ),
        # Deep crate-rooted path
        (
            "crate::a::b::c::Foo",
            "proj.src.lib",
            "proj.src.a.b.c.Foo",
        ),
        # Nested module's perspective still resolves relative to src/
        (
            "crate::a::Foo",
            "proj.src.deep.nested",
            "proj.src.a.Foo",
        ),
        # External / std paths return None
        ("std::collections::HashMap", "proj.src.lib", None),
        ("serde::Serialize", "proj.src.lib", None),
        # Empty / malformed return None
        ("", "proj.src.lib", None),
        ("crate::", "proj.src.lib", None),
    ],
)
def test_resolve_rust_full_path_qn_canonicalizes_crate_paths(
    tmp_path: Path,
    rust_path: str,
    module_qn: str,
    expected: str | None,
) -> None:
    """The new helper must convert ``crate::a::b::Foo`` to the dotted qn.

    This is the contract the chain walker relies on: its ``rsplit('.', 1)``
    only works when targets use ``.`` separators, and the crate root must
    line up with the layout ``definition_processor`` uses (i.e. the
    ``src/`` segment is included in the qn).
    """
    processor = ImportProcessor(repo_path=tmp_path, project_name="proj")
    result = processor._resolve_rust_full_path_qn(rust_path, module_qn)
    assert result == expected


# --------------------------------------------------------------------------
# Existing Rust suites unchanged smoke test
# --------------------------------------------------------------------------


def test_should_leave_existing_rust_singleton_topology_unchanged(
    temp_repo: Path,
) -> None:
    """Regression guard: BUC-1618 must not perturb a `use` (no pub) fixture.

    Mirrors a minimal cross-file Rust singleton — verifies that a vanilla
    ``use crate::storage::Storage`` import still populates only
    ``import_mapping`` and leaves ``re_export_mapping`` empty for that
    consumer (so the new code is strictly opt-in via ``pub``).
    """
    project = _make_rust_fixture(
        temp_repo,
        "rust_baseline_no_pub_use",
        {
            "storage.rs": "pub struct Storage;\nimpl Storage { pub fn get() {} }\n",
            "consumer.rs": (
                "use crate::storage::Storage;\n"
                "pub fn run() { Storage::get(); }\n"
            ),
            "lib.rs": "pub mod storage;\npub mod consumer;\n",
        },
    )

    proc = _run_and_extract_processor(project)

    consumer_qn = f"{project.name}.src.consumer"
    site = proc.re_export_mapping.get(consumer_qn, {})
    assert site == {}, (
        f"Vanilla `use` must not produce re-exports; got {site}"
    )
    # Sanity: the import did register on the import side.
    assert "Storage" in proc.import_mapping.get(consumer_qn, {})
