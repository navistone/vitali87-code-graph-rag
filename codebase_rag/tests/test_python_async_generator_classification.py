"""BUC-1602 — async / generator / async-generator classification on Python.

The graph_updater previously had no way to distinguish a plain ``async def``
from an async generator (``async def`` with ``yield``) or to flag plain
generators.  These tests assert that every function and method gets
``is_async`` and ``is_generator`` properties on the emitted node, and that
those flags are correct across the canonical shapes:

* plain function
* plain generator (``def`` with ``yield``)
* plain async function (``async def`` without ``yield``)
* async generator (``async def`` with ``yield``)
* methods of each of the above on a class
* generator inside a non-generator outer function (the outer is NOT a
  generator just because an inner function yields)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codebase_rag.tests.conftest import get_nodes, run_updater


def _make_repo(tmp_path: Path) -> Path:
    project = tmp_path / "async_gen_project"
    project.mkdir()
    (project / "__init__.py").write_text("", encoding="utf-8")
    (project / "lib.py").write_text(
        encoding="utf-8",
        data=(
            "def plain():\n"
            "    return 1\n"
            "\n"
            "def plain_gen():\n"
            "    yield 1\n"
            "    yield 2\n"
            "\n"
            "async def async_fn():\n"
            "    return 1\n"
            "\n"
            "async def async_gen():\n"
            "    yield 1\n"
            "    yield 2\n"
            "\n"
            "def outer_with_inner_gen():\n"
            "    def inner_gen():\n"
            "        yield 1\n"
            "    return inner_gen\n"
            "\n"
            "class Service:\n"
            "    def method(self):\n"
            "        return 1\n"
            "\n"
            "    def method_gen(self):\n"
            "        yield 1\n"
            "\n"
            "    async def method_async(self):\n"
            "        return 1\n"
            "\n"
            "    async def method_async_gen(self):\n"
            "        yield 1\n"
        ),
    )
    return project


def _flags_by_qn(mock_ingestor: MagicMock) -> dict[str, tuple[bool, bool]]:
    """Return {qualified_name: (is_async, is_generator)} for every Function + Method."""
    result: dict[str, tuple[bool, bool]] = {}
    for label in ("Function", "Method"):
        for c in get_nodes(mock_ingestor, label):
            props = c.args[1]
            qn = props["qualified_name"]
            result[qn] = (
                bool(props.get("is_async", False)),
                bool(props.get("is_generator", False)),
            )
    return result


def test_should_flag_async_and_generator_on_functions_when_indexed(
    tmp_path: Path, mock_ingestor: MagicMock
) -> None:
    repo = _make_repo(tmp_path)
    run_updater(repo, mock_ingestor)

    flags = _flags_by_qn(mock_ingestor)

    # plain
    assert flags["async_gen_project.lib.plain"] == (False, False)
    # plain generator
    assert flags["async_gen_project.lib.plain_gen"] == (False, True)
    # async function (not a generator)
    assert flags["async_gen_project.lib.async_fn"] == (True, False)
    # async generator — both flags must be True
    assert flags["async_gen_project.lib.async_gen"] == (True, True)


def test_should_not_propagate_inner_yield_to_outer_when_nested(
    tmp_path: Path, mock_ingestor: MagicMock
) -> None:
    """A ``yield`` inside an inner function does NOT make the outer function a generator."""
    repo = _make_repo(tmp_path)
    run_updater(repo, mock_ingestor)

    flags = _flags_by_qn(mock_ingestor)

    outer_qn = "async_gen_project.lib.outer_with_inner_gen"
    inner_qn = "async_gen_project.lib.outer_with_inner_gen.inner_gen"

    assert flags[outer_qn] == (False, False), (
        "outer function should not be flagged as generator just because "
        "inner function yields"
    )
    assert flags[inner_qn] == (False, True)


def test_should_flag_async_and_generator_on_methods_when_indexed(
    tmp_path: Path, mock_ingestor: MagicMock
) -> None:
    repo = _make_repo(tmp_path)
    run_updater(repo, mock_ingestor)

    flags = _flags_by_qn(mock_ingestor)

    assert flags["async_gen_project.lib.Service.method"] == (False, False)
    assert flags["async_gen_project.lib.Service.method_gen"] == (False, True)
    assert flags["async_gen_project.lib.Service.method_async"] == (True, False)
    # Async generator method — must carry both flags.
    assert flags["async_gen_project.lib.Service.method_async_gen"] == (True, True)
