"""BUC-1602 — comprehension scope must not leak loop vars to the outer scope.

In Python 3, the loop variable of a list/dict/set/generator comprehension
is local to the comprehension itself.  The variable analyzer used to write
comprehension loop vars directly into the enclosing function's
``local_var_types`` dict, which corrupted type inference for any later
reference to a variable of the same name in the outer scope.

The same traversal also only recognised ``list_comprehension`` nodes —
dict, set, and generator comprehensions were ignored entirely.  These
tests cover both fixes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from codebase_rag.parsers.import_processor import ImportProcessor
from codebase_rag.parsers.py.type_inference import PythonTypeInferenceEngine

if TYPE_CHECKING:
    from tree_sitter import Parser

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PYTHON_AVAILABLE = True
except ImportError:
    PYTHON_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not PYTHON_AVAILABLE, reason="tree-sitter-python not installed"
)


@pytest.fixture
def python_parser() -> Parser:
    return Parser(Language(tspython.language()))


@pytest.fixture
def import_processor() -> MagicMock:
    processor = MagicMock(spec=ImportProcessor)
    processor.import_mapping = {}
    return processor


@pytest.fixture
def mock_function_registry() -> MagicMock:
    registry = MagicMock()
    registry.__contains__ = MagicMock(return_value=False)
    registry.__getitem__ = MagicMock(return_value=None)
    registry.get = MagicMock(return_value=None)
    registry.find_with_prefix = MagicMock(return_value=[])
    registry.items = MagicMock(return_value=[])
    return registry


@pytest.fixture
def mock_ast_cache() -> MagicMock:
    cache = MagicMock()
    cache.__contains__ = MagicMock(return_value=False)
    cache.__getitem__ = MagicMock(return_value=(None, None))
    return cache


@pytest.fixture
def engine(
    import_processor: MagicMock,
    mock_function_registry: MagicMock,
    mock_ast_cache: MagicMock,
) -> PythonTypeInferenceEngine:
    return PythonTypeInferenceEngine(
        import_processor=import_processor,
        function_registry=mock_function_registry,
        repo_path=Path("/test/repo"),
        project_name="test_project",
        ast_cache=mock_ast_cache,
        queries={},
        module_qn_to_file_path={},
        class_inheritance={},
        simple_name_lookup=defaultdict(set),
        js_type_inference_getter=lambda: MagicMock(),
    )


def _find_function(root_node, name: str):
    if root_node.type == "function_definition":
        name_node = root_node.child_by_field_name("name")
        if name_node and name_node.text.decode() == name:
            return root_node
    for child in root_node.children:
        if result := _find_function(child, name):
            return result
    return None


class TestComprehensionScopeIsolation:
    """Loop variables of comprehensions must NOT bleed to the outer scope."""

    def test_should_not_leak_list_comprehension_loop_var_when_iterating(
        self, python_parser: Parser, engine: PythonTypeInferenceEngine
    ) -> None:
        # The comprehension binds ``user`` locally; in the outer function
        # ``user`` is never assigned, so it must not appear in the local
        # var map (or, if present, must not carry the leaked type from
        # the comprehension's User-iterable).
        python_code = b"""
class User: pass

def process(users):
    names = [user.name for user in users]
    return names
"""
        tree = python_parser.parse(python_code)
        func = _find_function(tree.root_node, "process")

        result = engine.build_local_variable_type_map(func, "test.module")

        assert "user" not in result, (
            "comprehension loop var 'user' must not be visible in the "
            f"outer function's local type map; got: {result!r}"
        )

    def test_should_not_leak_dict_comprehension_loop_var_when_iterating(
        self, python_parser: Parser, engine: PythonTypeInferenceEngine
    ) -> None:
        # Same case but via a dict comprehension.  Previously the variable
        # analyzer ignored ``dictionary_comprehension`` entirely AND leaked
        # its loop vars; now it must do neither.
        python_code = b"""
class User: pass

def process(users):
    lookup = {user.id: user for user in users}
    return lookup
"""
        tree = python_parser.parse(python_code)
        func = _find_function(tree.root_node, "process")

        result = engine.build_local_variable_type_map(func, "test.module")

        assert "user" not in result

    def test_should_not_leak_set_comprehension_loop_var_when_iterating(
        self, python_parser: Parser, engine: PythonTypeInferenceEngine
    ) -> None:
        python_code = b"""
class User: pass

def process(users):
    ids = {user.id for user in users}
    return ids
"""
        tree = python_parser.parse(python_code)
        func = _find_function(tree.root_node, "process")

        result = engine.build_local_variable_type_map(func, "test.module")

        assert "user" not in result

    def test_should_not_leak_generator_expression_loop_var_when_iterating(
        self, python_parser: Parser, engine: PythonTypeInferenceEngine
    ) -> None:
        python_code = b"""
class User: pass

def process(users):
    total = sum(user.score for user in users)
    return total
"""
        tree = python_parser.parse(python_code)
        func = _find_function(tree.root_node, "process")

        result = engine.build_local_variable_type_map(func, "test.module")

        assert "user" not in result

    def test_should_preserve_outer_scope_binding_when_name_collides_with_comp_var(
        self, python_parser: Parser, engine: PythonTypeInferenceEngine
    ) -> None:
        # The outer scope defines ``item: Widget`` explicitly via for-loop
        # iteration over a list literal.  A subsequent comprehension that
        # binds ``item`` to a different iterable must NOT overwrite the
        # outer binding.
        python_code = b"""
class Widget: pass
class Gadget: pass

def process():
    for item in [Widget(), Widget()]:
        pass
    extras = [item for item in [Gadget(), Gadget()]]
    return item
"""
        tree = python_parser.parse(python_code)
        func = _find_function(tree.root_node, "process")

        result = engine.build_local_variable_type_map(func, "test.module")

        # The outer for-loop should set item -> Widget.  The comprehension
        # binding to Gadget must stay inside the comprehension scope.
        assert result.get("item") == "Widget", (
            "comprehension must not overwrite outer-scope binding; "
            f"got result={result!r}"
        )

    def test_should_still_track_outer_for_loop_var_when_comprehensions_exist(
        self, python_parser: Parser, engine: PythonTypeInferenceEngine
    ) -> None:
        """Regression: the scope-copy fix must not break for-statement analysis."""
        python_code = b"""
class User: pass

def process():
    pairs = [(u, u.name) for u in [User(), User()]]
    for item in [User(), User()]:
        print(item)
    return pairs
"""
        tree = python_parser.parse(python_code)
        func = _find_function(tree.root_node, "process")

        result = engine.build_local_variable_type_map(func, "test.module")

        # outer for-loop var is still detected
        assert result.get("item") == "User"
        # comprehension var stays scoped
        assert "u" not in result
