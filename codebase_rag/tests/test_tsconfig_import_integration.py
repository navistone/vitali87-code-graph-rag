"""Integration tests for tsconfig path-alias wiring inside ImportProcessor.

These tests instantiate :class:`ImportProcessor` directly and exercise the
``_resolve_js_module_path`` -> ``_resolve_via_tsconfig`` pipeline against
on-disk fixtures. They guarantee that:

* Aliased import paths produce qualified names rooted at ``project_name`` and
  containing the resolved file's path-without-extension.
* Relative imports keep using the existing relative resolver.
* Bare specifiers that don't match any alias fall back to the legacy
  ``slash -> dot`` transform (so they end up as External nodes downstream).
* Non-existent alias targets fall back to the legacy behaviour rather than
  raising or silently swallowing the import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codebase_rag.parsers.import_processor import ImportProcessor

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tsconfig_paths"


@pytest.fixture()
def fixture_processor() -> ImportProcessor:
    """ImportProcessor wired up to the static tsconfig_paths fixture."""

    return ImportProcessor(
        repo_path=FIXTURE_ROOT,
        project_name="fixture",
        ingestor=None,
        function_registry=None,
    )


class TestResolveJsModulePathWithTsconfig:
    def test_should_resolve_aliased_import_to_project_qualified_module(
        self, fixture_processor: ImportProcessor
    ) -> None:
        resolved = fixture_processor._resolve_js_module_path(
            "@/components/Button", "fixture.src.App"
        )
        assert resolved == "fixture.src.components.Button"

    def test_should_resolve_components_alias(
        self, fixture_processor: ImportProcessor
    ) -> None:
        resolved = fixture_processor._resolve_js_module_path(
            "@components/Card", "fixture.src.App"
        )
        assert resolved == "fixture.src.components.Card"

    def test_should_resolve_nested_alias_path(
        self, fixture_processor: ImportProcessor
    ) -> None:
        resolved = fixture_processor._resolve_js_module_path(
            "@/lib/date", "fixture.src.App"
        )
        assert resolved == "fixture.src.lib.date"

    def test_should_resolve_exact_pattern_alias(
        self, fixture_processor: ImportProcessor
    ) -> None:
        resolved = fixture_processor._resolve_js_module_path(
            "shared", "fixture.src.App"
        )
        assert resolved == "fixture.packages.shared.index"

    def test_should_fall_back_to_dot_path_for_external_libraries(
        self, fixture_processor: ImportProcessor
    ) -> None:
        # ``lodash`` doesn't match any alias and isn't a relative import, so
        # the existing slash-to-dot logic still produces an External-shaped
        # qname downstream.
        resolved = fixture_processor._resolve_js_module_path(
            "lodash", "fixture.src.App"
        )
        assert resolved == "lodash"

    def test_should_fall_back_to_dot_path_when_alias_target_missing(
        self, fixture_processor: ImportProcessor
    ) -> None:
        # ``@/does/not/exist`` matches the alias prefix but no file is on
        # disk, so the resolver returns None and the legacy slash-to-dot path
        # runs.
        resolved = fixture_processor._resolve_js_module_path(
            "@/does/not/exist", "fixture.src.App"
        )
        assert resolved == "@.does.not.exist"

    def test_should_not_consult_tsconfig_for_relative_imports(
        self, fixture_processor: ImportProcessor
    ) -> None:
        # ``./components/Button`` from src/App.ts -> ``fixture.src.components.Button``
        resolved = fixture_processor._resolve_js_module_path(
            "./components/Button", "fixture.src.App"
        )
        assert resolved == "fixture.src.components.Button"

    def test_should_handle_unknown_source_module_gracefully(
        self, fixture_processor: ImportProcessor
    ) -> None:
        # ``something.weird`` has no on-disk file -- tsconfig resolution must
        # bail out and the legacy slash-to-dot path runs.
        resolved = fixture_processor._resolve_js_module_path(
            "@/components/Button", "fixture.does_not_exist"
        )
        assert resolved == "@.components.Button"


class TestResolveJsModulePathWithoutTsconfig:
    def test_should_fall_back_to_legacy_behaviour_when_no_tsconfig(
        self, tmp_path: Path
    ) -> None:
        # Set up a tiny repo with a JS source file but no tsconfig.json.
        src = tmp_path / "src" / "App.ts"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("export const x = 1;\n", encoding="utf-8")

        proc = ImportProcessor(
            repo_path=tmp_path,
            project_name="repo",
            ingestor=None,
            function_registry=None,
        )
        resolved = proc._resolve_js_module_path(
            "@/components/Button", "repo.src.App"
        )
        # No tsconfig anywhere -> alias logic returns None -> legacy fallback.
        assert resolved == "@.components.Button"

    def test_should_skip_tsconfig_when_malformed(self, tmp_path: Path) -> None:
        # A malformed tsconfig must not crash -- alias resolution silently
        # returns None and the legacy fallback path runs.
        (tmp_path / "tsconfig.json").write_text(
            "{ this is broken", encoding="utf-8"
        )
        src = tmp_path / "src" / "App.ts"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("export const x = 1;\n", encoding="utf-8")

        proc = ImportProcessor(
            repo_path=tmp_path,
            project_name="repo",
            ingestor=None,
            function_registry=None,
        )
        resolved = proc._resolve_js_module_path("@/x", "repo.src.App")
        assert resolved == "@.x"


class TestRepoLevelTsconfigDiscovery:
    """Walk-up behaviour: a tsconfig several directories above the source file
    is still consulted."""

    def test_should_walk_up_directory_tree_to_find_tsconfig(
        self, tmp_path: Path
    ) -> None:
        # tsconfig at the repo root; source file three levels deep.
        (tmp_path / "tsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "baseUrl": ".",
                        "paths": {"@/*": ["src/*"]},
                    }
                }
            ),
            encoding="utf-8",
        )
        deep_src = tmp_path / "src" / "feature" / "deep" / "App.ts"
        deep_src.parent.mkdir(parents=True, exist_ok=True)
        deep_src.write_text("export {};\n", encoding="utf-8")
        target = tmp_path / "src" / "components" / "Button.ts"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("export const Button = 1;\n", encoding="utf-8")

        proc = ImportProcessor(
            repo_path=tmp_path,
            project_name="repo",
            ingestor=None,
            function_registry=None,
        )
        resolved = proc._resolve_js_module_path(
            "@/components/Button", "repo.src.feature.deep.App"
        )
        assert resolved == "repo.src.components.Button"
