"""Unit tests for :mod:`codebase_rag.parsers.tsconfig_resolver`.

These tests exercise the resolver in isolation against a real on-disk
``tsconfig.json`` layout written into a temporary directory. They cover the
eight acceptance scenarios called out in BUC-1600 plus a handful of edge cases
the resolver is expected to handle gracefully (cycles, malformed configs,
ambient ``.d.ts`` shadowing real sources, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codebase_rag.parsers.tsconfig_resolver import (
    TsconfigResolver,
    parse_jsonc,
    strip_jsonc_comments,
)

# ---------------------------------------------------------------------------
# strip_jsonc_comments / parse_jsonc


class TestStripJsoncComments:
    def test_should_strip_line_comments_when_outside_strings(self) -> None:
        text = '{\n  "a": 1, // trailing\n  "b": 2 // end\n}'
        cleaned = strip_jsonc_comments(text)
        assert "// trailing" not in cleaned
        assert "// end" not in cleaned
        assert json.loads(cleaned) == {"a": 1, "b": 2}

    def test_should_strip_block_comments_across_multiple_lines(self) -> None:
        text = '{\n  /* multi\n  line */\n  "a": 1\n}'
        assert parse_jsonc(text) == {"a": 1}

    def test_should_preserve_comment_like_content_inside_strings(self) -> None:
        text = '{"url": "https://example.com/path"}'
        assert parse_jsonc(text) == {"url": "https://example.com/path"}

    def test_should_drop_trailing_commas_when_present(self) -> None:
        text = '{"paths": {"@/*": ["src/*",],},}'
        assert parse_jsonc(text) == {"paths": {"@/*": ["src/*"]}}

    def test_should_raise_when_top_level_is_not_an_object(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_jsonc("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Helpers


def _write_tsconfig(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _touch(path: Path, body: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Acceptance scenarios from BUC-1600


class TestResolveAlias:
    def test_should_resolve_at_alias_when_paths_points_to_src(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #1: `@/components/Button` -> `<repo>/src/components/Button.ts`."""

        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                }
            },
        )
        target = _touch(tmp_path / "src" / "components" / "Button.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/components/Button", source) == target

    def test_should_resolve_tilde_alias_when_paths_points_to_app(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #2: `~/util` with `~/*` -> `app/*` -> `<repo>/app/util.ts`."""

        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"~/*": ["app/*"]},
                }
            },
        )
        target = _touch(tmp_path / "app" / "util.ts")
        source = _touch(tmp_path / "src" / "main.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("~/util", source) == target

    def test_should_resolve_multiple_separate_aliases_in_one_config(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #3: multiple non-overlapping alias prefixes both work."""

        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                        "@app/*": ["app/*"],
                        "@lib/*": ["lib/*"],
                    },
                }
            },
        )
        app_target = _touch(tmp_path / "app" / "foo.ts")
        lib_target = _touch(tmp_path / "lib" / "bar.ts")
        source = _touch(tmp_path / "src" / "main.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@app/foo", source) == app_target
        assert resolver.resolve_alias("@lib/bar", source) == lib_target

    def test_should_inherit_paths_from_parent_when_child_has_none(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #4: child tsconfig with no `paths` inherits parent's."""

        _write_tsconfig(
            tmp_path / "tsconfig.base.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                }
            },
        )
        _write_tsconfig(
            tmp_path / "web" / "tsconfig.json",
            {
                "extends": "../tsconfig.base.json",
                "compilerOptions": {"target": "esnext"},
            },
        )
        target = _touch(tmp_path / "src" / "components" / "Button.ts")
        source = _touch(tmp_path / "web" / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/components/Button", source) == target

    def test_should_let_child_paths_override_parent_paths(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #5: a `paths` block in the child replaces the parent's."""

        _write_tsconfig(
            tmp_path / "tsconfig.base.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                }
            },
        )
        _write_tsconfig(
            tmp_path / "web" / "tsconfig.json",
            {
                "extends": "../tsconfig.base.json",
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["app/*"]},
                },
            },
        )
        # File that would match the parent's paths (but shouldn't, because the
        # child overrode it).
        _touch(tmp_path / "src" / "wrong.ts")
        override_target = _touch(tmp_path / "web" / "app" / "wrong.ts")
        source = _touch(tmp_path / "web" / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/wrong", source) == override_target

    def test_should_return_none_when_no_alias_matches(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #6: a bare specifier that doesn't match any alias yields None."""

        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                }
            },
        )
        source = _touch(tmp_path / "src" / "App.ts")
        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("lodash", source) is None
        assert resolver.resolve_alias("@babel/core", source) is None

    def test_should_parse_tsconfig_with_comments_and_trailing_commas(
        self, tmp_path: Path
    ) -> None:
        """Acceptance #7: jsonc input parses cleanly."""

        config_body = """
        {
          // root config for the web app
          "compilerOptions": {
            /* base url is repo root */
            "baseUrl": ".",
            "paths": {
              "@/*": ["src/*",],
            },
          },
        }
        """
        _write_tsconfig(tmp_path / "tsconfig.json", config_body)
        target = _touch(tmp_path / "src" / "x.ts")
        source = _touch(tmp_path / "src" / "y.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/x", source) == target

    def test_should_be_a_noop_when_paths_is_empty(self, tmp_path: Path) -> None:
        """Acceptance #8: empty `paths` config returns None without crashing."""

        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {"compilerOptions": {"baseUrl": ".", "paths": {}}},
        )
        source = _touch(tmp_path / "src" / "App.ts")
        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/anything", source) is None


# ---------------------------------------------------------------------------
# Edge cases


class TestEdgeCases:
    def test_should_return_none_for_relative_specifiers(self, tmp_path: Path) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}},
        )
        source = _touch(tmp_path / "src" / "App.ts")
        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("./util", source) is None
        assert resolver.resolve_alias("../lib", source) is None

    def test_should_resolve_directory_alias_to_index_file(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}},
        )
        target = _touch(tmp_path / "src" / "components" / "index.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/components", source) == target

    def test_should_pick_ts_over_dts_when_both_exist(self, tmp_path: Path) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}},
        )
        _touch(tmp_path / "src" / "x.d.ts")
        ts_target = _touch(tmp_path / "src" / "x.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/x", source) == ts_target

    def test_should_return_none_when_target_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}},
        )
        source = _touch(tmp_path / "src" / "App.ts")
        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/ghost", source) is None

    def test_should_try_alternative_targets_in_paths_array(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["missing/*", "src/*"]},
                }
            },
        )
        target = _touch(tmp_path / "src" / "Button.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/Button", source) == target

    def test_should_match_exact_pattern_without_wildcard(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"shared": ["packages/shared/index.ts"]},
                }
            },
        )
        target = _touch(tmp_path / "packages" / "shared" / "index.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("shared", source) == target

    def test_should_handle_baseurl_pointing_into_subdir(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": "./packages",
                    "paths": {"@/*": ["app/*"]},
                }
            },
        )
        target = _touch(tmp_path / "packages" / "app" / "foo.ts")
        source = _touch(tmp_path / "src" / "main.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/foo", source) == target

    def test_should_prefer_longer_prefix_when_two_aliases_overlap(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                        "@/*": ["src/*"],
                        "@/lib/*": ["packages/lib/*"],
                    },
                }
            },
        )
        # The more specific alias should win.
        specific_target = _touch(tmp_path / "packages" / "lib" / "core.ts")
        _touch(tmp_path / "src" / "lib" / "core.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/lib/core", source) == specific_target

    def test_should_break_extends_cycle_without_infinite_loop(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(
            tmp_path / "a.json",
            {"extends": "./b.json", "compilerOptions": {}},
        )
        _write_tsconfig(
            tmp_path / "b.json",
            {"extends": "./a.json", "compilerOptions": {}},
        )
        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "extends": "./a.json",
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                },
            },
        )
        target = _touch(tmp_path / "src" / "x.ts")
        source = _touch(tmp_path / "src" / "y.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/x", source) == target

    def test_should_cache_parsed_tsconfig_across_calls(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "tsconfig.json"
        _write_tsconfig(
            config_path,
            {"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}},
        )
        target = _touch(tmp_path / "src" / "x.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/x", source) == target

        # Corrupt the file on disk; resolution should still work because the
        # parsed config is cached on the resolver.
        config_path.write_text("THIS IS NOT JSON", encoding="utf-8")
        assert resolver.resolve_alias("@/x", source) == target

    def test_should_return_none_when_no_tsconfig_exists(
        self, tmp_path: Path
    ) -> None:
        source = _touch(tmp_path / "src" / "App.ts")
        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/anything", source) is None

    def test_should_skip_bare_module_extends_specifier(
        self, tmp_path: Path
    ) -> None:
        """``extends: '@tsconfig/strictest'`` is not supported (no node_modules walk)."""

        _write_tsconfig(
            tmp_path / "tsconfig.json",
            {
                "extends": "@tsconfig/strictest",
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                },
            },
        )
        target = _touch(tmp_path / "src" / "x.ts")
        source = _touch(tmp_path / "src" / "App.ts")

        resolver = TsconfigResolver(tmp_path)
        # Child's own paths still resolve even though extends was ignored.
        assert resolver.resolve_alias("@/x", source) == target

    def test_should_return_none_when_malformed_tsconfig(
        self, tmp_path: Path
    ) -> None:
        _write_tsconfig(tmp_path / "tsconfig.json", "{this is not json}")
        source = _touch(tmp_path / "src" / "App.ts")
        resolver = TsconfigResolver(tmp_path)
        assert resolver.resolve_alias("@/anything", source) is None
