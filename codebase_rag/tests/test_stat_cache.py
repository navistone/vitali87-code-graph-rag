"""Tests for the stat-cache layer in front of the SHA-256 hash cache.

See BUC-1612. The stat cache stores (mtime_ns, size, sha) per file so that
unchanged files skip the SHA-256 read+hash on subsequent runs.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag import constants as cs
from codebase_rag.graph_updater import (
    GraphUpdater,
    StatEntry,
    _load_stat_cache,
    _save_stat_cache,
    _stat_matches,
)
from codebase_rag.parser_loader import load_parsers


@pytest.fixture
def py_project(temp_repo: Path) -> Path:
    (temp_repo / "__init__.py").touch()
    (temp_repo / "module_a.py").write_text("def func_a():\n    pass\n")
    (temp_repo / "module_b.py").write_text("def func_b():\n    pass\n")
    return temp_repo


class TestStatMatches:
    def test_should_match_when_mtime_and_size_equal(self) -> None:
        cached = StatEntry(mtime_ns=1_000_000_000, size=42, sha="abc")
        assert _stat_matches(cached, mtime_ns=1_000_000_000, size=42)

    def test_should_not_match_when_size_differs(self) -> None:
        cached = StatEntry(mtime_ns=1_000_000_000, size=42, sha="abc")
        assert not _stat_matches(cached, mtime_ns=1_000_000_000, size=43)

    def test_should_not_match_when_mtime_differs_beyond_tolerance(self) -> None:
        cached = StatEntry(mtime_ns=1_000_000_000, size=42, sha="abc")
        # 1 second later — well beyond 1 ms tolerance.
        assert not _stat_matches(cached, mtime_ns=2_000_000_000, size=42)

    def test_should_match_within_one_millisecond_tolerance(self) -> None:
        """Some filesystems round mtime; tolerate <1ms jitter."""
        cached = StatEntry(mtime_ns=1_000_000_000, size=42, sha="abc")
        # 500_000 ns = 0.5 ms — within the 1 ms tolerance.
        assert _stat_matches(cached, mtime_ns=1_000_500_000, size=42)

    def test_should_not_match_when_mtime_regresses_more_than_tolerance(self) -> None:
        """git clone --depth resets mtime backwards; SHA must recompute."""
        cached = StatEntry(mtime_ns=2_000_000_000, size=42, sha="abc")
        assert not _stat_matches(cached, mtime_ns=1_000_000_000, size=42)


class TestStatCacheIO:
    def test_should_round_trip_save_and_load(self, temp_repo: Path) -> None:
        cache_path = temp_repo / cs.STAT_CACHE_FILENAME
        entries = {
            "a.py": StatEntry(mtime_ns=1, size=10, sha="hash-a"),
            "b.py": StatEntry(mtime_ns=2, size=20, sha="hash-b"),
        }
        _save_stat_cache(cache_path, entries)
        loaded = _load_stat_cache(cache_path)
        assert loaded == entries

    def test_should_return_empty_when_file_missing(self, temp_repo: Path) -> None:
        cache_path = temp_repo / cs.STAT_CACHE_FILENAME
        assert _load_stat_cache(cache_path) == {}

    def test_should_return_empty_when_json_corrupted(self, temp_repo: Path) -> None:
        cache_path = temp_repo / cs.STAT_CACHE_FILENAME
        cache_path.write_text("not valid json {{{")
        assert _load_stat_cache(cache_path) == {}

    def test_should_drop_rows_missing_required_fields(
        self, temp_repo: Path
    ) -> None:
        """Malformed rows degrade gracefully — they're dropped, not raised."""
        cache_path = temp_repo / cs.STAT_CACHE_FILENAME
        cache_path.write_text(
            json.dumps(
                {
                    "good.py": {"mtime_ns": 1, "size": 10, "sha": "h"},
                    "bad_no_sha.py": {"mtime_ns": 1, "size": 10},
                    "bad_wrong_type.py": "not a dict",
                }
            )
        )
        loaded = _load_stat_cache(cache_path)
        assert "good.py" in loaded
        assert "bad_no_sha.py" not in loaded
        assert "bad_wrong_type.py" not in loaded


class TestStatCacheSkipsShaOnUnchangedRun:
    """The headline guarantee — second run with no changes does zero SHA-256s."""

    def test_should_skip_all_sha_computation_on_second_unchanged_run(
        self, py_project: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()

        # First run primes both caches.
        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=py_project,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        # Confirm both sidecars now exist.
        assert (py_project / cs.HASH_CACHE_FILENAME).is_file()
        assert (py_project / cs.STAT_CACHE_FILENAME).is_file()

        mock_ingestor.reset_mock()

        # Second run — patch hashlib.sha256 to detect any SHA computation.
        with patch(
            "codebase_rag.graph_updater.hashlib.sha256",
            wraps=hashlib.sha256,
        ) as sha_spy:
            updater2 = GraphUpdater(
                ingestor=mock_ingestor,
                repo_path=py_project,
                parsers=parsers,
                queries=queries,
            )
            updater2.run()

            assert sha_spy.call_count == 0, (
                f"stat-cache should have skipped SHA entirely, "
                f"got {sha_spy.call_count} calls"
            )


class TestStatCacheRecomputesOnChange:
    """When a file is modified, only that file's SHA is recomputed."""

    def test_should_recompute_sha_only_for_modified_file(
        self, py_project: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()

        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=py_project,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        # Modify module_a — wait long enough to defeat mtime rounding on
        # filesystems with coarse precision.
        time.sleep(0.05)
        (py_project / "module_a.py").write_text("def func_a_updated():\n    pass\n")

        with patch(
            "codebase_rag.graph_updater.hashlib.sha256",
            wraps=hashlib.sha256,
        ) as sha_spy:
            updater2 = GraphUpdater(
                ingestor=mock_ingestor,
                repo_path=py_project,
                parsers=parsers,
                queries=queries,
            )
            updater2.run()

            # Exactly one SHA computation (for module_a.py). __init__.py and
            # module_b.py are stat-unchanged.
            assert sha_spy.call_count == 1, (
                f"expected 1 SHA call for the changed file, "
                f"got {sha_spy.call_count}"
            )


class TestStatCacheMtimeRegressionEdgeCase:
    """`git clone --depth` and `cp -p` style restores reset mtime backwards.

    The cached mtime is *greater* than the on-disk mtime. The check fails, SHA
    is recomputed — that's the safe behaviour, even if size matches.
    """

    def test_should_recompute_sha_when_mtime_regresses(
        self, py_project: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()
        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=py_project,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        # Simulate `git clone --depth` resetting mtime to an older timestamp.
        # Content unchanged → size unchanged. Only mtime moves backwards.
        target = py_project / "module_a.py"
        st = target.stat()
        old_atime_ns = st.st_atime_ns
        # 10 years earlier — well outside the 1 ms tolerance window.
        regressed_mtime_ns = st.st_mtime_ns - (10 * 365 * 24 * 3600 * 1_000_000_000)
        os.utime(target, ns=(old_atime_ns, regressed_mtime_ns))

        with patch(
            "codebase_rag.graph_updater.hashlib.sha256",
            wraps=hashlib.sha256,
        ) as sha_spy:
            updater2 = GraphUpdater(
                ingestor=mock_ingestor,
                repo_path=py_project,
                parsers=parsers,
                queries=queries,
            )
            updater2.run()

            # SHA was recomputed for module_a even though content is identical
            # — because mtime regressed. The hash still matches the prior
            # hash-cache entry, so _process_single_file is NOT called; this is
            # the "stat says maybe, sha says no" code path.
            assert sha_spy.call_count >= 1, (
                "mtime regression must force a SHA recompute"
            )

        # After the run, the stat-cache should reflect the new (regressed)
        # mtime — subsequent runs hit the fast path again.
        stat_cache = _load_stat_cache(py_project / cs.STAT_CACHE_FILENAME)
        assert stat_cache["module_a.py"].mtime_ns == regressed_mtime_ns


class TestStatCacheMtimeOnlyLiarSizeMismatch:
    """Some filesystems lie about mtime; if size changed, we still catch it."""

    def test_should_recompute_sha_when_size_differs_even_if_mtime_preserved(
        self, py_project: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()
        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=py_project,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        target = py_project / "module_a.py"
        original_st = target.stat()
        # Write content of a different length, then force the mtime back to
        # what it was — simulating a coarse-precision-FS "mtime liar" edit.
        target.write_text("def x(): pass  # MUCH longer content than before\n")
        os.utime(target, ns=(original_st.st_atime_ns, original_st.st_mtime_ns))

        new_st = target.stat()
        assert new_st.st_size != original_st.st_size, (
            "size guard prerequisite: rewrite must change file size"
        )

        with patch(
            "codebase_rag.graph_updater.hashlib.sha256",
            wraps=hashlib.sha256,
        ) as sha_spy:
            updater2 = GraphUpdater(
                ingestor=mock_ingestor,
                repo_path=py_project,
                parsers=parsers,
                queries=queries,
            )
            updater2.run()

            # At least the changed file got rehashed — size guard fired.
            assert sha_spy.call_count >= 1


class TestStatCacheForceBypass:
    """`force=True` skips both caches and re-hashes everything."""

    def test_should_recompute_all_shas_when_force_true(
        self, py_project: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()
        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=py_project,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        with patch(
            "codebase_rag.graph_updater.hashlib.sha256",
            wraps=hashlib.sha256,
        ) as sha_spy:
            updater2 = GraphUpdater(
                ingestor=mock_ingestor,
                repo_path=py_project,
                parsers=parsers,
                queries=queries,
            )
            updater2.run(force=True)

            # Force-mode hashes everything: __init__.py, module_a.py, module_b.py.
            assert sha_spy.call_count >= 3


class TestStatCacheSidecarIsIgnored:
    """The stat cache file itself must never become input to itself."""

    def test_should_not_index_stat_cache_file(
        self, py_project: Path, mock_ingestor: MagicMock
    ) -> None:
        parsers, queries = load_parsers()
        updater = GraphUpdater(
            ingestor=mock_ingestor,
            repo_path=py_project,
            parsers=parsers,
            queries=queries,
        )
        updater.run()

        stat_cache = _load_stat_cache(py_project / cs.STAT_CACHE_FILENAME)
        assert cs.STAT_CACHE_FILENAME not in stat_cache
        assert cs.HASH_CACHE_FILENAME not in stat_cache
