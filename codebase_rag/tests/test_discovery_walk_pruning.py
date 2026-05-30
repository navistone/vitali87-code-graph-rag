"""Regression tests for traversal-time directory pruning in file discovery.

Indexing a LOCAL filesystem repo path walks the real working tree. Such a
tree commonly contains heavy / ignored directories — ``node_modules``, the
repo's own ``.git``, and nested git worktrees under ``.claude/worktrees`` (each
a full repo checkout with its OWN ``node_modules``). The previous discovery
implementation used ``repo_path.rglob("*")``, which descends into and stats
every entry before rejecting it per-file. On a tree with tens of thousands of
noise files this stalled the "discovering" phase long enough for the job
watchdog to reap the index job.

The fix makes :meth:`GraphUpdater._collect_eligible_files` prune those
directories AT TRAVERSAL TIME (``os.walk`` with in-place ``dirs[:]``
filtering via :func:`should_prune_dir`), so they are never enumerated. These
tests assert:

* the discovered file set excludes everything under pruned dirs, and
* the walk never descends into those subtrees (bounded directory visits).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.utils.path_utils import should_prune_dir


def _make_heavy_tree(root: Path, files_per_noise_dir: int = 2000) -> None:
    """Build a repo tree with a couple of real source files plus large
    ``node_modules`` and ``.claude/worktrees/<nested>/node_modules`` subtrees."""
    # Real source the indexer SHOULD discover.
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("def main():\n    return 1\n")
    (root / "README.md").write_text("# repo\n")

    # The repo's own node_modules — must be pruned.
    nm = root / "node_modules" / "left-pad"
    nm.mkdir(parents=True)
    for i in range(files_per_noise_dir):
        (nm / f"f{i}.js").write_text("module.exports = 1;\n")

    # Nested git worktree under .claude/worktrees, each with its OWN
    # node_modules — must be pruned (the real-world stall reproduction).
    nested = root / ".claude" / "worktrees" / "agent-1" / "node_modules" / "dep"
    nested.mkdir(parents=True)
    for i in range(files_per_noise_dir):
        (nested / f"g{i}.js").write_text("module.exports = 2;\n")
    # A plausible source file buried inside the worktree — still pruned, because
    # the whole .claude subtree is excluded.
    (root / ".claude" / "worktrees" / "agent-1" / "src").mkdir(parents=True)
    (root / ".claude" / "worktrees" / "agent-1" / "src" / "buried.py").write_text(
        "def buried():\n    return 3\n"
    )

    # A .git dir with loose objects — must be pruned.
    git = root / ".git" / "objects"
    git.mkdir(parents=True)
    for i in range(50):
        (git / f"obj{i}").write_text("x")


@pytest.fixture
def heavy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "heavy_repo"
    repo.mkdir()
    _make_heavy_tree(repo)
    return repo


def test_should_prune_dir_prunes_builtin_heavy_dirs(tmp_path: Path) -> None:
    repo = tmp_path
    for name in (
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".next",
        ".turbo",
        ".cache",
        ".claude",
        ".cgr",
        ".tantivy",
        ".forge",
    ):
        assert should_prune_dir(repo / name, repo), f"{name} should be pruned"


def test_should_prune_dir_keeps_source_dirs(tmp_path: Path) -> None:
    repo = tmp_path
    for name in ("src", "app", "lib", "web"):
        assert not should_prune_dir(repo / name, repo), f"{name} should NOT be pruned"


def test_should_prune_dir_honors_caller_excludes(tmp_path: Path) -> None:
    repo = tmp_path
    excludes = frozenset({"vendored", "tests/fixtures"})
    assert should_prune_dir(repo / "vendored", repo, exclude_paths=excludes)
    assert should_prune_dir(repo / "tests" / "fixtures", repo, exclude_paths=excludes)
    assert not should_prune_dir(repo / "tests", repo, exclude_paths=excludes)


def test_should_prune_dir_respects_unignore(tmp_path: Path) -> None:
    repo = tmp_path
    # vendor is normally pruned, but an explicit unignore re-enables descent.
    assert should_prune_dir(repo / "vendor", repo)
    assert not should_prune_dir(
        repo / "vendor", repo, unignore_paths=frozenset({"vendor"})
    )
    # An ancestor of an unignored subtree must also remain walkable so the
    # whitelisted files are reachable.
    assert not should_prune_dir(
        repo / "vendor", repo, unignore_paths=frozenset({"vendor/keep/me"})
    )


def test_discovery_excludes_pruned_dirs(heavy_repo: Path) -> None:
    parsers, queries = load_parsers()
    updater = GraphUpdater(
        ingestor=object(),  # not used by _collect_eligible_files
        repo_path=heavy_repo,
        parsers=parsers,
        queries=queries,
    )

    eligible = updater._collect_eligible_files()
    rels = {p.relative_to(heavy_repo).as_posix() for p in eligible}

    # Real source is discovered.
    assert "src/main.py" in rels
    assert "README.md" in rels

    # Nothing under any pruned directory leaks in.
    for r in rels:
        assert not r.startswith("node_modules/"), r
        assert not r.startswith(".claude/"), r
        assert not r.startswith(".git/"), r

    # The buried-but-pruned source file is NOT discovered.
    assert "buried.py" not in {Path(r).name for r in rels}


def test_discovery_walk_is_bounded(heavy_repo: Path, monkeypatch) -> None:
    """The walk must never descend into pruned subtrees: spy on os.walk and
    assert no visited directory path falls under node_modules/.claude/.git."""
    parsers, queries = load_parsers()
    updater = GraphUpdater(
        ingestor=object(),
        repo_path=heavy_repo,
        parsers=parsers,
        queries=queries,
    )

    visited: list[str] = []
    real_walk = os.walk

    def _spy_walk(top, *args, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, *args, **kwargs):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr("codebase_rag.graph_updater.os.walk", _spy_walk)

    updater._collect_eligible_files()

    rel_visited = [Path(v).relative_to(heavy_repo).as_posix() for v in visited]
    for r in rel_visited:
        assert not r.startswith("node_modules"), f"descended into {r}"
        assert not r.startswith(".claude"), f"descended into {r}"
        assert not r.startswith(".git"), f"descended into {r}"

    # Sanity: the visited-dir count is tiny (repo root + src), proving we never
    # entered the ~4000-file noise subtrees.
    assert len(visited) <= 5, f"walk visited too many dirs: {rel_visited}"
