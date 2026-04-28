from fnmatch import fnmatch
from pathlib import Path

from .. import constants as cs


def should_skip_path(
    path: Path,
    repo_path: Path,
    exclude_paths: frozenset[str] | None = None,
    unignore_paths: frozenset[str] | None = None,
) -> bool:
    if path.is_file() and path.suffix in cs.IGNORE_SUFFIXES:
        return True
    rel_path = path.relative_to(repo_path)
    rel_path_str = rel_path.as_posix()
    dir_parts = rel_path.parent.parts if path.is_file() else rel_path.parts

    # Check directory-based exclusions FIRST. A filename unignore (e.g.
    # `!.env.example`) must NOT resurrect files that live inside a directory
    # that's been excluded wholesale (e.g. `.claude/worktrees` or `node_modules`).
    # Without this ordering, a safe-template basename would pull worktree
    # copies of itself back into the index.
    dir_excluded = bool(exclude_paths) and (
        not exclude_paths.isdisjoint(dir_parts)
        or rel_path_str in exclude_paths
        or any(rel_path_str.startswith(f"{p}/") for p in exclude_paths)
    )
    if dir_excluded:
        return True

    # Basename / glob unignore: re-enable files that a basename exclude
    # would otherwise catch (e.g. `!.env.example` overriding `.env.*`).
    if unignore_paths and path.is_file() and any(
        fnmatch(path.name, p) for p in unignore_paths
    ):
        return False

    # For files, let a bare filename / glob pattern (e.g. `.DS_Store`,
    # `.env`, `.env.*`, `*.pem`) in .cgrignore match the file anywhere in
    # the tree — gitignore-style basename + glob matching.
    if (
        exclude_paths
        and path.is_file()
        and any(fnmatch(path.name, p) for p in exclude_paths)
    ):
        return True

    # Path-based unignore (e.g. `!vendor/mylib`) — whitelists a subtree that
    # the built-in IGNORE_PATTERNS would otherwise skip.
    if unignore_paths and any(
        rel_path_str == p or rel_path_str.startswith(f"{p}/") for p in unignore_paths
    ):
        return False

    return not cs.IGNORE_PATTERNS.isdisjoint(dir_parts)
