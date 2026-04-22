"""Real-time code-graph updater.

Watches a repository for filesystem changes and incrementally updates the
LadybugDB graph so queries stay current without a full re-index. Designed
to run as a long-lived sidecar (e.g. alongside the Code Indexer Service).

Flow:
    1. Parse parsers/queries for every supported language.
    2. Open a ``LadybugIngestor`` context manager against the target DB.
    3. Run an initial full scan so in-memory state (AST cache, function
       registry) matches the graph.
    4. Start a Watchdog observer that dispatches file events to
       ``CodeChangeEventHandler``.
    5. The handler performs a 5-step update per changed file:
       delete old nodes, clear in-memory state, re-parse if modified/
       created, recalculate call edges globally, flush.

Designed for correctness over throughput: we re-run the entire CALLS
resolution pass on every change because partial updates cannot detect
cross-file call additions/removals ("island problem").
"""
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from codebase_rag import cli_help as ch
from codebase_rag import logs
from codebase_rag import tool_errors as te
from codebase_rag.config import settings
from codebase_rag.constants import (
    CYPHER_DELETE_CALLS,
    CYPHER_DELETE_MODULE,
    IGNORE_PATTERNS,
    IGNORE_SUFFIXES,
    KEY_PATH,
    LOG_LEVEL_INFO,
    REALTIME_LOGGER_FORMAT,
    WATCHER_SLEEP_INTERVAL,
    EventType,
    SupportedLanguage,
)
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.language_spec import get_language_spec
from codebase_rag.parser_loader import load_parsers
from codebase_rag.services import QueryProtocol
from codebase_rag.services.ladybug_ingestor import LadybugIngestor


class CodeChangeEventHandler(FileSystemEventHandler):
    """Watchdog handler that propagates filesystem events to the graph.

    A single handler instance is shared across all events emitted by the
    observer. It holds a reference to the ``GraphUpdater`` so AST cache
    and structural state stay consistent across events.
    """

    def __init__(self, updater: GraphUpdater):
        """Initialise with the updater whose state will mirror the repo.

        Args:
            updater: The GraphUpdater driving parse/ingest passes.
        """
        self.updater = updater
        self.ignore_patterns = IGNORE_PATTERNS
        self.ignore_suffixes = IGNORE_SUFFIXES
        logger.info(logs.WATCHER_ACTIVE)

    def _is_relevant(self, path_str: str) -> bool:
        """Return True when the path should trigger a graph update.

        Filters out non-source files (editor swap files, build artefacts)
        and paths under ignored directories (.git, node_modules, etc.).

        Args:
            path_str: Filesystem path as reported by Watchdog.

        Returns:
            bool: True when the path survives both the suffix and the
            component-name ignore filters.
        """
        path = Path(path_str)
        if any(path.name.endswith(suffix) for suffix in self.ignore_suffixes):
            return False
        return all(part not in self.ignore_patterns for part in path.parts)

    def dispatch(self, event: FileSystemEvent) -> None:
        """Handle a single filesystem event end-to-end.

        Overrides Watchdog's default dispatch so we can implement the full
        5-step update sequence (see header diagram) atomically per event
        rather than fanning out to separate on_modified/on_created/on_deleted
        hooks.

        Args:
            event: The Watchdog event describing the change.
        """
        # (H) ┌─────────────────────────────────────────────────────────────────────┐
        # (H) │                      Real-Time Graph Update Steps                   │
        # (H) ├─────────────────────────────────────────────────────────────────────┤
        # (H) │ Step 1: Delete all old data from the graph for this file           │
        # (H) │         Provides a clean slate for the updated information         │
        # (H) │ Step 2: Clear the specific in-memory state for the file            │
        # (H) │         Prevents stale in-memory representations                   │
        # (H) │ Step 3: Re-parse the file if it was modified or created            │
        # (H) │         Rebuilds in-memory state (AST, function registry)          │
        # (H) │ Step 4: Re-process all function calls across the entire codebase   │
        # (H) │         Fixes "island" problem - changes reflect in all relations  │
        # (H) │ Step 5: Flush all collected changes to the database                │
        # (H) └─────────────────────────────────────────────────────────────────────┘
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()

        if event.is_directory or not self._is_relevant(src_path):
            return

        ingestor = self.updater.ingestor
        if not isinstance(ingestor, QueryProtocol):
            logger.warning(logs.WATCHER_SKIP_NO_QUERY)
            return

        path = Path(src_path)
        relative_path_str = str(path.relative_to(self.updater.repo_path))

        logger.warning(
            logs.CHANGE_DETECTED.format(event_type=event.event_type, path=path)
        )

        # (H) Step 1
        ingestor.execute_write(CYPHER_DELETE_MODULE, {KEY_PATH: relative_path_str})
        logger.debug(logs.DELETION_QUERY.format(path=relative_path_str))

        # (H) Step 2
        self.updater.remove_file_from_state(path)

        # (H) Step 3
        if event.event_type in (EventType.MODIFIED, EventType.CREATED):
            lang_config = get_language_spec(path.suffix)
            if (
                lang_config
                and isinstance(lang_config.language, SupportedLanguage)
                and lang_config.language in self.updater.parsers
            ):
                if result := self.updater.factory.definition_processor.process_file(
                    path,
                    lang_config.language,
                    self.updater.queries,
                    self.updater.factory.structure_processor.structural_elements,
                ):
                    root_node, language = result
                    self.updater.ast_cache[path] = (root_node, language)

        # (H) Step 4
        logger.info(logs.RECALC_CALLS)
        ingestor.execute_write(CYPHER_DELETE_CALLS)
        self.updater._process_function_calls()

        # (H) Step 5
        self.updater.ingestor.flush_all()
        logger.success(logs.GRAPH_UPDATED.format(name=path.name))


def start_watcher(
    repo_path: str, db_path: str, batch_size: int | None = None
) -> None:
    """Bootstrap the watcher: load parsers, open ingestor, run loop.

    Args:
        repo_path: Path to the repository to watch. Resolved to an absolute
            path so Watchdog's relative-path events can be normalised.
        db_path: LadybugDB file to write updates into.
        batch_size: Optional override for ingestor batch size. ``None``
            defers to the project's ``settings.resolve_batch_size``.
    """
    repo_path_obj = Path(repo_path).resolve()
    parsers, queries = load_parsers()

    effective_batch_size = settings.resolve_batch_size(batch_size)

    # LadybugIngestor as a context manager ensures the DB connection is
    # cleanly closed (and any buffered writes flushed) if the watcher
    # loop exits — normal shutdown or exception.
    with LadybugIngestor(
        db_path=db_path,
        batch_size=effective_batch_size,
    ) as ingestor:
        _run_watcher_loop(ingestor, repo_path_obj, parsers, queries)


def _run_watcher_loop(ingestor, repo_path_obj, parsers, queries):
    """Run the initial scan then block on the Watchdog observer.

    Args:
        ingestor: Connected LadybugIngestor used for all writes.
        repo_path_obj: Absolute ``Path`` to the repo root.
        parsers: Tree-sitter parsers keyed by language.
        queries: Tree-sitter queries keyed by language.
    """
    updater = GraphUpdater(ingestor, repo_path_obj, parsers, queries)

    # (H) Initial full scan builds the complete context for real-time updates
    logger.info(logs.INITIAL_SCAN)
    updater.run()
    logger.success(logs.INITIAL_SCAN_DONE)

    event_handler = CodeChangeEventHandler(updater)
    observer = Observer()
    observer.schedule(event_handler, str(repo_path_obj), recursive=True)
    observer.start()
    logger.info(logs.WATCHING.format(path=repo_path_obj))

    # The observer runs on its own thread; we block the main thread so
    # Ctrl-C can cleanly stop the observer and flush pending writes.
    try:
        while True:
            time.sleep(WATCHER_SLEEP_INTERVAL)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _validate_positive_int(value: int | None) -> int | None:
    """Typer callback — accept ``None`` or any integer ≥ 1.

    Args:
        value: The value parsed from the CLI flag.

    Returns:
        int | None: Unchanged if valid.

    Raises:
        typer.BadParameter: When value is a non-positive integer.
    """
    if value is None:
        return None
    if value < 1:
        raise typer.BadParameter(te.INVALID_POSITIVE_INT.format(value=value))
    return value


def main(
    repo_path: Annotated[str, typer.Argument(help=ch.HELP_REPO_PATH_WATCH)],
    db_path: Annotated[
        str, typer.Option(help="Path to the LadybugDB database file.")
    ] = settings.LADYBUG_DB_PATH,
    batch_size: Annotated[
        int | None,
        typer.Option(
            help=ch.HELP_BATCH_SIZE,
            callback=_validate_positive_int,
        ),
    ] = None,
) -> None:
    """Typer entry point — configure logging then launch the watcher.

    Args:
        repo_path: Repository to watch (positional).
        db_path: LadybugDB file; defaults to the value from project settings.
        batch_size: Optional batch size override (must be positive).
    """
    # Reset loguru handlers so repeat invocations (e.g. in tests) don't
    # double-log, then install the realtime formatter.
    logger.remove()
    logger.add(sys.stdout, format=REALTIME_LOGGER_FORMAT, level=LOG_LEVEL_INFO)
    logger.info(logs.LOGGER_CONFIGURED)
    start_watcher(repo_path, db_path, batch_size)


if __name__ == "__main__":
    typer.run(main)
