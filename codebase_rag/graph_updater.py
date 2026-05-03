import hashlib
import json
import sys
import time as _time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, ItemsView, KeysView
from pathlib import Path

from loguru import logger
from tree_sitter import Node, Parser

from . import constants as cs
from . import logs as ls
from .config import settings
from .cypher_queries import (
    CYPHER_DELETE_MODULE_DEFINES,
    CYPHER_DELETE_MODULE_METHODS,
    CYPHER_DELETE_MODULE_NODE,
    CYPHER_DELETE_ORPHAN_PACKAGES,
)
from .language_spec import LANGUAGE_FQN_SPECS, get_language_spec
from .parsers.factory import ProcessorFactory
from .services import IngestorProtocol, QueryProtocol
from .types_defs import (
    EmbeddingQueryResult,
    FunctionRegistry,
    LanguageQueries,
    NodeType,
    QualifiedName,
    ResultRow,
    SimpleNameLookup,
    TrieNode,
)
from .utils.dependencies import has_semantic_dependencies
from .utils.fqn_resolver import find_function_source_by_fqn
from .utils.path_utils import should_skip_path
from .utils.source_extraction import extract_source_with_fallback

type FileHashCache = dict[str, str]


class FunctionRegistryTrie:
    __slots__ = ("root", "_entries", "_simple_name_lookup")

    def __init__(self, simple_name_lookup: SimpleNameLookup | None = None) -> None:
        self.root: TrieNode = {}
        self._entries: FunctionRegistry = {}
        self._simple_name_lookup = simple_name_lookup

    def insert(self, qualified_name: QualifiedName, func_type: NodeType) -> None:
        self._entries[qualified_name] = func_type

        parts = qualified_name.split(cs.SEPARATOR_DOT)
        current: TrieNode = self.root

        for part in parts:
            if part not in current:
                current[part] = {}
            child = current[part]
            assert isinstance(child, dict)
            current = child

        current[cs.TRIE_TYPE_KEY] = func_type
        current[cs.TRIE_QN_KEY] = qualified_name

    def get(
        self, qualified_name: QualifiedName, default: NodeType | None = None
    ) -> NodeType | None:
        return self._entries.get(qualified_name, default)

    def __contains__(self, qualified_name: QualifiedName) -> bool:
        return qualified_name in self._entries

    def __getitem__(self, qualified_name: QualifiedName) -> NodeType:
        return self._entries[qualified_name]

    def __setitem__(self, qualified_name: QualifiedName, func_type: NodeType) -> None:
        self.insert(qualified_name, func_type)

    def __delitem__(self, qualified_name: QualifiedName) -> None:
        if qualified_name not in self._entries:
            return

        del self._entries[qualified_name]

        parts = qualified_name.split(cs.SEPARATOR_DOT)
        self._cleanup_trie_path(parts, self.root)

    def _cleanup_trie_path(self, parts: list[str], node: TrieNode) -> bool:
        if not parts:
            node.pop(cs.TRIE_QN_KEY, None)
            node.pop(cs.TRIE_TYPE_KEY, None)
            return not node

        part = parts[0]
        if part not in node:
            return False

        child = node[part]
        assert isinstance(child, dict)
        if self._cleanup_trie_path(parts[1:], child):
            del node[part]

        is_endpoint = cs.TRIE_QN_KEY in node
        has_children = any(not key.startswith(cs.TRIE_INTERNAL_PREFIX) for key in node)
        return not has_children and not is_endpoint

    def _navigate_to_prefix(self, prefix: str) -> TrieNode | None:
        parts = prefix.split(cs.SEPARATOR_DOT) if prefix else []
        current: TrieNode = self.root
        for part in parts:
            if part not in current:
                return None
            child = current[part]
            assert isinstance(child, dict)
            current = child
        return current

    def _collect_from_subtree(
        self,
        node: TrieNode,
        filter_fn: Callable[[QualifiedName], bool] | None = None,
    ) -> list[tuple[QualifiedName, NodeType]]:
        results: list[tuple[QualifiedName, NodeType]] = []

        def dfs(n: TrieNode) -> None:
            if cs.TRIE_QN_KEY in n:
                qn = n[cs.TRIE_QN_KEY]
                func_type = n[cs.TRIE_TYPE_KEY]
                assert isinstance(qn, str) and isinstance(func_type, NodeType)
                if filter_fn is None or filter_fn(qn):
                    results.append((qn, func_type))

            for key, child in n.items():
                if not key.startswith(cs.TRIE_INTERNAL_PREFIX):
                    assert isinstance(child, dict)
                    dfs(child)

        dfs(node)
        return results

    def keys(self) -> KeysView[QualifiedName]:
        return self._entries.keys()

    def items(self) -> ItemsView[QualifiedName, NodeType]:
        return self._entries.items()

    def __len__(self) -> int:
        return len(self._entries)

    def find_with_prefix_and_suffix(
        self, prefix: str, suffix: str
    ) -> list[QualifiedName]:
        node = self._navigate_to_prefix(prefix)
        if node is None:
            return []
        suffix_pattern = f".{suffix}"
        matches = self._collect_from_subtree(
            node, lambda qn: qn.endswith(suffix_pattern)
        )
        return [qn for qn, _ in matches]

    def find_ending_with(self, suffix: str) -> list[QualifiedName]:
        if self._simple_name_lookup is not None and suffix in self._simple_name_lookup:
            # (H) O(1) lookup using the simple_name_lookup index
            return list(self._simple_name_lookup[suffix])
        # (H) Fallback to linear scan if no index available
        return [qn for qn in self._entries.keys() if qn.endswith(f".{suffix}")]

    def find_with_prefix(self, prefix: str) -> list[tuple[QualifiedName, NodeType]]:
        node = self._navigate_to_prefix(prefix)
        return [] if node is None else self._collect_from_subtree(node)


class BoundedASTCache:
    __slots__ = ("cache", "max_entries", "max_memory_bytes")

    def __init__(
        self,
        max_entries: int | None = None,
        max_memory_mb: int | None = None,
    ):
        self.cache: OrderedDict[Path, tuple[Node, cs.SupportedLanguage]] = OrderedDict()
        self.max_entries = (
            max_entries if max_entries is not None else settings.CACHE_MAX_ENTRIES
        )
        max_mem = (
            max_memory_mb if max_memory_mb is not None else settings.CACHE_MAX_MEMORY_MB
        )
        self.max_memory_bytes = max_mem * cs.BYTES_PER_MB

    def __setitem__(self, key: Path, value: tuple[Node, cs.SupportedLanguage]) -> None:
        if key in self.cache:
            del self.cache[key]

        self.cache[key] = value

        self._enforce_limits()

    def __getitem__(self, key: Path) -> tuple[Node, cs.SupportedLanguage]:
        value = self.cache[key]
        self.cache.move_to_end(key)
        return value

    def __delitem__(self, key: Path) -> None:
        if key in self.cache:
            del self.cache[key]

    def __contains__(self, key: Path) -> bool:
        return key in self.cache

    def items(self) -> ItemsView[Path, tuple[Node, cs.SupportedLanguage]]:
        return self.cache.items()

    def _enforce_limits(self) -> None:
        while len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)  # (H) Remove least recently used

        if self._should_evict_for_memory():
            entries_to_remove = max(
                1, len(self.cache) // settings.CACHE_EVICTION_DIVISOR
            )
            for _ in range(entries_to_remove):
                if self.cache:
                    self.cache.popitem(last=False)

    def _should_evict_for_memory(self) -> bool:
        try:
            cache_size = sum(sys.getsizeof(v) for v in self.cache.values())
            return cache_size > self.max_memory_bytes
        except Exception:
            return (
                len(self.cache)
                > self.max_entries * settings.CACHE_MEMORY_THRESHOLD_RATIO
            )


def _hash_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_hash_cache(cache_path: Path) -> FileHashCache:
    if not cache_path.is_file():
        return {}
    try:
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            logger.info(ls.HASH_CACHE_LOADED, count=len(data), path=cache_path)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(ls.HASH_CACHE_LOAD_FAILED, path=cache_path, error=e)
    return {}


def _save_hash_cache(cache_path: Path, hashes: FileHashCache) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)
        logger.info(ls.HASH_CACHE_SAVED, count=len(hashes), path=cache_path)
    except OSError as e:
        logger.warning(ls.HASH_CACHE_SAVE_FAILED, path=cache_path, error=e)


class GraphUpdater:
    def __init__(
        self,
        ingestor: IngestorProtocol,
        repo_path: Path,
        parsers: dict[cs.SupportedLanguage, Parser],
        queries: dict[cs.SupportedLanguage, LanguageQueries],
        unignore_paths: frozenset[str] | None = None,
        exclude_paths: frozenset[str] | None = None,
        progress_cb: Callable[[dict], None] | None = None,
        skip_embeddings: bool = False,  # set True when caller handles embedding externally
    ):
        self.ingestor = ingestor
        self.skip_embeddings = skip_embeddings
        self._single_file: Path | None = None
        if repo_path.is_file():
            resolved = repo_path.resolve()
            self._single_file = resolved
            repo_path = resolved.parent
        self.repo_path = repo_path
        self.parsers = parsers
        self.queries = queries
        self.project_name = repo_path.resolve().name
        self.simple_name_lookup: SimpleNameLookup = defaultdict(set)
        self.function_registry = FunctionRegistryTrie(
            simple_name_lookup=self.simple_name_lookup
        )
        self.ast_cache = BoundedASTCache()
        self.unignore_paths = unignore_paths
        self.exclude_paths = exclude_paths
        self._progress_cb = progress_cb

        self.factory = ProcessorFactory(
            ingestor=self.ingestor,
            repo_path=self.repo_path,
            project_name=self.project_name,
            queries=self.queries,
            function_registry=self.function_registry,
            simple_name_lookup=self.simple_name_lookup,
            ast_cache=self.ast_cache,
            unignore_paths=self.unignore_paths,
            exclude_paths=self.exclude_paths,
        )

    def _emit_progress(self, event: dict) -> None:
        """Fire the progress callback safely — errors never abort indexing."""
        if self._progress_cb is not None:
            try:
                self._progress_cb(event)
            except Exception:
                # Re-raise only CancelledError-style signals; swallow everything else
                # so a buggy callback cannot break the index run.
                raise

    def _is_dependency_file(self, file_name: str, filepath: Path) -> bool:
        return (
            file_name.lower() in cs.DEPENDENCY_FILES
            or filepath.suffix.lower() == cs.CSPROJ_SUFFIX
        )

    def run(self, force: bool = False) -> None:
        self._emit_progress({"phase": "discovering"})
        self.ingestor.ensure_node_batch(
            cs.NODE_PROJECT, {cs.KEY_NAME: self.project_name}
        )
        logger.info(ls.ENSURING_PROJECT, name=self.project_name)

        logger.info(ls.PASS_1_STRUCTURE)
        self.factory.structure_processor.identify_structure()

        logger.info(ls.PASS_2_FILES)
        self._process_files(force=force)

        logger.info(ls.FOUND_FUNCTIONS, count=len(self.function_registry))
        logger.info(ls.PASS_3_CALLS)
        self._process_function_calls()

        self.factory.definition_processor.process_all_method_overrides()

        logger.info(ls.ANALYSIS_COMPLETE)
        self._emit_progress({"phase": "writing"})
        self.ingestor.flush_all()

        if settings.SKIP_EMBEDDINGS:
            logger.info("Embedding pass skipped (SKIP_EMBEDDINGS=true)")
            self._emit_progress({"phase": "finalizing", "progress_pct": 98.0})
        else:
            self._generate_semantic_embeddings()

        self._emit_progress({"phase": "done", "progress_pct": 100.0})

    def remove_file_from_state(self, file_path: Path) -> None:
        logger.debug(ls.REMOVING_STATE, path=file_path)

        if file_path in self.ast_cache:
            del self.ast_cache[file_path]
            logger.debug(ls.REMOVED_FROM_CACHE)

        relative_path = file_path.relative_to(self.repo_path)
        path_parts = (
            relative_path.parent.parts
            if file_path.name == cs.INIT_PY
            else relative_path.with_suffix("").parts
        )
        module_qn_prefix = cs.SEPARATOR_DOT.join([self.project_name, *path_parts])

        qns_to_remove = set()

        for qn in list(self.function_registry.keys()):
            if qn.startswith(f"{module_qn_prefix}.") or qn == module_qn_prefix:
                qns_to_remove.add(qn)
                del self.function_registry[qn]

        if qns_to_remove:
            logger.debug(ls.REMOVING_QNS, count=len(qns_to_remove))

        for simple_name, qn_set in self.simple_name_lookup.items():
            original_count = len(qn_set)
            new_qn_set = qn_set - qns_to_remove
            if len(new_qn_set) < original_count:
                self.simple_name_lookup[simple_name] = new_qn_set
                logger.debug(ls.CLEANED_SIMPLE_NAME, name=simple_name)

        # ------------------------------------------------------------------
        # Graph DB cleanup — remove Module and its descendants, then prune
        # any Package nodes that have become orphans.
        # Skipped gracefully when the ingestor does not support writes (e.g.
        # in unit tests using a stub ingestor).
        # ------------------------------------------------------------------
        if hasattr(self.ingestor, "execute_write"):
            params = {"qn": module_qn_prefix}
            try:
                self.ingestor.execute_write(CYPHER_DELETE_MODULE_METHODS, params)
                self.ingestor.execute_write(CYPHER_DELETE_MODULE_DEFINES, params)
                self.ingestor.execute_write(CYPHER_DELETE_MODULE_NODE, params)
                self.ingestor.execute_write(CYPHER_DELETE_ORPHAN_PACKAGES, {})
                logger.debug(
                    "Removed graph nodes for module %s and pruned orphan packages",
                    module_qn_prefix,
                )
            except Exception as exc:
                logger.warning(
                    "graph_updater: could not remove graph nodes for %s: %s",
                    module_qn_prefix,
                    exc,
                )

    def _collect_eligible_files(self) -> list[Path]:
        if self._single_file is not None:
            if not should_skip_path(
                self._single_file,
                self.repo_path,
                exclude_paths=self.exclude_paths,
                unignore_paths=self.unignore_paths,
            ):
                return [self._single_file]
            return []

        eligible: list[Path] = []
        for filepath in self.repo_path.rglob("*"):
            try:
                if (
                    filepath.is_file()
                    and filepath.name != cs.HASH_CACHE_FILENAME
                    and not should_skip_path(
                        filepath,
                        self.repo_path,
                        exclude_paths=self.exclude_paths,
                        unignore_paths=self.unignore_paths,
                    )
                ):
                    eligible.append(filepath)
            except (UnicodeDecodeError, ValueError, OSError) as exc:
                # Filenames with non-UTF-8 bytes produce surrogate-escaped
                # Path objects on Linux; str() on them raises UnicodeDecodeError.
                # OSError can occur when the file disappears mid-scan.
                logger.warning(
                    "Skipping file with unreadable path during scan: %s", exc
                )
        return eligible

    def _process_files(self, force: bool = False) -> None:
        cache_path = self.repo_path / cs.HASH_CACHE_FILENAME
        old_hashes = _load_hash_cache(cache_path) if not force else {}
        if force:
            logger.info(ls.INCREMENTAL_FORCE)

        eligible_files = self._collect_eligible_files()
        # Emit discovery completion with total file count.
        self._emit_progress({"phase": "discovering", "files_total": len(eligible_files)})

        new_hashes: FileHashCache = {}
        skipped_count = 0
        changed_count = 0

        current_file_keys: set[str] = set()

        processed_since_flush = 0
        _total_files = len(eligible_files)
        _files_scanned = 0  # all files seen (including skipped)
        _last_cb_time = _time.monotonic()
        _files_since_cb = 0

        for filepath in eligible_files:
            file_key = str(filepath.relative_to(self.repo_path))
            current_file_keys.add(file_key)
            _files_scanned += 1

            current_hash = _hash_file(filepath)
            new_hashes[file_key] = current_hash

            if (
                not force
                and file_key in old_hashes
                and old_hashes[file_key] == current_hash
            ):
                logger.debug(ls.FILE_HASH_UNCHANGED, path=file_key)
                skipped_count += 1
                _files_since_cb += 1
                # Still emit progress so the bar moves even on skipped files.
                _now = _time.monotonic()
                if _files_since_cb >= 10 or (_now - _last_cb_time) >= 0.5:
                    self._emit_progress({
                        "phase": "parsing",
                        "files_done": _files_scanned,
                        "current_file": file_key,
                    })
                    _last_cb_time = _now
                    _files_since_cb = 0
                continue

            if file_key in old_hashes:
                logger.debug(ls.FILE_HASH_CHANGED, path=file_key)
                self.remove_file_from_state(filepath)
            else:
                logger.debug(ls.FILE_HASH_NEW, path=file_key)

            changed_count += 1
            self._process_single_file(filepath)
            _files_since_cb += 1

            _now = _time.monotonic()
            if _files_since_cb >= 10 or (_now - _last_cb_time) >= 0.5:
                self._emit_progress({
                    "phase": "parsing",
                    "files_done": _files_scanned,
                    "current_file": file_key,
                })
                _last_cb_time = _now
                _files_since_cb = 0

            processed_since_flush += 1
            if processed_since_flush >= settings.FILE_FLUSH_INTERVAL:
                logger.info(ls.PERIODIC_FLUSH.format(count=processed_since_flush))
                self.ingestor.flush_all()
                processed_since_flush = 0

        deleted_keys = set(old_hashes.keys()) - current_file_keys
        if deleted_keys:
            logger.info(ls.INCREMENTAL_DELETED, count=len(deleted_keys))
            for deleted_key in deleted_keys:
                deleted_path = self.repo_path / deleted_key
                self.remove_file_from_state(deleted_path)

        if skipped_count > 0:
            logger.info(ls.INCREMENTAL_SKIPPED, count=skipped_count)
        if changed_count > 0:
            logger.info(ls.INCREMENTAL_CHANGED, count=changed_count)

        _save_hash_cache(cache_path, new_hashes)

    def _process_single_file(self, filepath: Path) -> None:
        lang_config = get_language_spec(filepath.suffix)
        if (
            lang_config
            and isinstance(lang_config.language, cs.SupportedLanguage)
            and lang_config.language in self.parsers
        ):
            result = self.factory.definition_processor.process_file(
                filepath,
                lang_config.language,
                self.queries,
                self.factory.structure_processor.structural_elements,
            )
            if result:
                root_node, language = result
                self.ast_cache[filepath] = (root_node, language)
        elif self._is_dependency_file(filepath.name, filepath):
            self.factory.definition_processor.process_dependencies(filepath)

        self.factory.structure_processor.process_generic_file(filepath, filepath.name)

    def _process_function_calls(self) -> None:
        ast_cache_items = list(self.ast_cache.items())
        for file_path, (root_node, language) in ast_cache_items:
            self.factory.call_processor.process_calls_in_file(
                file_path, root_node, language, self.queries
            )

    def _generate_semantic_embeddings(self) -> None:
        if self.skip_embeddings:
            # Caller (e.g. code-indexer-service) handles embedding in a separate
            # subprocess that writes to the per-repo ``.duck`` file (DuckDB —
            # v5.3 §6.5 + §8.4).  Running the in-process embedding path here
            # would cause double-embedding and write to a store that is no
            # longer consulted at query time.
            logger.debug("Skipping built-in embedding pass (handled by caller)")
            return

        if not has_semantic_dependencies():
            logger.info(ls.SEMANTIC_NOT_AVAILABLE)
            return

        if not isinstance(self.ingestor, QueryProtocol):
            logger.info(ls.INGESTOR_NO_QUERY)
            return

        try:
            from .embedder import (
                embed_code,
                get_embedding_cache,
                get_lm_studio_embedder,
            )
            from .vector_store import (
                flush_embeddings,
                store_embedding_batch,
                verify_stored_ids,
            )

            logger.info(ls.PASS_4_EMBEDDINGS)

            results = self.ingestor.fetch_all(
                cs.CYPHER_QUERY_EMBEDDINGS, {"project_name": self.project_name}
            )

            if not results:
                logger.info(ls.NO_FUNCTIONS_FOR_EMBEDDING)
                return

            logger.info(ls.GENERATING_EMBEDDINGS, count=len(results))
            self._emit_progress({"phase": "embedding", "files_total": len(results)})

            # ------------------------------------------------------------------
            # Pass 1 — collect eligible symbols (source extraction + triviality
            # filter).  We do this before touching the embedder so we can hand
            # the full list of texts to batch_embed in one shot.
            # ------------------------------------------------------------------
            import re as _re_anon

            eligible: list[tuple[str, str, str]] = []  # (node_id, qname, embed_text)

            for row in results:
                parsed = self._parse_embedding_result(row)
                if parsed is None:
                    continue

                node_id = parsed[cs.KEY_NODE_ID]
                qualified_name = parsed[cs.KEY_QUALIFIED_NAME]
                start_line = parsed.get(cs.KEY_START_LINE)
                end_line = parsed.get(cs.KEY_END_LINE)
                file_path = parsed.get(cs.KEY_PATH)
                docstring = parsed.get(cs.KEY_DOCSTRING)

                if start_line is None or end_line is None or file_path is None:
                    logger.debug(ls.NO_SOURCE_FOR, name=qualified_name)
                    continue

                source_code = self._extract_source_code(
                    qualified_name, file_path, start_line, end_line
                )
                if not source_code:
                    logger.debug(ls.NO_SOURCE_FOR, name=qualified_name)
                    continue

                # Skip trivial functions — inline arrows, 1-line setters,
                # stub/mock helpers, empty-state placeholders. Their
                # embeddings are near-uniform and crowd real application
                # code out of top-k results for any natural-language
                # query. Thresholds:
                #   <150 non-whitespace chars OR <5 non-blank lines →
                #     too trivial (covers `emptyTask`-style React
                #     placeholders, one-line setters, stubs)
                #   anonymous_LINE_COL pattern → tree-sitter parser
                #     fallback for unnamed callbacks (always trivial)
                #   duplicated trailing segments (e.g.
                #     `useHook.useHook.connect.connect`) → inner-scope
                #     closure wrappers the parser emits; always trivial
                _stripped = "".join(source_code.split())
                _nonblank_lines = [
                    ln for ln in source_code.splitlines() if ln.strip()
                ]
                _is_anon = bool(
                    _re_anon.search(r"\banonymous_\d+_\d+\b", qualified_name)
                )
                _parts = qualified_name.split(".")
                _dup_tail = len(_parts) >= 2 and _parts[-1] == _parts[-2]
                if (
                    _is_anon
                    or _dup_tail
                    or len(_stripped) < 150
                    or len(_nonblank_lines) < 5
                ):
                    logger.debug(
                        "Skipping trivial function embedding",
                        name=qualified_name,
                        chars=len(_stripped),
                        lines=len(_nonblank_lines),
                    )
                    continue

                embed_text = self._build_embed_text(source_code, docstring)
                eligible.append((node_id, qualified_name, embed_text))

            if not eligible:
                logger.info(ls.NO_FUNCTIONS_FOR_EMBEDDING)
                return

            # ------------------------------------------------------------------
            # Pass 2 — embed.  Prefer LM Studio batched HTTP when available;
            # fall back to the in-process torch path (per-symbol) otherwise.
            # ------------------------------------------------------------------
            embedded_count = 0
            expected_ids: set[str] = set()
            batch_buffer: list[tuple[str, list[float], str]] = []
            batch_size = settings.VECTOR_BATCH_SIZE
            _total_to_embed = max(len(eligible), 1)
            _emit_interval = 50

            lm_embedder = get_lm_studio_embedder()

            if lm_embedder is not None:
                # Batched LM Studio path — single HTTP request per N symbols.
                logger.info(
                    "embedding.lm_studio.batched count=%d batch_size=%d",
                    len(eligible),
                    cs.LM_STUDIO_EMBED_BATCH_SIZE,
                )
                texts = [et for _, _, et in eligible]
                embeddings = lm_embedder.batch_embed(
                    texts, prefix=cs.CODERANK_CODE_PREFIX
                )
                if embeddings is not None:
                    for (node_id, qualified_name, _), embedding in zip(eligible, embeddings):
                        batch_buffer.append((node_id, embedding, qualified_name))
                        expected_ids.add(node_id)
                        if len(batch_buffer) >= batch_size:
                            embedded_count += store_embedding_batch(batch_buffer)
                            batch_buffer = []
                    if batch_buffer:
                        embedded_count += store_embedding_batch(batch_buffer)
                        batch_buffer = []
                    logger.info(
                        "embedding.lm_studio.done embedded=%d", embedded_count
                    )
                else:
                    # LM Studio batch failed — fall through to per-symbol torch path.
                    logger.warning(
                        "LM Studio batch_embed returned None; falling back to in-process embedder"
                    )
                    lm_embedder = None

            if lm_embedder is None:
                # Sequential in-process torch path (original behaviour, preserved).
                for _embed_idx, (node_id, qualified_name, embed_text) in enumerate(eligible):
                    try:
                        embedding = embed_code(embed_text)
                        batch_buffer.append((node_id, embedding, qualified_name))
                        expected_ids.add(node_id)

                        if len(batch_buffer) >= batch_size:
                            embedded_count += store_embedding_batch(batch_buffer)
                            batch_buffer = []

                        if (
                            embedded_count % settings.EMBEDDING_PROGRESS_INTERVAL == 0
                            and embedded_count > 0
                        ):
                            logger.debug(
                                ls.EMBEDDING_PROGRESS,
                                done=embedded_count,
                                total=len(eligible),
                            )

                        if (_embed_idx + 1) % _emit_interval == 0:
                            _pct = 70.0 + ((_embed_idx + 1) / _total_to_embed) * 28.0
                            self._emit_progress({
                                "phase": "embedding",
                                "files_done": _embed_idx + 1,
                                "progress_pct": min(97.9, _pct),
                            })

                    except Exception as e:
                        logger.warning(
                            ls.EMBEDDING_FAILED, name=qualified_name, error=e
                        )

                if batch_buffer:
                    embedded_count += store_embedding_batch(batch_buffer)

            # Persist all in-memory embeddings to disk (numpy files).
            db_path = str(self.ingestor._db_path) if hasattr(self.ingestor, "_db_path") else None
            flush_embeddings(db_path=db_path)

            logger.info(ls.EMBEDDINGS_COMPLETE, count=embedded_count)

            self._reconcile_embeddings(
                expected_ids,
                lambda ids: verify_stored_ids(ids, db_path=db_path),
            )

            get_embedding_cache().save()

        except Exception as e:
            logger.warning(ls.EMBEDDING_GENERATION_FAILED, error=e)

    def _reconcile_embeddings(
        self,
        expected_ids: set[str],
        verify_fn: Callable[[set[str]], set[str]],
    ) -> None:
        if not expected_ids:
            return
        try:
            stored_ids = verify_fn(expected_ids)
            missing = expected_ids - stored_ids
            if missing:
                sample = sorted(missing)[:10]
                logger.warning(
                    ls.EMBEDDING_RECONCILE_MISSING.format(
                        missing=len(missing),
                        expected=len(expected_ids),
                        sample_ids=sample,
                    )
                )
            else:
                logger.info(ls.EMBEDDING_RECONCILE_OK.format(count=len(expected_ids)))
        except Exception as e:
            logger.warning(ls.EMBEDDING_RECONCILE_FAILED.format(error=e))

    def _extract_source_code(
        self, qualified_name: str, file_path: str, start_line: int, end_line: int
    ) -> str | None:
        if not file_path or not start_line or not end_line:
            return None

        file_path_obj = self.repo_path / file_path

        ast_extractor = None
        if file_path_obj in self.ast_cache:
            root_node, language = self.ast_cache[file_path_obj]
            fqn_config = LANGUAGE_FQN_SPECS.get(language)

            if fqn_config:

                def ast_extractor_func(qname: str, path: Path) -> str | None:
                    return find_function_source_by_fqn(
                        root_node,
                        qname,
                        path,
                        self.repo_path,
                        self.project_name,
                        fqn_config,
                    )

                ast_extractor = ast_extractor_func

        return extract_source_with_fallback(
            file_path_obj, start_line, end_line, qualified_name, ast_extractor
        )

    @staticmethod
    def _build_embed_text(source_code: str, docstring: str | None) -> str:
        """Build the text sent to the embedder for a single symbol.

        The subprocess driver in code-indexer-service uses a richer version of
        this that includes qualified_name, module, and caller-count headers
        (Plan H).  This fallback version is used by the legacy in-process path
        only.
        """
        if docstring:
            return docstring.strip() + "\n# ---\n" + source_code
        return source_code

    def _parse_embedding_result(self, row: ResultRow) -> EmbeddingQueryResult | None:
        node_id = row.get(cs.KEY_NODE_ID)
        qualified_name = row.get(cs.KEY_QUALIFIED_NAME)

        # LadybugDB returns qualified_name as node_id (no integer id(n)).
        if not isinstance(node_id, str) or not isinstance(qualified_name, str):
            return None

        start_line = row.get(cs.KEY_START_LINE)
        end_line = row.get(cs.KEY_END_LINE)
        file_path = row.get(cs.KEY_PATH)
        docstring = row.get(cs.KEY_DOCSTRING)

        docstring = row.get(cs.KEY_DOCSTRING)
        return EmbeddingQueryResult(
            node_id=node_id,
            qualified_name=qualified_name,
            start_line=start_line if isinstance(start_line, int) else None,
            end_line=end_line if isinstance(end_line, int) else None,
            path=file_path if isinstance(file_path, str) else None,
            docstring=docstring if isinstance(docstring, str) else None,
        )
