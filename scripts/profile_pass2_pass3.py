"""BUC-1614 — Pass 2 vs Pass 3 wall-clock profiler.

Usage:
    PYTHONPATH=. python scripts/profile_pass2_pass3.py <repo_path>

Runs GraphUpdater with a no-op ingestor and measures the wall-clock for
Pass 2 (_process_files) and Pass 3 (_process_function_calls). Reports
ratio and recommends which pass to parallelise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Disable embeddings — irrelevant to Pass 2/3 comparison.
import os
os.environ.setdefault("SKIP_EMBEDDINGS", "true")

from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers


class NoopIngestor:
    """In-memory no-op ingestor — counts writes but does no I/O."""

    def __init__(self) -> None:
        self.nodes = 0
        self.rels = 0

    def ensure_node_batch(self, label: str, properties: Any) -> None:
        self.nodes += 1

    def ensure_relationship_batch(
        self, from_spec: Any, rel_type: str, to_spec: Any, properties: Any = None
    ) -> None:
        self.rels += 1

    def flush_all(self) -> None:
        pass

    def fetch_all(self, query: str, params: Any = None) -> list:
        return []

    def execute_write(self, query: str, params: Any = None) -> None:
        pass

    def ensure_constraints(self) -> None:
        pass

    def clean_database(self) -> None:
        pass


def profile(repo_path: Path, label: str = "") -> dict[str, float]:
    parsers, queries = load_parsers()
    ingestor = NoopIngestor()
    updater = GraphUpdater(
        ingestor=ingestor,  # type: ignore[arg-type]
        repo_path=repo_path,
        parsers=parsers,
        queries=queries,
    )

    # Pass 1 — Structure (small)
    t0 = time.monotonic()
    updater.factory.structure_processor.identify_structure()
    pass1 = time.monotonic() - t0

    # Pass 2 — Parse + definitions (force=True to bypass hash cache)
    t0 = time.monotonic()
    updater._process_files(force=True)
    pass2 = time.monotonic() - t0

    # Rebind discovery (Pass 2.5)
    t0 = time.monotonic()
    updater._discover_method_rebindings()
    pass25 = time.monotonic() - t0

    # Pass 3 — Call resolution
    t0 = time.monotonic()
    updater._process_function_calls()
    pass3 = time.monotonic() - t0

    return {
        "pass1_structure_s": pass1,
        "pass2_files_s": pass2,
        "pass2_5_rebind_s": pass25,
        "pass3_calls_s": pass3,
        "ratio_p2_over_p3": pass2 / pass3 if pass3 > 0 else float("inf"),
        "functions_found": len(updater.function_registry),
        "ast_cache_size": sum(1 for _ in updater.ast_cache.items()),
        "nodes_written": ingestor.nodes,
        "rels_written": ingestor.rels,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: profile_pass2_pass3.py <repo_path>", file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    if not repo.is_dir():
        print(f"not a directory: {repo}", file=sys.stderr)
        return 2

    print(f"Profiling: {repo}")
    metrics = profile(repo)
    print("\n=== BUC-1614 phase timings ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:>12.3f}")
        else:
            print(f"  {k:30s} {v:>12d}")
    print()
    if metrics["ratio_p2_over_p3"] > 2.0:
        print("DECISION: Pass 2 > 2x Pass 3 -> parallelise Pass 2 (proceed).")
    elif metrics["ratio_p2_over_p3"] < 0.5:
        print("DECISION: Pass 3 >> Pass 2 -> pivot, parallelise Pass 3.")
    else:
        print("DECISION: ratio in [0.5, 2.0] -> Pass 2 is still the larger absolute target; parallelise Pass 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
