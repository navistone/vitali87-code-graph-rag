"""BUC-1614 property tests: parallel Pass 3 must produce identical
ingestor writes to the serial path, modulo a stable replay order.

We don't compare ``function_registry`` directly because Pass 3 doesn't
mutate it — the registry is read-only by then. What we DO compare is
the multiset of ingestor calls (node + relationship batches) emitted
during call resolution, which is what actually lands in the graph.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers


class RecordingIngestor:
    """Captures every ingestor call as a normalised tuple."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def ensure_node_batch(self, label: str, properties: Any) -> None:
        # tuple-ise property dict for hashability
        self.calls.append(
            ("node", (label, tuple(sorted((properties or {}).items()))))
        )

    def ensure_relationship_batch(
        self,
        from_spec: Any,
        rel_type: str,
        to_spec: Any,
        properties: Any = None,
    ) -> None:
        self.calls.append(
            (
                "rel",
                (
                    tuple(from_spec),
                    rel_type,
                    tuple(to_spec),
                    tuple(sorted((properties or {}).items())),
                ),
            )
        )

    def flush_all(self) -> None:
        pass

    def fetch_all(self, query: str, params: Any = None) -> list:
        return []

    def execute_write(self, query: str, params: Any = None) -> None:
        pass

    def ensure_constraints(self) -> None:
        pass


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Small Python-only fixture with enough cross-file calls to make
    Pass 3 do real work."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text(
        textwrap.dedent(
            """
            from .b import helper, Service

            def greet(name: str) -> str:
                return helper(name)

            def use_service() -> None:
                svc = Service()
                svc.run()
                svc.run()
            """
        ).strip()
    )
    (tmp_path / "pkg" / "b.py").write_text(
        textwrap.dedent(
            """
            def helper(x: str) -> str:
                return x.upper()

            class Service:
                def run(self) -> None:
                    helper("ok")

                def shutdown(self) -> None:
                    self.run()
            """
        ).strip()
    )
    (tmp_path / "pkg" / "c.py").write_text(
        textwrap.dedent(
            """
            from .a import greet, use_service
            from .b import Service

            def entry() -> None:
                greet("world")
                use_service()
                Service().shutdown()
            """
        ).strip()
    )
    return tmp_path


def _run_pass3(repo: Path, parallelism: int | None) -> list[tuple[str, tuple[Any, ...]]]:
    """Run Pass 1 + 2 + 3 against ``repo`` and return the call sequence
    that *only* Pass 3 emitted."""
    if parallelism is None:
        os.environ.pop("PARSE_PARALLELISM", None)
    else:
        os.environ["PARSE_PARALLELISM"] = str(parallelism)

    parsers, queries = load_parsers()
    ingestor = RecordingIngestor()
    updater = GraphUpdater(
        ingestor=ingestor,  # type: ignore[arg-type]
        repo_path=repo,
        parsers=parsers,
        queries=queries,
    )

    updater.factory.structure_processor.identify_structure()
    updater._process_files(force=True)
    updater._discover_method_rebindings()

    # Snapshot Pass 1+2 writes so we can subtract them.
    before_pass3 = len(ingestor.calls)
    updater._process_function_calls()
    return ingestor.calls[before_pass3:]


def _multiset(calls: list[tuple[str, tuple[Any, ...]]]) -> list[str]:
    """Normalise a call sequence to a sorted multiset of repr strings.

    Pass 3 has a pre-existing ordering nondeterminism in the legacy
    serial path (running the same input twice can permute pairs of
    writes within a single file, e.g. ``Service()`` vs ``svc.run()``
    inside one function — verified during BUC-1614 spike). What matters
    for graph correctness is the multiset of writes, not the order in
    which CallProcessor happens to emit them.
    """
    return sorted(repr(c) for c in calls)


def test_should_produce_same_writes_when_serial_vs_parallel_pass3(
    sample_repo: Path,
) -> None:
    """The multiset of ingestor writes from Pass 3 must be identical
    whether we use the legacy serial path or the parallel pool."""
    serial_calls = _run_pass3(sample_repo, parallelism=1)
    parallel_calls = _run_pass3(sample_repo, parallelism=4)

    assert _multiset(serial_calls) == _multiset(parallel_calls), (
        f"serial vs parallel write set diverged: "
        f"serial={len(serial_calls)} parallel={len(parallel_calls)}"
    )


def test_should_match_legacy_when_parallelism_unset(sample_repo: Path) -> None:
    """PARSE_PARALLELISM unset must mean serial (default=1, BUC-1614).

    Both runs take the legacy code path, so the multiset of writes is
    identical. (Exact ordering can differ run-to-run due to a pre-existing
    intra-file nondeterminism — verified during the BUC-1614 spike.)
    """
    default_calls = _run_pass3(sample_repo, parallelism=None)
    serial_calls = _run_pass3(sample_repo, parallelism=1)
    assert _multiset(default_calls) == _multiset(serial_calls)


def test_should_be_stable_when_run_twice_in_parallel(
    sample_repo: Path,
) -> None:
    """Parallel mode write-set must be stable across runs (the underlying
    serial path has a known intra-file pair-swap nondeterminism that we
    inherit, but the total set of writes is invariant)."""
    a = _run_pass3(sample_repo, parallelism=4)
    b = _run_pass3(sample_repo, parallelism=4)
    assert _multiset(a) == _multiset(b), (
        "parallel pass produced different write set across two runs"
    )


def test_should_emit_some_calls_relationship_when_pass3_runs(
    sample_repo: Path,
) -> None:
    """Sanity: the fixture has cross-file calls, so Pass 3 must emit
    at least one CALLS relationship. Guards against a silent no-op
    regression."""
    calls = _run_pass3(sample_repo, parallelism=2)
    rel_types = {c[1][1] for c in calls if c[0] == "rel"}
    assert any("CALLS" in rt for rt in rel_types), (
        f"expected CALLS-* relationship from Pass 3; got rel types: {rel_types}"
    )
