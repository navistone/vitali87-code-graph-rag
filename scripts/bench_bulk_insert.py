"""Bench harness for DuckDB bulk_insert paths (Phase 4 decision input).

Measures median wall-clock for the existing executemany-based ``bulk_insert``
and (optionally) the experimental Arrow-staged ``bulk_insert_arrow``.

Reports:
    rows | median_ms | ms_per_row | MB_per_s_payload

Where ``MB_per_s_payload`` treats the embeddings as raw FLOAT32 bytes
(4 B per element x 768 dims x N rows), which is the lower bound on the
work the SQL layer has to push.

Usage::

    python scripts/bench_bulk_insert.py            # bulk_insert only
    python scripts/bench_bulk_insert.py --arrow    # both paths + speedup

Stdlib + numpy + duckdb only (no pytest, no third-party harness).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Make the repo importable when run from the scripts/ dir.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codebase_rag.storage.vector_store import (  # noqa: E402
    EmbeddingRow,
    bulk_insert,
    open_or_create,
)

_DIM = 768
_TRIALS = 3
_SIZES = (100, 500, 1000, 5000, 10000)


def _make_rows(n: int, seed: int = 0) -> list[EmbeddingRow]:
    """Generate ``n`` synthetic rows with random unit vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, _DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    units = (raw / norms).tolist()
    return [
        EmbeddingRow(
            qualified_name=f"bench.fn_{i}",
            embedding=units[i],
            file_path=f"/repo/m_{i % 64}.py",
            start_line=i,
            end_line=i + 5,
            symbol_type="Function",
        )
        for i in range(n)
    ]


def _time_once(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000.0


def _fresh_conn(tmp: Path, label: str, trial: int):
    db = tmp / f"{label}_{trial}.duck"
    if db.exists():
        db.unlink()
    return open_or_create(str(db))


def _bench_path(label: str, fn, sizes: tuple[int, ...]) -> dict[int, float]:
    """Run ``fn(conn, rows)`` against fresh DBs at each size; return median ms."""
    out: dict[int, float] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for size in sizes:
            samples: list[float] = []
            rows = _make_rows(size, seed=size)
            for trial in range(_TRIALS):
                conn = _fresh_conn(tmp, label, trial)
                try:
                    samples.append(_time_once(fn, conn, rows))
                finally:
                    conn.close()
            out[size] = statistics.median(samples)
            print(
                f"  [{label}] size={size:>5}  trials={samples}",
                file=sys.stderr,
            )
    return out


def _payload_mb_per_s(rows: int, ms: float) -> float:
    bytes_total = 4 * _DIM * rows
    seconds = ms / 1000.0
    if seconds <= 0:
        return float("inf")
    return (bytes_total / (1024 * 1024)) / seconds


def _print_table_single(results: dict[int, float]) -> None:
    print()
    print(f"{'rows':>6} | {'median_ms':>10} | {'ms_per_row':>11} | {'MB/s_payload':>13}")
    print("-" * 50)
    for size in sorted(results):
        ms = results[size]
        per = ms / size
        mbs = _payload_mb_per_s(size, ms)
        print(f"{size:>6} | {ms:>10.2f} | {per:>11.4f} | {mbs:>13.2f}")


def _print_table_dual(
    bulk: dict[int, float], arrow: dict[int, float]
) -> None:
    print()
    header = (
        f"{'rows':>6} | {'bulk_ms':>9} | {'bulk_ms/row':>11} | "
        f"{'bulk_MB/s':>9} | {'arrow_ms':>9} | {'arrow_ms/row':>12} | "
        f"{'arrow_MB/s':>10} | {'speedup':>7}"
    )
    print(header)
    print("-" * len(header))
    for size in sorted(bulk):
        b = bulk[size]
        a = arrow[size]
        bpr = b / size
        apr = a / size
        bmb = _payload_mb_per_s(size, b)
        amb = _payload_mb_per_s(size, a)
        speed = b / a if a > 0 else float("inf")
        print(
            f"{size:>6} | {b:>9.2f} | {bpr:>11.4f} | {bmb:>9.2f} | "
            f"{a:>9.2f} | {apr:>12.4f} | {amb:>10.2f} | {speed:>7.2f}x"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arrow",
        action="store_true",
        help="Also run the experimental Arrow-staged path.",
    )
    p.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in _SIZES),
        help="Comma-separated sizes to bench.",
    )
    args = p.parse_args()
    sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())

    print("== bulk_insert (executemany) ==", file=sys.stderr)
    bulk = _bench_path("bulk", bulk_insert, sizes)

    if args.arrow:
        try:
            from codebase_rag.storage.vector_store_arrow import bulk_insert_arrow
        except RuntimeError as exc:
            print(f"arrow path unavailable: {exc}", file=sys.stderr)
            _print_table_single(bulk)
            return 1
        print("== bulk_insert_arrow ==", file=sys.stderr)
        arrow = _bench_path("arrow", bulk_insert_arrow, sizes)
        _print_table_dual(bulk, arrow)
    else:
        _print_table_single(bulk)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
