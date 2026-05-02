# Phase 8 — HNSW VSS Scaffold

## Status: SCAFFOLD LANDED — NOT ACTIVATED

Scaffold merged: feat/phase-8-hnsw-scaffold
Activation triggers: NOT MET (see §10)

---

## §1 Motivation

DuckDB's brute-force `array_cosine_distance` scan is O(N) per query.
For repos approaching 50k symbols the p95 latency will exceed the 200 ms
target. The DuckDB VSS extension provides an HNSW (Hierarchical Navigable
Small World) index that reduces query time to O(log N) at the cost of an
upfront build.

---

## §2 Guiding Principles

- Conservative scaffold-first: no performance regression on small repos.
- Two-gate activation: global env **and** per-repo metadata must agree.
- Idempotent DDL: index creation is safe to re-run after restart.
- Brute-force remains the default; HNSW is opt-in per repo.

---

## §3 Global Env Gate

```
HNSW_ENABLED=false   # default — set to "true" to enable the HNSW path
```

This gate prevents any HNSW dispatch system-wide. It is evaluated by
`_hnsw_active(conn)` before the per-repo check.

---

## §4 Per-Repo Gate

`repo_metadata` key `"hnsw_active"` with value `"true"` (case-insensitive).

Set via the existing `write_metadata` helper:

```python
from codebase_rag.storage.vector_store import write_metadata
write_metadata(conn, hnsw_active="true")
```

Both gates must be true for `_hnsw_active(conn)` to return `True`.

---

## §5 HNSW Index DDL

```sql
CREATE INDEX IF NOT EXISTS hnsw_function_embed
ON embeddings USING HNSW (embedding)
WITH (metric = 'cosine', M = 16, ef_construction = 200)
```

Parameters chosen per DuckDB VSS documentation:
- `M = 16` — edges per node; balanced recall/memory for 768-dim vectors.
- `ef_construction = 200` — build-time beam width; higher = better recall,
  slower index build. At 50k symbols expected build time ~30–90 s.

---

## §6 Query Path Dispatch

```python
# In search_similar():
if _hnsw_active(conn):
    # HNSW: <=> is cosine-distance (lower = more similar)
    rows = conn.execute("""
        SELECT ... 1.0 - (embedding <=> ?::FLOAT[768]) AS score
        FROM embeddings
        ORDER BY embedding <=> ?::FLOAT[768]
        LIMIT ?
    """, (vec, vec, k)).fetchall()
else:
    # Brute-force (default, unchanged)
    rows = conn.execute("""
        SELECT ... 1.0 - array_cosine_distance(embedding, ?::FLOAT[768]) AS score
        FROM embeddings
        ORDER BY score DESC
        LIMIT ?
    """, (vec, k)).fetchall()
```

---

## §7 Schema Migration

`open_or_create()` now adds `hnsw_active BOOLEAN DEFAULT FALSE` to
`repo_metadata` idempotently on every open. The column is informational;
the live gate reads the string key `"hnsw_active"` from the key-value rows,
not this column directly.

---

## §8 DuckDB VSS Version Requirement

Minimum: `duckdb>=1.1.3` (first release with stable VSS extension).
Current CI install: 1.5.2 (verified 2026-05-01).

---

## §9 Test Coverage

`codebase_rag/tests/test_hnsw_scaffold.py` — 12 tests:
- Gate behaviour (_hnsw_active returns False in 4 scenarios, True in 1)
- Idempotency (3 tests: empty table, called twice, after insert)
- Smoke equivalence (3 tests: HNSW off, HNSW on, cross-comparison)
- Default invariant (1 regression guard)

---

## §10 Activation Triggers

Flip both gates only when **either** condition is met per repo:

| Metric | Threshold | Current state (2026-05-01) |
|---|---|---|
| cosine p95 latency | > 200 ms | 132 ms |
| repo symbol count | > 50,000 | 7,400 (max) |

Neither trigger is met. Scaffold lands off-by-default. Re-evaluate at next
performance review or when a repo crosses 50k symbols.

---

## §11 Activation Steps

When a repo crosses a trigger threshold, the operator should:

1. Confirm duckdb >= 1.1.3 is installed in the deployment environment.
2. Set `HNSW_ENABLED=true` in the service environment (restarts required).
3. Open the repo's `.duck` file and run:
   ```python
   from codebase_rag.storage.vector_store import (
       open_or_create, create_hnsw_index, write_metadata
   )
   conn = open_or_create("/path/to/repo.duck")
   create_hnsw_index(conn, table="embeddings", col="embedding")
   write_metadata(conn, hnsw_active="true")
   conn.close()
   ```
4. Verify with a timed `search_similar` call that p95 drops below 200 ms.
5. Monitor recall: run `scripts/bench_hnsw_query.py` (future PR) to confirm
   top-1 agreement rate >= 99% against brute-force on the live corpus.

**Index build time warning:** At large N the `create_hnsw_index` call is
blocking and can take 30–90 seconds per 50k symbols (DuckDB builds the full
graph before returning). Schedule this during a maintenance window or a
background task — do not call it inline on a user request. For repos above
200k symbols, consider an incremental build strategy (not yet implemented).

**Recall regression risk:** HNSW is approximate. With M=16 and
ef_construction=200, recall at top-10 is typically > 98% on
768-dim vectors, but is not guaranteed. Validate with the bench harness
before enabling in production on a high-stakes retrieval path.

**Persistence flag:** DuckDB VSS >= 1.1.3 requires `SET
hnsw_enable_experimental_persistence = true` before `CREATE INDEX ... USING
HNSW` on file-backed databases. `_ensure_vss_extension()` sets this
automatically. The "experimental" label is a DuckDB upstream warning —
the feature is stable in production use as of 1.1.3+.

**Connection isolation:** `_ensure_vss_extension` must be called on the
same connection that will issue the `CREATE INDEX`. Do not share the VSS
`LOAD` state across connections; each new connection requires its own call.
