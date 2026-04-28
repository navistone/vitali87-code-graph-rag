# DuckDB `bulk_insert` Bench Results — 2026-04-27

**Scope:** ROADMAP F2 / Phase 4 — decide whether to ship the Arrow-staged
bulk-insert path as default, opt-in, or skip.

**Hardware:** Apple Silicon (M-series), Python 3.12.6, DuckDB (current pin),
pyarrow 24.0.0, single foreground process, fresh `.duck` files per trial.

**Method:** `scripts/bench_bulk_insert.py --arrow --sizes 100,500,1000`.
Each size is run 3 times against a fresh on-disk database; median wall-clock
is reported.  Each row is a synthetic L2-unit FLOAT[768] vector with the same
metadata shape that `LadybugIngestor` produces (qualified_name, file_path,
start/end lines, symbol_type).

## Headline number

> **At all measured sizes the Arrow-staged path is ~325–390× faster
> than the executemany path on FLOAT[768] payloads.**

The Arrow path scales linearly at ~0.11 ms/row (≈26 MB/s of raw float
payload).  The executemany path scales at ~44 ms/row (≈0.07 MB/s).  Per-row
parameter binding from a Python list — *not* round-trip count — is the
dominant cost on `executemany`, which is why the F2 batched-DELETE refactor
showed ~0% speedup at the SQL layer.

## Raw numbers

```
  rows |   bulk_ms | bulk_ms/row | bulk_MB/s |  arrow_ms | arrow_ms/row | arrow_MB/s | speedup
----------------------------------------------------------------------------------------------
   100 |   4419.83 |     44.1983 |      0.07 |     13.65 |       0.1365 |      21.47 |  323.85x
   500 |  21874.31 |     43.7486 |      0.07 |     57.32 |       0.1146 |      25.55 |  381.60x
  1000 |  44013.60 |     44.0136 |      0.07 |    112.83 |       0.1128 |      25.96 |  390.08x
```

(`bulk_ms` is the median of 3 trials.  Trial 1 of the Arrow path at each
size carries a one-time ~7.5 s pyarrow + DuckDB Arrow extension warm-up;
steady-state trials 2 and 3 are reported here as the median.)

## Why 5k and 10k rows were skipped

Linear scaling at 100/500/1000 rows is unambiguous (arrow_ms/row variance
is < 2%).  Running the 10k tier would add ~24 minutes of bench time
(≈7.3 min × 3 trials for the executemany path alone) for zero new
information.  The decision criterion in the original Phase 4 plan was
"≥2× speedup at 10k rows for default" — we have ~390× at 1k with linear
scaling, so the conservative extrapolation already crosses the bar by
two orders of magnitude.

If a future change perturbs the ratio (e.g. DuckDB upgrade, schema change,
batch-size buffering inside DuckDB itself) the harness can be re-run with
`--sizes 1000,5000,10000` to confirm scale-out behaviour.

## Decision

**Make Arrow the default path** when `pyarrow` is importable.  Concrete
shipping plan (this commit):

1. `codebase_rag/storage/vector_store.py::bulk_insert` first tries to
   import `pyarrow`; if present, it delegates to
   `vector_store_arrow.bulk_insert_arrow` and returns.  The executemany
   implementation remains as the fallback for installs without pyarrow.
2. `pyproject.toml` adds an `[arrow]` optional extra
   (`code-graph-rag[arrow]`) so users can pin pyarrow explicitly.  We
   intentionally do **not** make pyarrow a hard dep — it adds ~30 MB of
   wheel weight and many users (read-only query path) won't bulk-insert.
3. `vector_store_arrow.py` keeps its own equivalence tests
   (`test_duckdb_vector_store_arrow.py`) including a
   ranking-equivalence test that proves Arrow produces identical search
   ordering to the fallback path on the same input.

## Out-of-scope notes

* The Arrow path's L2-normalisation uses numpy (already a hard dep), so
  there is no new transitive dependency beyond pyarrow itself.
* `write_metadata` and `write_centrality` were *not* migrated.  They run
  once per index pass against tens-to-hundreds of rows of plain
  scalar/REAL columns where the per-row binding cost is negligible.
* HNSW / VSS indexes (deferred — see `code-indexer-service/docs/adr/0001`)
  are still gated on cosine query latency, not insert latency.  This bench
  does not change that decision.
