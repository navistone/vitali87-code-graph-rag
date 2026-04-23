# Cleanup TODO — Residual Qdrant / Memgraph References

This fork was migrated from Memgraph + Qdrant to LadybugDB + numpy sidecars
(see phase CI-1 through CI-7 in the original build plan). The migration is
functionally complete — no Memgraph or Qdrant code is on the hot path — but
some dead symbols and stale strings remain.

**Safe to leave** (current code paths work fine) but should be removed when
someone does a dedicated cleanup pass with full test coverage.

---

## Dead Code (safe to remove with tests)

| File | What | Why dead |
|------|------|----------|
| `codebase_rag/vector_store.py:46` | `close_qdrant_client()` function | No-op stub kept for API compat; last caller removed 2026-04-23 |
| `codebase_rag/utils/dependencies.py:30` | `has_qdrant_client()` function | Deprecated helper, still imported by `tests/conftest.py` and `tests/test_vector_store.py` |
| `codebase_rag/tests/test_vector_store.py` | Entire file | Tests Qdrant-backed vector store; replaced by `test_ladybug_vector_store.py` — skipped in CI |
| `codebase_rag/constants.py:957` | `MODULE_QDRANT_CLIENT = "qdrant_client"` | No longer imported; can be removed with its usage sites |

## Stale Log Strings (cosmetic — rename on next touch)

| File:Line | Old string | Should become |
|-----------|-----------|---------------|
| `codebase_rag/logs.py:52-62` | "Qdrant upsert failed", "Stored batch of {count} embeddings in Qdrant", "Qdrant reconciliation...", "Deleting {count} Qdrant vectors..." | "Embedding upsert failed", "Stored batch of {count} embeddings", "Embedding reconciliation...", "Deleting {count} embeddings..." |
| `codebase_rag/config.py:246` | "# Embedding / vector search settings (replaces Qdrant — now LadybugDB native)" | "# Embedding / vector search settings (numpy sidecar store)" |

## Legacy Data Paths (leave — backward compat)

| File:Line | What | Why keep |
|-----------|------|----------|
| `codebase_rag/constants.py:816` | `".qdrant_code_embeddings"` in SKIP_DIRS | Prevents the ingestor from re-indexing old Qdrant data directories if a user upgrades from pre-migration checkout |

---

## Cleanup Procedure

When ready to remove (estimate ~2 hours with full test coverage):

1. Delete `close_qdrant_client`, `has_qdrant_client`, `MODULE_QDRANT_CLIENT`
2. Delete `tests/test_vector_store.py` (replaced by `test_ladybug_vector_store.py`)
3. Update `tests/conftest.py` — remove the `cleanup_qdrant_client` fixture
4. Rename the 7 log strings in `logs.py`
5. Run `uv run pytest codebase_rag/tests/ -x` — must be green
6. Smoke test: `cd ~/code-indexer-service && uv run uvicorn app.main:app --port 8000` and index a small repo end-to-end
7. Commit as a single `chore(cleanup): remove dead Qdrant references`

---

## Do NOT Remove

- Anything under `codebase_rag/tests/` that isn't `test_vector_store.py` — tests for edge cases in the LadybugDB / numpy path may still reference old names in test data
- `codebase_rag/constants.py:816` SKIP_DIRS entry — backward compat
- `realtime_updater.py` — uses the current vector_store API only, despite historical association with Qdrant

---

_Last audit: 2026-04-23. Migration tracked in original build spec CI-1..CI-7._
