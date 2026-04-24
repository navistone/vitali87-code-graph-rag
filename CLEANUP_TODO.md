# Cleanup TODO — Residual Qdrant / Memgraph References

> **Status: COMPLETE** — All items below were executed on 2026-04-23.
> This file is kept as a record of what was done.

This fork was migrated from Memgraph + Qdrant to LadybugDB + numpy sidecars
(see phase CI-1 through CI-7 in the original build plan).

---

## Completed

| Item | Commit |
|------|--------|
| Removed `close_qdrant_client()` stub from `vector_store.py` | cleanup pass 2026-04-23 |
| Added `store_embedding()`, `delete_project_embeddings()` to `vector_store.py` | cleanup pass 2026-04-23 |
| Fixed `verify_stored_ids` to check `_pending` + handle legacy int IDs | cleanup pass 2026-04-23 |
| Added auto-flush to `search_embeddings` (stores then searches without explicit flush) | cleanup pass 2026-04-23 |
| Removed `has_qdrant_client()` from `utils/dependencies.py` | cleanup pass 2026-04-23 |
| Removed `cleanup_qdrant_client` fixture from `tests/conftest.py` | cleanup pass 2026-04-23 |
| Added `_clear_vector_store_pending` autouse fixture for test isolation | cleanup pass 2026-04-23 |
| Deleted `tests/test_vector_store.py` (Qdrant-backed, replaced by test_ladybug) | cleanup pass 2026-04-23 |
| Deleted `tests/test_vector_store_batch.py` (skipped by has_qdrant_client guard, broken imports) | cleanup pass 2026-04-23 |
| Updated stale "Qdrant" log strings in `logs.py` → generic embedding strings | cleanup pass 2026-04-23 |
| Removed `MODULE_QDRANT_CLIENT` from `constants.py` | cleanup pass 2026-04-23 |
| Updated stale comment in `config.py` (embedding settings block) | cleanup pass 2026-04-23 |
| Removed `pymgclient` (Memgraph client) from `pyproject.toml` dependencies | cleanup pass 2026-04-23 |
| Added `numpy>=1.26.0` to `pyproject.toml` (was transitive, now explicit) | cleanup pass 2026-04-23 |
| Removed "memgraph" from PyPI keywords | cleanup pass 2026-04-23 |

## Do NOT Remove

- `codebase_rag/constants.py` `SKIP_DIRS` entry for `".qdrant_code_embeddings"` — prevents
  re-indexing of old Qdrant data directories on user upgrades from pre-migration checkouts.
- `realtime_updater.py` — fully migrated to current vector_store API.

---

_Audit completed: 2026-04-23. Migration tracked in original build spec CI-1..CI-7._
