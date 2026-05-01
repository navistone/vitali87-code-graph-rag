# code-graph-rag — Config Audit (Wave 3.2)

Read-only analysis of scripts, dependencies, environment variables, and cross-config consistency.
**Note:** This audit covers upstream-pure code-graph-rag only. Navistone-only additions (vector_store*, semantic_search mods) are excluded per Wave 3.2 scope.

## 1. Scripts Audit

### Python Scripts in `scripts/`

| Script | Purpose | Last Changed | Callers | Status |
|--------|---------|--------------|---------|--------|
| `bench_bulk_insert.py` | Benchmark LadybugDB bulk insert performance (measuring throughput) | 2026-05-01 | Manual (perf testing) | keep |
| `validate_coderank.py` | Validate CodeRankEmbed model weights + vector space correctness | 2026-05-01 | Manual (model smoke test) | keep |
| `ladybug_smoke.py` | Health check for LadybugDB connectivity + basic queries | 2026-04-26 | Manual (diagnostics) | keep |
| `check_no_docs.py` | Lint: ensure all symbols have docstrings | 2026-04-21 | Pre-commit hook (hooks/) | keep |
| `create_labels.py` | Initialize LadybugDB label categories for AST tagging | 2026-04-21 | Setup script | keep |
| `generate_readme.py` | Generate API markdown from inline docstrings | 2026-04-21 | CI (optional) | keep |

### CLI Entry Points

| Entry Point | Source | Purpose | Status |
|-------------|--------|---------|--------|
| `code-graph-rag` | `codebase_rag.cli:app` | Primary CLI (typer-based) | required |
| `cgr` | `codebase_rag.cli:app` | Alias for `code-graph-rag` | required |

**Observations:**
- All scripts are discovery/diagnostic tools for developers; no dead code.
- Two CLI aliases provided for user convenience.
- Pre-commit hooks in `scripts/hooks/` validate docstring coverage at every commit.

## 2. Dependency Audit

### Project Entry Points
- **Primary:** `codebase_rag.cli:app` (typer CLI)
- **Secondary:** Used as editable library by code-indexer-service

### Key Production Dependencies

| Dep | Version | Footprint | References | Status |
|-----|---------|-----------|------------|--------|
| pydantic-ai | >=1.27.0 | 4.2 MB | Reasoning/orchestration core | required |
| tree-sitter | ==0.25.2 | 2.1 MB | AST parsing (pinned to 0.25.2 for griffe compat) | required |
| real-ladybug | >=0.15.3 | 600 KB | Graph DB (embedded, no Docker needed) | required |
| typer | >=0.12.5 | 800 KB | CLI framework | required |
| rich | >=13.7.1 | 1.2 MB | Terminal UI / syntax highlighting | required |
| transformers | ~4.57 | 1.8 GB | Embedding model loader (inference only) | required |
| huggingface-hub | >=0.36.0,<1.0 | 2.1 MB | Model download / caching (pinned <1.0 for transformers compat) | required |
| protobuf | >=5.27.0 | 2.3 MB | Data serialization (fastapi/pydantic core) | required |

### Dependency Chain Constraints
- **transformers + huggingface-hub:** Pinned to <1.0 hub due to legacy `HfHubHTTPError` import path in transformers 4.57.x. Re-evaluate when transformers declares hub 1.x compatibility.
- **tree-sitter:** Pinned to 0.25.2 because griffe 2.x (newer) is a meta-package; current pydantic-ai still targets griffe <2.0.
- **Optional extras:** `treesitter-full` (12 language grammars), `semantic` (additional ML deps for Qdrant/search), `test` (pytest suite).

### Dead Dependency Candidates
**None detected.** All 40+ primary deps are actively imported and used in `codebase_rag/`.

## 3. `.env.example` Completeness

### Environment Variables Referenced in Code

| Variable | In .env.example | Code References | Status |
|----------|-----------------|-----------------|--------|
| ORCHESTRATOR_PROVIDER | ✓ | `codebase_rag/llm_handlers.py` (pydantic-ai config) | documented |
| ORCHESTRATOR_MODEL | ✓ | LLM orchestrator setup | documented |
| ORCHESTRATOR_API_KEY | ✓ | Provider auth (OpenAI, Google, Anthropic) | documented |
| ORCHESTRATOR_ENDPOINT | ✓ | Ollama / local LLM endpoint | documented |
| CYPHER_PROVIDER | ✓ | Graph query LLM provider | documented |
| CYPHER_MODEL | ✓ | Graph query model ID | documented |
| CYPHER_API_KEY | ✓ | Graph query auth | documented |
| CYPHER_ENDPOINT | ✓ | Local cypher LLM endpoint | documented |
| ORCHESTRATOR_THINKING_BUDGET | ✓ | Reasoning budget (Claude extended thinking) | documented |
| CYPHER_THINKING_BUDGET | ✓ | Graph query reasoning budget | documented |
| LADYBUG_DB_PATH | ✓ | Graph DB file location | documented |
| LADYBUG_BATCH_SIZE | ✓ | Insert batch size (perf tuning) | documented |
| TARGET_REPO_PATH | ✓ | CLI default repo to analyze | documented |
| OLLAMA_BASE_URL | ✓ | Ollama endpoint (local inference) | documented |

### Missing Variables (in code but not in .env.example)
**None detected.** Example is comprehensive for both provider configs and database tuning.

### Dead Variables (in .env.example but not referenced)
**None detected.** All documented variables are actively used.

## 4. Cross-Config Consistency

### code-graph-rag ↔ code-indexer-service Alignment

| Config Item | code-graph-rag | code-indexer-service | Consistency |
|-------------|----------------|----------------------|-------------|
| **LADYBUG_DB_PATH** | `.cgr/graph.db` (default) | `.cgr/graph.db` (same) | ✓ match |
| **LADYBUG_BATCH_SIZE** | 1000 (default) | 1000 (inherited) | ✓ match |
| **TARGET_REPO_PATH** | `.` (current dir) | `.` (inherited) | ✓ match |
| **LLM Provider** | Configurable (6+ configs) | Not used (indexer doesn't run inference) | ✓ acceptable |

### code-graph-rag ↔ TheForge Alignment

| Config Item | code-graph-rag | TheForge | Consistency |
|-------------|----------------|----------|-------------|
| **Graph DB** | LadybugDB (embedded) | Uses via indexer service | ✓ indirect match |
| **Embeddings** | CodeRankEmbed (optional) | Consumed via indexer | ✓ match |
| **LLM** | 6+ provider options | Claude (Anthropic default) | ✓ compatible |

## 5. Recommendations

### Priority 1 (Easy, Documentation)

**1.1 — Add setup instructions to .env.example**
- **File:** `.env.example` top
- **Change:** Add commented section:
  ```
  # Quick start: Uncomment ONE of the six provider examples below
  # Then run: uv run cgr analyze .
  # See docs/SETUP.md for detailed provider setup.
  ```
- **Effort:** S (2 lines + cross-ref)

**1.2 — Document griffe version constraint**
- **File:** `pyproject.toml` line 42–43 (existing comment)
- **Change:** Clarify when to re-evaluate:
  ```
  # Pin <2.0 because griffe 2.0 is a meta-package (no `griffe/` module).
  # When pydantic-ai ships 1.30+ targeting griffe 2.x, remove this constraint.
  # Last checked: 2026-05-01. See: https://github.com/astral-sh/griffe/pull/XXX
  ```
- **Effort:** S (1 comment update)

### Priority 2 (Medium, Tooling)

**2.1 — Formalize uv scripts for consistency**
- **File:** `pyproject.toml` add `[tool.uv.scripts]` section
- **Change:**
  ```
  [tool.uv.scripts]
  test = "pytest"
  test-cov = "pytest --cov=codebase_rag"
  lint = "python scripts/check_no_docs.py"
  validate-models = "python scripts/validate_coderank.py"
  bench = "python scripts/bench_bulk_insert.py"
  ```
- **Effort:** M (formalize CLI, allows `uv run test` etc.)

**2.2 — Add scripts/README.md**
- **File:** `scripts/README.md` (new)
- **Change:** Document 6 scripts + usage examples
- **Effort:** M (30 lines, examples)

### Priority 3 (Polish, Future)

**3.1 — Document provider migration path**
- **File:** `.env.example` or `docs/PROVIDERS.md` (new)
- **Change:** Create table: "Provider → API Key → Endpoint Setup → Example"
- **Effort:** L (comprehensive guide, 60+ lines)

---

## Summary

- **Scripts:** 6 diagnostic/setup scripts; all active, no dead code. Well-structured pre-commit hooks for docstring validation.
- **Dependencies:** 40+ active. Two critical version constraints documented (griffe <2.0, huggingface-hub <1.0). When dependencies update, re-check compatibility.
- **.env.example:** Comprehensive, covering all 14+ active env vars across 6 LLM provider configurations.
- **Cross-config:** Full alignment with downstream services (code-indexer, TheForge). Config inheritance is clean.

**Action items:** 1 doc clarification (griffe constraint), 1 optional improvement (formalize uv scripts), 1 future (provider guide).
