# AGENTS.md — code-graph-rag (navistone fork)

This repo's `CLAUDE.md` is gitignored (per-developer local notes). The
agent-skills index lives here so the whole team sees it.

This is a navistone-maintained fork of the upstream
`iflow-mcp/vitali87-code-graph-rag` project — the parsing/indexing engine
behind `code-indexer-service`.

## Agent skills

### Backlog

GitHub Issues on `navistone/vitali87-code-graph-rag` via the `gh` CLI. Backlog
lives on the navistone fork only — upstream issues are out of scope. See
`docs/agents/backlog.md`.

### Triage labels

Default canonical names (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). Created lazily on first use. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. `CONTEXT.md` and `docs/adr/` at the repo root, created lazily by `/grill-with-docs` when terms or decisions get resolved. See `docs/agents/domain.md`.

## Top-level docs

- `README.md` — engine overview + CLI/MCP usage
- `PYPI_README.md` — published package summary
- `scripts/BENCH_RESULTS_*.md` — performance benches (most recent: Arrow `bulk_insert` 380× speedup)
- `docs/` — upstream-style guides (architecture, SDK, advanced)
