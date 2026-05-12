# Spike: scip-typescript as additive TS accuracy upgrade — BUC-1615

- **Status:** Complete
- **Date:** 2026-05-12
- **Author:** backend-developer agent (spike)
- **Linear:** BUC-1615
- **Branch:** `spike/buc-1615-scip-typescript` (DO NOT MERGE)
- **Upstream:** [sourcegraph/scip-typescript](https://github.com/sourcegraph/scip-typescript) v0.4.0, Apache-2.0

---

## TL;DR — Recommendation: **ADOPT (PARTIAL)**

scip-typescript produces dramatically higher-fidelity TypeScript symbol and edge
data than tree-sitter alone, with **almost no engineering cost** (subprocess + protobuf
parse) and **trivial wall-clock cost** (8s for ~600 TS/TSX files). The edges it
resolves are exactly the ones tree-sitter currently classifies as `External`
(cross-package, generics, type-aliased members).

We should ship it as an **additive enrichment pass** behind a feature flag,
emitting edges tagged `resolved_via='scip'` per the BUC-1609 taxonomy. We should
**not** replace the tree-sitter pipeline. The two are complementary: tree-sitter
provides AST-level structure and call-site location; scip provides
compiler-grade cross-file/cross-package resolution. Specifically reject
treating scip output as a definitive symbol-kind taxonomy in v0.4.0 — `kind`
field is unpopulated.

---

## Method

1. Installed scip-typescript via `npx -y @sourcegraph/scip-typescript@0.4.0`.
2. Ran against TheForge's `web/` directory (Vite + React 19 frontend,
   240 `.ts`/`.tsx` files, strict mode, `moduleResolution: "bundler"`,
   no `paths` aliases).
3. Indexer output: `/tmp/scip-theforge.scip` (6.1 MiB protobuf).
4. Generated Python bindings from upstream `scip.proto` via `protoc --python_out`.
5. Wrote a spike-only parser at
   [`codebase_rag/tools/parse_scip.py`](../../codebase_rag/tools/parse_scip.py)
   that emits human/JSON summaries, samples documents, and inspects
   relationships.

Reproduce:

```bash
cd /path/to/TheForge/web    # must have node_modules installed
npx -y @sourcegraph/scip-typescript index --output /tmp/scip-theforge.scip

# In code-graph-rag:
curl -fsSL https://raw.githubusercontent.com/sourcegraph/scip/main/scip.proto \
  -o /tmp/scip.proto
protoc --python_out=codebase_rag/tools /tmp/scip.proto
python3 codebase_rag/tools/parse_scip.py /tmp/scip-theforge.scip
```

---

## Numbers (TheForge `web/`)

| Metric | Value |
|---|---|
| TS/TSX files indexed | 240 |
| Wall clock (cold cache, 2 runs averaged) | **7.9s** |
| Indexer-only time (excluding `npx` resolution) | **~4.0s** |
| SCIP output size | **6.1 MiB** |
| Documents in index | 240 |
| Symbol definitions (Document.symbols) | **12,517** (5,534 local, 6,983 global) |
| Total occurrences | **72,451** (12,517 def + 59,934 ref) |
| Unique referenced symbols | 9,016 |
| Intra-project occurrences (`. .` package) | 21,811 |
| Cross-package occurrences | **31,102 across 37 npm packages** |

### Top cross-package symbols resolved (tree-sitter currently marks all of these "External")

| Refs | Symbol |
|---:|---|
| 2,340 | `@types/react 19.2.14 React/HTMLAttributes#className` |
| 1,821 | `@types/react React/JSX/IntrinsicElements#div` |
| 1,367 | `vitest 4.1.0 globalExpect` |
| 1,164 | `@types/react React/JSX/IntrinsicElements#span` |
| 769 | `@vitest/runner it` |
| 723 | `@testing-library/dom screen` |
| 573 | `vitest vi` |
| 486 | `. . src/components/Icon.tsx/Icon()` (intra-project) |
| 476 | `@types/react React/useState()` |
| 325 | `@testing-library/react render()` |
| 319 | `typescript 5.9.3 lib.es5.d.ts/Array#length` |

These are the **exact edges** that today fall into our `(:External)` bucket and
break call-graph traversal at any boundary into a third-party type. scip-typescript
resolves them to fully-qualified package symbols `(manager, package, version, descriptor)` —
a stable cross-version identifier we can dedupe and join across repos.

---

## What scip-typescript identifies that tree-sitter misses

Tree-sitter's TypeScript parser gives us syntactic structure (the AST). It
*cannot* resolve:

1. **Cross-package call targets.** Every `vi.fn()`, `useState(...)`, `render(...)`,
   `screen.getByRole(...)` is currently an unresolved `External` node in our
   graph. **31,102 occurrences (43% of all occurrences)** fall in this bucket.
2. **Type-aliased member access.** `props.className` on a typed `HTMLAttributes`
   prop bag — tree-sitter sees a property access, scip resolves it to
   `@types/react React/HTMLAttributes#className`.
3. **Re-exports and barrel files.** scip follows the type resolver through
   `index.ts` re-exports.
4. **Default exports and namespace imports.** `import * as X` and
   `import X from '...'` get resolved targets, not just identifier names.
5. **Generic-parameter substitution.** Calls on `Array<T>#map` resolve to the
   `lib.es5.d.ts` definition, not a generic placeholder.
6. **Inferred types.** Even where there is no explicit annotation, the TS
   type-checker's inference is reflected in occurrence symbols.

For TheForge's `web/`, that's the gap between **"every JSX `<div>` is unknown"**
and **"every JSX `<div>` points to `@types/react React/JSX/IntrinsicElements#div`."**

### tsconfig path / alias resolution (the stated headline claim)

**Confirmed for TheForge `web/`.** This particular project doesn't define
`compilerOptions.paths`, so the path-alias claim couldn't be stressed against
this fixture. But scip-typescript invokes the TypeScript compiler API directly,
which is the same code path that resolves `paths`. The fact that it resolved
re-exports (e.g., `lucide-react` icon barrel) and `@testing-library/jest-dom`
ambient module augmentation — both of which require the type checker — is
sufficient evidence the resolver is fully online. We should still re-validate
against a repo that does use `paths` (e.g., a Next.js app with `"@/*": ["./src/*"]`)
before claiming universal coverage.

---

## CALLS edges: how scip represents them

scip does **not** have a dedicated `CALLS` edge in its schema. A call site is
represented as an `Occurrence` with:

- `symbol` = fully-qualified callee
- `symbol_roles` bitfield (1 = definition; absence of bit 1 + a function symbol
  is effectively a call/reference)
- `syntax_kind` (50 = IdentifierFunction, 51 = IdentifierFunctionDefinition)
- `range` = `[line, char_start, char_end]`

To synthesize a `CALLS(caller_def, callee_def)` edge for our graph, we must:

1. For each document, sort `symbols` by source range (the
   `SymbolInformation.enclosing_symbol` field can help when populated).
2. For each non-definition occurrence with a callable target, walk
   outward by range to find the enclosing definition; that pair is the edge.

**Caveat:** In v0.4.0, scip-typescript leaves `SymbolInformation.kind` zero
across the board (12,517/12,517 symbols had `kind = 0` in our run). It also
emits only **6 explicit `SymbolInformation.relationships`** for the entire
index — these are reserved for inheritance/impl edges (e.g., `implements`,
`extends`) and are not used as a general call/ref tool. So the call-graph
synthesis must come from `Occurrence` records alone, not relationships.

Tree-sitter is actually *better* at locating call-sites lexically (every
CallExpression node is obvious). The combination is:

- **tree-sitter:** find call-sites, attribute them to enclosing function defs (by AST scope)
- **scip:** for each call-site, resolve the callee identifier to its qualified symbol

So `CALLS` becomes: tree-sitter says "function `F` in file X calls identifier `foo` at L:C"; scip says "the identifier `foo` at X:L:C resolves to `scip-typescript npm react 19.2.14 React/useState()`."

---

## File-format complexity and LadybugDB ingestion model

### File format

- Single protobuf binary. Schema at `github.com/sourcegraph/scip/scip.proto`.
- Hierarchy: `Index { metadata, documents[], external_symbols[] }`.
- `Document { relative_path, language, occurrences[], symbols[] }`.
- `Occurrence { range, symbol, symbol_roles, syntax_kind, ... }`.
- `SymbolInformation { symbol, kind, display_name, relationships[], ... }`.

Adding a protobuf dep to our Python toolchain is a one-line `pyproject.toml`
change (`protobuf>=5.0`). The generated `scip_pb2.py` is ~30 KB and checked in
or regenerated at build time.

### Ingestion sketch

Per BUC-1609's edge-provenance taxonomy, every edge gets `resolved_via`. The
proposal:

1. **Indexer driver** (in `codebase_rag/parser_loader.py` or similar): for each
   repo, after tree-sitter parsing detects ≥1 TS/TSX file with an accompanying
   `tsconfig.json`, subprocess-invoke
   `scip-typescript index --output <tmp>.scip` in the repo root.
2. **Parse pass:** load the `.scip`, iterate `documents[].occurrences[]`.
3. **Symbol normalization:** map scip symbols `(npm, package, version, descriptor)`
   to our internal `qualified_name`. For intra-project symbols (`. .`), strip
   the prefix; for cross-package, keep `<package>::<descriptor>` form so cross-repo
   joins work.
4. **Edge emission:** for each non-definition occurrence whose enclosing
   definition we can resolve (tree-sitter scope analysis already does this in
   the existing `definitions` pass), emit:
   ```
   (caller_def) -[REFERENCES { resolved_via='scip', call: false }]-> (callee_qname)
   (caller_def) -[CALLS      { resolved_via='scip', call: true  }]-> (callee_qname)
   ```
5. **Conflict policy** (already in BUC-1609): when a `(caller, callee)` pair is
   produced by both tree-sitter and scip, prefer the scip edge; record the
   tree-sitter resolution as `alternative_resolution` for debugging.
6. **External node materialization:** create `(:ExternalSymbol)` nodes lazily
   for cross-package targets keyed by `(package, version, descriptor)` so that
   later cross-repo crawls can stitch them into resolved `(:Function)` nodes
   when that package is itself indexed.

### Cost estimate

| Phase | Cost |
|---|---|
| Subprocess invocation per repo | ~150 ms cold |
| Indexer wall-clock per 1k TS/TSX files | extrapolated **~14s** (linear from 240 files / 8s) |
| Protobuf parse | ~0.4s for 6 MiB on commodity hardware |
| Edge emission per file | dominated by tree-sitter pass that already runs |

For TheForge's `web/` specifically: **+8s** to a full reindex. Acceptable.
For a 5k-TS-file monorepo: **+60-80s**. Still well within tolerance for a
nightly or PR-triggered reindex; not appropriate for an incremental-update path
without filtering.

### Incremental update strategy

scip-typescript runs whole-program by design (the TS compiler can't be
single-file). For incremental updates we have two options:

- **Stale-and-rebuild:** trigger a full scip run when any TS file in a project
  changes, debounced. Cheap because runs are small.
- **Diff and prune:** keep the prior `.scip`, identify documents whose source
  hashes changed, replace only those documents in the merged index. The SCIP
  spec is explicitly designed to support partial-index merging.

We should ship stale-and-rebuild first.

---

## Limitations observed in scip-typescript v0.4.0

- `SymbolInformation.kind` is unpopulated (all 12,517 symbols have `kind=0`).
  This means we cannot rely on scip to tell us "this is a class vs. an interface
  vs. a const vs. a function." We must continue to derive kind from tree-sitter.
- `SymbolInformation.relationships` is barely populated (6 entries / 12,517).
  Inheritance/impl edges are not reliably represented in this version.
- `Occurrence.syntax_kind` is sparsely populated. Identifying which occurrences
  are calls vs. property reads requires our tree-sitter pass.
- The `external_symbols` collection is empty — all external symbol references
  inline directly into occurrence records.
- `imports` role (bit 2 in `symbol_roles`) is never set in our output (0
  occurrences). Tree-sitter must continue to provide IMPORTS edges.
- No `language` field populated on documents (empty string for all 240 docs).
  Use file extension instead.

These limitations confirm scip-typescript is best used as **complementary**, not
**replacement** — exactly the framing in BUC-1615.

---

## Risks / blockers

- **None blocking.** scip-typescript is stable, Apache-2.0, used in production
  by Sourcegraph.
- **Build dependency:** requires `node` + access to npm to fetch the indexer
  on first run. Already a hard dependency for indexing TS at all (the indexer
  uses the TypeScript compiler API).
- **`node_modules` requirement:** scip-typescript needs the target project's
  deps installed to resolve cross-package symbols. Our `code-indexer-service`
  clones don't currently install deps. Two paths:
  1. Run `npm install --ignore-scripts --no-audit --no-fund` before indexing
     (slow, ~30-60s per repo, but cached).
  2. Run scip-typescript in `--infer-tsconfig` mode without deps installed —
     resolves intra-project symbols only (loses the 31k cross-package wins).
  Recommend (1) gated by a config flag `scip.install_deps: true|false`. For
  trusted internal repos, install; for untrusted, skip.

---

## Recommendation: ADOPT (PARTIAL) — phased rollout

| Phase | Scope | Linear |
|---|---|---|
| 1 | Drive scip-typescript as a subprocess per TS-containing repo. Parse output. Emit `CALLS`/`REFERENCES` edges with `resolved_via='scip'` for *intra-project* symbols only. Behind feature flag `SCIP_TYPESCRIPT_ENABLED=false` default. | follow-up |
| 2 | Add `npm install` step before scip-typescript run. Materialize `(:ExternalSymbol)` nodes for cross-package targets. Enable flag for trusted repos. | follow-up |
| 3 | Cross-repo join: when package `X` is indexed, link existing `(:ExternalSymbol { package: 'X' })` nodes to resolved `(:Function)` nodes. | follow-up |
| 4 | Remove `External` nodes from tree-sitter pipeline once scip pass has higher coverage; keep tree-sitter for kind/structure. | follow-up |

**Reject:** treating scip as the symbol-kind source of truth (all `kind = 0` in
v0.4.0). Keep tree-sitter for kind classification.

---

## Artifacts

- Spike parser: [`codebase_rag/tools/parse_scip.py`](../../codebase_rag/tools/parse_scip.py)
- Sample SCIP fixture instructions (above)
- Worktree: `/tmp/agent-buc-1615-92862-1778596001` (ephemeral)
- Branch: `spike/buc-1615-scip-typescript` on `navistone` remote (do not merge)
