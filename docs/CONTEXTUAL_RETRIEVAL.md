# Contextual Retrieval

> Anthropic's "Contextual Retrieval" technique, implemented in the
> `code-graph-rag` ingestion pipeline.

## What it does

Before each code chunk is embedded, a cheap LLM (Claude Haiku 3.5 by
default) generates a 50-100 token summary of **how this chunk relates to
its parent file and project**. That summary is prepended to the chunk
text. The embedding model and the reranker then see the chunk **with
context** — drastically improving recall on natural-language queries that
do not share vocabulary with the raw code.

Anthropic's published research reports a **67% reduction in retrieval
failures** vs. naive chunked embeddings. Reference:
<https://www.anthropic.com/news/contextual-retrieval>

Example — without contextual retrieval, this chunk:

```python
def _flush(self) -> None:
    self._path.write_text(json.dumps(self._data))
```

…embeds as a generic "write JSON to disk" pattern. With contextual
retrieval the embedded text becomes:

```
This method persists the on-disk cache of contextual prefixes for the
ladybug ingestor. It atomically writes the JSON map of (file_hash,
qualified_name) → prefix used by ContextualPrefixGenerator so re-indexes
hit warm cache entries.

def _flush(self) -> None:
    self._path.write_text(json.dumps(self._data))
```

A query for "where do we cache contextual prefixes" now hits this chunk
on the first page; previously it was buried under every other JSON-write
helper in the repo.

## When to enable

| Repo size | First-run cost (Haiku 3.5) | Recall lift     |
| --------- | -------------------------- | --------------- |
| < 5k chunks   | ~$5             | Modest          |
| 5k - 50k      | ~$5 - $50       | Significant     |
| 50k - 500k    | ~$50 - $500     | Significant     |
| 500k+         | ~$500+          | Significant     |

The estimate assumes ~600 input tokens (chunk + prompt overhead) and
~90 output tokens per chunk at Haiku 3.5 published rates ($1.00 / 1M
input, $5.00 / 1M output as of 2026-05). 100k chunks ≈ $105 first-run;
re-indexes are free (disk cache).

The cost is a **one-time pass** — generated prefixes are cached on disk
keyed by `(file_hash, qualified_name)`, so re-indexing an unchanged file
is free. Only modified chunks are re-summarised.

**Enable it if:**

- Your users frequently ask natural-language questions ("where do we
  cache X?", "what handles Y?") that the current retrieval misses.
- You can afford the one-time spend (estimate via `--contextual-retrieval`
  CLI flag; the index command logs the live cost when it runs).
- You have an `ANTHROPIC_API_KEY` configured.

**Skip it if:**

- Your usage pattern is strictly symbol-name search ("find function
  `createUser`"). The graph already handles that.
- You cannot route egress to `api.anthropic.com` (compliance lockdown).
  The fallback prefix (`[from <path>]`) is automatic but provides only a
  minor lift over no prefix at all.

## How to enable

**One-shot (recommended for the first try):**

```bash
ANTHROPIC_API_KEY=sk-... \
  cgr index \
    --repo-path /path/to/repo \
    --output-proto-dir /tmp/proto \
    --contextual-retrieval
```

**Persistent (every subsequent index):**

```bash
# .env or shell rc
export CONTEXTUAL_RETRIEVAL_ENABLED=true
export ANTHROPIC_API_KEY=sk-...
```

The CLI flag is just a convenience wrapper that sets the env var for
that process.

### Tunables (env vars)

| Variable                         | Default                | Notes                                                  |
| -------------------------------- | ---------------------- | ------------------------------------------------------ |
| `CONTEXTUAL_RETRIEVAL_ENABLED`   | `false`                | Master switch.                                          |
| `CONTEXTUAL_RETRIEVAL_MODEL`     | `claude-haiku-4-5`     | Any Anthropic Messages-API model id.                    |
| `CONTEXTUAL_RETRIEVAL_MAX_TOKENS`| `150`                  | Cap on generated tokens per prefix.                     |
| `CONTEXTUAL_RETRIEVAL_TIMEOUT_S` | `10`                   | Per-call HTTP timeout. Failures fall back silently.     |
| `ANTHROPIC_API_KEY`              | -                      | Required when enabled. Same key used elsewhere in CGR.  |

## How to A/B test it on your own corpus

1. Build a small evaluation set (~20 queries) with known-good expected
   chunks. The Anthropic blog post describes a good labelling protocol.
2. Index the repo without the flag, run your queries, record recall@5.
3. Drop the embeddings (`rm -rf .cgr/embeddings*.npz`), re-index with
   `--contextual-retrieval`, run the same queries, record recall@5.
4. Compare. If recall@5 lifts by >20 points, ship it. If it lifts by <5
   points, your corpus probably has good vocabulary alignment already —
   the cost may not be worth it.

The `tests/test_contextual_prefix.py` unit suite is a good template for
building richer evaluation harnesses.

## What's NOT changed in this PR

- TheForge calls `/context-bundle` on the code-indexer-service. That
  endpoint already returns whatever the indexer has stored — no
  changes needed on the TheForge side. **Operators must re-index their
  repos** after enabling the flag for the change to take effect.
- Existing indexes continue to work with empty `contextual_prefix`
  columns. The migration ALTER adds them with `DEFAULT ''`.
- LM Studio batched-embedding path is unchanged — it just receives
  longer strings.

## Implementation notes

- Module: `codebase_rag/services/contextual_prefix.py`
- Wired in: `codebase_rag/graph_updater.py` (Pass 1.5, between
  eligibility filtering and the embedder).
- Schema: `Function.contextual_prefix STRING DEFAULT ''` +
  `Method.contextual_prefix STRING DEFAULT ''`. Backfill ALTERs in
  `ladybug_schema._NODE_ALTERS`.
- Cache: `.cgr/contextual_prefixes.json` (JSON map keyed by
  `sha256(file_hash:qualified_name)`).
- Fail-open: every failure mode (LLM down, no API key, malformed
  response) falls back to `[from <file_path>]` — embedding never breaks.
- No new heavy deps: uses `httpx` (already transitive via
  `mcp`/`pydantic-ai`).
