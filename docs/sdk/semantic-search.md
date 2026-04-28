---
description: "Semantic code search with CodeRankEmbed embeddings in Code-Graph-RAG."
---

# Semantic Search

Code-Graph-RAG supports intent-based code search using `nomic-ai/CodeRankEmbed` embeddings (768-dim, L2-normalised, asymmetric code/query prefixes). Find functions by describing what they do rather than by exact names. An optional listwise rerank stage via `nomic-ai/CodeRankLLM` further sharpens precision.

## Installation

Semantic search requires the `semantic` extra:

```bash
pip install 'code-graph-rag[semantic]'
```

## Usage

### Generate Code Embeddings

```python
from cgr import embed_code

embedding = embed_code("def authenticate(user, password): ...")
print(f"Embedding dimension: {len(embedding)}")
```

### Search by Description

In the interactive CLI, you can search semantically:

- "error handling functions"
- "authentication code"
- "database connection setup"

The system returns potential matches with similarity scores.

## How It Works

`nomic-ai/CodeRankEmbed` is a code-domain bi-encoder (a fine-tuned descendant of nomic-embed-text) that produces 768-dim L2-normalised embeddings. It uses asymmetric prefixes — `"Represent this code snippet: "` for code at index time and `"search_query: "` for queries — to optimise retrieval quality. Code-Graph-RAG uses it to capture the semantic meaning of code, enabling searches based on what code does rather than what it's named. UniXcoder was the v1 baseline before being superseded by CodeRankEmbed.
