"""End-to-end validation for the CodeRankEmbed migration.

Exercises codebase_rag.embedder.{embed_code, embed_query} without requiring
Memgraph or Qdrant. Verifies 768-dim output, unit L2 norm, and that each NL
query is closest (by cosine similarity) to its matching code snippet.

Run:  python scripts/validate_coderank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable when run as `python scripts/validate_coderank.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codebase_rag.embedder import embed_code, embed_query  # noqa: E402

CODE_SNIPPETS: list[str] = [
    """def merge_pull_request(pr_id):
    url = f"https://api.github.com/repos/acme/app/pulls/{pr_id}/merge"
    response = requests.put(url, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()
""",
    """def validate_jwt_token(token):
    header, payload, signature = token.split(".")
    decoded = base64.urlsafe_b64decode(payload + "==")
    expected = hmac.new(SECRET, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, base64.urlsafe_b64decode(signature + "==")):
        raise ValueError("invalid signature")
    return json.loads(decoded)
""",
    """def handle_websocket_reconnection(socket):
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            socket.connect()
            return socket
        except ConnectionError:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("websocket reconnect failed")
""",
]

QUERIES: list[str] = [
    "merge pull request",
    "validate JWT token",
    "handle websocket reconnection",
]

LABELS = ["merge_pr", "validate_jwt", "ws_reconnect"]


def main() -> int:
    code_vecs = np.stack([np.asarray(embed_code(s), dtype=np.float32) for s in CODE_SNIPPETS])
    query_vecs = np.stack([np.asarray(embed_query(q), dtype=np.float32) for q in QUERIES])

    print("=== Dimensions ===")
    print(f"code_vecs.shape  = {code_vecs.shape}")
    print(f"query_vecs.shape = {query_vecs.shape}")
    dim_ok = code_vecs.shape[1] == 768 and query_vecs.shape[1] == 768
    print(f"768-dim check: {'PASS' if dim_ok else 'FAIL'}")

    print("\n=== L2 norms (expect ~1.0) ===")
    code_norms = np.linalg.norm(code_vecs, axis=1)
    query_norms = np.linalg.norm(query_vecs, axis=1)
    for label, n in zip(LABELS, code_norms):
        print(f"  code  [{label:14s}] = {n:.6f}")
    for q, n in zip(QUERIES, query_norms):
        print(f"  query [{q:30s}] = {n:.6f}")
    norm_ok = bool(np.allclose(code_norms, 1.0, atol=1e-3) and np.allclose(query_norms, 1.0, atol=1e-3))
    print(f"Unit-norm check: {'PASS' if norm_ok else 'FAIL'}")

    # Cosine similarity = dot product since vectors are L2-normalized.
    sim = query_vecs @ code_vecs.T  # rows: queries, cols: code snippets

    print("\n=== Cosine similarity (rows=queries, cols=code) ===")
    header = "                              " + "  ".join(f"{l:>14s}" for l in LABELS)
    print(header)
    for q, row in zip(QUERIES, sim):
        cells = "  ".join(f"{v:>14.4f}" for v in row)
        print(f"{q:30s}{cells}")

    diagonal_ok = all(int(np.argmax(sim[i])) == i for i in range(len(QUERIES)))
    print(f"\nDiagonal dominance: {'PASS' if diagonal_ok else 'FAIL'}")

    overall = dim_ok and norm_ok and diagonal_ok
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
