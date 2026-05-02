#!/usr/bin/env python3
"""Benchmark: sequential vs batched LM Studio /v1/embeddings.

Times two embedding strategies against http://127.0.0.1:1234 on N=100
random-ish text inputs and prints a comparison table.  Designed to be run
on the local dev machine to validate the 50× improvement claimed in
SUCCESS.md §Outstanding-1.

Usage:
    uv run python scripts/bench_batched_embed.py [--url URL] [--n N] [--batch-size B]

If LM Studio is unreachable the script exits 0 with a diagnostic message so CI
does not fail; the benchmark is informational rather than a hard gate.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_code_snippet(rng: random.Random, length: int = 120) -> str:
    """Return a plausible-looking code snippet of roughly ``length`` chars."""
    names = ["process", "validate", "transform", "parse", "render", "fetch", "update"]
    args = ["data", "value", "items", "config", "ctx", "opts"]
    lines = [
        f"def {rng.choice(names)}_{rng.choice(args)}({rng.choice(args)}, {rng.choice(args)}):",
        f"    \"\"\"{''.join(rng.choices(string.ascii_lowercase + ' ', k=40))}.\"\"\"",
        f"    result = {rng.choice(args)} or []",
        f"    for item in {rng.choice(args)}:",
        f"        if isinstance(item, dict):",
        f"            result.append(item.get('{rng.choice(args)}'))",
        f"    return result",
    ]
    snippet = "\n".join(lines)
    # Pad or trim to roughly the requested length.
    while len(snippet) < length:
        snippet += f"\n    # {''.join(rng.choices(string.ascii_lowercase, k=20))}"
    return snippet[:max(length, 80)]


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_model(base_url: str, hint: str = "CodeRankEmbed") -> str | None:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hint_lc = hint.lower()
        for item in data.get("data", []):
            if hint_lc in item.get("id", "").lower():
                return item["id"]
    except Exception:
        pass
    return None


def _embed_sequential(
    base_url: str, model: str, texts: list[str], prefix: str, timeout: float
) -> float:
    """Embed ``texts`` one at a time.  Returns wall-clock seconds."""
    url = f"{base_url}/v1/embeddings"
    t0 = time.monotonic()
    for text in texts:
        _post_json(url, {"model": model, "input": prefix + text}, timeout)
    return time.monotonic() - t0


def _embed_batched(
    base_url: str,
    model: str,
    texts: list[str],
    prefix: str,
    timeout: float,
    batch_size: int,
) -> float:
    """Embed ``texts`` in chunks of ``batch_size``.  Returns wall-clock seconds."""
    url = f"{base_url}/v1/embeddings"
    t0 = time.monotonic()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        _post_json(url, {"model": model, "input": [prefix + t for t in chunk]}, timeout)
    return time.monotonic() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential vs batched LM Studio embed bench")
    parser.add_argument("--url", default="http://127.0.0.1:1234", help="LM Studio base URL")
    parser.add_argument("--n", type=int, default=100, help="Number of texts to embed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for batched path")
    parser.add_argument("--prefix", default="Represent this code snippet: ", help="Nomic asymmetric prefix")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout (s)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible texts")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Probe LM Studio availability
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3.0):
            pass
    except Exception as exc:
        print(f"LM Studio not reachable at {base_url}: {exc}")
        print("Skipping benchmark (exit 0 — informational only).")
        return 0

    model = _resolve_model(base_url)
    if model is None:
        print(f"No model loaded in LM Studio at {base_url} — cannot embed.")
        print("Skipping benchmark (exit 0).")
        return 0

    print(f"LM Studio: {base_url}")
    print(f"Model:     {model}")
    print(f"Texts:     {args.n}")
    print(f"Batch sz:  {args.batch_size}")
    print()

    rng = random.Random(args.seed)
    texts = [_random_code_snippet(rng) for _ in range(args.n)]

    # Warm up with a single call so model is hot.
    print("Warming up...")
    _post_json(f"{base_url}/v1/embeddings", {"model": model, "input": args.prefix + texts[0]}, args.timeout)

    print("Running sequential (1 text per request)...")
    seq_s = _embed_sequential(base_url, model, texts, args.prefix, args.timeout)
    seq_rate = args.n / seq_s

    print(f"Running batched  ({args.batch_size} texts per request)...")
    bat_s = _embed_batched(base_url, model, texts, args.prefix, args.timeout, args.batch_size)
    bat_rate = args.n / bat_s

    ratio = seq_s / bat_s if bat_s > 0 else float("inf")

    print()
    print("=" * 60)
    print(f"{'Strategy':<18} {'Time (s)':>10} {'sym/s':>10} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'Sequential':<18} {seq_s:>10.2f} {seq_rate:>10.1f} {'1.00x':>10}")
    print(f"{'Batched (N=' + str(args.batch_size) + ')':<18} {bat_s:>10.2f} {bat_rate:>10.1f} {ratio:>9.2f}x")
    print("=" * 60)
    print()
    print(f"Speedup: {ratio:.2f}x  (sequential {seq_s:.2f}s → batched {bat_s:.2f}s on {args.n} texts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
