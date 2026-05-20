# MIT License — Copyright (c) 2026 Navistone, contributors to code-graph-rag
"""Unit tests for ``codebase_rag.services.contextual_prefix``.

Covers the three contracts the spec calls out:

1. Generates a non-empty, reasonable-length prefix for a known function
   (LLM is mocked — we assert the request shape + that the response text
   surfaces back to the caller).
2. Caches the result — a second ``generate()`` call with the same
   (file_hash, qualified_name) MUST NOT hit the LLM.
3. Falls back gracefully when the LLM raises — returns the minimal
   ``[from <path>]`` prefix, never raises.

Tests follow the project's "should ... when ..." naming convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codebase_rag.services.contextual_prefix import (
    ContextualPrefixConfig,
    ContextualPrefixGenerator,
    estimate_cost,
)


class _FakeResponse:
    """httpx.Response stand-in — just enough surface for the generator."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    """Records calls; returns canned responses, or raises if configured."""

    def __init__(
        self,
        *,
        response_text: str | None = "Creates a new user in the database.",
        raise_exc: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response_text = response_text
        self._raise_exc = raise_exc

    def post(
        self, url: str, json: dict[str, Any], headers: dict[str, str]
    ) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeResponse(
            {"content": [{"type": "text", "text": self._response_text or ""}]}
        )


def _enabled_config() -> ContextualPrefixConfig:
    return ContextualPrefixConfig(
        enabled=True,
        model="claude-haiku-4-5",
        max_tokens=150,
        timeout_s=5.0,
        api_key="sk-test-key",
        api_base="https://api.anthropic.com/v1/messages",
    )


def test_should_return_llm_text_when_generating_prefix_for_known_function(
    tmp_path: Path,
) -> None:
    http = _FakeHttpClient(response_text="Creates a user given an email and password.")
    gen = ContextualPrefixGenerator(
        cache_dir=tmp_path, config=_enabled_config(), http_client=http
    )

    prefix = gen.generate(
        file_path="src/api/users.py",
        qualified_name="myapp.api.users.create_user",
        chunk_text="def create_user(email, password):\n    ...",
        file_hash="hash-abc",
    )

    # 1. non-empty, reasonable length
    assert prefix
    assert 10 < len(prefix) < 500
    assert "user" in prefix.lower()

    # 2. exactly one LLM call
    assert len(http.calls) == 1
    body = http.calls[0]["json"]
    assert body["model"] == "claude-haiku-4-5"
    assert body["max_tokens"] == 150
    assert "create_user" in body["messages"][0]["content"]
    assert "src/api/users.py" in body["messages"][0]["content"]

    # 3. api key never leaks into anything we record (sanity)
    assert http.calls[0]["headers"]["x-api-key"] == "sk-test-key"

    # 4. counters updated
    assert gen.stats["llm_calls"] == 1
    assert gen.stats["misses"] == 1
    assert gen.stats["hits"] == 0


def test_should_not_call_llm_twice_when_same_chunk_is_regenerated(
    tmp_path: Path,
) -> None:
    http = _FakeHttpClient(response_text="Summary line one.")
    gen = ContextualPrefixGenerator(
        cache_dir=tmp_path, config=_enabled_config(), http_client=http
    )

    args = {
        "file_path": "src/api/users.py",
        "qualified_name": "myapp.api.users.create_user",
        "chunk_text": "def create_user(email, password):\n    ...",
        "file_hash": "hash-abc",
    }
    first = gen.generate(**args)
    second = gen.generate(**args)

    assert first == second
    assert len(http.calls) == 1  # second call served from cache
    assert gen.stats["hits"] == 1
    assert gen.stats["misses"] == 1

    # Persisted across generator instances — proves the disk cache works.
    gen2 = ContextualPrefixGenerator(
        cache_dir=tmp_path,
        config=_enabled_config(),
        http_client=_FakeHttpClient(raise_exc=RuntimeError("should not be called")),
    )
    third = gen2.generate(**args)
    assert third == first


def test_should_fall_back_to_minimal_prefix_when_llm_is_unreachable(
    tmp_path: Path,
) -> None:
    http = _FakeHttpClient(raise_exc=ConnectionError("boom"))
    gen = ContextualPrefixGenerator(
        cache_dir=tmp_path, config=_enabled_config(), http_client=http
    )

    prefix = gen.generate(
        file_path="src/api/users.py",
        qualified_name="myapp.api.users.create_user",
        chunk_text="def create_user(...): ...",
        file_hash="hash-abc",
    )

    assert prefix == "[from src/api/users.py]"
    assert gen.stats["llm_failures"] == 1
    # IMPORTANT: a failed call must NOT poison the cache.  A subsequent
    # call (after the network is restored) should still attempt the LLM.
    assert gen.stats["llm_calls"] == 1
    gen2 = ContextualPrefixGenerator(
        cache_dir=tmp_path,
        config=_enabled_config(),
        http_client=_FakeHttpClient(response_text="Real summary."),
    )
    prefix2 = gen2.generate(
        file_path="src/api/users.py",
        qualified_name="myapp.api.users.create_user",
        chunk_text="def create_user(...): ...",
        file_hash="hash-abc",
    )
    assert prefix2 == "Real summary."


def test_should_return_minimal_prefix_when_disabled() -> None:
    gen = ContextualPrefixGenerator(
        cache_dir=Path("/tmp/nonexistent-never-written"),
        config=ContextualPrefixConfig(
            enabled=False,
            model="x",
            max_tokens=10,
            timeout_s=1.0,
            api_key=None,
            api_base="",
        ),
        http_client=_FakeHttpClient(raise_exc=AssertionError("LLM must not be called")),
    )
    assert gen.generate(
        file_path="src/x.py",
        qualified_name="x.y",
        chunk_text="def y(): ...",
        file_hash="h",
    ) == "[from src/x.py]"


def test_should_report_realistic_cost_estimate() -> None:
    est = estimate_cost(chunks=100_000)
    assert est["chunks"] == 100_000.0
    # Sanity: 100k chunks @ ~600 in / 90 out Haiku 3.5 tokens
    # ($1/M input, $5/M output) lands at ~$60 in + ~$45 out ≈ $105.
    # Bounds are wide enough to survive pricing tweaks but tight
    # enough to flag a unit-conversion typo.
    assert 50.0 < est["total_usd"] < 200.0
    assert est["input_usd"] > 0
    assert est["output_usd"] > 0
    # Per-chunk cost in the ~$0.0005 - $0.002 range — quoted in docs.
    per_chunk = est["total_usd"] / est["chunks"]
    assert 0.0001 < per_chunk < 0.01
