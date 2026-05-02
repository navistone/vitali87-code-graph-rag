"""Unit tests for LMStudioEmbedder.batch_embed (mocked HTTP).

These tests run without a real LM Studio instance — the HTTP layer is
patched at ``urllib.request.urlopen`` so they are fully offline.
"""
from __future__ import annotations

import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.embedder import LMStudioEmbedder, get_lm_studio_embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_urlopen(url: str | object, *, timeout: float = 5.0):
    """Context-manager stub that returns a 768-dim zero vector per input."""
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self) -> bytes:
            # Determine how many inputs were sent.
            if isinstance(url, str):
                # /v1/models probe
                data = {"data": [{"id": "nomic-ai/CodeRankEmbed-Q4"}]}
                return json.dumps(data).encode("utf-8")
            # /v1/embeddings POST — url is actually a Request object
            body = json.loads(url.data.decode("utf-8"))
            inputs = body["input"]
            n = len(inputs) if isinstance(inputs, list) else 1
            rows = [
                {"index": i, "embedding": [0.0] * 768}
                for i in range(n)
            ]
            return json.dumps({"data": rows}).encode("utf-8")

    return _FakeResp()


@pytest.fixture(autouse=True)
def clear_lm_cache() -> Generator[None, None, None]:
    """Reset the module-level LMStudioEmbedder singleton cache between tests."""
    get_lm_studio_embedder.cache_clear()
    yield
    get_lm_studio_embedder.cache_clear()


# ---------------------------------------------------------------------------
# LMStudioEmbedder.batch_embed — unit tests
# ---------------------------------------------------------------------------


def test_batch_embed_returns_empty_list_for_empty_input() -> None:
    """should return [] when texts is empty."""
    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    result = embedder.batch_embed([])
    assert result == []


def test_batch_embed_returns_correct_vector_count() -> None:
    """should return one vector per input text."""
    texts = ["def foo(): pass", "def bar(): pass", "class Baz: pass"]

    def _mock_post(self_ignored: object, payload: dict) -> dict:
        inputs = payload["input"]
        return {
            "data": [
                {"index": i, "embedding": [float(i)] * 768}
                for i in range(len(inputs))
            ]
        }

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    with patch.object(LMStudioEmbedder, "_post", _mock_post):
        result = embedder.batch_embed(texts)

    assert result is not None
    assert len(result) == 3
    assert all(len(v) == 768 for v in result)


def test_batch_embed_preserves_order() -> None:
    """should return vectors in the same order as input texts, even if server reorders."""
    texts = ["a", "b", "c"]

    def _mock_post_reversed(self_ignored: object, payload: dict) -> dict:
        # Server returns rows in reverse order — client must sort by ``index``.
        inputs = payload["input"]
        rows = [
            {"index": i, "embedding": [float(i + 10)] * 4}
            for i in range(len(inputs))
        ]
        return {"data": list(reversed(rows))}

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    with patch.object(LMStudioEmbedder, "_post", _mock_post_reversed):
        result = embedder.batch_embed(texts)

    assert result is not None
    # index 0 → [10.0]*4, index 1 → [11.0]*4, etc.
    assert result[0] == [10.0] * 4
    assert result[1] == [11.0] * 4
    assert result[2] == [12.0] * 4


def test_batch_embed_prepends_prefix() -> None:
    """should prepend the caller-supplied prefix to every text before sending."""
    captured: list[dict] = []

    def _mock_post(self_ignored: object, payload: dict) -> dict:
        captured.append(payload)
        inputs = payload["input"]
        return {"data": [{"index": i, "embedding": [0.0] * 4} for i in range(len(inputs))]}

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    prefix = "Represent this code snippet: "
    with patch.object(LMStudioEmbedder, "_post", _mock_post):
        embedder.batch_embed(["def foo(): pass"], prefix=prefix)

    assert len(captured) == 1
    assert captured[0]["input"] == [prefix + "def foo(): pass"]


def test_batch_embed_chunks_at_batch_size() -> None:
    """should split into multiple HTTP requests when len(texts) > batch_size."""
    call_count = [0]

    def _mock_post(self_ignored: object, payload: dict) -> dict:
        call_count[0] += 1
        inputs = payload["input"]
        return {"data": [{"index": i, "embedding": [0.0] * 4} for i in range(len(inputs))]}

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    texts = [f"text_{i}" for i in range(10)]
    with patch.object(LMStudioEmbedder, "_post", _mock_post):
        result = embedder.batch_embed(texts, batch_size=3)

    # 10 texts / chunk_size 3 → ceil(10/3) = 4 requests
    assert call_count[0] == 4
    assert result is not None
    assert len(result) == 10


def test_batch_embed_returns_none_on_network_error() -> None:
    """should return None (not raise) when _post raises RuntimeError."""

    def _mock_post_fail(self_ignored: object, payload: dict) -> dict:
        raise RuntimeError("connection refused")

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    with patch.object(LMStudioEmbedder, "_post", _mock_post_fail):
        result = embedder.batch_embed(["hello"])

    assert result is None


def test_batch_embed_returns_none_on_count_mismatch() -> None:
    """should return None when server returns fewer embeddings than inputs."""

    def _mock_post_short(self_ignored: object, payload: dict) -> dict:
        # Return only 1 row for a 2-text input.
        return {"data": [{"index": 0, "embedding": [0.0] * 4}]}

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    with patch.object(LMStudioEmbedder, "_post", _mock_post_short):
        result = embedder.batch_embed(["a", "b"])

    assert result is None


def test_embed_single_delegates_to_batch() -> None:
    """should call batch_embed internally and return the first vector."""

    def _mock_post(self_ignored: object, payload: dict) -> dict:
        inputs = payload["input"]
        return {"data": [{"index": i, "embedding": [float(i + 1)] * 4} for i in range(len(inputs))]}

    embedder = LMStudioEmbedder(
        base_url="http://127.0.0.1:1234", model="test-model"
    )
    with patch.object(LMStudioEmbedder, "_post", _mock_post):
        result = embedder.embed("some text")

    assert result == [1.0] * 4


# ---------------------------------------------------------------------------
# LMStudioEmbedder.from_env
# ---------------------------------------------------------------------------


def test_from_env_returns_none_when_url_unset() -> None:
    """should return None when LM_STUDIO_URL env var is absent."""
    with patch.dict("os.environ", {}, clear=True):
        assert LMStudioEmbedder.from_env() is None


def test_from_env_returns_none_when_url_empty() -> None:
    """should return None when LM_STUDIO_URL is set to empty string."""
    with patch.dict("os.environ", {"LM_STUDIO_URL": ""}):
        assert LMStudioEmbedder.from_env() is None


def test_from_env_returns_embedder_when_model_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """should return LMStudioEmbedder when LM_STUDIO_URL is set and model resolves."""
    monkeypatch.setenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("LM_STUDIO_EMBED_MODEL", "CodeRankEmbed")

    with patch.object(
        LMStudioEmbedder,
        "_resolve_model",
        staticmethod(lambda base_url, hint: "nomic-ai/CodeRankEmbed-Q4_K_M.gguf"),
    ):
        embedder = LMStudioEmbedder.from_env()

    assert embedder is not None
    assert embedder._model == "nomic-ai/CodeRankEmbed-Q4_K_M.gguf"


def test_from_env_returns_none_when_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """should return None when no loaded model matches the hint."""
    monkeypatch.setenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("LM_STUDIO_EMBED_MODEL", "CodeRankEmbed")

    with patch.object(
        LMStudioEmbedder,
        "_resolve_model",
        staticmethod(lambda base_url, hint: None),
    ):
        result = LMStudioEmbedder.from_env()

    assert result is None


# ---------------------------------------------------------------------------
# get_lm_studio_embedder singleton
# ---------------------------------------------------------------------------


def test_get_lm_studio_embedder_returns_none_without_env() -> None:
    """should return None when LM_STUDIO_URL is not set."""
    with patch.dict("os.environ", {}, clear=True):
        result = get_lm_studio_embedder()
    assert result is None


def test_get_lm_studio_embedder_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """should return the same object on repeated calls (lru_cache)."""
    monkeypatch.setenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("LM_STUDIO_EMBED_MODEL", "CodeRankEmbed")

    with patch.object(
        LMStudioEmbedder,
        "_resolve_model",
        staticmethod(lambda base_url, hint: "nomic-ai/CodeRankEmbed-Q4_K_M.gguf"),
    ):
        result1 = get_lm_studio_embedder()
        result2 = get_lm_studio_embedder()

    assert result1 is result2


# ---------------------------------------------------------------------------
# Integration smoke (skipped when LM Studio is not reachable)
# ---------------------------------------------------------------------------


def _lm_studio_reachable() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2.0):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _lm_studio_reachable(), reason="LM Studio not running on :1234")
def test_batch_embed_integration_live() -> None:
    """Integration: embed 5 texts against a live LM Studio instance.

    Skipped unless http://127.0.0.1:1234 responds — runs only in environments
    where LM Studio is available (e.g. the local dev machine or a CI runner
    with the model pre-loaded).
    """
    embedder = LMStudioEmbedder.from_env()
    if embedder is None:
        pytest.skip("LM_STUDIO_URL not set or model not loaded")

    texts = [
        "def add(a, b): return a + b",
        "class FooBar: pass",
        "import os; path = os.getcwd()",
        "async def fetch(url): ...",
        "x = [i**2 for i in range(10)]",
    ]
    result = embedder.batch_embed(texts, prefix="Represent this code snippet: ")

    assert result is not None
    assert len(result) == len(texts)
    assert all(isinstance(v, list) and len(v) > 0 for v in result)
