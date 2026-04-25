from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.embedder import EmbeddingCache, clear_embedding_cache
from codebase_rag.utils.dependencies import has_torch, has_transformers

CODE_PREFIX = "Represent this code snippet: "
QUERY_PREFIX = "search_query: "


def _has_semantic_deps() -> bool:
    return has_torch() and has_transformers()


@pytest.fixture
def reset_model_cache() -> Generator[None, None, None]:
    if _has_semantic_deps():
        from codebase_rag.embedder import (
            get_model,  # ty: ignore[possibly-missing-import]
        )

        get_model.cache_clear()
    yield
    if _has_semantic_deps():
        from codebase_rag.embedder import (
            get_model,  # ty: ignore[possibly-missing-import]
        )

        get_model.cache_clear()


@pytest.fixture(autouse=True)
def reset_cache() -> Generator[None, None, None]:
    clear_embedding_cache()
    yield
    clear_embedding_cache()


@pytest.fixture
def mock_embed_texts() -> Generator[MagicMock, None, None]:
    """Patches _embed_texts to return a single fixed 768-dim vector per input text."""
    mock = MagicMock(side_effect=lambda texts, max_length: [[0.0] * 768 for _ in texts])
    with patch("codebase_rag.embedder._embed_texts", mock):
        yield mock


# ---------------------------------------------------------------------------
# embed_code
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_returns_768_dimensional_vector(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code

    result = embed_code("def hello(): pass")

    assert isinstance(result, list)
    assert len(result) == 768


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_prepends_code_prefix(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code

    embed_code("def test(): return 42")

    args, _ = mock_embed_texts.call_args
    assert args[0] == [CODE_PREFIX + "def test(): return 42"]


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_uses_default_max_length(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code

    embed_code("x = 1")

    args, _ = mock_embed_texts.call_args
    assert args[1] == 8192


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_respects_custom_max_length(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code

    embed_code("x = 1", max_length=256)

    args, _ = mock_embed_texts.call_args
    assert args[1] == 256


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_query_returns_768_dimensional_vector(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_query

    result = embed_query("find authentication handlers")

    assert isinstance(result, list)
    assert len(result) == 768


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_query_prepends_query_prefix(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_query

    embed_query("merge pull request")

    args, _ = mock_embed_texts.call_args
    assert args[0] == [QUERY_PREFIX + "merge pull request"]


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_query_uses_different_prefix_than_embed_code(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code, embed_query

    embed_code("def foo(): pass")
    code_args, _ = mock_embed_texts.call_args
    code_text = code_args[0][0]

    embed_query("find foo function")
    query_args, _ = mock_embed_texts.call_args
    query_text = query_args[0][0]

    assert code_text.startswith(CODE_PREFIX)
    assert query_text.startswith(QUERY_PREFIX)
    assert CODE_PREFIX != QUERY_PREFIX


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_get_model_is_cached(reset_model_cache: None) -> None:
    from codebase_rag.embedder import get_model  # ty: ignore[possibly-missing-import]

    mock_tok = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance

    with (
        patch("codebase_rag.embedder.AutoTokenizer") as mock_tokenizer_cls,
        patch("codebase_rag.embedder.AutoModel") as mock_model_cls,
        patch("codebase_rag.embedder.torch.cuda.is_available", return_value=False),
    ):
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = mock_model_instance

        result1 = get_model()
        result2 = get_model()

    assert result1 is result2
    mock_tokenizer_cls.from_pretrained.assert_called_once()
    mock_model_cls.from_pretrained.assert_called_once()


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_get_model_uses_cuda_when_available(reset_model_cache: None) -> None:
    from codebase_rag.embedder import get_model  # ty: ignore[possibly-missing-import]

    mock_tok = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.cuda.return_value = mock_model_instance

    with (
        patch("codebase_rag.embedder.AutoTokenizer") as mock_tokenizer_cls,
        patch("codebase_rag.embedder.AutoModel") as mock_model_cls,
        patch("codebase_rag.embedder.torch.cuda.is_available", return_value=True),
    ):
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = mock_model_instance

        get_model()

    mock_model_instance.cuda.assert_called_once()


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_get_model_does_not_use_cuda_when_unavailable(reset_model_cache: None) -> None:
    from codebase_rag.embedder import get_model  # ty: ignore[possibly-missing-import]

    mock_tok = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance

    with (
        patch("codebase_rag.embedder.AutoTokenizer") as mock_tokenizer_cls,
        patch("codebase_rag.embedder.AutoModel") as mock_model_cls,
        patch("codebase_rag.embedder.torch.cuda.is_available", return_value=False),
    ):
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = mock_model_instance

        get_model()

    mock_model_instance.cuda.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests (marked slow — require real model download)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
@pytest.mark.slow
def test_embed_code_integration(reset_model_cache: None) -> None:
    from codebase_rag.embedder import embed_code

    code = "def add(a, b): return a + b"
    result = embed_code(code)

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
@pytest.mark.slow
def test_similar_code_has_similar_embeddings(reset_model_cache: None) -> None:
    from codebase_rag.embedder import embed_code

    code1 = "def add(a, b): return a + b"
    code2 = "def sum(x, y): return x + y"
    code3 = "class DatabaseConnection: pass"

    emb1 = embed_code(code1)
    emb2 = embed_code(code2)
    emb3 = embed_code(code3)

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)

    assert sim_1_2 > sim_1_3


# ---------------------------------------------------------------------------
# Without-dependencies fallback
# ---------------------------------------------------------------------------


def test_embed_code_raises_without_dependencies() -> None:
    if _has_semantic_deps():
        pytest.skip("Dependencies are installed")

    from codebase_rag.embedder import embed_code

    with pytest.raises(RuntimeError, match="Semantic search requires"):
        embed_code("x = 1")


def test_embed_query_raises_without_dependencies() -> None:
    if _has_semantic_deps():
        pytest.skip("Dependencies are installed")

    from codebase_rag.embedder import embed_query

    with pytest.raises(RuntimeError, match="Semantic search requires"):
        embed_query("find something")


# ---------------------------------------------------------------------------
# EmbeddingCache unit tests (model-agnostic)
# ---------------------------------------------------------------------------


def test_embedding_cache_put_and_get() -> None:
    cache = EmbeddingCache()
    embedding = [0.1, 0.2, 0.3]
    cache.put("def foo(): pass", embedding)
    assert cache.get("def foo(): pass") == embedding


def test_embedding_cache_miss_returns_none() -> None:
    cache = EmbeddingCache()
    assert cache.get("unknown code") is None


def test_embedding_cache_different_content_different_key() -> None:
    cache = EmbeddingCache()
    cache.put("code_a", [1.0])
    cache.put("code_b", [2.0])
    assert cache.get("code_a") == [1.0]
    assert cache.get("code_b") == [2.0]


def test_embedding_cache_overwrite() -> None:
    cache = EmbeddingCache()
    cache.put("code_a", [1.0])
    cache.put("code_a", [9.9])
    assert cache.get("code_a") == [9.9]


def test_embedding_cache_len() -> None:
    cache = EmbeddingCache()
    assert len(cache) == 0
    cache.put("a", [1.0])
    assert len(cache) == 1
    cache.put("b", [2.0])
    assert len(cache) == 2


def test_embedding_cache_clear() -> None:
    cache = EmbeddingCache()
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_embedding_cache_get_many() -> None:
    cache = EmbeddingCache()
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    results = cache.get_many(["a", "c", "b"])
    assert results == {0: [1.0], 2: [2.0]}


def test_embedding_cache_put_many() -> None:
    cache = EmbeddingCache()
    cache.put_many(["x", "y"], [[1.0], [2.0]])
    assert cache.get("x") == [1.0]
    assert cache.get("y") == [2.0]


def test_embedding_cache_save_and_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.json"
        cache = EmbeddingCache(path=cache_path)
        cache.put("hello", [0.5, 0.6])
        cache.save()

        assert cache_path.exists()

        cache2 = EmbeddingCache(path=cache_path)
        cache2.load()
        assert cache2.get("hello") == [0.5, 0.6]


def test_embedding_cache_load_nonexistent_path() -> None:
    cache = EmbeddingCache(path=Path("/nonexistent/path/cache.json"))
    cache.load()
    assert len(cache) == 0


def test_embedding_cache_load_corrupt_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "corrupt.json"
        cache_path.write_text("not valid json data", encoding="utf-8")
        cache = EmbeddingCache(path=cache_path)
        cache.load()
        assert len(cache) == 0


def test_embedding_cache_save_no_path() -> None:
    cache = EmbeddingCache(path=None)
    cache.put("a", [1.0])
    cache.save()


def test_embedding_cache_load_no_path() -> None:
    cache = EmbeddingCache(path=None)
    cache.load()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# embed_code cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_uses_cache(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code, get_embedding_cache

    cache = get_embedding_cache()
    cache.put("cached_code", [0.42] * 768)

    result = embed_code("cached_code")

    assert result == [0.42] * 768
    mock_embed_texts.assert_not_called()


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_populates_cache(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code, get_embedding_cache

    embed_code("new_code")

    cache = get_embedding_cache()
    assert cache.get("new_code") is not None


# ---------------------------------------------------------------------------
# embed_code_batch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_empty_list(reset_model_cache: None) -> None:
    from codebase_rag.embedder import embed_code_batch

    assert embed_code_batch([]) == []


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_returns_correct_count(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code_batch

    snippets = ["def a(): pass", "def b(): pass", "def c(): pass"]
    results = embed_code_batch(snippets)

    assert len(results) == 3
    assert all(len(emb) == 768 for emb in results)


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_prepends_code_prefix(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code_batch

    snippets = ["short", "longer code here"]
    embed_code_batch(snippets)

    args, _ = mock_embed_texts.call_args
    assert args[0] == [CODE_PREFIX + s for s in snippets]


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_cache_hit(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code_batch, get_embedding_cache

    cache = get_embedding_cache()
    cache.put("a", [1.0] * 768)
    cache.put("b", [2.0] * 768)

    results = embed_code_batch(["a", "b"])

    mock_embed_texts.assert_not_called()
    assert results == [[1.0] * 768, [2.0] * 768]


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_partial_cache(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code_batch, get_embedding_cache

    cache = get_embedding_cache()
    cache.put("a", [1.0] * 768)

    mock_embed_texts.side_effect = lambda texts, max_length: [[3.0] * 768 for _ in texts]

    results = embed_code_batch(["a", "b"])

    assert results[0] == [1.0] * 768
    assert results[1] == [3.0] * 768
    args, _ = mock_embed_texts.call_args
    assert args[0] == [CODE_PREFIX + "b"]


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_populates_cache(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code_batch, get_embedding_cache

    embed_code_batch(["new_snippet"])

    cache = get_embedding_cache()
    assert cache.get("new_snippet") is not None


@pytest.mark.skipif(not _has_semantic_deps(), reason="torch/transformers not installed")
def test_embed_code_batch_respects_batch_size(
    mock_embed_texts: MagicMock, reset_model_cache: None
) -> None:
    from codebase_rag.embedder import embed_code_batch

    snippets = [f"def f{i}(): pass" for i in range(5)]

    results = embed_code_batch(snippets, batch_size=2)

    assert len(results) == 5
    assert mock_embed_texts.call_count == 3


def test_embed_code_batch_raises_without_dependencies() -> None:
    if _has_semantic_deps():
        pytest.skip("Dependencies are installed")

    from codebase_rag.embedder import embed_code_batch

    with pytest.raises(RuntimeError, match="Semantic search requires"):
        embed_code_batch(["x = 1"])


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


def test_embedding_cache_persistence_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "subdir" / "cache.json"

        cache1 = EmbeddingCache(path=cache_path)
        cache1.put("fn_a", [0.1, 0.2])
        cache1.put("fn_b", [0.3, 0.4])
        cache1.save()

        cache2 = EmbeddingCache(path=cache_path)
        cache2.load()
        assert cache2.get("fn_a") == [0.1, 0.2]
        assert cache2.get("fn_b") == [0.3, 0.4]
        assert cache2.get("fn_c") is None
        assert len(cache2) == 2
