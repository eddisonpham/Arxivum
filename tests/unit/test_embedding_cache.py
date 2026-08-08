"""Unit tests for the EmbeddingCache (Tier 2.1)."""

from __future__ import annotations

import math
import os
import tempfile

import pytest


@pytest.fixture
def cache(tmp_path):
    from src.inference.embed_cache import EmbeddingCache
    return EmbeddingCache(
        capacity=8,
        persistent_path=tmp_path / "embed_cache.sqlite",
    )


def test_miss_then_hit_returns_same_vector(cache):
    cache.store("model-A", "alpha", [0.1, 0.2, 0.3])
    out = cache.lookup("model-A", "alpha")
    assert out == [0.1, 0.2, 0.3]
    s = cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 0
    assert s["inserts"] == 1


def test_different_models_have_separate_keys(cache):
    cache.store("model-A", "x", [1.0])
    cache.store("model-B", "x", [2.0])
    assert cache.lookup("model-A", "x") == [1.0]
    assert cache.lookup("model-B", "x") == [2.0]
    assert cache.stats()["hits"] == 2


def test_lru_eviction_at_capacity():
    """Capacity=8 -> the 9th insert evicts the LRU entry.

    The persistent layer (SQLite) is *not* pruned at eviction time:
    its job is to warm-start the next session, so we deliberately
    leave evicted entries there and accept that on a cache miss
    after eviction the lookup re-finds the entry via persistent_hits.
    This test uses a process-local cache (no persistent_path) so a
    miss is a clean None and we can pin the LRU invariant.
    """
    from src.inference.embed_cache import EmbeddingCache
    c = EmbeddingCache(capacity=8, persistent_path=None)
    for i in range(9):
        c.store("m", f"k{i}", [float(i)])
    # First key 'k0' is evicted from in-process LRU.
    assert c.lookup("m", "k0") is None
    assert c.lookup("m", "k8") == [8.0]
    s = c.stats()
    assert s["evictions"] >= 1


def test_persistent_layer_round_trips(tmp_path):
    """Insert, close, reopen with a fresh cache; key hits via the SQL layer."""
    from src.inference.embed_cache import EmbeddingCache

    p = tmp_path / "embed_cache.sqlite"
    c1 = EmbeddingCache(capacity=4, persistent_path=p)
    c1.store("model-A", "hello", [0.1, 0.2])
    c1.close()

    c2 = EmbeddingCache(capacity=4, persistent_path=p)
    out = c2.lookup("model-A", "hello")
    assert out is not None
    assert pytest.approx(out[0], abs=1e-6) == 0.1
    s = c2.stats()
    assert s["persistent_hits"] == 1
    c2.close()


def test_default_cache_singleton():
    """The default-cache helper returns the same object across calls."""
    from src.inference.embed_cache import (
        default_cache, reset_default_cache,
    )
    reset_default_cache()
    a = default_cache()
    b = default_cache()
    assert a is b
    reset_default_cache()


def test_lookup_returns_none_for_unknown_model(cache):
    assert cache.lookup("never-stored", "anything") is None


def test_empty_vector_is_not_stored(cache):
    """An empty vector is rejected from the store path."""
    cache.store("m", "x", [])
    assert cache.lookup("m", "x") is None


def test_long_vector_round_trips():
    """1024-d BGE-large vectors fit through the cache unchanged."""
    from src.inference.embed_cache import EmbeddingCache
    c = EmbeddingCache(capacity=2, persistent_path=None)
    v = [math.sin(i * 0.01) for i in range(1024)]
    c.store("BAAI/bge-large-en-v1.5", "abstract", v)
    out = c.lookup("BAAI/bge-large-en-v1.5", "abstract")
    assert out is not None
    assert len(out) == 1024
    assert max(abs(o - e) for o, e in zip(out, v)) < 1e-12
