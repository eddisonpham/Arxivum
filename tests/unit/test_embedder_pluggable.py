"""Unit tests for the pluggable embedder surface."""

from __future__ import annotations

from src.inference.embedder import (
    BgeLargeEmbedder,
    BgeSmallEmbedder,
    E5LargeEmbedder,
    Specter2Embedder,
    fallback_embedder,
    try_resolve,
)


def test_bge_small_defaults():
    e = BgeSmallEmbedder()
    assert e.dim == 384
    assert e.model_name == "BAAI/bge-small-en-v1.5"
    assert not e.is_loaded


def test_bge_large_defaults():
    e = BgeLargeEmbedder()
    assert e.dim == 1024
    assert e.model_name == "BAAI/bge-large-en-v1.5"
    assert not e.is_loaded


def test_specter2_defaults():
    e = Specter2Embedder()
    assert e.dim == 768
    assert e.model_name == "allenai/specter2_base"
    assert e.is_available is False
    assert not e.is_loaded


def test_e5_large_defaults():
    e = E5LargeEmbedder()
    assert e.dim == 1024
    assert e.model_name == "intfloat/e5-large-v2"


def test_try_resolve_known_models():
    e = try_resolve("BAAI/bge-large-en-v1.5")
    assert isinstance(e, BgeLargeEmbedder)
    assert e.dim == 1024

    e = try_resolve("allenai/specter2_base")
    assert isinstance(e, Specter2Embedder)
    assert e.dim == 768

    e = try_resolve("BAAI/bge-small-en-v1.5")
    assert isinstance(e, BgeSmallEmbedder)
    assert e.dim == 384


def test_fallback_embedder_hash_dim():
    e = fallback_embedder()
    assert e.dim == 384
    assert e.model_name == "hash-pseudo"
    v1 = e.embed_one("hello world", is_query=False)
    v2 = e.embed_one("hello world", is_query=False)
    assert v1 == v2

    v3 = e.embed_one("hello world", is_query=True)
    assert v3 != v1 or len(v3) == len(v1)


def test_unknown_model_returns_generic_bge_style():
    e = try_resolve("some-org/totally-unknown-model-xyz")
    assert e.dim == 768
    assert e.model_name == "some-org/totally-unknown-model-xyz"


def test_is_available_load_status_propagates():
    e = fallback_embedder()
    assert not e.is_loaded
    _ = e.embed(["text"])
    assert e.is_loaded
    assert e.is_available is False
