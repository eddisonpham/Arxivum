"""Benchmark runner utilities.

Builds an ephemeral AppContext with the synthetic library pre-loaded
into both SQLite and ChromaDB. Uses deterministic hash-based pseudo-
embeddings so that retrieval benchmarks are reproducible offline
(no embedder model load, no arXiv network).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from src.app import AppContext, create_app, shutdown_app
from src.db.chroma_store import CHUNK_ABSTRACT, CHUNK_TITLE, Chunk, ChromaStore
from src.db.models import MetricsRow, PaperRow

from tests.benchmark.synthetic import SYNTHETIC_LIBRARY

logger = logging.getLogger("bench")
logging.basicConfig(level=logging.WARNING)


EMBED_DIM = 384


def _hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    out = [0.0] * dim
    for k in range(8):
        h = hashlib.sha256(f"{k}::{text}".encode()).digest()
        for i in range(dim):
            out[i] += (h[i % len(h)] - 128) / 128.0
    norm = math.sqrt(sum(x * x for x in out))
    if norm == 0:
        return out
    return [x / norm for x in out]


class StubEmbedder:
    """Stub replacing Embedder for offline benchmarks."""

    def __init__(self) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return "stub-embedder"

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def _ensure_loaded(self) -> None:
        pass

    def embed(self, texts: Sequence[str], is_query: bool = False, normalize: bool = True) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]

    def embed_one(self, text: str, is_query: bool = False, normalize: bool = True) -> list[float]:
        return _hash_embed(text)

    def unload(self) -> None:
        self._loaded = False


@dataclass
class BenchEnv:
    ctx: AppContext
    library_keys: list[str]
    chroma: ChromaStore

    def close(self) -> None:
        shutdown_app(self.ctx)


def build_env(stub_llm: object | None = None) -> BenchEnv:
    """Build an in-memory AppContext seeded with the synthetic library."""
    ctx = create_app(db_path="", chroma_path="")
    if stub_llm is not None:
        ctx.models.set_llm(stub_llm)
    ctx.models._embedder = StubEmbedder()
    ctx.models._resident = "embedder"

    chroma = ctx.chroma
    chroma.reset()

    for p in SYNTHETIC_LIBRARY:
        row = PaperRow(
            arxiv_id=p["arxiv_id"],
            title=p["title"],
            authors=p["authors"],
            abstract=p["abstract"],
            published=p["published"],
            categories=[p["primary_category"]],
            primary_category=p["primary_category"],
            pdf_url=f"https://arxiv.org/pdf/{p['arxiv_id']}",
            abs_url=f"https://arxiv.org/abs/{p['arxiv_id']}",
        )
        ctx.db.upsert_paper(row)
        ctx.db.upsert_metrics(MetricsRow(
            arxiv_id=p["arxiv_id"],
            citation_count=p["citation_count"],
            influential_citation_count=max(0, p["citation_count"] // 4),
            venue=p["venue"],
        ))
        chroma.upsert_chunks([
            Chunk(
                id=ChromaStore.chunk_id(p["arxiv_id"], CHUNK_ABSTRACT),
                document=p["abstract"],
                arxiv_id=p["arxiv_id"],
                chunk_type=CHUNK_ABSTRACT,
                title=p["title"],
                venue=p["venue"],
                citation_count=p["citation_count"],
                primary_category=p["primary_category"],
                embedding=_hash_embed(p["abstract"]),
            ),
            Chunk(
                id=ChromaStore.chunk_id(p["arxiv_id"], CHUNK_TITLE),
                document=p["title"],
                arxiv_id=p["arxiv_id"],
                chunk_type=CHUNK_TITLE,
                title=p["title"],
                venue=p["venue"],
                citation_count=p["citation_count"],
                primary_category=p["primary_category"],
                embedding=_hash_embed(p["title"]),
            ),
        ])

    return BenchEnv(
        ctx=ctx,
        library_keys=[p["arxiv_id"] for p in SYNTHETIC_LIBRARY],
        chroma=chroma,
    )


def timed(fn: Callable, *args, **kwargs) -> tuple[float, object]:
    """Return (wall_ms, result) of fn(*args, **kwargs)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000.0, out
