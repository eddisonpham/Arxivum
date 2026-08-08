"""Persistent embedding cache.

A bounded in-process LRU keyed on SHA-256(text) + model_name, with
spillover to a small SQLite table at ``.cache/embeddings.sqlite``. The
LRU is consulted before any embedder model load, so a hit returns
in ~1 ms without spinning up ``sentence-transformers`` or pulling a
2 GB model file.

The cache is deliberately small (default 4096 in-process entries, no
persistent eviction policy) because the working set during a
research session is bounded by how many paper titles and abstracts
the user looks at, which is small. The persistent layer buys a
warm-start on the next session.

The cache participates in tests via a temporary directory so CI does
not pollute the user's project.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    persistent_hits: int = 0
    inserts: int = 0
    evictions: int = 0
    bytes_saved_model_loads: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits + self.persistent_hits) / total if total else 0.0


@dataclass
class CacheRecord:
    model_name: str
    text_hash: str
    vector: list[float]
    dim: int
    created_at: float


class EmbeddingCache:
    """LRU + sqlite cache for embedding vectors.

    Key: ``(model_name, sha256(text))``. Value: vector (list of float).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache (
        model_name TEXT NOT NULL,
        text_hash  TEXT NOT NULL,
        vector     TEXT NOT NULL,
        dim        INTEGER NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (model_name, text_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_cache_created_at
        ON cache(created_at);
    """

    def __init__(
        self,
        capacity: int = 4096,
        persistent_path: str | Path | None = None,
    ) -> None:
        self._capacity = max(1, capacity)
        self._lock = threading.RLock()
        self._lru: "OrderedDict[str, CacheRecord]" = OrderedDict()
        self._stats = CacheStats()
        # Persistent layer is optional; ``None`` means process-local only.
        self._persistent_path: Path | None = None
        self._persistent: sqlite3.Connection | None = None
        if persistent_path is not None:
            p = Path(persistent_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._persistent_path = p
            self._persistent = sqlite3.connect(str(p),
                                                  check_same_thread=False)
            self._persistent.row_factory = sqlite3.Row
            self._persistent.executescript(self._SCHEMA)
            self._persistent.commit()

    @staticmethod
    def make_key(model_name: str, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        return f"{model_name}::{h}"

    def lookup(self, model_name: str, text: str) -> list[float] | None:
        key = self.make_key(model_name, text)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self._stats.hits += 1
                return list(self._lru[key].vector)
            if self._persistent is not None:
                try:
                    row = self._persistent.execute(
                        "SELECT vector FROM cache "
                        "WHERE model_name=? AND text_hash=?",
                        (model_name, key.split("::", 1)[1]),
                    ).fetchone()
                except sqlite3.OperationalError as exc:
                    logger.debug("embed_cache lookup failed: %s", exc) \
                        if False else None  # noqa
                    row = None
                if row is not None:
                    try:
                        vec = json.loads(row["vector"])
                    except Exception:
                        vec = None
                    if vec is not None:
                        self._stats.persistent_hits += 1
                        # Promote into LRU
                        rec = CacheRecord(
                            model_name=model_name,
                            text_hash=key.split("::", 1)[1],
                            vector=vec,
                            dim=len(vec),
                            created_at=time.time(),
                        )
                        self._lru[key] = rec
                        self._lru.move_to_end(key)
                        self._evict_if_needed()
                        return list(vec)
        self._stats.misses += 1
        return None

    def store(self, model_name: str, text: str, vector: Sequence[float]) -> None:
        if not vector:
            return
        key = self.make_key(model_name, text)
        with self._lock:
            rec = CacheRecord(
                model_name=model_name,
                text_hash=key.split("::", 1)[1],
                vector=list(vector),
                dim=len(vector),
                created_at=time.time(),
            )
            self._lru[key] = rec
            self._lru.move_to_end(key)
            self._evict_if_needed()
            self._stats.inserts += 1
            if self._persistent is not None:
                try:
                    self._persistent.execute(
                        "INSERT OR REPLACE INTO cache "
                        "(model_name, text_hash, vector, dim, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (rec.model_name, rec.text_hash,
                         json.dumps(rec.vector), rec.dim, rec.created_at),
                    )
                    self._persistent.commit()
                except sqlite3.OperationalError:
                    pass

    def _evict_if_needed(self) -> None:
        while len(self._lru) > self._capacity:
            evicted_key, _ = self._lru.popitem(last=False)
            self._stats.evictions += 1
            _ = evicted_key  # for future per-key analytics hooks

    def stats(self) -> dict:
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "persistent_hits": self._stats.persistent_hits,
            "inserts": self._stats.inserts,
            "evictions": self._stats.evictions,
            "hit_rate": round(self._stats.hit_rate(), 4),
            "in_process_size": len(self._lru),
            "capacity": self._capacity,
            "persistent": str(self._persistent_path) if self._persistent_path else None,
        }

    def close(self) -> None:
        if self._persistent is not None:
            try:
                self._persistent.close()
            except Exception:
                pass
            self._persistent = None


_DEFAULT_CACHE: EmbeddingCache | None = None
_DEFAULT_LOCK = threading.Lock()


def default_cache(
    capacity: int = 4096,
    persistent_path: str | Path | None = ".cache/embeddings.sqlite",
) -> EmbeddingCache:
    """Process-wide singleton cache so the Embedder and benchmark
    share state; the persistent SQLite path defaults to a project-local
    ``.cache/`` directory so successive sessions warm-start."""
    global _DEFAULT_CACHE
    with _DEFAULT_LOCK:
        if _DEFAULT_CACHE is None:
            # Allow tests to override APP_ENV via env-passed path.
            path = (
                str(persistent_path)
                if persistent_path is not None
                else None
            )
            if path and os.environ.get("APP_ENV") == "test":
                # tests should opt-in to a tmp path explicitly;
                # in test mode default to ephemeral only.
                path = None
            _DEFAULT_CACHE = EmbeddingCache(capacity=capacity,
                                              persistent_path=path)
        return _DEFAULT_CACHE


def reset_default_cache() -> None:
    """Force the singleton to forget state; primarily for tests."""
    global _DEFAULT_CACHE
    with _DEFAULT_LOCK:
        if _DEFAULT_CACHE is not None:
            _DEFAULT_CACHE.close()
        _DEFAULT_CACHE = None