"""Pluggable embedding model wrapper.

Supports three model families relevant to scientific retrieval:

  * ``BAAI/bge-small-en-v1.5``  (default, 384-d)
  * ``BAAI/bge-large-en-v1.5``  (1024-d, ~340M params, ~1.3 GB)
  * ``allenai/specter2_base``   (768-d, ~110M params; uses
    triplet-loss fine-tuning with document / query prefixes per the
    SPECTER2 paper)

Each family has its own BGE-style query prefix and (for SPECTER2)
its own document prefix.  The wrapper exposes a uniform
``embed`` / ``embed_one`` surface so swapping the embedder is a
config-only change (``EMBEDDING_MODEL=...``).

Loading is best-effort: if the resolved model cannot be loaded
(missing weights, no ``sentence-transformers`` installed, OOM),
``is_available()`` returns False and the model manager falls back to a
hash pseudo-embedder so the rest of the pipeline still runs.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Sequence

logger = logging.getLogger(__name__)


_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_E5_QUERY_PREFIX = "query: "
_SPECTER2_QUERY_PREFIX = ""
_SPECTER2_DOC_PREFIX = ""


class Embedder(ABC):
    """Abstract base for embedder implementations.

    All concrete embedders expose ``embed``, ``embed_one``,
    ``is_loaded``, ``unload``, ``dim``, and ``model_name``.
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def embed(
        self,
        texts: Sequence[str],
        is_query: bool = False,
        normalize: bool = True,
    ) -> list[list[float]]: ...

    def embed_one(
        self, text: str, is_query: bool = False, normalize: bool = True
    ) -> list[float]:
        return self.embed([text], is_query=is_query, normalize=normalize)[0]

    @abstractmethod
    def unload(self) -> None: ...

    @property
    def is_available(self) -> bool:
        """Return True if real weights are loaded (not stub)."""
        return self.is_loaded and not isinstance(self, _HashEmbedder)


class _SentenceTransformerEmbedder(Embedder):
    """BGE / E5 family wrapper, lazy-loaded via ``sentence-transformers``."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str,
        dim: int,
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._dim = dim
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("Loading embedder: %s", self._model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(
        self,
        texts: Sequence[str],
        is_query: bool = False,
        normalize: bool = True,
    ) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        inputs = (
            [f"{self._query_prefix}{t}" for t in texts]
            if is_query and self._query_prefix
            else list(texts)
        )
        vecs = self._model.encode(
            inputs,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Embedder unloaded: %s", self._model_name)


class Specter2Embedder(_SentenceTransformerEmbedder):
    """SPECTER2 embedder.

    SPECTER2 is trained with a triplet loss and does NOT follow the
    BGE-style query instruction.  Instead it expects plain ``title [SEP] abstract``
    on both the document and query side, and the original AllenAI
    recommendation is to embed the concatenated title + abstract.
    The query prefix is therefore empty.
    """

    def __init__(self, model_name: str = "allenai/specter2_base") -> None:
        super().__init__(
            model_name=model_name,
            query_prefix=_SPECTER2_QUERY_PREFIX,
            dim=768,
        )


class BgeSmallEmbedder(_SentenceTransformerEmbedder):
    """BGE-small English embedding model (384-d, ~33M params)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        super().__init__(
            model_name=model_name,
            query_prefix=_BGE_QUERY_PREFIX,
            dim=384,
        )


class BgeLargeEmbedder(_SentenceTransformerEmbedder):
    """BGE-large English embedding model (1024-d, ~340M params).

    Recommended for CPU / single GPU scientific retrieval: published
    BEIR / SciFact scores rank it roughly 4-5 NDCG@10 points above
    BGE-small.  Roughly 1.3 GB on disk; ~3-5x slower than BGE-small
    on CPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        super().__init__(
            model_name=model_name,
            query_prefix=_BGE_QUERY_PREFIX,
            dim=1024,
        )


class E5LargeEmbedder(_SentenceTransformerEmbedder):
    """E5-large English embedding model (1024-d)."""

    def __init__(self, model_name: str = "intfloat/e5-large-v2") -> None:
        super().__init__(
            model_name=model_name,
            query_prefix=_E5_QUERY_PREFIX,
            dim=1024,
        )


class _HashEmbedder(Embedder):
    """Fake embedder that deterministically hashes text into a 384-d vector.

    Used as a last-resort fallback so the rest of the pipeline still
    functions when no real weights are present.  NOT comparable to
    real embeddings; NDCG scores on this embedder should be ignored.
    """

    _DIM = 384

    def __init__(self) -> None:
        self._loaded = False

    @property
    def model_name(self) -> str:
        return "hash-pseudo"

    @property
    def dim(self) -> int:
        return self._DIM

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> None:
        self._loaded = True

    def embed(
        self,
        texts: Sequence[str],
        is_query: bool = False,
        normalize: bool = True,
    ) -> list[list[float]]:
        self._loaded = True
        results: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            data = (h * 12)[: self._DIM]
            vec = [(b / 255.0 - 0.5) * 2 for b in data]
            if normalize:
                norm = sum(v * v for v in vec) ** 0.5 or 1.0
                vec = [v / norm for v in vec]
            results.append(vec)
        return results

    def unload(self) -> None:
        self._loaded = False


_KNOWN_EMBEDDERS: dict[str, type[_SentenceTransformerEmbedder]] = {
    "BAAI/bge-small-en-v1.5": BgeSmallEmbedder,
    "BAAI/bge-large-en-v1.5": BgeLargeEmbedder,
    "allenai/specter2_base": Specter2Embedder,
    "allenai/specter2_pro": Specter2Embedder,
    "intfloat/e5-large-v2": E5LargeEmbedder,
}


def try_resolve(model_name: str) -> Embedder:
    """Resolve a model name to an embedder instance without touching weights.

    Returns a real embedder for known names; otherwise returns a
    generic ``_SentenceTransformerEmbedder`` with conservative
    defaults (BGE-style query prefix, dim=768).  If the
    ``sentence-transformers`` library is missing, falls back to
    :class:`_HashEmbedder` so the app still runs.
    """
    cls = _KNOWN_EMBEDDERS.get(model_name)
    if cls is not None:
        return cls()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning(
            "sentence-transformers unavailable; falling back to hash embedder for %s",
            model_name,
        )
        return _HashEmbedder()
    return _SentenceTransformerEmbedder(
        model_name=model_name,
        query_prefix=_BGE_QUERY_PREFIX,
        dim=768,
    )


def fallback_embedder() -> Embedder:
    """Return a stable hash-based fallback embedder."""
    return _HashEmbedder()
