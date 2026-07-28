"""Embedding model wrapper (lazy-loaded).

Uses ``BAAI/bge-small-en-v1.5`` (384-d) via ``sentence-transformers``.
The model is only loaded on first use and can be unloaded to free memory.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# BGE models recommend a query prefix for asymmetric retrieval.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Lazy-loading sentence-transformers embedder."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

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
        """Embed a list of texts.

        If ``is_query`` is True, prepend the BGE query prefix for better
        retrieval performance on asymmetric search.
        """
        self._ensure_loaded()
        assert self._model is not None
        inputs = (
            [f"{_QUERY_PREFIX}{t}" for t in texts] if is_query else list(texts)
        )
        vecs = self._model.encode(
            inputs,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_one(
        self, text: str, is_query: bool = False, normalize: bool = True
    ) -> list[float]:
        """Embed a single text and return a 1-D vector."""
        return self.embed([text], is_query=is_query, normalize=normalize)[0]

    @property
    def dim(self) -> int:
        """Return the embedding dimension (loads the model if needed)."""
        self._ensure_loaded()
        assert self._model is not None
        return int(self._model.get_sentence_embedding_dimension())

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Embedder unloaded: %s", self._model_name)
