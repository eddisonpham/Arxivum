"""Reranker model wrapper (lazy-loaded).

Uses ``BAAI/bge-reranker-base`` (cross-encoder) via ``sentence-transformers``.
Only run on a small top-k (≤10) to avoid CPU bottlenecks.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

MAX_RERANK_CANDIDATES = 10

class Reranker:
    """Lazy-loading cross-encoder reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("Loading reranker: %s", self._model_name)
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank candidate texts against a query.

        Returns a list of ``(original_index, score)`` sorted by score
        descending.  Caps candidates at ``MAX_RERANK_CANDIDATES`` to
        protect CPU.
        """
        self._ensure_loaded()
        assert self._model is not None
        capped = list(candidates[:MAX_RERANK_CANDIDATES])
        if not capped:
            return []
        pairs = [(query, c) for c in capped]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            enumerate(float(s) for s in scores),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Reranker unloaded: %s", self._model_name)
