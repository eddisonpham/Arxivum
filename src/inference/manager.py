"""Model manager — coordinates lazy loading and memory policy.

On constrained machines (≤ 4 GB) only one heavy model should be resident
at a time.  The :class:`ModelManager` enforces this by unloading the
previous heavy model before loading the next one when
``constrained_memory`` is True (default).

It also provides a single ``StubLLM`` injection point so unit tests can
swap in a deterministic LLM without touching the real GGUF model.
"""

from __future__ import annotations

import logging
from typing import Protocol

from src.config import get_settings
from src.inference.embedder import Embedder
from src.inference.reranker import Reranker

logger = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    """Minimal protocol any LLM (real or stub) must satisfy."""

    def chat(
        self,
        messages: list[dict],
        temperature: float = ...,
        max_tokens: int = ...,
        stop: list[str] | None = ...,
    ) -> str: ...

    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...


class ModelManager:
    """Owns the embedder, reranker, and LLM instances.

    Usage::

        mgr = ModelManager()
        vec = mgr.embedder.embed_one("text")   # loads embedder on demand
        out = mgr.llm.chat([...])              # loads LLM on demand
        mgr.shutdown()                         # free everything
    """

    def __init__(self, constrained_memory: bool = True) -> None:
        settings = get_settings()
        self._constrained = constrained_memory
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._llm: LLMProtocol | None = None
        self._settings = settings
        # Track which heavy model is currently resident.
        self._resident: str | None = None

    # ── embedder ───────────────────────────────────────────────────────
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self._settings.embedding_model)
        if self._constrained and self._resident != "embedder":
            self._unload_others("embedder")
            self._embedder._ensure_loaded()
            self._resident = "embedder"
        else:
            self._embedder._ensure_loaded()
            if self._resident is None:
                self._resident = "embedder"
        return self._embedder

    # ── reranker ───────────────────────────────────────────────────────
    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self._settings.reranker_model)
        if self._constrained and self._resident != "reranker":
            self._unload_others("reranker")
            self._reranker._ensure_loaded()
            self._resident = "reranker"
        else:
            self._reranker._ensure_loaded()
            if self._resident is None:
                self._resident = "reranker"
        return self._reranker

    # ── LLM ────────────────────────────────────────────────────────────
    @property
    def llm(self) -> LLMProtocol:
        if self._llm is None:
            from src.inference.llm import LocalLLM

            self._llm = LocalLLM(
                model_path=self._settings.llm_model_path,
                n_ctx=self._settings.llm_n_ctx,
                n_threads=self._settings.llm_n_threads,
                n_gpu_layers=self._settings.llm_n_gpu_layers,
            )
        if self._constrained and self._resident != "llm":
            self._unload_others("llm")
            self._llm._ensure_loaded()
            self._resident = "llm"
        return self._llm

    def set_llm(self, llm: LLMProtocol) -> None:
        """Inject a custom LLM (e.g. ``StubLLM`` for tests)."""
        self._llm = llm
        self._resident = "llm"

    # ── memory policy ──────────────────────────────────────────────────
    def _unload_others(self, keep: str) -> None:
        """Unload all heavy models except the one we're about to load."""
        if keep != "embedder" and self._embedder is not None and self._embedder.is_loaded:
            logger.debug("Unloading embedder (constrained memory)")
            self._embedder.unload()
        if keep != "reranker" and self._reranker is not None and self._reranker.is_loaded:
            logger.debug("Unloading reranker (constrained memory)")
            self._reranker.unload()
        if keep != "llm" and self._llm is not None and self._llm.is_loaded:
            logger.debug("Unloading LLM (constrained memory)")
            self._llm.unload()
        self._resident = None

    @property
    def resident_model(self) -> str | None:
        """Name of the currently-resident heavy model (for UI indicator)."""
        return self._resident

    @property
    def model_state(self) -> dict:
        """Return load state for all models (used by the visual panel)."""
        return {
            "embedder_loaded": self._embedder is not None and self._embedder.is_loaded,
            "reranker_loaded": self._reranker is not None and self._reranker.is_loaded,
            "llm_loaded": self._llm is not None and self._llm.is_loaded,
            "resident": self._resident,
            "constrained_memory": self._constrained,
        }

    # ── shutdown ───────────────────────────────────────────────────────
    def shutdown(self) -> None:
        """Unload all models and free memory."""
        for model in (self._embedder, self._reranker, self._llm):
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    logger.debug("Error unloading model", exc_info=True)
        self._resident = None
