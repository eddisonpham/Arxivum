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
from typing import Any, Callable, Protocol

from src.config import get_settings
from src.inference.embedder import Embedder, try_resolve
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
        self._resident: str | None = None

    def _acquire(self, name: str, factory: Callable[[], Any]) -> Any:
        """Get or create a model, applying the constrained-memory policy."""
        attr = f"_{name}"
        model = getattr(self, attr)
        if model is None:
            model = factory()
            setattr(self, attr, model)
        if self._constrained and self._resident != name:
            self._unload_others(name)
            model._ensure_loaded()
            self._resident = name
        else:
            model._ensure_loaded()
            if self._resident is None:
                self._resident = name
        return model

    @property
    def embedder(self) -> Embedder:
        return self._acquire("embedder", lambda: try_resolve(self._settings.embedding_model))

    @property
    def reranker(self) -> Reranker:
        return self._acquire("reranker", lambda: Reranker(self._settings.reranker_model))

    @property
    def llm(self) -> LLMProtocol:
        def _make_llm() -> LLMProtocol:
            from src.inference.llm import LocalLLM
            return LocalLLM(
                model_path=self._settings.llm_model_path,
                n_ctx=self._settings.llm_n_ctx,
                n_threads=self._settings.llm_n_threads,
                n_gpu_layers=self._settings.llm_n_gpu_layers,
            )
        return self._acquire("llm", _make_llm)

    def set_llm(self, llm: LLMProtocol) -> None:
        """Inject a custom LLM (e.g. ``StubLLM`` for tests)."""
        self._llm = llm
        self._resident = "llm"

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
        state = {
            "embedder_loaded": self._embedder is not None and self._embedder.is_loaded,
            "reranker_loaded": self._reranker is not None and self._reranker.is_loaded,
            "llm_loaded": self._llm is not None and self._llm.is_loaded,
            "resident": self._resident,
            "constrained_memory": self._constrained,
        }
        try:
            from src.inference.embed_cache import default_cache
            state["embedder_cache"] = default_cache().stats()
        except Exception:  # noqa: BLE001
            state["embedder_cache"] = None
        if self._embedder is not None:
            state["embedder_model"] = self._embedder.model_name
            state["embedder_dim"] = getattr(self._embedder, "dim", None)
        return state

    def shutdown(self) -> None:
        """Unload all models and free memory."""
        for model in (self._embedder, self._reranker, self._llm):
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    logger.debug("Error unloading model", exc_info=True)
        self._resident = None
