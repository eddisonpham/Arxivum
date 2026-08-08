"""Local LLM wrapper via ``llama-cpp-python`` (lazy-loaded).

Runs a quantized GGUF model (default: Qwen2.5-1.5B-Instruct Q4_K_M)
entirely on CPU or a small GPU.  The model is only loaded on first use
and can be unloaded to free memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

class LocalLLM:
    """Lazy-loading local LLM backed by llama.cpp."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ) -> None:
        self._model_path = str(model_path)
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    def _ensure_loaded(self) -> None:
        if self._llm is None:
            path = Path(self._model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"LLM model file not found: {self._model_path}\n"
                    f"Run `python scripts/download_models.py` to download it."
                )
            logger.info("Loading LLM: %s (ctx=%d, threads=%d, gpu_layers=%d)",
                        self._model_path, self._n_ctx, self._n_threads, self._n_gpu_layers)
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is not installed. Install it with:\n"
                    '  pip install -e ".[llm]"\n'
                    "On Windows you may need CMake + a C++ compiler.\n"
                    "See README.md for details."
                ) from None

            self._llm = Llama(
                model_path=str(path),
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )

    def chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        """Run a chat completion and return the assistant's text.

        ``messages`` follows the OpenAI chat format::

            [{"role": "system", "content": "..."},
             {"role": "user",   "content": "..."}]
        """
        self._ensure_loaded()
        assert self._llm is not None
        resp = self._llm.create_chat_completion(
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        return resp["choices"][0]["message"]["content"]

    def stream_chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ):
        """Yield text chunks as the model generates them.

        Used by ``/api/v1/library/<id>/summarize/stream`` to give the
        user time-to-first-token.  Falls back to single-yield when
        llama-cpp-python is older than 0.2 and lacks the stream kwarg.
        """
        self._ensure_loaded()
        assert self._llm is not None
        try:
            stream = self._llm.create_chat_completion(
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop or [],
                stream=True,
            )
        except TypeError:
            yield self.chat(messages, temperature=temperature,
                             max_tokens=max_tokens, stop=stop)
            return
        for chunk in stream:
            try:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
            except (KeyError, IndexError, TypeError):
                content = None
            if content:
                yield content

    def unload(self) -> None:
        """Release the model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("LLM unloaded: %s", self._model_path)

class StubLLM:
    """Deterministic stub LLM for unit tests.

    Returns a canned response or calls a user-supplied function, so tests
    never need to download or run the real GGUF model.
    """

    def __init__(
        self,
        responder: Callable[[list[dict]], str] | None = None,
        default: str = '{"overall": "stub summary"}',
    ) -> None:
        self._responder = responder
        self._default = default
        self.calls: list[list[dict]] = []

    @property
    def is_loaded(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        """No-op for interface compatibility with :class:`LocalLLM`."""
        pass

    def chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        msgs = list(messages)
        self.calls.append(msgs)
        if self._responder is not None:
            return self._responder(msgs)
        return self._default

    def unload(self) -> None:
        pass
